from faultspool import Trace, detect, is_failing
from faultspool.detect.judge import parse_verdict, render_trace


def kinds(trace, **kw):
    return {f.kind for f in detect(trace, **kw) if f.severity != "low"}


def make(steps, output=None, status="ok", **meta):
    t = Trace(input="task", status=status, meta=meta)
    t.add("input", content="task")
    for s in steps:
        t.add(**s)
    if output is not None:
        t.output = output
        t.add("output", content=output)
    return t


def test_toy_agent_failures(v1_traces):
    assert not is_failing(detect(v1_traces["ada"]))
    assert kinds(v1_traces["Grace"]) == {"loop", "timeout"}
    assert kinds(v1_traces["grace"]) == {"tool_error", "unsupported_claim"}
    assert kinds(v1_traces["linus"]) == {"exception"}


def test_recovered_tool_error_is_low_severity():
    t = make([{"kind": "tool", "name": "fetch", "args": {"u": 1}, "error": "503"},
              {"kind": "tool", "name": "fetch", "args": {"u": 1}, "content": "ok"}], output="got it")
    fs = detect(t)
    assert [f.severity for f in fs if f.kind == "tool_error"] == ["low"]
    assert not is_failing(fs)


def test_malformed_output_and_schema():
    assert "malformed_output" in kinds(make([], output="  "))
    assert "malformed_output" in kinds(make([], output='{"a": 1'))
    schema = {"type": "object", "required": ["answer"]}
    assert "malformed_output" in kinds(make([], output={"x": 1}, output_schema=schema))
    assert not kinds(make([], output={"answer": 1}, output_schema=schema))


def test_cycle_loop_detection():
    steps = [{"kind": "tool", "name": n, "args": {}, "content": 1} for n in ["a", "b"] * 3]
    fs = [f for f in detect(make(steps, output="ok")) if f.kind == "loop"]
    assert any("cycle a->b" in f.message for f in fs)


def test_self_contradiction():
    steps = [{"kind": "tool", "name": "create_file", "args": {"path": "x"}, "content": True},
             {"kind": "tool", "name": "delete_file", "args": {"path": "x"}, "content": True}]
    assert "self_contradiction" in kinds(make(steps, output="All set"))


def test_abandonment():
    assert "abandoned" in kinds(make([], output="I cannot complete this task."))
    assert "abandoned" in kinds(make([{"kind": "tool", "name": "x", "args": {}, "content": 1}]))


def test_ground_truth_label():
    t = make([], output="42", benchmark_success=False)
    assert "wrong_answer" in kinds(t)


def test_judge_with_injected_client():
    t = make([{"kind": "tool", "name": "search", "args": {"q": "paris"}, "content": "rain"}], output="Sunny!")
    client = lambda prompt: '{"off_track": true, "step": 2, "reason": "ignored tool result"}'  # noqa: E731
    fs = detect(t, use_judge=True, judge_client=client)
    judged = [f for f in fs if f.kind == "off_track"]
    assert judged and judged[0].step_idx == 2 and judged[0].detector == "judge"
    ok = detect(t, use_judge=True, judge_client=lambda p: '{"off_track": false}')
    assert not [f for f in ok if f.kind == "off_track"]
    assert "TOOL search" in render_trace(t)


def test_parse_verdict_tolerates_chatter():
    assert parse_verdict('Sure! {"off_track": false, "step": null} hope that helps')["off_track"] is False
    assert parse_verdict("no json here") is None
