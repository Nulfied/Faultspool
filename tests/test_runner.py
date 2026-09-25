from faultspool import detect, record, run_suite, run_test, trace_to_test
from faultspool.assertions import check
from faultspool.runner import load_agent, should_fail, summarize


def build_tests(toy, v1_traces, with_retries=True):
    out = []
    for name in ("Grace", "grace", "linus"):
        t = v1_traces[name]
        retry = record(toy.run_v2, t.input, toy.TOOLS) if with_retries else None
        out.append(trace_to_test(t, detect(t), retry))
    return out


def test_v1_still_fails_v2_passes(toy, v1_traces):
    tests = build_tests(toy, v1_traces)
    assert {r.outcome for r in run_suite(tests, toy.run_v1, max_tool_calls=8)} == {"still_fails"}
    assert {r.outcome for r in run_suite(tests, toy.run_v2, max_tool_calls=8)} == {"passes"}


def test_missing_recording_is_unreplayable_not_still_failing(toy, v1_traces):
    t = v1_traces["linus"]  # v1 crashed before refund(), so there is no recording of it
    r = run_test(trace_to_test(t, detect(t)), toy.run_v2)
    assert r.outcome == "unreplayable" and r.replay_misses[0]["name"] == "refund"


def test_fails_differently(toy, v1_traces):
    t = v1_traces["linus"]
    test = trace_to_test(t, detect(t))

    def v3(task, tools):
        tools["get_order"](tools["search_orders"](task["customer"])[0])
        raise TypeError("new bug")

    r = run_test(test, v3)
    assert r.outcome == "fails_differently" and "exception:TypeError" in r.detail


def test_regression_gate(toy, v1_traces):
    tests = build_tests(toy, v1_traces)
    previous = {t.test_id: "passes" for t in tests}
    results = run_suite(tests, toy.run_v1, previous, max_tool_calls=8)
    assert all(r.regressed for r in results)
    assert should_fail(results, "regression") and not should_fail(results, "none")
    s = summarize(results)
    assert s["still_fails"] == 3 and s["pass_rate"] == 0 and len(s["regressions"]) == 3


def test_assertion_types(toy):
    t = record(toy.run_v2, {"customer": "ada"}, toy.TOOLS)
    assert check({"type": "output_contains", "value": "a-1001"}, t, [])[0]
    assert check({"type": "output_matches", "pattern": r"R-\d+"}, t, [])[0]
    assert check({"type": "tool_called", "name": "refund"}, t, [])[0]
    assert not check({"type": "tool_not_called", "name": "refund"}, t, [])[0]
    assert check({"type": "max_tool_calls", "value": 3}, t, [])[0]
    assert not check({"type": "nope"}, t, [])[0]


def test_load_agent():
    assert load_agent("examples.toy_agent:run_v2").__name__ == "run_v2"
