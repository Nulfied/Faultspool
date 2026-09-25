import json
import time

from faultspool import Recorder, Trace, record
from faultspool.capture import StepBudgetExceeded


def add(a, b):
    return a + b


def boom(x):
    raise ValueError(f"bad {x}")


def test_record_ok_logs_every_step():
    def agent(task, tools, recorder):
        recorder.reasoning("adding")
        return tools["add"](task, 2)

    t = record(agent, 1, {"add": add})
    assert t.status == "ok" and t.output == 3
    assert [s.kind for s in t.steps] == ["input", "reasoning", "tool", "output"]
    tool = t.steps[2]
    assert tool.name == "add" and tool.args == {"a": 1, "b": 2} and tool.content == 3
    assert tool.meta["params"] == ["a", "b"]
    assert t.meta["tools"] == ["add"]


def test_tool_exception_is_recorded_and_status_error():
    t = record(lambda task, tools: tools["boom"](task), 5, {"boom": boom})
    assert t.status == "error"
    assert t.steps[1].error == "ValueError: bad 5"
    assert t.steps[-1].kind == "error"


def test_tool_budget_turns_loops_into_timeouts():
    def looper(task, tools):
        while True:
            tools["add"](1, 1)

    t = record(looper, None, {"add": add}, max_tool_calls=4)
    assert t.status == "timeout"
    assert len(t.tool_steps()) == 4


def test_wall_clock_timeout():
    t = record(lambda task, tools: time.sleep(2), None, {}, timeout_s=0.1)
    assert t.status == "timeout"
    assert "Timeout" in t.steps[-1].error


def test_context_manager_and_jsonl_sink(tmp_path):
    sink = tmp_path / "traces.jsonl"
    with Recorder("q", sink=str(sink)) as rec:
        tools = rec.wrap_tools({"add": add})
        tools["add"](2, b=3)
        rec.llm("five", prompt="2+3?", model="local")
        rec.output("5")
    with Recorder("q2", sink=str(sink)) as rec:
        try:
            with rec:
                raise StepBudgetExceeded("x")
        except StepBudgetExceeded:
            pass
    rows = [json.loads(line) for line in sink.read_text().splitlines()]
    assert len(rows) == 2
    t = Trace.from_dict(rows[0])
    assert t.status == "ok" and t.steps[1].args == {"a": 2, "b": 3}
    assert t.steps[2].meta["prompt"] == "2+3?"
    assert rows[1]["status"] == "timeout"


def test_roundtrip_and_task_id_groups_retries():
    a = record(lambda task, tools: "x", {"q": 1}, {})
    b = record(lambda task, tools: "y", {"q": 1}, {})
    assert a.task_id == b.task_id
    assert Trace.from_json(a.to_json()).to_dict() == a.to_dict()
