import pytest

from faultspool import MockTools, detect, record, trace_to_test
from faultspool.convert import TestCase, diff_traces, failure_cut, find_retry
from faultspool.replay import RecordedToolError, ReplayMiss


def test_mocks_exact_then_by_name_then_miss():
    mocks = [{"name": "get", "args": {"k": 1}, "result": "a", "params": ["k"]},
             {"name": "get", "args": {"k": 1}, "result": "b", "params": ["k"]},
             {"name": "get", "args": {"k": 2}, "result": "c", "params": ["k"]}]
    m = MockTools(mocks)
    assert m.call("get", 1) == "a"          # positional args bound via recorded params
    assert m.call("get", k=1) == "b"        # same call, next recording
    assert m.call("get", k=99) == "c"       # no exact match: next unused by name
    with pytest.raises(ReplayMiss):
        m.call("get", k=1)
    assert m.misses == [{"name": "get", "args": {"k": 1}}]


def test_strict_replay_and_recorded_errors():
    m = MockTools([{"name": "pay", "args": {}, "error": "Declined"}], fallback=None)
    with pytest.raises(RecordedToolError, match="Declined"):
        m.call("pay")
    with pytest.raises(ReplayMiss):
        MockTools([{"name": "pay", "args": {"a": 1}, "result": 1}], fallback=None).call("pay", a=2)


def test_minimal_unit_cuts_at_first_failure(v1_traces):
    t = v1_traces["Grace"]
    fs = detect(t)
    cut = failure_cut(t, fs)
    assert cut < len(t.steps) - 1 or any(f.step_idx is None for f in fs)
    test = trace_to_test(t, fs)
    assert test.input == {"customer": "Grace"}
    assert test.assertions[0]["type"] == "no_failure"
    assert set(test.tool_names) == {"search_orders", "get_order", "refund"}
    assert TestCase.from_dict(test.to_dict()).to_dict() == test.to_dict()


def test_retry_inference_self_play(toy, v1_traces):
    failed = v1_traces["grace"]
    retry = record(toy.run_v2, {"customer": "grace"}, toy.TOOLS)
    assert find_retry(failed, [(failed, detect(failed)), (retry, detect(retry))]) is retry
    d = diff_traces(failed, retry)
    assert d["tools_only_in_failure"] == ["refund"]
    test = trace_to_test(failed, detect(failed), retry)
    assert test.origin == "retry"
    assert {"type": "output_contains", "value": "A-1002"} in test.assertions
    assert any(a["type"] == "max_tool_calls" for a in test.assertions)
