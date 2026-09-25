"""The MCP capture -> replay loop. Needs no `mcp` install: the session is duck-typed."""
import asyncio

import pytest

from examples import mcp_agent
from faultspool import detect, is_failing, record, run_test, trace_to_test
from faultspool.integrations.mcp import ReplaySession, is_error, result_text


def capture(agent, city):
    return record(agent, {"city": city}, {}, agent="weather-bot", agent_version="v1")


def test_result_text_and_is_error_duck_type_dicts_and_objects():
    assert result_text({"content": [{"type": "text", "text": "18C"}]}) == "18C"
    assert result_text(mcp_agent.FakeServer._Result("hi")) == "hi"
    assert is_error({"isError": True}) and is_error({"is_error": True})
    assert not is_error({"content": []})
    assert result_text({"structuredContent": {"t": 1}}) == '{"t": 1}'


def test_recording_session_logs_calls_names_and_errors():
    t = capture(mcp_agent.capture_v1, "Berlin")
    tools = t.tool_steps()
    assert [s.name for s in tools] == ["weather.get_forecast"]
    assert tools[0].args == {"city": "Berlin"} and tools[0].meta["transport"] == "mcp"
    assert "Rate limit exceeded" in tools[0].error  # isError result, not an exception
    assert t.meta["tools"] == ["weather.get_forecast", "weather.get_cached"]


def test_isError_result_is_detected_as_a_failure():
    ok = capture(mcp_agent.capture_v1, "Paris")
    assert not is_failing(detect(ok))
    bad = capture(mcp_agent.capture_v1, "Berlin")
    kinds = {f.kind for f in detect(bad) if f.severity != "low"}
    assert kinds == {"tool_error", "unsupported_claim"}


def test_replay_session_serves_recorded_calls_offline():
    t = capture(mcp_agent.capture_v1, "Berlin")
    session = ReplaySession.from_test(trace_to_test(t, detect(t)), server="weather")
    result = asyncio.run(session.call_tool("get_forecast", {"city": "Berlin"}))
    assert is_error(result) and "Rate limit exceeded" in result_text(result)
    listed = asyncio.run(session.list_tools())
    assert {x.name for x in listed.tools} == {"get_forecast", "get_cached"}


def test_full_loop_capture_convert_replay_against_fixed_agent():
    failed = capture(mcp_agent.capture_v1, "Berlin")
    failures = detect(failed)
    # self-play: the fixed agent's successful run supplies get_cached's recording
    retry = record(mcp_agent.capture_v2, {"city": "Berlin"}, {}, agent="weather-bot",
                   agent_version="v2", task_id=failed.task_id)
    assert not is_failing(detect(retry))

    test = trace_to_test(failed, failures, retry)
    assert "weather.get_cached" in test.tool_names
    assert run_test(test, mcp_agent.replay_v1).outcome == "still_fails"
    passed = run_test(test, mcp_agent.replay_v2)
    assert passed.outcome == "passes", passed.detail


def test_unrecorded_call_is_flagged_not_silently_passed():
    t = capture(mcp_agent.capture_v1, "Berlin")
    test = trace_to_test(t, detect(t))  # no retry, so get_cached was never recorded
    r = run_test(test, mcp_agent.replay_v2)
    assert r.outcome == "unreplayable"
    assert r.replay_misses[0]["name"] == "weather.get_cached"


def test_mock_lookup_by_explicit_dict_allows_an_argument_called_name():
    from faultspool.replay import MockTools
    m = MockTools([{"name": "greet", "args": {"name": "ada"}, "result": "hi ada"}])
    assert m.call_with("greet", {"name": "ada"}) == "hi ada"
    with pytest.raises(TypeError):  # why call_with exists: kwargs would collide
        m.call("greet", name="ada")
