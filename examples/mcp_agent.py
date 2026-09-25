"""An MCP-driven agent, captured from a live session and replayed without one.

The agent logic is one coroutine that only ever talks to an MCP session, so the
same code runs two ways:

* against a real ``ClientSession`` wrapped in ``RecordingSession`` -> captured
* against ``ReplaySession``, which serves the recording back -> replayed in CI

``FakeServer`` stands in for a real MCP server here so the example runs with
nothing installed; swap it for ``mcp.ClientSession`` and the rest is unchanged.

v1's bug is the one MCP makes easy to write: a failed ``tools/call`` comes back
as a normal result with ``isError`` set, not as an exception, so ignoring the
flag silently turns a failure into a confident wrong answer.
"""
from __future__ import annotations

import asyncio

from faultspool.integrations.mcp import RecordingSession, ReplaySession, is_error, result_text

FORECASTS = {"paris": "18C, rain", "oslo": "4C, snow"}
CACHE = {"paris": "17C, rain (cached 1h ago)", "berlin": "21C, clear (cached 3h ago)"}


class FakeServer:
    """Stands in for an MCP ClientSession. get_forecast is rate limited for Berlin."""

    class _Result:
        def __init__(self, text, isError=False):
            self.content = [type("Part", (), {"text": text, "type": "text"})()]
            self.isError = isError

    class _Tool:
        def __init__(self, name):
            self.name, self.description, self.annotations = name, "", None

    async def list_tools(self):
        return type("R", (), {"tools": [self._Tool("get_forecast"), self._Tool("get_cached")]})()

    async def call_tool(self, name, arguments=None):
        city = str((arguments or {}).get("city", "")).lower()
        if name == "get_forecast":
            if city not in FORECASTS:
                return self._Result(f"Rate limit exceeded for {city}", isError=True)
            return self._Result(FORECASTS[city])
        if name == "get_cached":
            if city not in CACHE:
                return self._Result(f"nothing cached for {city}", isError=True)
            return self._Result(CACHE[city])
        return self._Result(f"unknown tool {name}", isError=True)


async def research_v1(session, task) -> str:
    """BUG: never checks isError, so a rate-limited call is reported as a success."""
    await session.list_tools()
    result = await session.call_tool("get_forecast", {"city": task["city"]})
    return f"Successfully checked the forecast for {task['city']}: {result_text(result)}"


async def research_v2(session, task) -> str:
    """Checks isError and falls back to the cache before answering."""
    await session.list_tools()
    result = await session.call_tool("get_forecast", {"city": task["city"]})
    if is_error(result):
        cached = await session.call_tool("get_cached", {"city": task["city"]})
        if is_error(cached):
            return f"I could not get a forecast for {task['city']}: {result_text(cached)}"
        return f"Forecast for {task['city']} (from cache): {result_text(cached)}"
    return f"Forecast for {task['city']}: {result_text(result)}"


def capture(agent_coro, server=None):
    """Agent callable for faultspool.record: talks to a live MCP server."""
    def run(task, tools, recorder=None):
        session = RecordingSession(recorder, server or FakeServer(), server="weather")
        return asyncio.run(agent_coro(session, task))
    return run


def replay(agent_coro):
    """Agent callable for faultspool.run_test: served from the recording."""
    def run(task, tools, recorder=None):
        return asyncio.run(agent_coro(ReplaySession.from_tools(tools, server="weather"), task))
    return run


capture_v1, capture_v2 = capture(research_v1), capture(research_v2)
replay_v1, replay_v2 = replay(research_v1), replay(research_v2)
