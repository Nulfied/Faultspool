"""MCP client adapter: capture every ``tools/call`` an agent makes, then replay
those same calls with no server running.

Capture, against a live MCP server::

    from mcp import ClientSession
    from faultspool import Recorder
    from faultspool.integrations.mcp import RecordingSession

    with Recorder(task, agent="researcher", sink=store) as rec:
        async with ClientSession(read, write) as raw:
            await raw.initialize()
            session = RecordingSession(rec, raw, server="github")
            await session.list_tools()
            await session.call_tool("create_issue", {...})   # recorded
            rec.output(answer)

Replay, with the server switched off::

    from faultspool.integrations.mcp import ReplaySession

    session = ReplaySession.from_test(test)          # standalone
    session = ReplaySession.from_tools(tools)        # inside faultspool.run_test

Nothing here imports ``mcp``. Results are read by duck typing (attribute or
dict, camelCase or snake_case), so this works against a real ``ClientSession``,
a fake in a test, or a dict decoded straight off the wire.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..replay import MockTools, RecordedToolError, ReplayMiss


def _get(obj: Any, *names: str) -> Any:
    """First present attribute/key out of `names` (for isError vs is_error)."""
    for name in names:
        if isinstance(obj, dict):
            if name in obj:
                return obj[name]
        elif hasattr(obj, name):
            return getattr(obj, name)
    return None


def is_error(result: Any) -> bool:
    return bool(_get(result, "isError", "is_error"))


def result_text(result: Any) -> str:
    """Flatten an MCP tool result into the text an agent would read."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    structured = _get(result, "structuredContent", "structured_content")
    parts = _get(result, "content") or []
    texts = []
    for part in parts if isinstance(parts, (list, tuple)) else [parts]:
        text = _get(part, "text")
        if text is not None:
            texts.append(text)
        elif part is not None:
            texts.append(json.dumps(part, default=str, ensure_ascii=False)
                         if isinstance(part, dict) else str(part))
    if texts:
        return "\n".join(texts)
    if structured is not None:
        return json.dumps(structured, default=str, ensure_ascii=False)
    return str(result)


# --------------------------------------------------------------------- capture


class RecordingSession:
    """Wraps an MCP ``ClientSession``; every call but ``call_tool`` passes through.

    A tool result with ``isError`` set is recorded as a failed step, which is
    what MCP servers use to report a tool failure instead of raising.
    """

    def __init__(self, recorder: Any, session: Any, *, server: str = "mcp", qualify: bool = True):
        self._rec, self._session, self._server, self._qualify = recorder, session, server, qualify

    def _name(self, tool: str) -> str:
        return f"{self._server}.{tool}" if self._qualify else tool

    async def list_tools(self, *args: Any, **kwargs: Any) -> Any:
        result = await self._session.list_tools(*args, **kwargs)
        names = [self._name(_get(t, "name")) for t in (_get(result, "tools") or [])]
        if names:
            known = self._rec.trace.meta.setdefault("tools", [])
            known.extend(n for n in names if n not in known)
        return result

    async def call_tool(self, name: str, arguments: Optional[dict] = None,
                        *args: Any, **kwargs: Any) -> Any:
        recorded, t0 = self._name(name), time.perf_counter()
        try:
            result = await self._session.call_tool(name, arguments or {}, *args, **kwargs)
        except Exception as exc:
            self._rec._add("tool", name=recorded, args=dict(arguments or {}),
                           error=f"{type(exc).__name__}: {exc}",
                           duration_ms=(time.perf_counter() - t0) * 1000,
                           meta={"server": self._server, "transport": "mcp"})
            raise
        text = result_text(result)
        self._rec._add("tool", name=recorded, args=dict(arguments or {}),
                       content=text, error=text if is_error(result) else None,
                       duration_ms=(time.perf_counter() - t0) * 1000,
                       meta={"server": self._server, "transport": "mcp"})
        return result

    def __getattr__(self, item: str) -> Any:
        return getattr(self._session, item)


# ---------------------------------------------------------------------- replay


@dataclass
class TextPart:
    text: str
    type: str = "text"


@dataclass
class ReplayedResult:
    """The shape of an MCP ``CallToolResult``, rebuilt from a recording."""

    content: list = field(default_factory=list)
    isError: bool = False

    @property
    def is_error(self) -> bool:
        return self.isError

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.content)


@dataclass
class ToolInfo:
    name: str
    description: str = ""


@dataclass
class ListToolsResult:
    tools: list = field(default_factory=list)


class ReplaySession:
    """Serves recorded MCP responses back, so a captured run replays offline.

    Build it from a test's recordings (``from_test``) to drive an agent
    directly, or from the tool callables the regression runner hands the agent
    (``from_tools``), which keeps replay misses visible to the runner.
    """

    def __init__(self, mocks: Optional[list] = None, *, server: str = "mcp", qualify: bool = True,
                 fallback: Optional[str] = "by_name", invoke: Optional[Callable] = None,
                 names: Optional[list] = None):
        self._server, self._qualify = server, qualify
        self._mock = MockTools(mocks or [], fallback=fallback) if invoke is None else None
        self._invoke = invoke or (lambda n, a: self._mock.call_with(n, a))
        self._names = names if names is not None else (self._mock.names() if self._mock else [])

    @classmethod
    def from_test(cls, test: Any, **kw) -> "ReplaySession":
        return cls(test.mocks, names=list(getattr(test, "tool_names", []) or []), **kw)

    @classmethod
    def from_tools(cls, tools: dict, **kw) -> "ReplaySession":
        """Wrap the ``{name: callable}`` dict passed to an agent under replay."""
        return cls(invoke=lambda n, a: tools[n](**(a or {})), names=sorted(tools), **kw)

    def _name(self, tool: str) -> str:
        return f"{self._server}.{tool}" if self._qualify else tool

    @property
    def misses(self) -> list:
        return self._mock.misses if self._mock else []

    async def list_tools(self, *args: Any, **kwargs: Any) -> ListToolsResult:
        prefix = f"{self._server}." if self._qualify else ""
        return ListToolsResult([ToolInfo(n[len(prefix):] if n.startswith(prefix) else n)
                                for n in self._names])

    async def call_tool(self, name: str, arguments: Optional[dict] = None,
                        *args: Any, **kwargs: Any) -> ReplayedResult:
        recorded = self._name(name)
        try:
            result = self._invoke(recorded, dict(arguments or {}))
        except RecordedToolError as exc:
            # The server reported this as a tool error, so replay it as one too.
            return ReplayedResult([TextPart(str(exc))], isError=True)
        except KeyError:
            raise ReplayMiss(f"no recorded response for {recorded}({arguments or {}})") from None
        text = result if isinstance(result, str) else json.dumps(result, default=str, ensure_ascii=False)
        return ReplayedResult([TextPart(text)])
