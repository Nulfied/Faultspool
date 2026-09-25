"""Deterministic tool replay: recorded tool responses served back as mocks.

Lookup order for each call the agent makes during replay:

1. exact match on tool name + canonical args, consumed in recorded order
   (so the same call can return different results on its 1st and 2nd use);
2. with ``fallback="by_name"``, the next unused recording of that tool name;
3. otherwise ``ReplayMiss`` is raised inside the agent, which shows up in the
   new trace as a tool error. That is deliberate: a test should not silently
   pass because the agent wandered somewhere the recording never went.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional

from .capture import canonical_args
from .schema import stable_hash


class ReplayMiss(LookupError):
    pass


class RecordedToolError(RuntimeError):
    """Re-raised in place of an exception that was recorded from the real tool."""


class MockTools:
    def __init__(self, mocks: list, fallback: Optional[str] = "by_name"):
        self.fallback = fallback
        self._exact = defaultdict(deque)
        self._by_name = defaultdict(deque)
        self._params: dict = {}
        self.misses: list = []
        for m in mocks:
            key = (m["name"], stable_hash(m.get("args") or {}))
            self._exact[key].append(m)
            self._by_name[m["name"]].append(m)
            if m.get("params"):
                self._params[m["name"]] = m["params"]

    def names(self) -> list:
        return list(self._by_name)

    def _take(self, name: str, args: dict) -> dict:
        q = self._exact.get((name, stable_hash(args)))
        if q:
            m = q.popleft()
            try:
                self._by_name[name].remove(m)
            except ValueError:
                pass
            return m
        if self.fallback == "by_name" and self._by_name.get(name):
            m = self._by_name[name].popleft()
            key = (name, stable_hash(m.get("args") or {}))
            try:
                self._exact[key].remove(m)
            except ValueError:
                pass
            return m
        self.misses.append({"name": name, "args": args})
        raise ReplayMiss(f"no recorded response for {name}({args})")

    def call(self, name: str, *args, **kwargs):
        return self.call_with(name, canonical_args(self._params.get(name), args, kwargs))

    def call_with(self, name: str, args: dict):
        """Look a call up by an explicit argument dict.

        Callers that already have the arguments as a mapping (an MCP
        ``tools/call``, say) use this instead of ``call``, where an argument
        named ``name`` would collide with the positional parameter.
        """
        m = self._take(name, dict(args or {}))
        if m.get("error"):
            raise RecordedToolError(m["error"])
        return m.get("result")

    def as_dict(self, extra_names=()) -> dict:
        """Tool callables keyed by name, ready to hand to an agent."""
        def make(n):
            def tool(*args, **kwargs):
                return self.call(n, *args, **kwargs)
            tool.__name__ = n
            return tool
        return {n: make(n) for n in list(self._by_name) + list(extra_names)}


def mocks_from_steps(steps) -> list:
    return [{"name": s.name, "args": s.args or {}, "result": s.content, "error": s.error,
             "params": (s.meta or {}).get("params")} for s in steps if s.kind == "tool"]
