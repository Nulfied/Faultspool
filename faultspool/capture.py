"""Trace capture: hook an agent's execution and log every step.

Two ways in:

* ``record(agent_fn, task, tools)`` runs an agent callable with instrumented
  tools and returns a finished ``Trace`` (timeouts and exceptions included).
* ``Recorder`` for frameworks where you own the loop: wrap tools, and call
  ``rec.llm(...)``, ``rec.reasoning(...)``, ``rec.output(...)`` yourself.

The agent contract used everywhere in Faultspool is simply::

    def agent(task, tools: dict[str, Callable]) -> output

If the agent also accepts a ``recorder`` keyword it is passed the live Recorder,
so it can log reasoning and model calls.
"""
from __future__ import annotations

import functools
import inspect
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .schema import Trace, write_jsonl


class StepBudgetExceeded(RuntimeError):
    """Raised inside a tool call once the agent exceeds its tool-call budget."""


class RecorderClosed(RuntimeError):
    """Raised when a timed-out agent thread keeps calling tools."""


def _param_names(fn: Callable) -> Optional[list]:
    try:
        return [p.name for p in inspect.signature(fn).parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    except (TypeError, ValueError):
        return None


def canonical_args(params: Optional[list], args: tuple, kwargs: dict) -> dict:
    """Turn a call's positional + keyword args into one name->value dict.

    Recording and replay both go through here so recorded calls and replayed
    calls produce identical keys.
    """
    out = {}
    names = params or []
    for i, value in enumerate(args):
        out[names[i] if i < len(names) else f"_{i}"] = value
    out.update(kwargs)
    return out


def accepts_kwarg(fn: Callable, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind == p.VAR_KEYWORD for p in params.values())


class Recorder:
    def __init__(self, task: Any = None, *, agent: str = "agent", agent_version: str = "0",
                 task_id: Optional[str] = None, source: str = "live",
                 max_tool_calls: Optional[int] = None, sink: Any = None,
                 meta: Optional[dict] = None):
        self.trace = Trace(input=task, agent=agent, agent_version=str(agent_version),
                           task_id=task_id, source=source, meta=dict(meta or {}))
        self.trace.add("input", content=task)
        self.max_tool_calls = max_tool_calls
        self.sink = sink
        self.closed = False
        self._lock = threading.Lock()

    # -- instrumentation ---------------------------------------------------
    def tool(self, name: str, fn: Callable) -> Callable:
        params = _param_names(fn)

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            if self.closed:
                raise RecorderClosed(f"recorder closed; call to {name} dropped")
            call_args = canonical_args(params, args, kwargs)
            with self._lock:
                n_calls = len(self.trace.tool_steps())
            if self.max_tool_calls is not None and n_calls >= self.max_tool_calls:
                raise StepBudgetExceeded(f"tool-call budget of {self.max_tool_calls} exhausted")
            t0 = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                self._add("tool", name=name, args=call_args, error=f"{type(exc).__name__}: {exc}",
                          duration_ms=(time.perf_counter() - t0) * 1000, meta={"params": params})
                raise
            self._add("tool", name=name, args=call_args, content=result,
                      duration_ms=(time.perf_counter() - t0) * 1000, meta={"params": params})
            return result

        return wrapped

    def wrap_tools(self, tools: dict) -> dict:
        self.trace.meta.setdefault("tools", sorted(tools))
        return {name: self.tool(name, fn) for name, fn in tools.items()}

    def llm(self, response: Any, prompt: Any = None, model: Optional[str] = None, **meta) -> None:
        if prompt is not None:
            meta["prompt"] = prompt
        self._add("llm", name=model, content=response, meta=meta)

    def reasoning(self, text: str) -> None:
        self._add("reasoning", content=text)

    def output(self, value: Any) -> None:
        self.trace.output = value
        self._add("output", content=value)

    def error(self, exc: BaseException) -> None:
        self._add("error", error=f"{type(exc).__name__}: {exc}")

    def _add(self, kind: str, **kw) -> None:
        with self._lock:
            if not self.closed:
                self.trace.add(kind, **kw)

    # -- lifecycle --------------------------------------------------------
    def finish(self, status: Optional[str] = None) -> Trace:
        with self._lock:
            if self.closed:
                return self.trace
            self.closed = True
        t = self.trace
        t.ended_at = time.time()
        if status:
            t.status = status
        elif t.status == "incomplete":
            t.status = "ok" if any(s.kind == "output" for s in t.steps) else "incomplete"
        emit(t, self.sink)
        return t

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.error(exc)
            self.finish("timeout" if isinstance(exc, StepBudgetExceeded) else "error")
        else:
            self.finish()
        return False


def emit(trace: Trace, sink: Any) -> None:
    """Send a finished trace to a sink: a JSONL path, a Store, or any callable."""
    if sink is None:
        return
    if isinstance(sink, (str, Path)):
        Path(sink).parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(sink, [trace.to_dict()], append=True)
    elif hasattr(sink, "add_trace"):
        sink.add_trace(trace)
    elif callable(sink):
        sink(trace)
    else:
        raise TypeError(f"unsupported sink {sink!r}")


def record(agent_fn: Callable, task: Any, tools: dict, *, timeout_s: Optional[float] = None,
           sink: Any = None, **recorder_kw) -> Trace:
    """Run ``agent_fn(task, tools)`` with every tool call logged; never raises."""
    rec = Recorder(task, sink=None, **recorder_kw)
    wrapped = rec.wrap_tools(tools)
    kwargs = {"recorder": rec} if accepts_kwarg(agent_fn, "recorder") else {}
    box: dict = {}

    def target():
        try:
            box["out"] = agent_fn(task, wrapped, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - we record everything
            box["exc"] = exc

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(timeout_s)

    if th.is_alive():
        rec._add("error", error=f"Timeout: agent exceeded {timeout_s}s")
        status = "timeout"
    elif "exc" in box:
        rec.error(box["exc"])
        status = "timeout" if isinstance(box["exc"], StepBudgetExceeded) else "error"
    else:
        if not any(s.kind == "output" for s in rec.trace.steps):
            rec.output(box.get("out"))
        status = "ok"
    trace = rec.finish(status)
    emit(trace, sink)
    return trace
