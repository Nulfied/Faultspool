"""Heuristic failure detection: behaviour that is usually, not always, wrong."""
from __future__ import annotations

import re

from ..schema import Failure, Trace

# Pairs of tool-name verbs that undo each other.
INVERSE_VERBS = [
    ("create", "delete"), ("add", "remove"), ("open", "close"), ("insert", "delete"),
    ("enable", "disable"), ("start", "stop"), ("lock", "unlock"), ("install", "uninstall"),
    ("book", "cancel"), ("subscribe", "unsubscribe"), ("write", "delete"),
]

GIVE_UP = re.compile(
    r"\b(i (can ?not|can't|am unable to|was unable to|could not|couldn't)|unable to (complete|find|proceed)"
    r"|giving up|i give up|not possible to complete|task (failed|aborted))\b",
    re.I,
)
SUCCESS_CLAIM = re.compile(r"\b(success(fully)?|done|completed|finished|resolved|refunded|booked|sent)\b", re.I)


def detect_loops(trace: Trace, repeat: int = 3) -> list:
    """Same call (name + args) repeated, or a short cycle of calls repeating."""
    keys = [s.call_key for s in trace.tool_steps()]
    tools = trace.tool_steps()
    out, seen = [], set()
    # identical call repeated `repeat` times anywhere
    counts: dict = {}
    for s in tools:
        counts[s.call_key] = counts.get(s.call_key, 0) + 1
        if counts[s.call_key] == repeat and s.call_key not in seen:
            seen.add(s.call_key)
            out.append(Failure("loop", "heuristic",
                               f"{s.name} called {repeat}+ times with identical args", s.idx, tool=s.name))
    # cycles of period 2..4 repeated `repeat` times (A B A B A B)
    for period in (2, 3, 4):
        span = period * repeat
        for i in range(len(keys) - span + 1):
            window = keys[i:i + span]
            if len(set(window[:period])) == period and all(window[j] == window[j % period] for j in range(span)):
                names = "->".join(t.name for t in tools[i:i + period])
                if names not in seen:
                    seen.add(names)
                    out.append(Failure("loop", "heuristic", f"cycle {names} repeated {repeat}x",
                                       tools[i].idx, tool=tools[i].name))
                break
    return out


def _verb_obj(name: str):
    parts = re.split(r"[_\-.\s]+|(?<=[a-z])(?=[A-Z])", name or "")
    parts = [p.lower() for p in parts if p]
    return (parts[0], "_".join(parts[1:])) if parts else ("", "")


def detect_contradictions(trace: Trace) -> list:
    out = []
    tools = trace.tool_steps()
    # 1) an action later undone by its inverse on the same object and args
    for i, a in enumerate(tools):
        va, oa = _verb_obj(a.name)
        for b in tools[i + 1:]:
            vb, ob = _verb_obj(b.name)
            if oa == ob and ((va, vb) in INVERSE_VERBS) and not a.error and not b.error:
                shared = set((a.args or {}).items()) & set((b.args or {}).items()) if _hashable(a, b) else set()
                if shared or not (a.args or b.args):
                    out.append(Failure("self_contradiction", "heuristic",
                                       f"{b.name} undid earlier {a.name} (step {a.idx})", b.idx, tool=b.name))
                    break
    # 2) final answer claims success although the last attempt of some tool failed
    output = trace.output if isinstance(trace.output, str) else ""
    if output and SUCCESS_CLAIM.search(output) and not GIVE_UP.search(output):
        last_by_tool: dict = {}
        for s in tools:
            last_by_tool[s.name] = s
        failed = [s for s in last_by_tool.values() if s.error]
        if failed:
            f = failed[-1]
            out.append(Failure("unsupported_claim", "heuristic",
                               f"output claims success but last {f.name} call failed: {f.error}",
                               f.idx, tool=f.name, severity="high"))
    return out


def _hashable(a, b) -> bool:
    try:
        set((a.args or {}).items()) | set((b.args or {}).items())
        return True
    except TypeError:
        return False


def detect_abandonment(trace: Trace) -> list:
    if trace.status in ("error", "timeout"):
        return []
    has_output = any(s.kind == "output" for s in trace.steps)
    if not has_output:
        return [Failure("abandoned", "heuristic", "trace ended without a final output")]
    out = trace.output
    if isinstance(out, str) and GIVE_UP.search(out):
        return [Failure("abandoned", "heuristic", f"agent gave up: {out[:120]!r}")]
    if isinstance(out, str) and SUCCESS_CLAIM.search(out):
        return []  # a success claim after a failed call is reported as unsupported_claim
    non_meta = [s for s in trace.steps if s.kind not in ("input", "output")]
    if non_meta and non_meta[-1].kind == "tool" and non_meta[-1].error:
        s = non_meta[-1]
        return [Failure("abandoned", "heuristic", f"stopped right after failed {s.name} call", s.idx, tool=s.name)]
    return []


def run_heuristics(trace: Trace, loop_repeat: int = 3) -> list:
    return detect_loops(trace, loop_repeat) + detect_contradictions(trace) + detect_abandonment(trace)
