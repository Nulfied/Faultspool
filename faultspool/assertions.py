"""Assertions a regression test makes about a replayed run."""
from __future__ import annotations

import json
import re

from .detect.rules import check_schema
from .schema import Trace


def _text(v) -> str:
    return v if isinstance(v, str) else json.dumps(v, sort_keys=True, default=str)


def _norm(v):
    return " ".join(v.split()).lower() if isinstance(v, str) else v


def check(assertion: dict, trace: Trace, failures: list) -> tuple:
    """Return (passed, message) for one assertion."""
    t = assertion["type"]
    out = trace.output
    called = [s.name for s in trace.tool_steps()]
    real = [f for f in failures if f.severity != "low"]

    if t == "no_failures":
        return (not real, "no failures" if not real else f"failures: {sorted({f.kind for f in real})}")
    if t == "no_failure":
        hit = sorted({f.kind for f in real if f.kind in assertion["kinds"]})
        return (not hit, "original failure gone" if not hit else f"still has {hit}")
    if t == "output_equals":
        ok = _norm(out) == _norm(assertion["value"])
        return ok, "output matches" if ok else f"expected {assertion['value']!r}, got {out!r}"
    if t == "output_contains":
        ok = _norm(str(assertion["value"])) in _norm(_text(out) or "")
        return ok, f"output {'contains' if ok else 'missing'} {assertion['value']!r}"
    if t == "output_matches":
        ok = re.search(assertion["pattern"], _text(out) or "") is not None
        return ok, f"output {'matches' if ok else 'does not match'} /{assertion['pattern']}/"
    if t == "output_schema":
        err = check_schema(out, assertion["schema"])
        return err is None, err or "output matches schema"
    if t == "tool_called":
        ok = assertion["name"] in called
        return ok, f"{assertion['name']} {'called' if ok else 'never called'}"
    if t == "tool_not_called":
        ok = assertion["name"] not in called
        return ok, f"{assertion['name']} {'not called' if ok else 'was called'}"
    if t == "max_tool_calls":
        ok = len(called) <= assertion["value"]
        return ok, f"{len(called)} tool calls (max {assertion['value']})"
    return False, f"unknown assertion type {t!r}"


def check_all(assertions: list, trace: Trace, failures: list) -> list:
    return [{"assertion": a, "passed": p, "message": m}
            for a in assertions for p, m in [check(a, trace, failures)]]
