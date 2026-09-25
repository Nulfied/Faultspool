"""Rule-based failure detection: things that are unambiguously broken."""
from __future__ import annotations

import json
from typing import Optional

from ..schema import Failure, Trace


def detect_exceptions(trace: Trace) -> list:
    if trace.status == "timeout":
        return []  # the budget/timeout error step is reported by detect_timeout
    out = []
    for s in trace.steps:
        if s.kind == "error" and not (s.error or "").startswith("Timeout"):
            out.append(Failure("exception", "rule", s.error or "error step", s.idx, severity="high"))
    if trace.status == "error" and not out:
        out.append(Failure("exception", "rule", "trace ended with status=error", severity="high"))
    return out


def detect_tool_errors(trace: Trace) -> list:
    """Every tool error; ones the agent later recovered from are marked low severity."""
    tools = trace.tool_steps()
    out = []
    for i, s in enumerate(tools):
        if not s.error:
            continue
        recovered = any(t.name == s.name and not t.error for t in tools[i + 1:])
        out.append(Failure("tool_error", "rule", s.error, s.idx, tool=s.name,
                           severity="low" if recovered else "medium"))
    return out


def detect_timeout(trace: Trace, budget_s: Optional[float] = None) -> list:
    if trace.status == "timeout":
        msg = next((s.error for s in reversed(trace.steps) if s.error), "trace timed out")
        return [Failure("timeout", "rule", msg, severity="high")]
    budget_s = budget_s or trace.meta.get("budget_s")
    if budget_s and trace.duration_s and trace.duration_s > budget_s:
        return [Failure("timeout", "rule", f"took {trace.duration_s:.1f}s > budget {budget_s}s")]
    return []


def _type_ok(value, expected: str) -> bool:
    return {
        "string": isinstance(value, str), "number": isinstance(value, (int, float)),
        "integer": isinstance(value, int), "boolean": isinstance(value, bool),
        "object": isinstance(value, dict), "array": isinstance(value, list),
        "null": value is None,
    }.get(expected, True)


def check_schema(value, schema: dict) -> Optional[str]:
    """Tiny JSON-schema subset: type, required, properties (recursive), enum."""
    if "type" in schema and not _type_ok(value, schema["type"]):
        return f"expected {schema['type']}, got {type(value).__name__}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{value!r} not in {schema['enum']}"
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                return f"missing required key {key!r}"
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                err = check_schema(value[key], sub)
                if err:
                    return f"{key}: {err}"
    return None


def detect_malformed_output(trace: Trace) -> list:
    if trace.status in ("error", "timeout"):
        return []  # already covered; no output is expected
    out = trace.output
    if out is None or (isinstance(out, str) and not out.strip()):
        return [Failure("malformed_output", "rule", "empty output")]
    if isinstance(out, str) and out.lstrip()[:1] in "{[":
        try:
            out = json.loads(out)
        except ValueError as e:
            return [Failure("malformed_output", "rule", f"output looks like JSON but does not parse: {e}")]
    schema = trace.meta.get("output_schema")
    if schema:
        err = check_schema(out, schema)
        if err:
            return [Failure("malformed_output", "rule", f"schema violation: {err}")]
    return []


def detect_ground_truth(trace: Trace) -> list:
    """Imported benchmark runs carry their own pass/fail label."""
    if trace.meta.get("benchmark_success") is False:
        return [Failure("wrong_answer", "rule", "benchmark ground truth marks this run as failed",
                        severity="high")]
    return []


def run_rules(trace: Trace, budget_s: Optional[float] = None) -> list:
    return (detect_exceptions(trace) + detect_tool_errors(trace) + detect_timeout(trace, budget_s)
            + detect_malformed_output(trace) + detect_ground_truth(trace))
