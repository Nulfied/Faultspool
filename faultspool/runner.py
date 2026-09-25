"""Regression runner: replay captured failures against the current agent.

Each test ends in one of five outcomes:

  passes             no real failures detected and every assertion holds
  still_fails        at least one of the original failure signatures recurs
  fails_differently  it breaks, but not in the way it originally did
  unreplayable       the agent asked for tool responses that were never recorded
                     and then failed, so the result says nothing either way;
                     record a successful retry or annotate the test
  error              the harness itself could not run the test
"""
from __future__ import annotations

import importlib
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from .assertions import check_all
from .capture import record
from .convert import TestCase
from .detect import detect, signatures
from .replay import MockTools

OUTCOMES = ("passes", "still_fails", "fails_differently", "unreplayable", "error")


@dataclass
class TestResult:
    __test__ = False

    test_id: str
    outcome: str
    failures: list = field(default_factory=list)
    checks: list = field(default_factory=list)
    replay_misses: list = field(default_factory=list)
    trace: Optional[dict] = None
    duration_ms: float = 0.0
    previous: Optional[str] = None
    detail: str = ""

    @property
    def regressed(self) -> bool:
        return self.previous == "passes" and self.outcome != "passes"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["regressed"] = self.regressed
        return d


def load_agent(spec: str) -> Callable:
    """Import ``package.module:function`` (cwd is put on sys.path first)."""
    if ":" not in spec:
        raise ValueError(f"agent spec must look like 'module:function', got {spec!r}")
    mod_name, fn_name = spec.split(":", 1)
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    obj = importlib.import_module(mod_name)
    for part in fn_name.split("."):
        obj = getattr(obj, part)
    return obj


def run_test(test: TestCase, agent_fn: Callable, *, agent_version: str = "dev",
             timeout_s: float = 30.0, max_tool_calls: int = 50, fallback: Optional[str] = "by_name",
             use_judge: bool = False, judge_model: Optional[str] = None,
             previous: Optional[str] = None) -> TestResult:
    t0 = time.perf_counter()
    try:
        mock = MockTools(test.mocks, fallback=fallback)
        tools = mock.as_dict(extra_names=[n for n in test.tool_names if n not in mock.names()])
        trace = record(agent_fn, test.input, tools, timeout_s=timeout_s, max_tool_calls=max_tool_calls,
                       agent=test.agent, agent_version=agent_version, task_id=test.task_id,
                       source="replay", meta={"test_id": test.test_id,
                                              "output_schema": test.meta.get("output_schema")})
        failures = detect(trace, use_judge=use_judge, judge_model=judge_model)
        checks = check_all(test.assertions, trace, failures)
    except Exception as exc:  # noqa: BLE001 - harness failure, not agent failure
        return TestResult(test.test_id, "error", detail=f"{type(exc).__name__}: {exc}",
                          duration_ms=(time.perf_counter() - t0) * 1000, previous=previous)

    new_sigs = signatures(failures)
    all_ok = all(c["passed"] for c in checks)
    if not new_sigs and all_ok:
        outcome, detail = "passes", "no failures, all assertions hold"
    elif mock.misses:
        outcome = "unreplayable"
        detail = "no recording for " + ", ".join(sorted({m["name"] for m in mock.misses}))
    elif new_sigs & test.signatures:
        outcome, detail = "still_fails", "recurring: " + ", ".join(sorted(new_sigs & test.signatures))
    else:
        bad = sorted(new_sigs) + [c["message"] for c in checks if not c["passed"]]
        outcome, detail = "fails_differently", "; ".join(bad)
    return TestResult(test.test_id, outcome, [f.to_dict() for f in failures], checks, mock.misses,
                      trace.to_dict(), (time.perf_counter() - t0) * 1000, previous, detail)


def run_suite(tests: list, agent_fn: Callable, previous: Optional[dict] = None, **kw) -> list:
    previous = previous or {}
    return [run_test(t, agent_fn, previous=previous.get(t.test_id), **kw) for t in tests]


def summarize(results: list) -> dict:
    counts = {o: 0 for o in OUTCOMES}
    for r in results:
        counts[r.outcome] += 1
    total = len(results)
    return {"total": total, **counts,
            "pass_rate": round(counts["passes"] / total, 4) if total else None,
            "regressions": [r.test_id for r in results if r.regressed]}


def should_fail(results: list, fail_on: str) -> bool:
    """CI gate. regression: something that passed before broke. any: anything not passing."""
    if fail_on == "none":
        return False
    if fail_on == "any":
        return any(r.outcome != "passes" for r in results)
    return any(r.regressed or r.outcome == "error" for r in results)
