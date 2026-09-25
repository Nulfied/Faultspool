"""Trace -> regression test conversion.

A test case is the minimal reproducible unit of a failure:

* the task input,
* recorded tool responses up to (and including) the first failing step, served
  back as mocks so the test needs no live APIs,
* assertions describing "correct": at minimum that the original failure kinds do
  not recur; richer ones are inferred from a successful retry of the same task
  (self-play) or supplied by a human during annotation.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .detect import is_failing
from .replay import mocks_from_steps
from .schema import Failure, Trace, new_id, stable_hash


@dataclass
class TestCase:
    __test__ = False  # keep pytest from collecting this class

    input: Any
    mocks: list
    assertions: list
    failures: list                       # failure dicts from the source trace
    source_trace_id: str
    task_id: str
    agent: str = "agent"
    tool_names: list = field(default_factory=list)
    test_id: str = field(default_factory=lambda: new_id("t_"))
    status: str = "pending"              # pending | confirmed | rejected
    origin: str = "inferred"             # inferred | retry | human
    cluster_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    notes: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def signatures(self) -> set:
        return {f["signature"] for f in self.failures if f.get("severity") != "low"}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TestCase":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


def failure_cut(trace: Trace, failures: list) -> int:
    """Index of the last step needed to reproduce the earliest real failure."""
    idxs = [f.step_idx for f in failures if f.severity != "low" and f.step_idx is not None]
    has_unlocated = any(f.step_idx is None for f in failures if f.severity != "low")
    if not idxs or has_unlocated:
        return len(trace.steps) - 1
    return min(idxs)


def diff_traces(failed: Trace, succeeded: Trace) -> dict:
    """What the successful attempt did differently."""
    fk = [s.call_key for s in failed.tool_steps()]
    sk = [s.call_key for s in succeeded.tool_steps()]
    diverge = next((i for i, (a, b) in enumerate(zip(fk, sk)) if a != b), min(len(fk), len(sk)))
    fn = {s.name for s in failed.tool_steps()}
    sn = {s.name for s in succeeded.tool_steps()}
    return {
        "first_divergent_call": diverge,
        "tools_only_in_success": sorted(sn - fn),
        "tools_only_in_failure": sorted(fn - sn),
        "failed_tool_calls": len(fk),
        "success_tool_calls": len(sk),
        "failed_output": failed.output,
        "success_output": succeeded.output,
    }


_FACT = re.compile(r"\b[A-Za-z_#-]*\d[\w.\-]*\b")


def _grounded_facts(output: str, trace: Trace, limit: int = 3) -> list:
    """Id/number-like tokens in the output that also appear in tool results."""
    evidence = " ".join(str(s.content) for s in trace.tool_steps())
    facts = []
    for tok in _FACT.findall(output):
        if tok in evidence and tok not in facts:
            facts.append(tok)
    return facts[:limit]


def infer_assertions(trace: Trace, failures: list, retry: Optional[Trace] = None) -> tuple:
    kinds = sorted({f.kind for f in failures if f.severity != "low"})
    assertions = [{"type": "no_failure", "kinds": kinds}] if kinds else []
    if retry is None:
        return assertions, "inferred", None
    d = diff_traces(trace, retry)
    out = retry.output
    if isinstance(out, str):
        facts = _grounded_facts(out, retry)
        if facts:
            assertions += [{"type": "output_contains", "value": f} for f in facts]
        elif len(out) <= 80:
            assertions.append({"type": "output_equals", "value": out})
    elif out is not None:
        assertions.append({"type": "output_equals", "value": out})
    assertions += [{"type": "tool_called", "name": n} for n in d["tools_only_in_success"]]
    assertions.append({"type": "max_tool_calls", "value": max(3, d["success_tool_calls"] * 2)})
    return assertions, "retry", d


def find_retry(trace: Trace, others: list) -> Optional[Trace]:
    """A later successful attempt at the same task. `others` is [(trace, failures)]."""
    cands = [t for t, fs in others
             if t.task_id == trace.task_id and t.trace_id != trace.trace_id
             and t.started_at >= trace.started_at and t.status == "ok" and not is_failing(fs)]
    return min(cands, key=lambda t: t.started_at) if cands else None


def trace_to_test(trace: Trace, failures: list, retry: Optional[Trace] = None) -> TestCase:
    cut = failure_cut(trace, failures)
    mocks = mocks_from_steps(trace.steps[: cut + 1])
    if retry is not None:  # the fixed behaviour may need responses the failing run never saw
        seen = {(m["name"], stable_hash(m["args"]), stable_hash(m["result"])) for m in mocks}
        for m in mocks_from_steps(retry.steps):
            k = (m["name"], stable_hash(m["args"]), stable_hash(m["result"]))
            if k not in seen:
                seen.add(k)
                mocks.append(m)
    assertions, origin, d = infer_assertions(trace, failures, retry)
    names = set(trace.meta.get("tools") or []) | {m["name"] for m in mocks}
    return TestCase(
        input=trace.input, mocks=mocks, assertions=assertions,
        failures=[f.to_dict() for f in failures], source_trace_id=trace.trace_id,
        task_id=trace.task_id, agent=trace.agent, tool_names=sorted(names), origin=origin,
        meta={"cut_step": cut, "source": trace.source, "agent_version": trace.agent_version,
              "retry_trace_id": retry.trace_id if retry else None, "diff": d,
              "output_schema": trace.meta.get("output_schema")},
    )


def failures_from_dicts(rows: list) -> list:
    return [Failure.from_dict(r) for r in rows]
