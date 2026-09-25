"""Failure detection: rules, heuristics, and an optional local LLM judge."""
from __future__ import annotations

from typing import Optional

from ..schema import Failure, Trace
from .heuristics import run_heuristics
from .judge import judge, ollama_available
from .rules import run_rules

__all__ = ["detect", "is_failing", "judge", "ollama_available"]


def detect(trace: Trace, *, use_judge: bool = False, judge_model: Optional[str] = None,
           judge_client=None, budget_s: Optional[float] = None, loop_repeat: int = 3) -> list:
    failures = run_rules(trace, budget_s) + run_heuristics(trace, loop_repeat)
    if use_judge:
        kw = {"client": judge_client}
        if judge_model:
            kw["model"] = judge_model
        failures += judge(trace, **kw)
    return failures


def is_failing(failures: list) -> bool:
    """A trace fails if any finding is above low severity (recovered tool errors are low)."""
    return any(f.severity != "low" for f in failures)


def signatures(failures: list) -> set:
    return {f.signature for f in failures if f.severity != "low"}
