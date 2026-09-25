"""Standardized trace schema.

A trace is one agent run: the task input, an ordered list of steps, and the final
output. Every step has the same shape so detectors, converters and the replayer
never need to know which framework produced it.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

SCHEMA_VERSION = "1"

# input      - the task handed to the agent
# llm        - a model call (content = response, meta.prompt = prompt)
# reasoning  - intermediate thoughts / plans
# tool       - a tool invocation (name, args, content = result, error = failure)
# output     - the agent's final answer
# error      - an exception or fatal condition outside a tool
STEP_KINDS = ("input", "llm", "reasoning", "tool", "output", "error")

TRACE_STATUSES = ("ok", "error", "timeout", "incomplete")


def new_id(prefix: str = "") -> str:
    return prefix + uuid.uuid4().hex[:12]


def stable_hash(value: Any) -> str:
    """Deterministic short hash of any JSON-able value (dict key order ignored)."""
    blob = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class Step:
    idx: int
    kind: str
    ts: float = field(default_factory=time.time)
    name: Optional[str] = None          # tool / model name
    args: Optional[dict] = None         # tool arguments
    content: Any = None                 # result, text, output...
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in STEP_KINDS:
            raise ValueError(f"unknown step kind {self.kind!r}; expected one of {STEP_KINDS}")

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None and v != {}}

    @classmethod
    def from_dict(cls, d: dict) -> "Step":
        return cls(
            idx=d["idx"], kind=d["kind"], ts=d.get("ts", 0.0), name=d.get("name"),
            args=d.get("args"), content=d.get("content"), error=d.get("error"),
            duration_ms=d.get("duration_ms"), meta=d.get("meta") or {},
        )

    @property
    def call_key(self) -> str:
        """Identity of a tool call: name + canonical args."""
        return f"{self.name}:{stable_hash(self.args or {})}"


@dataclass
class Trace:
    input: Any = None
    steps: list = field(default_factory=list)
    output: Any = None
    status: str = "incomplete"
    trace_id: str = field(default_factory=lambda: new_id("tr_"))
    task_id: Optional[str] = None       # groups retries of the same task
    agent: str = "agent"
    agent_version: str = "0"
    source: str = "live"                # live | toolbench | swe-agent | agentbench | webarena | ...
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    meta: dict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.task_id is None:
            self.task_id = "task_" + stable_hash(self.input)

    # -- building ---------------------------------------------------------
    def add(self, kind: str, **kw) -> Step:
        step = Step(idx=len(self.steps), kind=kind, **kw)
        self.steps.append(step)
        return step

    # -- views ------------------------------------------------------------
    def tool_steps(self) -> list:
        return [s for s in self.steps if s.kind == "tool"]

    @property
    def duration_s(self) -> Optional[float]:
        return None if self.ended_at is None else self.ended_at - self.started_at

    # -- (de)serialization -------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["steps"] = [s.to_dict() for s in self.steps]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "Trace":
        return cls(
            input=d.get("input"), steps=[Step.from_dict(s) for s in d.get("steps", [])],
            output=d.get("output"), status=d.get("status", "incomplete"),
            trace_id=d.get("trace_id") or new_id("tr_"), task_id=d.get("task_id"),
            agent=d.get("agent", "agent"), agent_version=str(d.get("agent_version", "0")),
            source=d.get("source", "live"), started_at=d.get("started_at", 0.0),
            ended_at=d.get("ended_at"), meta=d.get("meta") or {},
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )

    @classmethod
    def from_json(cls, s: str) -> "Trace":
        return cls.from_dict(json.loads(s))


@dataclass
class Failure:
    kind: str                 # exception | tool_error | timeout | malformed_output | loop | ...
    detector: str             # rule | heuristic | judge
    message: str
    step_idx: Optional[int] = None
    tool: Optional[str] = None
    severity: str = "medium"  # low | medium | high

    @property
    def signature(self) -> str:
        """Coarse identity used to compare failures across agent versions.

        kind + tool, or kind + exception class for exceptions, so a KeyError
        that turns into a TypeError counts as failing *differently*.
        """
        if self.kind == "exception":
            m = re.match(r"([A-Za-z_][\w.]*):", self.message or "")
            return f"exception:{m.group(1) if m else '-'}"
        return f"{self.kind}:{self.tool or '-'}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["signature"] = self.signature
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Failure":
        return cls(kind=d["kind"], detector=d["detector"], message=d.get("message", ""),
                   step_idx=d.get("step_idx"), tool=d.get("tool"), severity=d.get("severity", "medium"))


def read_jsonl(path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows, append: bool = False) -> None:
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
