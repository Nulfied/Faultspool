"""LLM-judge failure detection using a local Ollama model (free, offline).

The judge reads a compact rendering of the trace and answers one question:
did the agent go off track? Nothing is sent anywhere but your own Ollama.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional

from ..schema import Failure, Trace

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
DEFAULT_MODEL = os.environ.get("FAULTSPOOL_JUDGE_MODEL", "llama3.1")

PROMPT = """You are auditing an AI agent's execution trace for failures.
The agent was given a task, took the steps below, and produced a final output.

Decide whether the agent went off track. Off track includes: pursuing the wrong goal,
ignoring or misreading tool results, hallucinating facts not in tool results,
contradicting itself, giving up early, or a final answer that does not satisfy the task.
Minor inefficiency is NOT off track.

Respond with JSON only:
{{"off_track": true|false, "step": <index of first bad step or null>, "reason": "<one sentence>"}}

TRACE
{trace}
"""


def render_trace(trace: Trace, max_chars: int = 400, max_steps: int = 60) -> str:
    def clip(v) -> str:
        s = v if isinstance(v, str) else json.dumps(v, default=str, ensure_ascii=False)
        return s if len(s) <= max_chars else s[:max_chars] + "…"

    lines = [f"TASK: {clip(trace.input)}"]
    steps = trace.steps
    if len(steps) > max_steps:  # keep the start and the end, where failures live
        steps = steps[: max_steps // 3] + steps[-(max_steps - max_steps // 3):]
    for s in steps:
        if s.kind == "input":
            continue
        if s.kind == "tool":
            res = f"ERROR {s.error}" if s.error else clip(s.content)
            lines.append(f"[{s.idx}] TOOL {s.name}({clip(s.args or {})}) -> {res}")
        elif s.kind == "error":
            lines.append(f"[{s.idx}] ERROR {s.error}")
        else:
            lines.append(f"[{s.idx}] {s.kind.upper()}: {clip(s.content)}")
    lines.append(f"STATUS: {trace.status}")
    return "\n".join(lines)


def _post(path: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(OLLAMA_URL + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ollama_available(timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=timeout):
            return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def parse_verdict(text: str) -> Optional[dict]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        v = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return v if isinstance(v, dict) and "off_track" in v else None


def judge(trace: Trace, model: str = DEFAULT_MODEL, timeout: float = 120.0, client=None) -> list:
    """Return a one-element list with an `off_track` Failure, or [] if on track / unavailable.

    `client(prompt) -> str` can be injected for tests or for a different local runtime.
    """
    prompt = PROMPT.format(trace=render_trace(trace))
    try:
        if client is not None:
            text = client(prompt)
        else:
            resp = _post("/api/generate", {"model": model, "prompt": prompt, "stream": False,
                                           "format": "json", "options": {"temperature": 0}}, timeout)
            text = resp.get("response", "")
    except (urllib.error.URLError, OSError, ValueError):
        return []
    verdict = parse_verdict(text)
    if not verdict or not verdict.get("off_track"):
        return []
    step = verdict.get("step")
    return [Failure("off_track", "judge", str(verdict.get("reason", "judge flagged trace"))[:300],
                    step if isinstance(step, int) else None)]
