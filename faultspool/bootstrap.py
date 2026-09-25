"""Import open agent-benchmark trajectories into the Faultspool trace schema.

This lets you build and validate the pipeline before you have real traffic.
Supported inputs (JSON file, JSONL file, or a directory of either):

  native      Faultspool traces (one Trace dict per line / file)
  openai      OpenAI-style chat messages with function/tool calls. This is the
              shape of ToolBench's ``answer_generation.train_messages`` and of
              most function-calling datasets.
  toolbench   ToolBench answer files ({"answer_generation": {...}})
  swe-agent   SWE-agent / SWE-bench ``.traj`` files ({"trajectory": [...], "info": {...}})
  agentbench  AgentBench run outputs ({"output": {"history": [...], "status": ...}})
  webarena    WebArena-style action logs ({"task_id", "intent", "actions"/"trajectory", "success"/"score"})

Formats drift between releases; every adapter is small and forgiving, so adapt
the one closest to your data rather than reshaping the data.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Iterator

from .schema import Trace

# ---------------------------------------------------------------- file walking


def iter_records(path) -> Iterator[tuple]:
    """Yield (name, record) for every JSON object in a file or directory."""
    p = Path(path)
    files = sorted(x for x in p.rglob("*") if x.suffix in (".json", ".jsonl", ".traj")) if p.is_dir() else [p]
    for f in files:
        text = f.read_text(encoding="utf-8")
        if f.suffix == ".jsonl":
            for i, line in enumerate(text.splitlines()):
                if line.strip():
                    yield f"{f.stem}:{i}", json.loads(line)
        else:
            data = json.loads(text)
            if isinstance(data, list):
                for i, rec in enumerate(data):
                    yield f"{f.stem}:{i}", rec
            elif isinstance(data, dict) and not _looks_like_single(data):
                for k, rec in data.items():  # e.g. {"task_id": {...}, ...}
                    if isinstance(rec, dict):
                        yield f"{f.stem}:{k}", rec
            else:
                yield f.stem, data


def _looks_like_single(d: dict) -> bool:
    single_keys = {"steps", "trajectory", "messages", "answer_generation", "output", "history",
                   "actions", "train_messages", "info", "intent", "input"}
    return bool(single_keys & set(d))


def _args(raw):
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
            return v if isinstance(v, dict) else {"input": v}
        except ValueError:
            return {"input": raw}
    return {} if raw is None else {"input": raw}


def _content(c):
    if isinstance(c, list):  # content parts
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return c


# ---------------------------------------------------------------- adapters


def from_native(rec: dict, name: str = "") -> Trace:
    return Trace.from_dict(rec)


def from_openai_messages(messages: list, *, name: str = "", source: str = "openai",
                         task_id=None, meta=None) -> Trace:
    task = next((_content(m.get("content")) for m in messages if m.get("role") == "user"), None)
    t = Trace(input=task, source=source, task_id=task_id or None, status="ok", meta=dict(meta or {}))
    t.meta["import_name"] = name
    t.add("input", content=task)
    pending: dict = {}  # tool_call_id -> step
    last_fn_step = None
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            content = _content(m.get("content"))
            calls = m.get("tool_calls") or ([{"function": m["function_call"]}] if m.get("function_call") else [])
            if content:
                t.add("llm", content=content)
            for c in calls:
                fn = c.get("function", c)
                step = t.add("tool", name=fn.get("name"), args=_args(fn.get("arguments")))
                pending[c.get("id") or f"_{step.idx}"] = step
                last_fn_step = step
        elif role in ("tool", "function"):
            step = pending.pop(m.get("tool_call_id"), None) if m.get("tool_call_id") else None
            step = step or last_fn_step
            if step is None:
                continue
            body = _content(m.get("content"))
            step.content = body
            if isinstance(body, str) and re.search(r'"error"\s*:\s*"[^"]+"|^error\b|exception', body, re.I):
                parsed = _args(body)
                err = parsed.get("error") if isinstance(parsed, dict) else None
                step.error = str(err or body[:300])
    final = next((m for m in reversed(messages) if m.get("role") == "assistant"
                  and not m.get("tool_calls") and not m.get("function_call")), None)
    if final is not None:
        out = _content(final.get("content"))
        t.output = out
        t.add("output", content=out)
    else:
        t.status = "incomplete"
    t.task_id = task_id or t.task_id
    return t


def from_toolbench(rec: dict, name: str = "") -> Trace:
    ag = rec.get("answer_generation", rec)
    msgs = ag.get("train_messages") or ag.get("messages") or []
    if msgs and isinstance(msgs[0], list):  # list of conversations; the last is the full one
        msgs = msgs[-1]
    t = from_openai_messages(msgs, name=name, source="toolbench",
                             meta={"valid": ag.get("valid_data"), "win": rec.get("win")})
    if ag.get("query"):
        t.input = ag["query"]
        t.steps[0].content = ag["query"]
        t.task_id = "tb_" + str(rec.get("query_id") or name)
    final = ag.get("final_answer")
    if final is not None:
        if isinstance(final, str):
            try:
                final = json.loads(final).get("final_answer", final)
            except (ValueError, AttributeError):
                pass
        t.output = final
        if not any(s.kind == "output" for s in t.steps):
            t.add("output", content=final)
    if ag.get("valid_data") is False:
        t.status = "incomplete"
    return t


def from_swe_agent(rec: dict, name: str = "") -> Trace:
    info = rec.get("info", {})
    task = rec.get("problem_statement") or info.get("problem_statement") or rec.get("instance_id") or name
    t = Trace(input=task, source="swe-agent", task_id="swe_" + str(rec.get("instance_id") or name),
              meta={"exit_status": info.get("exit_status"), "import_name": name})
    t.add("input", content=task)
    for st in rec.get("trajectory", []):
        if st.get("thought"):
            t.add("reasoning", content=st["thought"])
        action = (st.get("action") or "").strip()
        if action:
            cmd = action.split()[0]
            obs = st.get("observation")
            err = None
            if isinstance(obs, str) and re.search(r"Traceback|command not found|Error:|No such file", obs):
                err = obs.strip().splitlines()[-1][:300]
            t.add("tool", name=cmd, args={"command": action}, content=obs, error=err)
    status = (info.get("exit_status") or "").lower()
    sub = info.get("submission")
    if sub:
        t.output = sub
        t.add("output", content=sub)
    if "submitted" in status and sub:
        t.status = "ok"
    elif any(k in status for k in ("cost", "context", "limit", "timeout")):
        t.status = "timeout"
        t.add("error", error=f"Timeout: {info.get('exit_status')}")
    elif "error" in status or "exit" in status:
        t.status = "error"
        t.add("error", error=str(info.get("exit_status")))
    else:
        t.status = "ok" if sub else "incomplete"
    return t


_ACTION = re.compile(r"Action:\s*([\w\-.]+)\s*(?:\((.*)\)|Input:\s*(.*)|:(.*))?", re.S)


def from_agentbench(rec: dict, name: str = "") -> Trace:
    out = rec.get("output", rec)
    hist = out.get("history", [])
    task = next((h.get("content") for h in hist if h.get("role") == "user"), rec.get("input"))
    t = Trace(input=task, source="agentbench", task_id="ab_" + str(rec.get("index", name)),
              meta={"agentbench_status": out.get("status"), "import_name": name})
    t.add("input", content=task)
    last_tool = None
    for h in hist[1:]:
        content = h.get("content") or ""
        if h.get("role") == "agent":
            m = _ACTION.search(content)
            if m:
                thought = content[: m.start()].strip()
                if thought:
                    t.add("reasoning", content=thought)
                arg = next((g for g in m.groups()[1:] if g), "")
                last_tool = t.add("tool", name=m.group(1), args={"input": arg.strip()})
            else:
                t.add("llm", content=content)
        elif h.get("role") == "user" and last_tool is not None and last_tool.content is None:
            last_tool.content = content
            if re.search(r"\berror\b|invalid|not found", content, re.I):
                last_tool.error = content[:300]
    status = (out.get("status") or "").lower()
    result = out.get("result")
    final = hist[-1].get("content") if hist and hist[-1].get("role") == "agent" else None
    if status == "completed":
        t.status = "ok"
        t.output = result if result is not None else final
        t.add("output", content=t.output)
    elif "limit" in status:
        t.status = "timeout"
        t.add("error", error=f"Timeout: {status}")
    else:
        t.status = "error" if status else "incomplete"
        if status:
            t.add("error", error=status)
    return t


def from_webarena(rec: dict, name: str = "") -> Trace:
    task = rec.get("intent") or rec.get("task") or rec.get("instruction")
    t = Trace(input=task, source="webarena", task_id="wa_" + str(rec.get("task_id", name)),
              meta={"start_url": rec.get("start_url"), "import_name": name})
    t.add("input", content=task)
    steps = rec.get("actions") or rec.get("trajectory") or []
    final = None
    for st in steps:
        if isinstance(st, str):
            st = {"action": st}
        if st.get("thought") or st.get("reasoning"):
            t.add("reasoning", content=st.get("thought") or st.get("reasoning"))
        action = st.get("action") or st.get("action_str") or ""
        kind = str(st.get("action_type") or (action.split("[")[0].split() or ["noop"])[0]).strip()
        if kind.lower() == "stop":
            m = re.search(r"\[(.*)\]", action)
            final = m.group(1) if m else st.get("answer", "")
            continue
        t.add("tool", name=kind, args={"action": action}, content=st.get("observation") or st.get("url"),
              error=st.get("error"))
    final = rec.get("answer", final)
    success = rec.get("success")
    if success is None and "score" in rec:
        success = float(rec["score"]) > 0
    if final is not None:
        t.output = final
        t.add("output", content=final)
    t.status = "ok" if final is not None else "incomplete"
    t.meta["benchmark_success"] = success
    return t


def from_openai_record(rec: dict, name: str = "") -> Trace:
    return from_openai_messages(rec.get("messages") or rec.get("conversation") or [], name=name,
                                task_id=rec.get("id"))


ADAPTERS: dict = {
    "native": from_native, "openai": from_openai_record, "toolbench": from_toolbench,
    "swe-agent": from_swe_agent, "agentbench": from_agentbench, "webarena": from_webarena,
}


def sniff(rec: dict) -> str:
    if "steps" in rec and "schema_version" in rec:
        return "native"
    if "answer_generation" in rec or "train_messages" in rec:
        return "toolbench"
    if "trajectory" in rec and ("info" in rec or "history" in rec):
        return "swe-agent"
    if "output" in rec and isinstance(rec["output"], dict) and "history" in rec["output"]:
        return "agentbench"
    if "intent" in rec or ("actions" in rec and "task_id" in rec):
        return "webarena"
    if "messages" in rec or "conversation" in rec:
        return "openai"
    if "steps" in rec:
        return "native"
    raise ValueError(f"cannot tell the format of record with keys {sorted(rec)[:8]}")


def import_path(path, fmt: str = "auto") -> Iterator[Trace]:
    """Convert every record under `path` to a Trace.

    Benchmarks that ship ground truth (WebArena ``success``/``score``) record it
    in ``meta.benchmark_success``; the rule detector turns ``False`` into a
    ``wrong_answer`` failure, so labelled data needs no judge at all.
    """
    for name, rec in iter_records(path):
        adapter: Callable = ADAPTERS[sniff(rec) if fmt == "auto" else fmt]
        yield adapter(rec, name=name)
