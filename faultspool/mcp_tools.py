"""Tool logic behind the MCP server, as plain functions with zero dependency on
the ``mcp`` package.

Everything here is ordinary Python over a :class:`~faultspool.storage.Store`, so
it is fully testable without installing ``mcp`` or speaking JSON-RPC; only the
protocol wiring in :mod:`faultspool.mcp_server` needs the package. Each function
returns a plain string, because that is what an MCP client renders back to the
model.
"""
from __future__ import annotations

import json
from typing import Optional

from . import pipeline
from .annotate import expected_assertion
from .runner import load_agent, run_suite, summarize

MAX_ROWS = 50


def _clip(v, n: int = 160) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str, ensure_ascii=False)
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _kinds(failures: list) -> str:
    real = [f for f in failures if f.get("severity") != "low"]
    return ", ".join(sorted({f["kind"] for f in real})) or "none"


def do_stats(store) -> str:
    s = store.stats()
    runs = store.runs()
    lines = [
        f"traces={s['traces']} (failing={s['failing_traces']})",
        f"tests={s['tests']} (confirmed={s['tests_confirmed']}, pending={s['tests_pending']})",
        f"clusters={s['clusters']}  runs={s['runs']}",
    ]
    if runs:
        last = runs[-1]
        sm = last["summary"]
        rate = sm.get("pass_rate")
        lines.append(f"latest run: version={last['agent_version']} "
                     f"{sm.get('passes', 0)}/{sm.get('total', 0)} passing"
                     + (f" ({rate:.0%})" if rate is not None else "")
                     + (f", {len(sm.get('regressions') or [])} regression(s)" if sm.get("regressions") else ""))
    counts = store.failure_counts()
    if counts:
        top = ", ".join(f"{c['kind']}={c['n']}" for c in counts if c["severity"] != "low")
        lines.append(f"failure kinds: {top or 'none above low severity'}")
    return "\n".join(lines)


def do_list_clusters(store, limit: int = 10) -> str:
    clusters = store.clusters()[: max(1, min(limit, MAX_ROWS))]
    if not clusters:
        return "No clusters yet. Ingest traces first, then the pipeline clusters them."
    out = ["Failure clusters, biggest first (size = how many captured failures share this root cause):"]
    for c in clusters:
        out.append(f"  [{c['size']}x] {c['cluster_id']}  {_clip(c['label'], 120)}\n"
                   f"        representative test: {c['representative']}")
    return "\n".join(out)


def do_list_tests(store, status: str = "", limit: int = 20) -> str:
    tests = store.tests(status=status or None)
    if not tests:
        return f"No tests with status={status or 'any'}."
    tests = tests[: max(1, min(limit, MAX_ROWS))]
    out = [f"{len(tests)} test(s):"]
    for t in tests:
        out.append(f"  {t.test_id}  [{t.status}/{t.origin}] cluster={t.cluster_id or '-'}  "
                   f"fails: {_kinds(t.failures)}\n        task: {_clip(t.input, 120)}")
    return "\n".join(out)


def do_show_test(store, test_id: str) -> str:
    t = store.get_test(test_id)
    if t is None:
        return f"No test {test_id!r}. Use list_tests to see the ids."
    out = [f"test {t.test_id}  status={t.status}  origin={t.origin}  cluster={t.cluster_id or '-'}",
           f"agent: {t.agent}   source trace: {t.source_trace_id}",
           f"task input: {_clip(t.input, 400)}",
           "failures:"]
    for f in t.failures:
        mark = "  " if f.get("severity") == "low" else "! "
        out.append(f"  {mark}{f['kind']} (step {f.get('step_idx')}, {f['detector']}): {_clip(f.get('message'), 200)}")
    out.append(f"recorded tool responses ({len(t.mocks)}), replayed as mocks:")
    for m in t.mocks:
        result = f"ERROR {_clip(m['error'], 120)}" if m.get("error") else _clip(m.get("result"), 120)
        out.append(f"    {m['name']}({_clip(m.get('args'), 100)}) -> {result}")
    out.append("assertions:")
    for a in t.assertions:
        out.append(f"    {json.dumps(a, default=str)}")
    if t.notes:
        out.append(f"notes: {t.notes}")
    return "\n".join(out)


def do_show_trace(store, trace_id: str, max_steps: int = 40) -> str:
    t = store.get_trace(trace_id)
    if t is None:
        return f"No trace {trace_id!r}."
    failures = store.failures_for(trace_id)
    out = [f"trace {t.trace_id}  status={t.status}  agent={t.agent}@{t.agent_version}  source={t.source}",
           f"task: {_clip(t.input, 300)}"]
    steps = t.steps[: max(1, min(max_steps, 200))]
    for s in steps:
        if s.kind == "tool":
            result = f"ERROR {_clip(s.error, 120)}" if s.error else _clip(s.content, 120)
            out.append(f"  [{s.idx}] TOOL {s.name}({_clip(s.args or {}, 100)}) -> {result}")
        elif s.kind == "error":
            out.append(f"  [{s.idx}] ERROR {_clip(s.error, 160)}")
        elif s.kind != "input":
            out.append(f"  [{s.idx}] {s.kind.upper()}: {_clip(s.content, 160)}")
    if len(t.steps) > len(steps):
        out.append(f"  … {len(t.steps) - len(steps)} more step(s)")
    out.append(f"output: {_clip(t.output, 300)}")
    out.append("detected failures: " + (", ".join(f"{f.kind} ({f.severity})" for f in failures) or "none"))
    return "\n".join(out)


def do_ingest(store, path: str, fmt: str = "auto", use_judge: bool = False) -> str:
    """Ingest traces from a file or directory, then detect, convert and cluster."""
    n = pipeline.ingest(store, path, fmt)
    det = pipeline.detect_all(store, use_judge=use_judge)
    conv = pipeline.convert_all(store)
    clu = pipeline.cluster_all(store)
    return (f"Ingested {n} trace(s) from {path}.\n"
            f"Detected failures in {det['failing']} of {det['scanned']} scanned trace(s).\n"
            f"Created {conv['created']} new test(s) ({conv['already_had_test']} trace(s) already had one).\n"
            f"Clustered into {clu['clusters']} group(s) over {clu['tests']} test(s) using {clu['model']}.\n"
            "New tests start as pending; confirm the real ones with annotate_test.")


def do_annotate(store, test_id: str, status: str = "", expected: str = "", note: str = "") -> str:
    t = store.get_test(test_id)
    if t is None:
        return f"No test {test_id!r}."
    changed = []
    if expected:
        t.assertions = [a for a in t.assertions if not a["type"].startswith("output_")]
        t.assertions.append(expected_assertion(expected))
        t.origin = "human"
        changed.append(f"expected output -> {json.dumps(t.assertions[-1], default=str)}")
    if status:
        if status not in ("confirmed", "rejected", "pending"):
            return f"status must be confirmed, rejected or pending (got {status!r})."
        t.status = status
        changed.append(f"status -> {status}")
    if note:
        t.notes = note
        changed.append("note saved")
    if not changed:
        return "Nothing to change: pass status, expected or note."
    store.upsert_test(t)
    return f"Updated {test_id}: " + "; ".join(changed)


def do_run_regression(store, agent: str, agent_version: str = "dev", tests_path: str = "",
                      status: str = "confirmed", max_tool_calls: int = 50, timeout_s: float = 30.0,
                      allowlist: Optional[list] = None) -> str:
    """Replay captured failures against `agent` (a ``module:function`` spec).

    Tool calls are served from the recording, so no live API is touched; the
    agent's own code does run, exactly as ``faultspool run`` would run it.
    """
    if allowlist and agent not in allowlist:
        return (f"Refused: {agent!r} is not in FAULTSPOOL_MCP_AGENTS "
                f"({', '.join(allowlist)}). Set that variable to allow it.")
    try:
        agent_fn = load_agent(agent)
    except (ImportError, AttributeError, ValueError) as exc:
        return f"Could not load agent {agent!r}: {type(exc).__name__}: {exc}"

    tests = pipeline.load_tests(tests_path) if tests_path else store.tests(status=status or None)
    tests = [t for t in tests if t.status != "rejected"]
    if not tests:
        return "No tests to run. Confirm some tests first, or pass tests_path."

    results = run_suite(tests, agent_fn, store.last_outcomes(), agent_version=agent_version,
                        max_tool_calls=max_tool_calls, timeout_s=timeout_s)
    s = summarize(results)
    store.add_run(agent_version, s, results)
    out = [f"{s['passes']}/{s['total']} passing against {agent} (version {agent_version}): "
           f"{s['still_fails']} still failing, {s['fails_differently']} failing differently, "
           f"{s['unreplayable']} unreplayable, {s['error']} harness error(s)."]
    for r in results:
        flag = "  <- REGRESSION (this passed before)" if r.regressed else ""
        out.append(f"  {r.outcome:<18} {r.test_id}  {_clip(r.detail, 140)}{flag}")
    if s["regressions"]:
        out.append(f"{len(s['regressions'])} regression(s): CI with --fail-on regression would fail on this.")
    return "\n".join(out)


def do_export(store, out_path: str, status: str = "confirmed", dedup: bool = False) -> str:
    n = pipeline.export_tests(store, out_path, status=status, one_per_cluster=dedup)
    return (f"Wrote {n} test(s) to {out_path}. Commit it and point "
            f"`faultspool run --tests {out_path}` at it from CI.")
