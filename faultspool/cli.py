"""faultspool command line."""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__, pipeline
from .annotate import annotate
from .dashboard import build as build_dashboard
from .runner import load_agent, run_suite, should_fail, summarize
from .storage import Store

DEFAULT_DB = os.environ.get("FAULTSPOOL_DB", ".faultspool/faultspool.db")
ICONS = {"passes": "PASS ", "still_fails": "FAIL ", "fails_differently": "DIFF ", "unreplayable": "MISS ",
         "error": "ERROR"}


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_init(a):
    with Store(a.db) as s:
        print(f"initialized {s.path}")


def cmd_ingest(a):
    with Store(a.db) as s:
        n = sum(pipeline.ingest(s, p, a.format) for p in a.paths)
        print(f"ingested {n} trace(s)")


def cmd_detect(a):
    with Store(a.db) as s:
        _print(pipeline.detect_all(s, use_judge=a.judge, judge_model=a.model, redetect=a.all))


def cmd_convert(a):
    with Store(a.db) as s:
        _print(pipeline.convert_all(s))


def cmd_cluster(a):
    with Store(a.db) as s:
        _print(pipeline.cluster_all(s, threshold=a.threshold, prefer_ollama=not a.no_ollama))


def cmd_process(a):
    with Store(a.db) as s:
        for p in a.paths:
            print(f"ingest   {p}: {pipeline.ingest(s, p, a.format)} trace(s)")
        print("detect  ", json.dumps(pipeline.detect_all(s, use_judge=a.judge, judge_model=a.model)))
        print("convert ", json.dumps(pipeline.convert_all(s)))
        r = pipeline.cluster_all(s, threshold=a.threshold, prefer_ollama=not a.no_ollama)
        print(f"cluster  {r['clusters']} cluster(s) over {r['tests']} test(s) using {r['model']}")


def cmd_annotate(a):
    with Store(a.db) as s:
        _print(annotate(s, limit=a.limit))


def cmd_export(a):
    with Store(a.db) as s:
        n = pipeline.export_tests(s, a.out, status=a.status, one_per_cluster=a.dedup)
        print(f"wrote {n} test(s) to {a.out}")


def cmd_run(a):
    agent_spec = a.agent or os.environ.get("FAULTSPOOL_AGENT")
    if not agent_spec:
        sys.exit("error: pass --agent module:function or set FAULTSPOOL_AGENT")
    agent = load_agent(agent_spec)
    store = None if (a.tests and a.no_db) else Store(a.db)
    if a.tests:
        tests = pipeline.load_tests(a.tests)
    else:
        tests = store.tests(status=None) if a.include_pending else store.tests(status="confirmed")
        tests = [t for t in tests if t.status != "rejected"]
    previous = pipeline.load_baseline(a.baseline) if a.baseline else (store.last_outcomes() if store else {})
    results = run_suite(tests, agent, previous, agent_version=a.version, timeout_s=a.timeout,
                        max_tool_calls=a.max_tool_calls, use_judge=a.judge, judge_model=a.model,
                        fallback=None if a.strict_replay else "by_name")
    summary = summarize(results)
    for r in results:
        flag = "  <- REGRESSION" if r.regressed else ""
        print(f"{ICONS[r.outcome]}  {r.test_id}  {r.detail}{flag}")
    print(f"\n{summary['passes']}/{summary['total']} passing | {summary['still_fails']} still failing | "
          f"{summary['fails_differently']} failing differently | {summary['unreplayable']} unreplayable | "
          f"{summary['error']} errors | "
          f"{len(summary['regressions'])} regression(s)")
    if store:
        store.add_run(a.version, summary, results)
        store.close()
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": [r.to_dict() for r in results]}, f, indent=2, default=str)
    if a.baseline and a.update_baseline:
        pipeline.save_baseline(a.baseline, results)
        print(f"baseline updated: {a.baseline}")
    if should_fail(results, a.fail_on):
        sys.exit(1)


def cmd_dashboard(a):
    with Store(a.db) as s:
        print(f"wrote {build_dashboard(s, a.out, a.title)}")


def cmd_stats(a):
    with Store(a.db) as s:
        _print({**s.stats(), "failure_kinds": s.failure_counts(), "top_clusters": s.clusters()[:5]})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="faultspool", description="Turn agent failure traces into regression tests.")
    p.add_argument("--version", action="version", version=f"faultspool {__version__}")
    p.add_argument("--db", default=DEFAULT_DB, help=f"SQLite database (default {DEFAULT_DB})")
    sub = p.add_subparsers(dest="cmd", required=True)

    def judge_args(sp):
        sp.add_argument("--judge", action="store_true", help="also ask a local Ollama model to judge traces")
        sp.add_argument("--model", default=None, help="Ollama model for the judge (default llama3.1)")

    sub.add_parser("init", help="create the database").set_defaults(fn=cmd_init)

    sp = sub.add_parser("ingest", help="import traces (native JSONL or benchmark trajectories)")
    sp.add_argument("paths", nargs="+")
    sp.add_argument("--format", default="auto",
                    choices=["auto", "native", "openai", "toolbench", "swe-agent", "agentbench", "webarena"])
    sp.set_defaults(fn=cmd_ingest)

    sp = sub.add_parser("detect", help="find failures in ingested traces")
    judge_args(sp)
    sp.add_argument("--all", action="store_true", help="re-run on traces already scanned")
    sp.set_defaults(fn=cmd_detect)

    sub.add_parser("convert", help="turn failing traces into test cases").set_defaults(fn=cmd_convert)

    sp = sub.add_parser("cluster", help="group near-duplicate failures")
    sp.add_argument("--threshold", type=float, default=0.85)
    sp.add_argument("--no-ollama", action="store_true", help="use the built-in hashing embedder")
    sp.set_defaults(fn=cmd_cluster)

    sp = sub.add_parser("process", help="ingest + detect + convert + cluster in one go")
    sp.add_argument("paths", nargs="*")
    sp.add_argument("--format", default="auto")
    sp.add_argument("--threshold", type=float, default=0.85)
    sp.add_argument("--no-ollama", action="store_true")
    judge_args(sp)
    sp.set_defaults(fn=cmd_process)

    sp = sub.add_parser("annotate", help="confirm/reject tests and fix expected outputs")
    sp.add_argument("--limit", type=int, default=None)
    sp.set_defaults(fn=cmd_annotate)

    sp = sub.add_parser("export", help="write tests to JSONL for git / CI")
    sp.add_argument("out")
    sp.add_argument("--status", default="confirmed", choices=["confirmed", "pending", "all"])
    sp.add_argument("--dedup", action="store_true", help="one test per failure cluster")
    sp.set_defaults(fn=cmd_export)

    sp = sub.add_parser("run", help="replay tests against the current agent")
    sp.add_argument("--agent", help="agent callable as module:function (or FAULTSPOOL_AGENT)")
    sp.add_argument("--tests", help="JSONL of tests (default: confirmed tests in the database)")
    sp.add_argument("--include-pending", action="store_true")
    sp.add_argument("--agent-version", dest="version", default=os.environ.get("FAULTSPOOL_AGENT_VERSION", "dev"))
    sp.add_argument("--baseline", help="JSON of previous outcomes; used to detect regressions")
    sp.add_argument("--update-baseline", action="store_true")
    sp.add_argument("--fail-on", default="regression", choices=["regression", "any", "none"])
    sp.add_argument("--timeout", type=float, default=30.0)
    sp.add_argument("--max-tool-calls", type=int, default=50)
    sp.add_argument("--strict-replay", action="store_true", help="only exact recorded calls are answered")
    sp.add_argument("--no-db", action="store_true", help="with --tests, don't record the run in the database")
    sp.add_argument("--json", help="write full results as JSON")
    judge_args(sp)
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("dashboard", help="write the static HTML report")
    sp.add_argument("-o", "--out", default="faultspool-report/index.html")
    sp.add_argument("--title", default="Faultspool")
    sp.set_defaults(fn=cmd_dashboard)

    sub.add_parser("stats", help="counts and top clusters").set_defaults(fn=cmd_stats)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
