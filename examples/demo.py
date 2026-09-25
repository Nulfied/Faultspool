"""End-to-end demo: capture -> detect -> convert -> cluster -> run -> dashboard.

    python examples/demo.py

Writes everything under examples/.demo/ (database, exported tests, report).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from examples import toy_agent  # noqa: E402
from faultspool import Store, record  # noqa: E402
from faultspool import pipeline  # noqa: E402
from faultspool.dashboard import build  # noqa: E402
from faultspool.runner import run_suite, summarize  # noqa: E402

OUT = ROOT / "examples" / ".demo"


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    store = Store(str(OUT / "faultspool.db"))

    print("1. capture: running agent v1 on", len(toy_agent.TASKS), "tasks")
    for task in toy_agent.TASKS:
        t = record(toy_agent.run_v1, task, toy_agent.TOOLS, agent="refund-desk", agent_version="v1",
                   max_tool_calls=8, timeout_s=5, sink=store)
        print(f"   {task['customer']:<8} -> {t.status:<8} {str(t.output)[:60]}")

    # Self-play: a successful retry of a failed task gives the expected behaviour for free.
    # (Here the "retry" is v2; in practice it is the same agent re-sampled or re-prompted.)
    for customer in ("Grace", "linus"):
        retry = record(toy_agent.run_v2, {"customer": customer}, toy_agent.TOOLS, agent="refund-desk",
                       agent_version="v1-retry", sink=store)
        print(f"   retry {customer:<6} -> {retry.status}: {retry.output}")

    print("\n2. detect:", pipeline.detect_all(store))
    print("3. convert:", pipeline.convert_all(store))
    c = pipeline.cluster_all(store, prefer_ollama=False)
    print(f"4. cluster: {c['clusters']} clusters over {c['tests']} tests")
    for size, label in c["top"]:
        print(f"   [{size}] {label}")

    # In real use a human confirms tests with `faultspool annotate`; the demo accepts them all.
    for t in store.tests(status="pending"):
        t.status = "confirmed"
        store.upsert_test(t)

    tests = store.tests(status="confirmed")
    for version, agent in (("v1", toy_agent.run_v1), ("v2", toy_agent.run_v2)):
        results = run_suite(tests, agent, store.last_outcomes(), agent_version=version, max_tool_calls=8)
        s = summarize(results)
        store.add_run(version, s, results)
        print(f"\n5. regression run against {version}: {s['passes']}/{s['total']} passing")
        for r in results:
            print(f"   {r.outcome:<18} {r.test_id}  {r.detail}")

    n = pipeline.export_tests(store, OUT / "tests.jsonl")
    report = build(store, str(OUT / "report" / "index.html"), "Faultspool demo")
    print(f"\n6. exported {n} tests to {OUT / 'tests.jsonl'}")
    print(f"   dashboard: {report}")
    store.close()


if __name__ == "__main__":
    main()
