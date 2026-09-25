"""The MCP tool logic, tested directly (no `mcp` install needed)."""
import json

import pytest

from conftest import BENCH
from faultspool import Store, mcp_tools as T, record
from faultspool import pipeline


@pytest.fixture
def spool(tmp_path, toy):
    store = Store(str(tmp_path / "fs.db"))
    for task in toy.TASKS:
        record(toy.run_v1, task, toy.TOOLS, max_tool_calls=8, agent_version="v1", sink=store)
    for customer in ("Grace", "linus", "grace"):
        record(toy.run_v2, {"customer": customer}, toy.TOOLS, agent_version="v1-retry", sink=store)
    pipeline.detect_all(store)
    pipeline.convert_all(store)
    pipeline.cluster_all(store, prefer_ollama=False)
    yield store
    store.close()


def test_stats_reports_the_spool(spool):
    out = T.do_stats(spool)
    assert "traces=8 (failing=3)" in out and "tests=3" in out
    assert "tool_error" in out or "exception" in out


def test_list_clusters_and_tests(spool):
    clusters = T.do_list_clusters(spool)
    assert "Failure clusters" in clusters and "[1x]" in clusters
    pending = T.do_list_tests(spool, status="pending")
    assert pending.count("[pending/") == 3
    assert T.do_list_tests(spool, status="confirmed") == "No tests with status=confirmed."


def test_show_test_includes_mocks_and_assertions(spool):
    tid = spool.tests()[0].test_id
    out = T.do_show_test(spool, tid)
    assert tid in out and "recorded tool responses" in out and "assertions:" in out
    assert "search_orders" in out
    assert "No test 'nope'" in T.do_show_test(spool, "nope")


def test_show_trace_renders_steps_and_failures(spool):
    trace = next(t for t in spool.traces() if t.status == "error")
    out = T.do_show_trace(spool, trace.trace_id)
    assert "TOOL search_orders" in out and "detected failures: exception" in out
    assert "No trace 'nope'" in T.do_show_trace(spool, "nope")


def test_annotate_confirms_and_sets_expected(spool):
    t = spool.tests()[0]
    assert "status must be" in T.do_annotate(spool, t.test_id, status="bogus")
    assert "Nothing to change" in T.do_annotate(spool, t.test_id)
    out = T.do_annotate(spool, t.test_id, status="confirmed", expected="=All good.", note="checked")
    assert "status -> confirmed" in out and "expected output" in out
    saved = spool.get_test(t.test_id)
    assert saved.status == "confirmed" and saved.origin == "human" and saved.notes == "checked"
    assert {"type": "output_equals", "value": "All good."} in saved.assertions


def test_run_regression_reports_outcomes_and_respects_allowlist(spool):
    for t in spool.tests():
        t.status = "confirmed"
        spool.upsert_test(t)

    refused = T.do_run_regression(spool, "examples.toy_agent:run_v2", allowlist=["other:agent"])
    assert refused.startswith("Refused:")

    bad = T.do_run_regression(spool, "examples.toy_agent:nope")
    assert bad.startswith("Could not load agent")

    out = T.do_run_regression(spool, "examples.toy_agent:run_v2", agent_version="v2", max_tool_calls=8)
    assert "3/3 passing" in out
    out = T.do_run_regression(spool, "examples.toy_agent:run_v1", agent_version="v1", max_tool_calls=8)
    assert "0/3 passing" in out and "REGRESSION" in out
    assert len(spool.runs()) == 2


def test_run_regression_needs_confirmed_tests(spool):
    assert "No tests to run" in T.do_run_regression(spool, "examples.toy_agent:run_v2")


def test_ingest_runs_the_whole_pipeline(tmp_path):
    with Store(str(tmp_path / "b.db")) as store:
        out = T.do_ingest(store, str(BENCH))
        assert "Ingested 8 trace(s)" in out and "Clustered into" in out
        assert store.stats()["tests"] >= 4
        exported = tmp_path / "t.jsonl"
        assert "Wrote 0 test(s)" in T.do_export(store, str(exported))
        assert "test(s) to" in T.do_export(store, str(exported), status="all")
        assert len(exported.read_text().splitlines()) >= 4


def test_export_dedups_by_cluster(spool, tmp_path):
    out = str(tmp_path / "dedup.jsonl")
    T.do_export(spool, out, status="all", dedup=True)
    rows = [json.loads(line) for line in open(out, encoding="utf-8")]
    assert len({r["cluster_id"] for r in rows}) == len(rows)
