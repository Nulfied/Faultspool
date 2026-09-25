"""Real, in-process tests of mcp_server.py's MCP protocol wiring, through
MCPServer.call_tool()/.list_tools() rather than the do_* functions in
mcp_tools.py (those have their own coverage in test_mcp_tools.py).

Skips itself entirely if `mcp` isn't installed (`pip install "faultspool[mcp]"`),
so the base suite still runs without the optional extra -- except when
FAULTSPOOL_REQUIRE_MCP is set, which CI does: there the extra is installed on
purpose, so a skip would quietly stop testing the protocol wiring at all.
"""
import asyncio
import os

import pytest

from faultspool import Store, record

if os.environ.get("FAULTSPOOL_REQUIRE_MCP"):
    import mcp  # noqa: F401
else:
    mcp = pytest.importorskip("mcp", reason='needs: pip install "faultspool[mcp]"')
import faultspool.mcp_server as srv  # noqa: E402

TOOL_NAMES = {"spool_stats", "list_clusters", "list_tests", "show_test", "show_trace",
              "ingest_traces", "annotate_test", "run_regression", "export_tests"}


@pytest.fixture
def db(tmp_path, toy, monkeypatch):
    """Point the server at a fresh database; each call opens its own connection."""
    path = str(tmp_path / "mcp.db")
    monkeypatch.setattr(srv, "DB_PATH", path)
    monkeypatch.setattr(srv, "DEFAULT_AGENT", "")
    monkeypatch.delenv("FAULTSPOOL_MCP_AGENTS", raising=False)
    with Store(path) as store:
        for task in toy.TASKS:
            record(toy.run_v1, task, toy.TOOLS, max_tool_calls=8, agent_version="v1", sink=store)
        for customer in ("Grace", "linus", "grace"):
            record(toy.run_v2, {"customer": customer}, toy.TOOLS, agent_version="v1-retry", sink=store)
    return path


def call(name, **kwargs):
    result = asyncio.run(srv.mcp.call_tool(name, kwargs))
    assert not result.is_error, f"{name}({kwargs}) errored: {result}"
    return result.content[0].text


def test_lists_every_tool_with_schemas():
    tools = asyncio.run(srv.mcp.list_tools())
    assert {t.name for t in tools} == TOOL_NAMES
    by_name = {t.name: t for t in tools}
    assert "test_id" in by_name["show_test"].input_schema["required"]
    assert by_name["list_clusters"].input_schema["properties"]["limit"]["default"] == 10
    assert all(t.description for t in tools)


def test_triage_flow_over_the_protocol(db):
    assert "traces=8" in call("spool_stats")
    assert "No clusters yet" in call("list_clusters")
    assert "tests=0" in call("spool_stats")

    # run detection/conversion through the same entry point the CLI uses
    from faultspool import pipeline
    with Store(db) as store:
        pipeline.detect_all(store)
        pipeline.convert_all(store)
        pipeline.cluster_all(store, prefer_ollama=False)

    assert "[1x]" in call("list_clusters", limit=5)
    listing = call("list_tests", status="pending")
    test_id = listing.split("\n")[1].strip().split()[0]
    assert "recorded tool responses" in call("show_test", test_id=test_id)

    assert "status -> confirmed" in call("annotate_test", test_id=test_id, status="confirmed")
    assert "[confirmed/" in call("list_tests", status="confirmed")


def test_run_regression_over_the_protocol(db):
    from faultspool import pipeline
    with Store(db) as store:
        pipeline.detect_all(store)
        pipeline.convert_all(store)
        for t in store.tests():
            t.status = "confirmed"
            store.upsert_test(t)

    assert "No agent given" in call("run_regression")
    out = call("run_regression", agent="examples.toy_agent:run_v2", agent_version="v2", max_tool_calls=8)
    assert "3/3 passing" in out
    out = call("run_regression", agent="examples.toy_agent:run_v1", agent_version="v1", max_tool_calls=8)
    assert "0/3 passing" in out and "REGRESSION" in out


def test_allowlist_blocks_unlisted_agents(db, monkeypatch):
    monkeypatch.setenv("FAULTSPOOL_MCP_AGENTS", "safe.mod:agent")
    out = call("run_regression", agent="examples.toy_agent:run_v1")
    assert out.startswith("Refused:") and "FAULTSPOOL_MCP_AGENTS" in out


def test_ingest_and_export_over_the_protocol(tmp_path, monkeypatch):
    from conftest import BENCH
    monkeypatch.setattr(srv, "DB_PATH", str(tmp_path / "bench.db"))
    out = call("ingest_traces", path=str(BENCH))
    assert "Ingested 8 trace(s)" in out
    trace_id = None
    with Store(str(tmp_path / "bench.db")) as store:
        trace_id = store.traces()[0].trace_id
    assert "status=" in call("show_trace", trace_id=trace_id, max_steps=5)
    assert "Wrote" in call("export_tests", out_path=str(tmp_path / "t.jsonl"), status="all")


def test_bad_arguments_raise_tool_errors():
    from mcp.server.mcpserver.exceptions import ToolError
    with pytest.raises(ToolError):
        asyncio.run(srv.mcp.call_tool("show_test", {}))
