"""MCP server: drive the whole Faultspool pipeline from an MCP client.

Any MCP client (Claude Code, Claude Desktop, ...) gets the spool as tools: look
at what failed, read a captured trace, triage failures into confirmed regression
tests, replay them against the current agent, and export a suite for CI.

Built against ``mcp`` 2.x's ``MCPServer`` (``mcp.server.mcpserver``), NOT the
older ``FastMCP`` (``mcp.server.fastmcp``) that most tutorials still show --
that class was renamed in 2.x, so ``from mcp.server.fastmcp import FastMCP``
raises ModuleNotFoundError there. Verified against a real ``pip install mcp``
(2.2.0): ``@mcp.tool()`` takes the description from the docstring and the
argument schema from the type hints, and ``MCPServer.call_tool()`` /
``.list_tools()`` are what tests/test_mcp_server.py exercises in-process.

Setup (still $0 -- it runs locally, nothing is hosted):

    pip install "faultspool[mcp]"
    python -m faultspool.mcp_server        # or: faultspool-mcp

Claude Code:  claude mcp add faultspool -- python -m faultspool.mcp_server
Claude Desktop (claude_desktop_config.json):

    {
      "mcpServers": {
        "faultspool": {
          "command": "python",
          "args": ["-m", "faultspool.mcp_server"],
          "env": {"FAULTSPOOL_DB": "/path/to/project/.faultspool/faultspool.db"}
        }
      }
    }

Environment:
  FAULTSPOOL_DB       SQLite file to work against (default .faultspool/faultspool.db)
  FAULTSPOOL_AGENT    default ``module:function`` spec for run_regression
  FAULTSPOOL_MCP_AGENTS  comma-separated allowlist of agent specs run_regression
                      may import; unset means any importable spec is allowed

CRITICAL: never print() in this file. stdio IS the JSON-RPC transport, so
anything on stdout that is not a protocol message corrupts the connection. All
diagnostics go to stderr through ``logging``.
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager

from mcp.server.mcpserver import MCPServer

from . import __version__
from . import mcp_tools as T
from .storage import Store

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("faultspool-mcp")

DB_PATH = os.environ.get("FAULTSPOOL_DB", ".faultspool/faultspool.db")
DEFAULT_AGENT = os.environ.get("FAULTSPOOL_AGENT", "")

mcp = MCPServer("faultspool", version=__version__,
                instructions="Triage AI agent failures: look at what failed, turn failing runs "
                             "into regression tests, and replay them against the current agent.")


def _allowlist():
    raw = os.environ.get("FAULTSPOOL_MCP_AGENTS", "").strip()
    return [s.strip() for s in raw.split(",") if s.strip()] or None


@contextmanager
def _store():
    """A fresh Store (and sqlite connection) per call, never a cached global.

    MCPServer dispatches synchronous tool functions onto a thread pool, so two
    calls are not guaranteed the same thread, and a sqlite3 connection reused
    across threads raises "SQLite objects created in a thread can only be used
    in that same thread". Opening per call is cheap and is what the CLI already
    does once per invocation.
    """
    store = Store(DB_PATH)
    try:
        yield store
    finally:
        store.close()


@mcp.tool()
def spool_stats() -> str:
    """Overview of the spool: how many traces were captured, how many failed,
    how many regression tests exist, and how the latest replay run went.
    Start here to see whether there is anything to triage."""
    with _store() as s:
        return T.do_stats(s)


@mcp.tool()
def list_clusters(limit: int = 10) -> str:
    """List failure clusters, biggest first. Near-duplicate failures are grouped
    by a local embedding, so cluster size is a priority signal: fixing the root
    cause of the biggest cluster clears the most captured failures.

    Args:
        limit: How many clusters to return.
    """
    with _store() as s:
        return T.do_list_clusters(s, limit)


@mcp.tool()
def list_tests(status: str = "", limit: int = 20) -> str:
    """List regression tests generated from failing traces.

    Args:
        status: Filter by pending, confirmed or rejected. Empty means all.
        limit: How many tests to return.
    """
    with _store() as s:
        return T.do_list_tests(s, status, limit)


@mcp.tool()
def show_test(test_id: str) -> str:
    """Show one regression test in full: the task input, the detected failures,
    the recorded tool responses that are replayed as mocks, and the assertions
    it checks. Use this before confirming or correcting a test.

    Args:
        test_id: Id from list_tests, e.g. "t_9c739e3e0dcc".
    """
    with _store() as s:
        return T.do_show_test(s, test_id)


@mcp.tool()
def show_trace(trace_id: str, max_steps: int = 40) -> str:
    """Show a captured agent run step by step (tool calls, results, errors, the
    final output) together with the failures detected in it. Use this to work
    out WHY a run went wrong.

    Args:
        trace_id: Id of the trace, e.g. "tr_1a2b3c4d5e6f".
        max_steps: How many steps to include.
    """
    with _store() as s:
        return T.do_show_trace(s, trace_id, max_steps)


@mcp.tool()
def ingest_traces(path: str, format: str = "auto", use_judge: bool = False) -> str:
    """Ingest agent traces from a file or directory and run the whole pipeline:
    detect failures, convert each failing run into a regression test, and
    cluster near-duplicates. Reads Faultspool JSONL and open benchmark
    trajectories (ToolBench, SWE-agent, AgentBench, WebArena, OpenAI messages).

    Args:
        path: File or directory of traces.
        format: auto, native, openai, toolbench, swe-agent, agentbench or webarena.
        use_judge: Also ask a local Ollama model whether each run went off track.
    """
    with _store() as s:
        return T.do_ingest(s, path, format, use_judge)


@mcp.tool()
def annotate_test(test_id: str, status: str = "", expected: str = "", note: str = "") -> str:
    """Triage a test: confirm it is a real failure, reject it as a false
    positive, and/or correct what the right answer should have been.

    Args:
        test_id: Id from list_tests.
        status: confirmed, rejected or pending.
        expected: Expected output. Prefix with "=" for an exact match or "re:"
            for a regex; bare JSON is compared as a value; anything else must
            appear in the output.
        note: Free-text note stored with the test.
    """
    with _store() as s:
        return T.do_annotate(s, test_id, status, expected, note)


@mcp.tool()
def run_regression(agent: str = "", agent_version: str = "dev", tests_path: str = "",
                   status: str = "confirmed", max_tool_calls: int = 50) -> str:
    """Replay captured failures against the current agent and report, per test,
    whether it now passes, still fails, fails differently, or cannot be replayed.
    Recorded tool responses are served back as mocks, so no live API is called --
    but the agent's own code does run, exactly as `faultspool run` would run it.

    Args:
        agent: The agent to run, as "module:function". Defaults to $FAULTSPOOL_AGENT.
        agent_version: Label for this run, e.g. a git sha.
        tests_path: Optional exported tests JSONL; default is the stored tests.
        status: Which stored tests to run when tests_path is empty.
        max_tool_calls: Budget per run; exceeding it is recorded as a timeout.
    """
    spec = agent or DEFAULT_AGENT
    if not spec:
        return "No agent given. Pass agent=\"module:function\" or set FAULTSPOOL_AGENT."
    with _store() as s:
        return T.do_run_regression(s, spec, agent_version, tests_path, status,
                                   max_tool_calls, allowlist=_allowlist())


@mcp.tool()
def export_tests(out_path: str, status: str = "confirmed", dedup: bool = False) -> str:
    """Write regression tests to a JSONL file to commit and run in CI.

    Args:
        out_path: Where to write, e.g. "spool/tests.jsonl".
        status: confirmed, pending or all.
        dedup: Keep only one test per failure cluster.
    """
    with _store() as s:
        return T.do_export(s, out_path, status, dedup)


def main() -> None:
    logger.info("faultspool MCP server starting, db=%s", DB_PATH)
    mcp.run()


if __name__ == "__main__":
    main()
