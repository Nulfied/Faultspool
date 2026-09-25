"""SQLite storage for traces, failures, tests, embeddings, clusters and runs.

One file, no server. Tests can also be exported to JSONL so they live in git
and CI can run them without the database.
"""
from __future__ import annotations

import array
import json
import sqlite3
import time
from pathlib import Path
from typing import Iterable, Optional

from .convert import TestCase
from .schema import Failure, Trace

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
  trace_id TEXT PRIMARY KEY, task_id TEXT, agent TEXT, agent_version TEXT, source TEXT,
  status TEXT, started_at REAL, n_steps INTEGER, detected INTEGER DEFAULT 0, body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS traces_task ON traces(task_id);
CREATE TABLE IF NOT EXISTS failures (
  trace_id TEXT, kind TEXT, detector TEXT, severity TEXT, step_idx INTEGER, tool TEXT,
  signature TEXT, message TEXT
);
CREATE INDEX IF NOT EXISTS failures_trace ON failures(trace_id);
CREATE TABLE IF NOT EXISTS tests (
  test_id TEXT PRIMARY KEY, source_trace_id TEXT UNIQUE, task_id TEXT, status TEXT,
  origin TEXT, cluster_id TEXT, created_at REAL, body TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
  test_id TEXT PRIMARY KEY, model TEXT, dim INTEGER, vec BLOB
);
CREATE TABLE IF NOT EXISTS clusters (
  cluster_id TEXT PRIMARY KEY, label TEXT, size INTEGER, representative TEXT, updated_at REAL
);
CREATE TABLE IF NOT EXISTS runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT, agent_version TEXT, created_at REAL, summary TEXT
);
CREATE TABLE IF NOT EXISTS results (
  run_id INTEGER, test_id TEXT, outcome TEXT, regressed INTEGER, detail TEXT
);
"""

DEFAULT_DIR = ".faultspool"


class Store:
    def __init__(self, path: str = f"{DEFAULT_DIR}/faultspool.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- traces ---------------------------------------------------------------
    def add_trace(self, t: Trace) -> None:
        self.add_traces([t])

    def add_traces(self, traces: Iterable[Trace]) -> int:
        n = 0
        for t in traces:
            self.db.execute(
                "INSERT OR REPLACE INTO traces VALUES (?,?,?,?,?,?,?,?,0,?)",
                (t.trace_id, t.task_id, t.agent, t.agent_version, t.source, t.status,
                 t.started_at, len(t.steps), t.to_json()))
            n += 1
        self.db.commit()
        return n

    def traces(self, where: str = "1=1", params: tuple = ()) -> list:
        rows = self.db.execute(f"SELECT body FROM traces WHERE {where} ORDER BY started_at", params)
        return [Trace.from_json(r["body"]) for r in rows]

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        r = self.db.execute("SELECT body FROM traces WHERE trace_id=?", (trace_id,)).fetchone()
        return Trace.from_json(r["body"]) if r else None

    # -- failures -------------------------------------------------------------
    def set_failures(self, trace_id: str, failures: list) -> None:
        self.db.execute("DELETE FROM failures WHERE trace_id=?", (trace_id,))
        self.db.executemany(
            "INSERT INTO failures VALUES (?,?,?,?,?,?,?,?)",
            [(trace_id, f.kind, f.detector, f.severity, f.step_idx, f.tool, f.signature, f.message)
             for f in failures])
        self.db.execute("UPDATE traces SET detected=1 WHERE trace_id=?", (trace_id,))
        self.db.commit()

    def failures_for(self, trace_id: str) -> list:
        rows = self.db.execute("SELECT * FROM failures WHERE trace_id=?", (trace_id,))
        return [Failure(r["kind"], r["detector"], r["message"], r["step_idx"], r["tool"], r["severity"])
                for r in rows]

    def failure_counts(self) -> list:
        return [dict(r) for r in self.db.execute(
            "SELECT kind, severity, COUNT(*) AS n FROM failures GROUP BY kind, severity ORDER BY n DESC")]

    # -- tests ----------------------------------------------------------------
    def upsert_test(self, t: TestCase) -> bool:
        """Insert or update; returns False if a test for that trace already existed."""
        existing = self.db.execute("SELECT test_id FROM tests WHERE source_trace_id=?",
                                   (t.source_trace_id,)).fetchone()
        if existing and existing["test_id"] != t.test_id:
            return False
        self.db.execute("INSERT OR REPLACE INTO tests VALUES (?,?,?,?,?,?,?,?)",
                        (t.test_id, t.source_trace_id, t.task_id, t.status, t.origin,
                         t.cluster_id, t.created_at, json.dumps(t.to_dict(), default=str)))
        self.db.commit()
        return not existing

    def tests(self, status: Optional[str] = None) -> list:
        q, p = "SELECT body FROM tests", ()
        if status:
            q, p = q + " WHERE status=?", (status,)
        return [TestCase.from_dict(json.loads(r["body"])) for r in self.db.execute(q + " ORDER BY created_at", p)]

    def get_test(self, test_id: str) -> Optional[TestCase]:
        r = self.db.execute("SELECT body FROM tests WHERE test_id=?", (test_id,)).fetchone()
        return TestCase.from_dict(json.loads(r["body"])) if r else None

    # -- embeddings & clusters ------------------------------------------------
    def set_embedding(self, test_id: str, model: str, vec: list) -> None:
        blob = array.array("f", vec).tobytes()
        self.db.execute("INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?)", (test_id, model, len(vec), blob))
        self.db.commit()

    def embeddings(self) -> dict:
        out = {}
        for r in self.db.execute("SELECT test_id, vec FROM embeddings"):
            a = array.array("f")
            a.frombytes(r["vec"])
            out[r["test_id"]] = a.tolist()
        return out

    def set_clusters(self, clusters: list) -> None:
        self.db.execute("DELETE FROM clusters")
        now = time.time()
        self.db.executemany("INSERT INTO clusters VALUES (?,?,?,?,?)",
                            [(c["cluster_id"], c["label"], c["size"], c["representative"], now) for c in clusters])
        self.db.commit()

    def clusters(self) -> list:
        return [dict(r) for r in self.db.execute("SELECT * FROM clusters ORDER BY size DESC")]

    # -- runs -----------------------------------------------------------------
    def add_run(self, agent_version: str, summary: dict, results: list) -> int:
        cur = self.db.execute("INSERT INTO runs (agent_version, created_at, summary) VALUES (?,?,?)",
                              (agent_version, time.time(), json.dumps(summary)))
        run_id = cur.lastrowid
        self.db.executemany("INSERT INTO results VALUES (?,?,?,?,?)",
                            [(run_id, r.test_id, r.outcome, int(r.regressed), r.detail) for r in results])
        self.db.commit()
        return run_id

    def runs(self) -> list:
        return [{**dict(r), "summary": json.loads(r["summary"])}
                for r in self.db.execute("SELECT * FROM runs ORDER BY created_at")]

    def last_outcomes(self) -> dict:
        """Most recent outcome per test, used to spot regressions."""
        rows = self.db.execute(
            "SELECT test_id, outcome FROM results WHERE rowid IN "
            "(SELECT MAX(rowid) FROM results GROUP BY test_id)")
        return {r["test_id"]: r["outcome"] for r in rows}

    def stats(self) -> dict:
        one = lambda q: self.db.execute(q).fetchone()[0]  # noqa: E731
        return {
            "traces": one("SELECT COUNT(*) FROM traces"),
            "failing_traces": one("SELECT COUNT(DISTINCT trace_id) FROM failures WHERE severity!='low'"),
            "tests": one("SELECT COUNT(*) FROM tests"),
            "tests_confirmed": one("SELECT COUNT(*) FROM tests WHERE status='confirmed'"),
            "tests_pending": one("SELECT COUNT(*) FROM tests WHERE status='pending'"),
            "clusters": one("SELECT COUNT(*) FROM clusters"),
            "runs": one("SELECT COUNT(*) FROM runs"),
        }
