"""The pipeline stages as plain functions over a Store (the CLI is a thin wrapper)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .bootstrap import import_path
from .cluster import cluster_tests, get_embedder
from .convert import TestCase, find_retry, trace_to_test
from .detect import detect, is_failing
from .schema import read_jsonl, write_jsonl


def ingest(store, path, fmt: str = "auto") -> int:
    return store.add_traces(import_path(path, fmt))


def detect_all(store, *, use_judge: bool = False, judge_model: Optional[str] = None,
               redetect: bool = False, judge_client=None) -> dict:
    traces = store.traces() if redetect else store.traces("detected=0")
    failing = 0
    for t in traces:
        fs = detect(t, use_judge=use_judge, judge_model=judge_model, judge_client=judge_client)
        store.set_failures(t.trace_id, fs)
        failing += is_failing(fs)
    return {"scanned": len(traces), "failing": failing}


def convert_all(store) -> dict:
    by_task = defaultdict(list)
    for t in store.traces("source != 'replay'"):
        by_task[t.task_id].append((t, store.failures_for(t.trace_id)))
    created = skipped = 0
    for group in by_task.values():
        for t, fs in group:
            if not is_failing(fs):
                continue
            retry = find_retry(t, group)
            if store.upsert_test(trace_to_test(t, fs, retry)):
                created += 1
            else:
                skipped += 1
    return {"created": created, "already_had_test": skipped}


def cluster_all(store, *, threshold: float = 0.85, prefer_ollama: bool = True) -> dict:
    tests = [t for t in store.tests() if t.status != "rejected"]
    if not tests:
        store.set_clusters([])
        return {"model": None, "clusters": 0, "tests": 0}
    model, embed = get_embedder(prefer_ollama)
    cache = store.embeddings()
    clusters, vectors = cluster_tests(tests, embed, threshold, cache)
    for tid, vec in vectors.items():
        if tid not in cache:
            store.set_embedding(tid, model, vec)
    by_id = {t.test_id: t for t in tests}
    for c in clusters:
        for tid in c["members"]:
            t = by_id[tid]
            if t.cluster_id != c["cluster_id"]:
                t.cluster_id = c["cluster_id"]
                store.upsert_test(t)
    store.set_clusters(clusters)
    return {"model": model, "clusters": len(clusters), "tests": len(tests),
            "top": [(c["size"], c["label"]) for c in clusters[:5]]}


def export_tests(store, path, status: str = "confirmed", one_per_cluster: bool = False) -> int:
    tests = store.tests(status=None if status == "all" else status)
    if one_per_cluster:
        reps = {c["representative"] for c in store.clusters()}
        seen_clusters = set()
        picked = []
        for t in sorted(tests, key=lambda t: t.test_id not in reps):
            key = t.cluster_id or t.test_id
            if key not in seen_clusters:
                seen_clusters.add(key)
                picked.append(t)
        tests = picked
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, [t.to_dict() for t in tests])
    return len(tests)


def load_tests(path) -> list:
    return [TestCase.from_dict(d) for d in read_jsonl(path)]


def load_baseline(path) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_baseline(path, results) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({r.test_id: r.outcome for r in results}, indent=2, sort_keys=True) + "\n",
                 encoding="utf-8")
