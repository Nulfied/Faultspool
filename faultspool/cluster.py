"""Dedup near-duplicate failures with free local embeddings.

Embeddings come from Ollama (``nomic-embed-text`` by default) when it is
running, otherwise from a dependency-free hashing embedder that works well for
the short, templated text of error messages. Clustering is single-pass leader
clustering on cosine similarity; cluster size doubles as priority.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from typing import Callable, Optional

from .convert import TestCase
from .detect.judge import OLLAMA_URL
from .schema import stable_hash

HASH_DIM = 512
EMBED_MODEL = os.environ.get("FAULTSPOOL_EMBED_MODEL", "nomic-embed-text")

_VOLATILE = [
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "<uuid>"),
    (re.compile(r"\b0x[0-9a-f]+\b|\b[0-9a-f]{12,}\b", re.I), "<hex>"),
    (re.compile(r"(/|[A-Za-z]:\\)[^\s'\"]+"), "<path>"),
    (re.compile(r"\d+(\.\d+)?"), "<n>"),
    (re.compile(r"'[^']{1,80}'|\"[^\"]{1,80}\""), "<str>"),
]


def normalize(text: str) -> str:
    """Strip ids, numbers, paths and literals so the same root cause maps to the same text."""
    for pat, rep in _VOLATILE:
        text = pat.sub(rep, text)
    return " ".join(text.lower().split())


def failure_text(test: TestCase) -> str:
    parts = []
    for f in test.failures:
        if f.get("severity") == "low":
            continue
        parts.append(f"{f['kind']} {f.get('tool') or ''} {normalize(f.get('message', ''))}")
    return " | ".join(sorted(set(parts))) or "no-failure"


def hash_embed(text: str, dim: int = HASH_DIM) -> list:
    vec = [0.0] * dim
    words = re.findall(r"<\w+>|\w+", text.lower())
    feats = words + [f"{a} {b}" for a, b in zip(words, words[1:])]
    feats += [w[i:i + 3] for w in words if len(w) > 3 for i in range(len(w) - 2)]
    for feat in feats:
        h = int.from_bytes(hashlib.md5(feat.encode()).digest()[:8], "little")
        vec[h % dim] += 1.0 if (h >> 63) & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def ollama_embed(text: str, model: str = EMBED_MODEL, timeout: float = 30.0) -> Optional[list]:
    for path, payload, key in (("/api/embed", {"model": model, "input": text}, "embeddings"),
                               ("/api/embeddings", {"model": model, "prompt": text}, "embedding")):
        try:
            req = urllib.request.Request(OLLAMA_URL + path, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                v = json.loads(r.read()).get(key)
            if v:
                v = v[0] if isinstance(v[0], list) else v
                n = math.sqrt(sum(x * x for x in v)) or 1.0
                return [x / n for x in v]
        except (urllib.error.URLError, OSError, ValueError):
            continue
    return None


def get_embedder(prefer_ollama: bool = True) -> tuple:
    """Return (name, fn). Probes Ollama once; falls back to the hashing embedder."""
    if prefer_ollama and ollama_embed("probe", timeout=3.0) is not None:
        return f"ollama:{EMBED_MODEL}", ollama_embed
    return f"hash{HASH_DIM}", hash_embed


def cosine(a: list, b: list) -> float:
    return sum(x * y for x, y in zip(a, b))  # vectors are unit-normalized


def leader_cluster(items: list, threshold: float) -> list:
    """items: [(id, vec)] in priority order. Returns [[ids...]] with the leader first."""
    leaders: list = []
    groups: list = []
    for item_id, vec in items:
        best, best_sim = None, threshold
        for gi, lvec in enumerate(leaders):
            sim = cosine(vec, lvec)
            if sim >= best_sim:
                best, best_sim = gi, sim
        if best is None:
            leaders.append(vec)
            groups.append([item_id])
        else:
            groups[best].append(item_id)
    return groups


def cluster_tests(tests: list, embed: Callable = hash_embed, threshold: float = 0.85,
                  cache: Optional[dict] = None) -> tuple:
    """Returns (clusters, vectors). Each cluster: id, label, size, representative, members."""
    cache = cache or {}
    vectors = {t.test_id: cache.get(t.test_id) or embed(failure_text(t)) for t in tests}
    by_id = {t.test_id: t for t in tests}
    ordered = sorted(tests, key=lambda t: t.created_at)
    groups = leader_cluster([(t.test_id, vectors[t.test_id]) for t in ordered], threshold)
    clusters = []
    for g in groups:
        rep = by_id[g[0]]
        sig = Counter(s for tid in g for s in by_id[tid].signatures).most_common(2)
        example = next((f["message"] for f in rep.failures if f.get("severity") != "low"), "")
        label = ", ".join(s for s, _ in sig) or "unknown"
        if example:
            label += f" | {example[:90]}"
        clusters.append({"cluster_id": "c_" + stable_hash(sorted(g))[:10], "label": label,
                         "size": len(g), "representative": rep.test_id, "members": g})
    clusters.sort(key=lambda c: -c["size"])
    return clusters, vectors
