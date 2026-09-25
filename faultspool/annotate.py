"""Terminal annotation: confirm real failures and correct expected output.

Goes through pending tests largest-cluster first (so one decision can cover
a whole root cause) and asks for a verdict on each.
"""
from __future__ import annotations

import json
from typing import Callable, Optional

from .convert import TestCase

HELP = """  [c] confirm   [r] reject (not a real failure)   [e] set expected output
  [a] add assertion (JSON)   [x] apply verdict to whole cluster   [s] skip   [q] quit"""


def _clip(v, n=300) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "…"


def render(test: TestCase, cluster_size: int = 1) -> str:
    lines = [f"── {test.test_id}  (cluster {test.cluster_id or '-'} × {cluster_size}, origin={test.origin})",
             f"task:  {_clip(test.input, 200)}", "failures:"]
    for f in test.failures:
        mark = " " if f.get("severity") == "low" else "!"
        lines.append(f"  {mark} {f['kind']:<18} step {f.get('step_idx')}  {_clip(f.get('message', ''), 140)}")
    lines.append(f"recorded tool calls: {len(test.mocks)}")
    for m in test.mocks[-4:]:
        res = f"ERROR {m['error']}" if m.get("error") else _clip(m.get("result"), 100)
        lines.append(f"  {m['name']}({_clip(m.get('args'), 80)}) -> {res}")
    lines.append("assertions:")
    for a in test.assertions:
        lines.append(f"  - {json.dumps(a, default=str)}")
    return "\n".join(lines)


def expected_assertion(text: str) -> dict:
    """Parse a typed expectation: JSON value -> equals; 're:...' -> regex; else contains."""
    if text.startswith("re:"):
        return {"type": "output_matches", "pattern": text[3:]}
    if text.startswith("="):
        return {"type": "output_equals", "value": text[1:]}
    try:
        return {"type": "output_equals", "value": json.loads(text)}
    except ValueError:
        return {"type": "output_contains", "value": text}


def annotate(store, *, ask: Callable = input, say: Callable = print, limit: Optional[int] = None) -> dict:
    tests = store.tests(status="pending")
    sizes = {c["cluster_id"]: c["size"] for c in store.clusters()}
    tests.sort(key=lambda t: (-sizes.get(t.cluster_id, 1), t.created_at))
    done = {"confirmed": 0, "rejected": 0, "skipped": 0}
    say(HELP)
    for i, t in enumerate(tests):
        if limit is not None and i >= limit:
            break
        if t.status != "pending":  # may have been decided via a cluster verdict
            continue
        say("\n" + render(t, sizes.get(t.cluster_id, 1)))
        while True:
            choice = ask("verdict> ").strip().lower()
            if choice in ("c", "r"):
                t.status = "confirmed" if choice == "c" else "rejected"
                store.upsert_test(t)
                done[t.status] += 1
                break
            if choice == "e":
                exp = ask("expected output (JSON, =exact text, re:regex, or text it must contain)> ").strip()
                if exp:
                    t.assertions = [a for a in t.assertions if not a["type"].startswith("output_")]
                    t.assertions.append(expected_assertion(exp))
                    t.origin = "human"
                    say(f"  + {json.dumps(t.assertions[-1])}")
                continue
            if choice == "a":
                raw = ask("assertion JSON> ").strip()
                try:
                    a = json.loads(raw)
                    assert isinstance(a, dict) and "type" in a
                    t.assertions.append(a)
                    t.origin = "human"
                except (ValueError, AssertionError):
                    say("  not a valid assertion (need a JSON object with a 'type')")
                continue
            if choice == "x":
                verdict = ask("apply to cluster: [c]onfirm or [r]eject> ").strip().lower()
                if verdict in ("c", "r") and t.cluster_id:
                    status = "confirmed" if verdict == "c" else "rejected"
                    for other in tests:
                        if other.cluster_id == t.cluster_id and other.status == "pending":
                            other.status = status
                            store.upsert_test(other)
                            done[status] += 1
                    t.status = status
                    store.upsert_test(t)
                    break
                continue
            if choice == "s":
                done["skipped"] += 1
                break
            if choice == "q":
                return done
            say(HELP)
    return done
