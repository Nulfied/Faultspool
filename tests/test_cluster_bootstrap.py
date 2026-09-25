from conftest import BENCH

from faultspool.bootstrap import import_path, sniff
from faultspool.cluster import cluster_tests, cosine, hash_embed, normalize
from faultspool.convert import TestCase


def fake_test(msg, kind="tool_error", tool="fetch", created=0.0):
    f = {"kind": kind, "detector": "rule", "message": msg, "tool": tool, "severity": "medium",
         "signature": f"{kind}:{tool}"}
    return TestCase(input=None, mocks=[], assertions=[], failures=[f], source_trace_id=msg,
                    task_id="t", created_at=created)


def test_normalize_strips_volatile_parts():
    a = normalize("Timeout after 30.5s calling /api/v1/users/8812 id=3f2a9c1b-1111-2222-3333-444455556666")
    b = normalize("Timeout after 12s calling /api/v1/users/17 id=aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    assert a == b


def test_near_duplicates_cluster_and_size_is_priority():
    tests = [fake_test(f"HTTP 503 from upstream after {i} retries", created=i) for i in range(5)]
    tests += [fake_test("KeyError: 'amount'", kind="exception", tool=None, created=10)]
    clusters, _ = cluster_tests(tests, hash_embed, threshold=0.85)
    assert [c["size"] for c in clusters] == [5, 1]
    assert clusters[0]["representative"] == tests[0].test_id


def test_hash_embed_is_unit_and_deterministic():
    v = hash_embed("loop search_orders")
    assert abs(cosine(v, v) - 1) < 1e-9 and v == hash_embed("loop search_orders")


def test_benchmark_adapters():
    traces = list(import_path(BENCH))
    by_source = {}
    for t in traces:
        by_source.setdefault(t.source, []).append(t)
    assert set(by_source) == {"toolbench", "swe-agent", "agentbench", "webarena", "openai"}

    tb = by_source["toolbench"][0]
    assert tb.input == "What's the weather in Paris tomorrow?"
    assert tb.tool_steps()[0].error == "Rate limit exceeded"

    swe = by_source["swe-agent"][0]
    assert swe.status == "timeout" and swe.tool_steps()[0].name == "find_file"

    ab = sorted(by_source["agentbench"], key=lambda t: t.task_id)
    assert ab[0].status == "timeout" and ab[1].output == "3"

    wa = sorted(by_source["webarena"], key=lambda t: t.task_id)
    assert wa[0].output == "$39.99" and wa[0].meta["benchmark_success"] is False
    assert wa[1].meta["benchmark_success"] is True

    oa = by_source["openai"][0]
    assert [s.name for s in oa.tool_steps()] == ["book_table", "cancel_table"]


def test_sniff():
    assert sniff({"answer_generation": {}}) == "toolbench"
    assert sniff({"trajectory": [], "info": {}}) == "swe-agent"
    assert sniff({"output": {"history": []}}) == "agentbench"
    assert sniff({"intent": "x"}) == "webarena"
    assert sniff({"messages": []}) == "openai"
