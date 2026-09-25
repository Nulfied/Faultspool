import json

import pytest

from conftest import BENCH
from faultspool import Store, record
from faultspool.annotate import annotate, expected_assertion
from faultspool.cli import main
from faultspool.dashboard import build


@pytest.fixture
def store_with_failures(tmp_path, toy):
    store = Store(str(tmp_path / "fs.db"))
    for task in toy.TASKS:
        record(toy.run_v1, task, toy.TOOLS, max_tool_calls=8, sink=store)
    yield store
    store.close()


def test_full_cli_flow(tmp_path, toy, store_with_failures, capsys, monkeypatch):
    db = str(store_with_failures.path)
    store_with_failures.close()
    main(["--db", db, "process", "--no-ollama"])
    out = capsys.readouterr().out
    assert '"failing": 3' in out and "3 cluster(s)" in out

    with Store(db) as s:  # confirm everything, as `faultspool annotate` would
        for t in s.tests():
            t.status = "confirmed"
            s.upsert_test(t)

    tests_file = tmp_path / "spool" / "tests.jsonl"
    main(["--db", db, "export", str(tests_file)])
    assert len(tests_file.read_text().splitlines()) == 3

    baseline = tmp_path / "spool" / "baseline.json"
    main(["--db", db, "run", "--agent", "examples.toy_agent:run_v2", "--tests", str(tests_file),
          "--baseline", str(baseline), "--update-baseline", "--max-tool-calls", "8"])
    # linus has no successful retry recorded, so v2's refund() call has nothing to replay
    assert sorted(json.loads(baseline.read_text()).values()) == ["passes", "passes", "unreplayable"]

    with pytest.raises(SystemExit) as e:  # v1 regresses against a passing baseline
        main(["--db", db, "run", "--agent", "examples.toy_agent:run_v1", "--tests", str(tests_file),
              "--baseline", str(baseline), "--max-tool-calls", "8"])
    assert e.value.code == 1
    assert "REGRESSION" in capsys.readouterr().out

    report = tmp_path / "report.html"
    main(["--db", db, "dashboard", "-o", str(report)])
    html = report.read_text(encoding="utf-8")
    assert "Pass rate by agent version" in html and "still fails" in html


def test_annotate_scripted(store_with_failures):
    from faultspool import pipeline
    pipeline.detect_all(store_with_failures)
    pipeline.convert_all(store_with_failures)
    answers = iter(["e", "=No orders found for customer Grace.", "c", "r", "s"])
    done = annotate(store_with_failures, ask=lambda _: next(answers), say=lambda *_: None)
    assert done == {"confirmed": 1, "rejected": 1, "skipped": 1}
    confirmed = store_with_failures.tests(status="confirmed")[0]
    assert confirmed.origin == "human"
    assert {"type": "output_equals", "value": "No orders found for customer Grace."} in confirmed.assertions


def test_expected_assertion_parsing():
    assert expected_assertion("re:R-\\d+")["type"] == "output_matches"
    assert expected_assertion('{"a": 1}') == {"type": "output_equals", "value": {"a": 1}}
    assert expected_assertion("refunded")["type"] == "output_contains"


def test_ingest_benchmarks_and_dashboard(tmp_path):
    db = str(tmp_path / "b.db")
    main(["--db", db, "process", str(BENCH), "--no-ollama"])
    with Store(db) as s:
        assert s.stats()["traces"] == 8
        assert s.stats()["tests"] >= 4
        path = build(s, str(tmp_path / "r" / "index.html"))
    assert "Top clusters" in path.read_text(encoding="utf-8")
