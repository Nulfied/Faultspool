import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from examples import toy_agent  # noqa: E402
from faultspool import record  # noqa: E402

BENCH = ROOT / "examples" / "benchmarks"


@pytest.fixture
def toy():
    return toy_agent


@pytest.fixture
def v1_traces():
    return {t["customer"]: record(toy_agent.run_v1, t, toy_agent.TOOLS, max_tool_calls=8, timeout_s=5,
                                  agent_version="v1")
            for t in toy_agent.TASKS}
