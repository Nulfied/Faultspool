"""Faultspool: auto-generate regression tests from agent traces."""
from .capture import Recorder, record
from .convert import TestCase, trace_to_test
from .detect import detect, is_failing
from .replay import MockTools
from .runner import run_suite, run_test
from .schema import Failure, Step, Trace
from .storage import Store

__version__ = "0.1.0"

__all__ = ["Failure", "MockTools", "Recorder", "Step", "Store", "TestCase", "Trace", "detect",
           "is_failing", "record", "run_suite", "run_test", "trace_to_test", "__version__"]
