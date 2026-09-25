# Faultspool

**Turn your agent's failures into regression tests, automatically.**

Faultspool records what your AI agent does, finds the runs that went wrong, and turns each
one into a small deterministic test that replays the recorded tool responses. Run the tests
on every commit to see which past failures are fixed, which still fail, and which fail in a
new way.

Everything runs on your own machine and free CI. It has no runtime dependencies (stdlib only),
uses SQLite for storage, and uses a local Ollama model when you want an LLM judge or
embeddings.

```
 capture ──► detect ──► convert ──► cluster ──► annotate ──► run in CI ──► dashboard
 (traces)   (rules,     (minimal    (dedup by   (confirm,    (still fails /
            heuristics,  repro +     local       fix         fails differently /
            local judge) mocks)      embeddings) expected)   passes)
```

## Quickstart

```bash
pip install git+https://github.com/Nulfied/Faultspool
```

```bash
git clone https://github.com/Nulfied/Faultspool && cd Faultspool && python examples/demo.py
```

The demo runs a small refund-desk agent (`examples/toy_agent.py`) that has three planted bugs.
Faultspool captures them, turns them into 3 tests, confirms that v1 fails all 3 and v2 passes
all 3, and writes a dashboard to `examples/.demo/report/index.html`.

## 1. Capture traces

The agent contract is plain Python: `agent(task, tools) -> output`, where `tools` is a dict of
callables.

```python
from faultspool import record, Store

store = Store()  # .faultspool/faultspool.db
trace = record(my_agent, task, {"search": search, "refund": refund},
               agent="support-bot", agent_version="1.4.0",
               max_tool_calls=30, timeout_s=60, sink=store)
```

`record` never raises. Exceptions, timeouts, and runaway loops (via the tool-call budget) all
end up in the trace. If your agent takes a `recorder` keyword argument, you can also log
reasoning and model calls:

```python
def my_agent(task, tools, recorder=None):
    recorder.reasoning("look up the order first")
    recorder.llm(response, prompt=prompt, model="llama3.1")
```

If you control the loop yourself, use `Recorder` directly:

```python
with Recorder(task, agent="bot", sink="traces.jsonl") as rec:
    tools = rec.wrap_tools(TOOLS)
    ...
    rec.output(answer)
```

**Schema.** A trace has `input`, `steps`, `output`, `status` (`ok|error|timeout|incomplete`),
`task_id`, `agent`, `agent_version`, `source`, and `meta`. Every step has the same shape:
`idx, kind, ts, name, args, content, error, duration_ms, meta`. `kind` is one of `input`,
`llm`, `reasoning`, `tool`, `output`, or `error`. See [`faultspool/schema.py`](faultspool/schema.py).

## 2. Detect failures

| detector | finds |
|---|---|
| **rule** | `exception`, `tool_error` (low severity if the agent recovered), `timeout`, `malformed_output` (empty output, broken JSON, or `meta.output_schema` violations), `wrong_answer` (benchmark ground truth) |
| **heuristic** | `loop` (the same call 3+ times, or A→B→A→B cycles), `self_contradiction` (create→delete, book→cancel on the same args), `unsupported_claim` (claims success after the last call failed), `abandoned` (no output, gave up, or stopped right after an error) |
| **judge** (opt-in) | `off_track`: a local Ollama model reads a compact version of the trace and decides whether the agent went off track |

```bash
faultspool detect                 # rules + heuristics
faultspool detect --judge --model llama3.1   # plus the local LLM judge (ollama serve)
```

## 3. Trace → test case

`faultspool convert` turns each failing trace into a test with:

- **the minimal reproducible unit**: the task input plus the recorded tool responses up to
  the first failing step, served back as mocks, so the test needs no live APIs.
- **assertions**: at minimum, the original failure kinds must not come back. If the same task
  has a later successful attempt (a **self-play retry**), Faultspool diffs the two traces and
  infers more: grounded facts the output must contain, tools that must be called, and a
  tool-call budget. The retry's tool responses are added to the mocks so the fixed behavior
  can be replayed too.

During replay, each call is matched by tool name and arguments first, then by the next unused
recording of that tool. `--strict-replay` turns the second fallback off. If the agent asks for
something that was never recorded, the call fails loudly and the test is marked
`unreplayable`. It is never silently passed.

## 4. Bootstrap without your own traffic

Import open benchmark trajectories to build and validate the pipeline before you have users:

```bash
faultspool process path/to/trajectories/     # ingest + detect + convert + cluster
```

| `--format` | reads |
|---|---|
| `toolbench` | ToolBench `answer_generation` files (`train_messages`, `final_answer`) |
| `swe-agent` | SWE-agent / SWE-bench `.traj` files (`trajectory`, `info.exit_status`) |
| `agentbench` | AgentBench run outputs (`output.history`, `output.status`) |
| `webarena` | WebArena-style action logs (`intent`, `actions`, `success`/`score`) |
| `openai` | any OpenAI-style `messages` with `tool_calls` / `function_call` |
| `native` | Faultspool JSONL |

`auto` (the default) detects the format for each record. Small samples of each format are in
[`examples/benchmarks/`](examples/benchmarks). Benchmark formats change between releases, and
each adapter is a single short function in [`faultspool/bootstrap.py`](faultspool/bootstrap.py).

## 5. Dedup / clustering

```bash
faultspool cluster                # Ollama nomic-embed-text if running, else built-in hashing embedder
```

Before embedding, failure text is normalized: ids, numbers, paths, and quoted literals are
stripped out. Near-duplicates are then grouped by cosine similarity. Cluster size is the
priority signal. `faultspool export --dedup` keeps one test per root cause.

## 6. Annotate

```bash
faultspool annotate
```

Tests are shown largest cluster first. For each one you can **c**onfirm, **r**eject, set the
**e**xpected output (`=exact`, `re:regex`, JSON, or text it must contain), **a**dd any
assertion, or apply a verdict to the whole cluster (**x**).

## 7. Regression runner + CI

```bash
faultspool export spool/tests.jsonl                  # confirmed tests → git
faultspool run --agent mypkg.agent:run \
  --tests spool/tests.jsonl --baseline spool/baseline.json --update-baseline
```

| outcome | meaning |
|---|---|
| `passes` | no real failures, and all assertions hold |
| `still_fails` | an original failure signature came back |
| `fails_differently` | it breaks, but in a new way |
| `unreplayable` | the agent needed tool responses that were never recorded |

`--fail-on regression` (the default) exits 1 only when a test that passed in the baseline no
longer passes. `--fail-on any` makes every test that isn't passing fail CI.
[`.github/workflows/regression.yml`](.github/workflows/regression.yml) is a ready-to-copy
GitHub Actions job that also uploads the dashboard as an artifact.

## 8. Dashboard

```bash
faultspool dashboard -o faultspool-report/index.html
```

This writes one self-contained HTML file with no JavaScript and no CDN. It shows pass rate per
agent version, failing traces per day, failure kinds, top clusters, and the results of the
latest run.

## Storage

- **SQLite** (`.faultspool/faultspool.db`) holds traces, failures, tests, embeddings (float32
  blobs, brute-force cosine), clusters, and runs.
- **JSONL** is used for trace sinks and for exported tests, so tests can be diffed in git.

## CLI

```
faultspool [--db PATH] {init,ingest,detect,convert,cluster,process,annotate,export,run,dashboard,stats}
```

Environment variables: `FAULTSPOOL_DB`, `FAULTSPOOL_AGENT`, `FAULTSPOOL_AGENT_VERSION`,
`OLLAMA_HOST`, `FAULTSPOOL_JUDGE_MODEL` (default `llama3.1`), `FAULTSPOOL_EMBED_MODEL`
(default `nomic-embed-text`).

## Development

```bash
pip install -e ".[dev]"
```

```bash
pytest -q
```

## License

MIT
