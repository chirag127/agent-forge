# Agent Forge

**Live demo:** `pip install agent-forge && agent-forge run "Calculate 2^10"` | **Python 3.13** | [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://python.org)

Multi-agent LLM orchestration platform — Planner → Executor → Critic pipeline with pluggable tool-calling, an eval harness, structured observability, and a keyless LLM backend (no API key required).

---

## Architecture

```mermaid
flowchart TD
    User([User Task]) --> CLI
    CLI --> Orchestrator

    subgraph Orchestrator
        P[Planner Agent\nDecomposes task → Steps]
        E[Executor Agent\nRuns each step]
        C[Critic Agent\nVerifies result]
        BB[(Shared Blackboard\nMessage Bus)]
        P -->|plan| E
        E -->|write results| BB
        BB -->|context| C
        P -.->|read context| BB
    end

    subgraph Tools
        T1[web_fetch\nhttpx GET]
        T2[python_eval\nAST sandbox]
        T3[file_read\nworkdir-restricted]
        T4[calculator\nmath eval]
    end

    subgraph LLM Provider
        K[kilo.ai\nfree auto-router]
        PL[pollinations.ai\nfallback]
        G[g4f\nlast-resort]
        K -->|fail| PL
        PL -->|fail| G
    end

    E --> Tools
    P & E & C --> LLM Provider
    Orchestrator --> RunResult([RunResult\nanswer + verdict])
```

## How it works

**Planner** receives a task, queries the LLM, and decomposes it into ordered steps — each step either invokes a tool (JSON-specified name + args) or is a pure reasoning step. Output is parsed JSON.

**Executor** runs each step in sequence. If a tool is specified, it validates the args against the tool's JSON Schema, calls the function, then asks the LLM to synthesize the raw tool output into a natural-language finding. Reasoning steps go straight to the LLM. Context from prior steps is threaded forward.

**Critic** reviews the full step log and produces a `pass`/`fail` verdict with a reason and confidence score. This mirrors real LLM-as-judge eval patterns used in production systems.

**Blackboard** is an async-safe shared key-value store with pub/sub — the communication layer for multi-agent runs where multiple named agents collaborate.

---

## Quickstart

```bash
pip install agent-forge
# or from source
git clone https://github.com/chirag127/agent-forge
cd agent-forge
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .

# Run a task
agent-forge run "What is the square root of 2025?"

# Run with verbose logs + OpenTelemetry spans
agent-forge run "Calculate compound interest: 10000 at 8% for 5 years" --verbose --otel

# Run eval harness over gold dataset
agent-forge eval

# Custom dataset / report path
agent-forge eval --dataset path/to/gold.json --report path/to/report.md
```

---

## Tool system

Tools are registered with a decorator:

```python
from agent_forge.tools import tool

@tool(
    name="my_tool",
    description="Does something useful.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)
def my_tool(query: str) -> str:
    return f"Result for: {query}"
```

Built-in tools:

| Tool | What it does |
|---|---|
| `calculator` | Evaluates math expressions (`sqrt`, `**`, trig) |
| `python_eval` | Runs sandboxed Python — no imports, no dunder access |
| `web_fetch` | HTTP GET via httpx, strips HTML, truncates at 4 000 chars |
| `file_read` | Reads files restricted to a working directory |

---

## LLM provider

No API keys. Provider chain: **kilo.ai** (free auto-router) → **pollinations.ai** (fallback) → **g4f** (last resort). Any provider failure silently cascades to the next. Swap the chain via `LLMConfig`:

```python
from agent_forge.providers import LLMProvider, LLMConfig

provider = LLMProvider(LLMConfig(timeout=30))
resp = provider.complete([Message("user", "Hello")])
print(resp.content, resp.provider, resp.latency_ms)
```

---

## Eval harness

`evals/gold_dataset.json` — 8 tasks spanning math, reasoning, Python eval, multi-step. Each case has typed assertions (`contains`, `contains_ci`, `contains_any`, `not_contains`, `llm_judge`).

`evals/run_eval.py` — runs the orchestrator over every case, scores pass/fail, writes `evals/report.md`.

```bash
python evals/run_eval.py
# or via CLI:
agent-forge eval
```

The eval harness is the pattern used in production LLM systems to measure agent capability over a fixed benchmark. It supports adding an `llm_judge` assertion type for cases where string matching is insufficient.

---

## Observability

Every agent step, tool call, token count, and latency is logged via **structlog** (structured JSON or pretty console). **OpenTelemetry** spans (planner.plan, executor.step, critic.verify, orchestrator.run) are emitted to a console exporter by default — drop in any OTLP-compatible backend (Jaeger, Honeycomb, etc.) without code changes.

```python
from agent_forge.observability import configure
configure(level="DEBUG", otel=True, json_logs=True)
```

---

## Multi-agent usage

```python
from agent_forge.agents import Orchestrator
from agent_forge.blackboard import Blackboard

bb = Blackboard()
orch = Orchestrator(blackboard=bb)
result = orch.run("Summarize and evaluate: the Fibonacci sequence")

# inspect shared state
print(bb.snapshot())
print(bb.history())
```

---

## Tests

```bash
pip install -e ".[dev]"
# unit + integration (no network):
pytest tests/unit/ tests/integration/ -v
# live network tests (hit real LLM):
pytest tests/test_llm_live.py -v -s
# full suite with coverage:
pytest
```

Coverage target: 80% of `agent_forge/` core.

---

## Resume keywords this repo backs

`multi-agent orchestration` · `LLM evals / LLM-as-judge` · `planner-executor-critic` · `tool-calling / function-calling` · `JSON Schema validation` · `OpenTelemetry instrumentation` · `structlog structured logging` · `AST sandboxed code execution` · `provider failover / resilience` · `async Python / asyncio` · `pytest unit + integration` · `keyless LLM inference`
