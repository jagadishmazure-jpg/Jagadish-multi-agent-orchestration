# Jagadish Multi-Agent Orchestration

Multi-agent orchestration patterns as code: **supervisor**, **router**,
**sequential**, **blackboard**, and **hierarchical** — plus a tool-calling
agent loop, a deterministic fake LLM, trajectory recording, and task-success
checks. Everything runs **fully offline**: no network access, no API keys,
no flaky tests.

## Architecture

```
                        ┌──────────────────────────┐
                        │        scenarios/        │
                        │  research_team  triage   │
                        └────────────┬─────────────┘
                                     │ composes
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
   │ orchestrator/    │   │ orchestrator/    │   │ orchestrator/    │
   │ patterns/        │   │ tools/           │   │ eval/            │
   │                  │   │                  │   │                  │
   │ • supervisor     │   │ • ToolRegistry   │   │ • Trajectory     │
   │ • router         │   │ • ToolCallingLoop│   │   Recorder       │
   │ • sequential     │   │ • calculator     │   │ • checks         │
   │ • blackboard     │   │ • search stub    │   │   (keywords,     │
   │ • hierarchical   │   │ • clock stub     │   │    budgets, ...) │
   └────────┬─────────┘   └──────────────────┘   └──────────────────┘
            │ uses
   ┌────────▼─────────┐   ┌──────────────────┐
   │ orchestrator/    │   │ orchestrator/    │
   │ core/            │   │ llm/             │
   │                  │   │                  │
   │ • Agent          │   │ • LLMClient      │
   │ • Message        │   │   (interface)    │
   │ • TaskState      │   │ • FakeLLM        │
   │ • Handoff        │   │   (scripted,     │
   │   protocol       │   │    deterministic)│
   └──────────────────┘   └──────────────────┘
```

Pattern walkthroughs live in [docs/patterns.md](docs/patterns.md).

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev

# Run a scenario in a Python shell
uv run python -c "
from scenarios import run_research_scenario, triage_ticket
print(run_research_scenario()['final'])
print(triage_ticket('please refund my last invoice')['output'])
"
```

Or piece the primitives together directly:

```python
from orchestrator.core import Agent
from orchestrator.llm import FakeLLM
from orchestrator.patterns import SequentialPipeline

planner = Agent("planner", "planner", FakeLLM(default="PLAN: ..."))
writer = Agent("writer", "writer", FakeLLM(default="DRAFT: ..."))
result = SequentialPipeline([planner, writer]).run("Write a launch note")
print(result.final)
```

## How to run the tests

```bash
uv run pytest            # full suite (120+ tests, all offline)
uv run pytest -k router  # subset, e.g. router pattern tests
```

The suite covers every pattern, the tool-calling loop, handoffs, trajectory
recording, both end-to-end scenarios, and failure modes (max-iteration
cutoff, unknown tools, agent errors, unparseable plans).

## Layout

| Path | Contents |
| ---- | -------- |
| `orchestrator/core/` | `Agent`, `Message`, `TaskState`, handoff protocol |
| `orchestrator/patterns/` | supervisor, router, sequential, blackboard, hierarchical |
| `orchestrator/tools/` | tool registry, tool-calling loop, offline built-ins |
| `orchestrator/llm/` | `LLMClient` interface + deterministic `FakeLLM` |
| `orchestrator/eval/` | trajectory recorder + task-success checks |
| `scenarios/` | research team, support-triage pipeline (offline) |
| `evals/` | regression scenario definitions |
| `docs/` | pattern overviews |

## License

MIT — see [LICENSE](LICENSE).
