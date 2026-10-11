# Jagadish Multi-Agent Orchestration

[![CI](https://github.com/jagadishmazure-jpg/Jagadish-multi-agent-orchestration/actions/workflows/ci.yml/badge.svg)](https://github.com/jagadishmazure-jpg/Jagadish-multi-agent-orchestration/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![Tests](https://img.shields.io/badge/tests-124%20passing-brightgreen)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Multi-agent orchestration **patterns as code**: supervisor, router,
sequential, blackboard, and hierarchical — plus a tool-calling agent loop, a
deterministic fake LLM, trajectory recording, and task-success checks.
Everything runs **fully offline**: no network access, no API keys, no flaky
tests.

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
uv run pytest -q
uv run python -c "
from scenarios import run_research_scenario, triage_ticket
print(run_research_scenario()['final'])
print(triage_ticket('please refund my last invoice')['output'])
"
```

## Project structure

```
orchestrator/
  core/        Agent, Message, TaskState, Handoff protocol
  patterns/    supervisor, router, sequential, blackboard, hierarchical
  tools/       ToolRegistry, ToolCallingLoop, stub tools (calculator, search, clock)
  llm/         LLMClient interface + deterministic FakeLLM
  eval/        TrajectoryRecorder, task-success checks
scenarios/     research_team, support_triage — end-to-end compositions
docs/          patterns.md — pattern walkthroughs
tests/         124 offline pytest tests
```

## Tests & CI

**124 tests, all passing, fully offline.** GitHub Actions runs the suite on
every push and pull request (Python 3.12, via `uv`). Releases ship to PyPI
from version tags (`v*`) via trusted publishing — see
`.github/workflows/release.yml` (add manually; workflow files can't be
pushed via the GitHub API).

## Portfolio

Part of an agentic-AI engineering portfolio by
[jagadishmazure-jpg](https://github.com/jagadishmazure-jpg):

| Repo | What it is | Tests |
|---|---|---|
| **This repo** | Orchestration patterns as code | 124 |
| [Jagadish-agent-security-redteam](https://github.com/jagadishmazure-jpg/Jagadish-agent-security-redteam) | Red-teaming suite: attack catalog, layered defenses, risk scoring | 206 |
| [Jagadish-llm-evals-observability](https://github.com/jagadishmazure-jpg/Jagadish-llm-evals-observability) | LLM evals + observability as code | 111 |
| [Jagadish-ingredion-agentic-ai](https://github.com/jagadishmazure-jpg/Jagadish-ingredion-agentic-ai) | End-to-end multi-agent platform (Microsoft Agent Framework 1.0) + live demo | 564 |

## License

MIT — see [LICENSE](LICENSE).
