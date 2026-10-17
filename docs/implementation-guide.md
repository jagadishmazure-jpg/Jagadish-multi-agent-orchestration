# Implementation guide: from offline build to Azure

> Status: **planned**. Nothing here has been deployed. The offline build in this
> repository is what exists today; this guide is how it would be taken to a
> real Azure environment.

## Azure service mapping

| Offline component | Azure service (planned) | Notes |
|---|---|---|
| `LLMClient` | Azure OpenAI chat completions with tool calling | Managed identity |
| Patterns | Microsoft Agent Framework or Azure AI Foundry Agent Service workflows | Keep the same pattern choice |
| Tools | Functions or MCP servers behind APIM |  |
| Trajectories | OpenTelemetry spans to Application Insights |  |
| Human handoff | Service Bus queue to a support tool |  |

## Steps

1. Implement `LLMClient` against Azure OpenAI; keep `FakeLLM` for tests.
2. Pick the pattern per use case: router for triage, sequential for fixed pipelines, supervisor for open tasks.
3. Add a default human-queue route for unmatched tickets.
4. Export trajectories as traces and load the YAML expectations in the regression tests.

## Guardrails to keep when going live

- Tool loop has a hard iteration budget and raises `MaxIterationsError` when exceeded.
- Router and supervisor raise on unknown routes or workers instead of guessing (a default route is opt-in).
- Blackboard stops at a round limit and reports `converged=False`.
- Every run leaves a trajectory that the checks can assert on.

## What to measure before calling it production-ready

- Task success rate per pattern on the same task set.
- Steps and tokens per completed task.
- Share of tickets routed to the human queue.
