# Implementation guide: from offline build to Azure

> Status: **partly built**. The Azure OpenAI chat client exists and is selected
> when its env vars are set. Nothing has been deployed, and CI does not call
> Azure. This guide is how the offline build would be taken to a real tenant.

## Azure service mapping

| Offline component | Azure service (planned) | Notes |
|---|---|---|
| `LLMClient` | Azure OpenAI chat completions with tool calling | Client is in `orchestrator/llm/azure.py`. CI uses the mock. |
| Patterns | Microsoft Agent Framework or Microsoft Foundry Agent Service workflows | Keep the same pattern choice |
| Tools | Functions or MCP servers behind APIM |  |
| Trajectories | OpenTelemetry spans to Application Insights |  |
| Human handoff | Service Bus queue to a support tool |  |

## Steps

1. Point `AzureOpenAIClient` at a deployment (`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`) and fill the live rows with `python examples/compare_frameworks.py --live` and `python examples/compare_control.py --live`. Keep the mock for tests.
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
