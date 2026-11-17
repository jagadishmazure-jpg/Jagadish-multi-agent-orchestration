# Orchestration Patterns

Five multi-agent orchestration patterns, implemented in
`orchestrator/patterns/`. Every pattern is deterministic and runs offline
against `FakeLLM`, so each one is covered by the test suite with no API keys.

## 1. Supervisor — decompose → delegate → synthesize

A decomposer agent splits the task into `worker: subtask` lines, each subtask
is delegated to the named worker, and a synthesizer merges the outputs.

```
              ┌─────────────┐
              │ Decomposer  │  "researcher: find facts
              └──────┬──────┘   writer: write summary"
                     ├──────────────────┬───────────────
              ┌──────▼──────┐    ┌──────▼──────┐
              │ Researcher  │    │   Writer    │
              └──────┬──────┘    └──────┬──────┘
                     └────────┬────────┘
                       ┌──────▼──────┐
                       │ Synthesizer │
                       └─────────────┘
```

Use when: the task splits into independent subtasks owned by specialists.
Watch for: plans naming unknown workers (`UnknownWorkerError`), unparseable
plans (`ValueError`).

## 2. Router — classify + route to specialist

A classifier emits `CATEGORY: <name>`; the router dispatches to the matching
specialist agent. An optional default specialist handles unknown categories.

```
              ┌─────────────┐
              │ Classifier  │── "CATEGORY: billing"
              └──────┬──────┘
        ┌───────────┼───────────┐
  ┌─────▼─────┐ ┌───▼────┐ ┌────▼─────┐
  │  Billing  │ │Technical│ │ Account  │
  └───────────┘ └────────┘ └──────────┘
```

Use when: inputs fall into distinct kinds needing different expertise.
Watch for: missing `CATEGORY:` in the classifier output, categories with no
specialist (`UnknownRouteError`).

## 3. Sequential — pipeline with handoffs

Agents run in order; each receives the previous agent's output through the
formal handoff protocol (`Handoff` → `perform_handoff`).

```
  ┌─────────┐   handoff   ┌────────────┐   handoff   ┌────────┐
  │ Planner │ ──────────▶ │ Researcher │ ──────────▶ │ Writer │ ──▶ Critic
  └─────────┘             └────────────┘             └────────┘
```

Use when: the work is a fixed chain of transformations (research team,
ETL-style flows). Watch for: one weak stage degrading everything downstream —
evaluate each stage independently.

## 4. Blackboard — shared workspace

Agents take turns reading and writing a shared board dict. An agent signals it
is done by emitting the done marker (`DONE`); the run stops early on full
convergence or after `max_rounds`.

```
              ┌─────────────────────┐
              │   Shared board      │
              │ task / drafts / ... │
              └────────┬────────────┘
        ┌─────────────┼──────────────┐
   ┌────▼───┐   ┌────▼───┐    ┌─────▼──┐
   │ Agent A│   │ Agent B│    │ Agent C│
   └────────┘   └────────┘    └────────┘
```

Use when: agents must build on each other's partial work opportunistically.
Watch for: non-convergence — always set a `max_rounds` budget.

## 5. Hierarchical — manager + sub-teams

A manager decomposes into `team: subtask` assignments; each sub-team runs a
sequential pipeline; the manager aggregates the team results.

```
                    ┌─────────┐
                    │ Manager │
                    └────┬────┘
            ┌───────────┴────────────┐
     ┌──────▼──────┐          ┌──────▼──────┐
     │ Team A pipe │          │ Team B pipe │
     └──────┬──────┘          └──────┬──────┘
            └───────────┬───────────┘
                  ┌─────▼─────┐
                  │ Aggregate │
                  └───────────┘
```

Use when: the task has natural sub-domains, each needing its own pipeline.
Watch for: plans naming unknown teams (`UnknownTeamError`).

## Cross-cutting pieces

- **Tool loop** (`orchestrator/tools/loop.py`): ReAct-style loop with an
  iteration budget (`MaxIterationsError`), strict tool validation, and visible
  failure modes (unknown tool, agent errors). This is the ReAct baseline in
  the control-pattern comparison.
- **Trajectory recorder** (`orchestrator/eval/recorder.py`): ordered event log
  for every run — the basis of the regression evals.
- **Checks** (`orchestrator/eval/checks.py`): keyword presence, subtask
  completion, error-free runs, step budgets, agent participation.

## 6. Plan-and-execute — planner, executor, replanner

A planner writes a list of `tool: query` steps. The executor runs the first
step (one model call, one tool). The replanner either returns the remaining
plan or `FINAL: <answer>`. On two of the three tasks the first plan is
incomplete, and the replanner adds the missing lookups and the calculator
step. The graph is the LangGraph shape (`StateGraph` when that package is
installed, the same nodes on the builtin graph otherwise).

```
  planner --> executor --> replanner -- plan left --> executor
                              |
                              +-- FINAL --> end
```

## 7. ReWOO — plan, then tools, then solve

The planner emits every call at once:

```
#E1 = search[capital of France]
#E3 = search[population of #E1]
#E5 = calculator[#E3+#E4]
```

The worker substitutes `#E` references and calls tools with no model turn in
between. The solver writes the answer from the evidence. On this task set
that is two model calls, against 13 for ReAct and 23 for plan-and-execute.
Numbers are in the README and come from `python examples/compare_control.py`.
