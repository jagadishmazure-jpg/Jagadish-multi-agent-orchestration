# Interview points

Short, honest talking points for Jagadish-multi-agent-orchestration. Every number quoted is from the
real run shown in the README.

- Pattern choice is about who decides the next step; I can show all five on one core.
- Fail loudly: an unmatched ticket raises instead of going to the wrong specialist.
- Budgets everywhere: tool loop iterations and blackboard rounds are capped.
- Trajectories make agent runs testable, not just their final answers.

## Likely follow-up questions

- When is a supervisor worse than a pipeline? (When the steps are fixed; the supervisor adds cost and variance.)
- How do you test non-deterministic agents? (Scripted fakes for logic, evals with real models for quality.)
- Where does a human come in? (Default route and approval steps on risky actions.)
