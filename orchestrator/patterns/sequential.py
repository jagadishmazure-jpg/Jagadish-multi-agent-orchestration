"""Sequential pattern: an ordered pipeline of agents with handoffs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..core.handoff import Handoff, perform_handoff
from ..core.state import TaskState
from ..eval.recorder import TrajectoryRecorder

if TYPE_CHECKING:
    from ..core.agent import Agent


@dataclass
class PipelineResult:
    outputs: list[str]
    final: str
    trajectory: TrajectoryRecorder = field(repr=False)


class SequentialPipeline:
    """Run agents in order; each receives the previous output via handoff."""

    def __init__(
        self, agents: list[Agent], recorder: TrajectoryRecorder | None = None
    ) -> None:
        if not agents:
            raise ValueError("pipeline needs at least one agent")
        self.agents = agents
        self.recorder = recorder if recorder is not None else TrajectoryRecorder()
        self._by_name = {a.name: a for a in agents}

    def run(self, task: str) -> PipelineResult:
        if not task or not task.strip():
            raise ValueError("pipeline task must be a non-empty string")
        state = TaskState(task=task, status="running")
        self.recorder.record("task_start", task=task)
        outputs: list[str] = []
        current = task
        previous = "pipeline"
        for index, agent in enumerate(self.agents, start=1):
            handoff = Handoff(
                from_agent=previous,
                to_agent=agent.name,
                payload={"input": current},
                reason=f"pipeline step {index}/{len(self.agents)}",
            )
            transfer = perform_handoff(handoff, self._by_name, state)
            reply = agent.respond([transfer])
            state.add_message(reply)
            outputs.append(reply.content)
            current = reply.content
            previous = agent.name
            self.recorder.record("step", agent=agent.name, index=index)
        state.mark_done()
        self.recorder.record("task_done", steps=len(self.agents))
        return PipelineResult(
            outputs=outputs, final=current, trajectory=self.recorder
        )
