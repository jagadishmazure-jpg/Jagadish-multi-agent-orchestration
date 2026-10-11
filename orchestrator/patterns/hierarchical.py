"""Hierarchical pattern: a manager delegates to sub-teams, then aggregates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..core.messages import Message, MessageRole
from ..core.state import TaskState
from ..eval.recorder import TrajectoryRecorder
from .sequential import SequentialPipeline
from .supervisor import parse_subtasks

if TYPE_CHECKING:
    from ..core.agent import Agent


class UnknownTeamError(Exception):
    """Raised when the manager's plan names an unknown team."""


@dataclass
class HierarchicalResult:
    team_outputs: dict[str, str]
    final: str
    trajectory: TrajectoryRecorder = field(repr=False)


class Hierarchical:
    """Manager decomposes work; each sub-team runs a sequential pipeline."""

    def __init__(
        self,
        manager: Agent,
        teams: dict[str, list[Agent]],
        recorder: TrajectoryRecorder | None = None,
    ) -> None:
        if not teams:
            raise ValueError("hierarchical orchestration needs at least one team")
        self.manager = manager
        self.teams = teams
        self.recorder = recorder if recorder is not None else TrajectoryRecorder()

    def run(self, task: str) -> HierarchicalResult:
        state = TaskState(task=task, status="running")
        self.recorder.record("task_start", agent=self.manager.name, task=task)

        plan_msg = self.manager.respond([Message(MessageRole.USER, task)])
        state.add_message(plan_msg)
        assignments = parse_subtasks(plan_msg.content)
        self.recorder.record("decompose", agent=self.manager.name)

        team_outputs: dict[str, str] = {}
        for team_name, subtask in assignments:
            if team_name not in self.teams:
                raise UnknownTeamError(f"unknown team {team_name!r}")
            pipeline = SequentialPipeline(
                self.teams[team_name], recorder=self.recorder
            )
            result = pipeline.run(subtask)
            team_outputs[team_name] = result.final
            self.recorder.record("team_done", agent=team_name, subtask=subtask[:120])

        combined = "\n".join(
            f"[team {name}] {text}" for name, text in team_outputs.items()
        )
        final_msg = self.manager.respond(
            [Message(MessageRole.USER, f"Aggregate these team results:\n{combined}")]
        )
        state.add_message(final_msg)
        state.mark_done()
        self.recorder.record("aggregate", agent=self.manager.name)
        return HierarchicalResult(
            team_outputs=team_outputs,
            final=final_msg.content,
            trajectory=self.recorder,
        )
