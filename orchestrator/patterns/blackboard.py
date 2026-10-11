"""Blackboard pattern: agents collaborate through a shared workspace."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..core.messages import Message, MessageRole
from ..core.state import TaskState
from ..eval.recorder import TrajectoryRecorder

if TYPE_CHECKING:
    from ..core.agent import Agent


@dataclass
class BlackboardResult:
    board: dict[str, Any]
    rounds: int
    converged: bool
    trajectory: TrajectoryRecorder = field(repr=False)


class Blackboard:
    """Agents take turns reading and writing a shared board.

    An agent signals completion by including ``done_marker`` in its response.
    The run stops early when every agent is done, or after ``max_rounds``.
    """

    def __init__(
        self,
        agents: list[Agent],
        max_rounds: int = 3,
        done_marker: str = "DONE",
        recorder: TrajectoryRecorder | None = None,
    ) -> None:
        if not agents:
            raise ValueError("blackboard needs at least one agent")
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        self.agents = agents
        self.max_rounds = max_rounds
        self.done_marker = done_marker
        self.recorder = recorder if recorder is not None else TrajectoryRecorder()

    def _board_view(self, board: dict[str, Any]) -> str:
        return "\n".join(f"{key}: {value}" for key, value in board.items())

    def run(self, task: str) -> BlackboardResult:
        state = TaskState(task=task, status="running")
        board: dict[str, Any] = {"task": task}
        done: set[str] = set()
        rounds = 0
        self.recorder.record("task_start", task=task)

        for round_no in range(1, self.max_rounds + 1):
            rounds = round_no
            for agent in self.agents:
                if agent.name in done:
                    continue
                prompt = (
                    f"Shared blackboard:\n{self._board_view(board)}\n"
                    f"Add your contribution, or reply with {self.done_marker} "
                    "if you have nothing more to add."
                )
                reply = agent.respond(
                    [Message(MessageRole.USER, prompt, sender="blackboard")]
                )
                state.add_message(reply)
                if self.done_marker in reply.content:
                    done.add(agent.name)
                    board[f"{agent.name}_status"] = "done"
                    self.recorder.record("agent_done", agent=agent.name)
                else:
                    board[agent.name] = reply.content
                    self.recorder.record("contribute", agent=agent.name)
            if len(done) == len(self.agents):
                break

        converged = len(done) == len(self.agents)
        state.mark_done()
        self.recorder.record("task_done", rounds=rounds, converged=converged)
        return BlackboardResult(
            board=board,
            rounds=rounds,
            converged=converged,
            trajectory=self.recorder,
        )
