"""Supervisor pattern: decompose -> delegate -> synthesize."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..core.messages import Message, MessageRole
from ..core.state import TaskState
from ..eval.recorder import TrajectoryRecorder

if TYPE_CHECKING:
    from ..core.agent import Agent


class UnknownWorkerError(Exception):
    """Raised when the plan names a worker that is not registered."""


def parse_subtasks(text: str) -> list[tuple[str, str]]:
    """Parse ``worker: subtask`` lines from a decomposition response.

    Accepts plain lines and bullet lists (``-``, ``*``, ``•``). Lines without
    a colon are ignored.
    """
    tasks: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-*• ").strip()
        if ":" not in line:
            continue
        name, desc = line.split(":", 1)
        name, desc = name.strip(), desc.strip()
        if name and desc:
            tasks.append((name, desc))
    if not tasks:
        raise ValueError(f"no subtasks could be parsed from: {text!r}")
    return tasks


@dataclass
class SupervisorResult:
    subtasks: list[tuple[str, str]]
    worker_outputs: dict[str, str]
    final: str
    trajectory: TrajectoryRecorder = field(repr=False)


class Supervisor:
    """Decompose a task, delegate subtasks to workers, synthesize results."""

    def __init__(
        self,
        decomposer: Agent,
        workers: dict[str, Agent],
        synthesizer: Agent,
        recorder: TrajectoryRecorder | None = None,
    ) -> None:
        if not workers:
            raise ValueError("supervisor needs at least one worker")
        self.decomposer = decomposer
        self.workers = workers
        self.synthesizer = synthesizer
        self.recorder = recorder if recorder is not None else TrajectoryRecorder()

    def run(self, task: str) -> SupervisorResult:
        state = TaskState(task=task, status="running")
        self.recorder.record("task_start", agent=self.decomposer.name, task=task)

        plan_msg = self.decomposer.respond([Message(MessageRole.USER, task)])
        state.add_message(plan_msg)
        self.recorder.record("decompose", agent=self.decomposer.name)
        subtasks = parse_subtasks(plan_msg.content)

        outputs: dict[str, str] = {}
        for worker_name, subtask in subtasks:
            if worker_name not in self.workers:
                raise UnknownWorkerError(
                    f"plan references unknown worker {worker_name!r}"
                )
            worker = self.workers[worker_name]
            out = worker.respond(
                [Message(MessageRole.USER, subtask, sender="supervisor")]
            )
            state.add_message(out)
            outputs[worker_name] = out.content
            self.recorder.record(
                "delegate", agent=worker_name, subtask=subtask[:120]
            )

        combined = "\n".join(f"[{name}] {text}" for name, text in outputs.items())
        final_msg = self.synthesizer.respond(
            [Message(MessageRole.USER, f"Synthesize these worker outputs:\n{combined}")]
        )
        state.add_message(final_msg)
        state.mark_done()
        self.recorder.record("synthesize", agent=self.synthesizer.name)
        return SupervisorResult(
            subtasks=subtasks,
            worker_outputs=outputs,
            final=final_msg.content,
            trajectory=self.recorder,
        )
