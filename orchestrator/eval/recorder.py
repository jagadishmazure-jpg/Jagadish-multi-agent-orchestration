"""Trajectory recorder: ordered, inspectable event log for agent runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Event:
    seq: int
    kind: str
    agent: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


class TrajectoryRecorder:
    """Append-only recorder of what happened during an orchestration run."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self._seq = 0

    def record(self, kind: str, agent: str = "", **detail: Any) -> Event:
        self._seq += 1
        event = Event(seq=self._seq, kind=kind, agent=agent, detail=detail)
        self.events.append(event)
        return event

    def filter(self, kind: str) -> list[Event]:
        return [e for e in self.events if e.kind == kind]

    def agents(self) -> list[str]:
        seen: list[str] = []
        for event in self.events:
            if event.agent and event.agent not in seen:
                seen.append(event.agent)
        return seen

    def to_dict(self) -> dict[str, Any]:
        return {"events": [asdict(e) for e in self.events]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def __len__(self) -> int:
        return len(self.events)
