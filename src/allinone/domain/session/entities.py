"""Session entities."""

from __future__ import annotations

from dataclasses import dataclass, field

from allinone.domain.session.errors import SessionStateError
from allinone.domain.session.events import StageStarted, WorkSessionOpened
from allinone.domain.shared.value_objects import SessionId, StageType


@dataclass
class WorkSession:
    session_id: SessionId
    task_type: str
    status: str = "draft"
    current_stage: StageType | None = None
    pending_events: list[object] = field(default_factory=list)

    @classmethod
    def open(cls, *, session_id: SessionId, task_type: str) -> "WorkSession":
        if not task_type or not task_type.strip():
            raise ValueError("task_type must not be empty")
        session = cls(session_id=session_id, task_type=task_type.strip(), status="open")
        session.pending_events.append(
            WorkSessionOpened(session_id=session.session_id, task_type=session.task_type)
        )
        return session

    def start_stage(self, stage_type: StageType) -> None:
        if self.status != "open":
            raise SessionStateError("cannot start stage unless session is open")
        if self.current_stage is not None:
            raise SessionStateError("cannot start a second stage while one is active")
        self.current_stage = stage_type
        self.pending_events.append(
            StageStarted(session_id=self.session_id, stage_type=stage_type)
        )
