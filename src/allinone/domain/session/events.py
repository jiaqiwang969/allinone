"""Session events."""

from __future__ import annotations

from dataclasses import dataclass

from allinone.domain.shared.value_objects import SessionId, StageType


@dataclass(frozen=True)
class WorkSessionOpened:
    session_id: SessionId
    task_type: str


@dataclass(frozen=True)
class StageStarted:
    session_id: SessionId
    stage_type: StageType
