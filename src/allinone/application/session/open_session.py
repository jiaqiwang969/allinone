"""Open a work session from an application boundary."""

from __future__ import annotations

from allinone.domain.session.entities import WorkSession
from allinone.domain.shared.value_objects import SessionId


def open_session(*, session_id: str, task_type: str) -> WorkSession:
    """Create a new open work session aggregate for a runtime workflow."""
    return WorkSession.open(
        session_id=SessionId(session_id),
        task_type=task_type,
    )
