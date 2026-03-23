import pytest

from allinone.domain.session.entities import WorkSession
from allinone.domain.session.errors import SessionStateError
from allinone.domain.session.events import StageStarted, WorkSessionOpened
from allinone.domain.shared.value_objects import SessionId, StageType


def test_open_session_requires_task_type():
    with pytest.raises(ValueError):
        WorkSession.open(session_id=SessionId("session-001"), task_type="")


def test_open_session_emits_opened_event():
    session = WorkSession.open(
        session_id=SessionId("session-001"),
        task_type="remote_quality_check",
    )

    assert isinstance(session.pending_events[0], WorkSessionOpened)
    assert session.status == "open"


def test_start_stage_sets_current_stage_and_emits_event():
    session = WorkSession.open(
        session_id=SessionId("session-001"),
        task_type="remote_quality_check",
    )

    session.start_stage(StageType("overview"))

    assert session.current_stage.value == "overview"
    assert isinstance(session.pending_events[-1], StageStarted)


def test_starting_second_stage_while_active_is_rejected():
    session = WorkSession.open(
        session_id=SessionId("session-001"),
        task_type="remote_quality_check",
    )
    session.start_stage(StageType("overview"))

    with pytest.raises(SessionStateError):
        session.start_stage(StageType("meter"))
