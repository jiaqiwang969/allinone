import pytest

from allinone.domain.shared.errors import DomainValidationError
from allinone.domain.shared.value_objects import (
    BoundingBox,
    CenterOffset,
    PromptAction,
    SessionId,
    StageType,
)


def test_session_id_wraps_non_empty_value():
    session_id = SessionId("session-001")
    assert session_id.value == "session-001"


def test_stage_type_rejects_empty_value():
    with pytest.raises(DomainValidationError):
        StageType("")


def test_prompt_action_rejects_unknown_value():
    with pytest.raises(DomainValidationError):
        PromptAction("spin_around")


def test_bounding_box_rejects_invalid_coordinates():
    with pytest.raises(DomainValidationError):
        BoundingBox(x1=3.0, y1=0.0, x2=2.0, y2=1.0)


def test_center_offset_exposes_components():
    offset = CenterOffset(dx=0.1, dy=-0.2)
    assert offset.dx == 0.1
    assert offset.dy == -0.2
