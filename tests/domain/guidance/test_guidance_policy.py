from allinone.domain.guidance.entities import GuidanceDecision
from allinone.domain.guidance.services import GuidancePolicyService
from allinone.domain.perception.entities import PerceptionObservation
from allinone.domain.shared.value_objects import BoundingBox, CenterOffset


def test_shifted_right_target_yields_left_action():
    observation = PerceptionObservation(
        visibility_score=0.6,
        readable_ratio=0.8,
        fill_ratio=0.5,
        center_offset=CenterOffset(dx=0.3, dy=0.0),
        roi=BoundingBox(x1=0.1, y1=0.1, x2=0.9, y2=0.9),
    )

    decision = GuidancePolicyService().decide(observation)

    assert isinstance(decision, GuidanceDecision)
    assert decision.action.value == "left"


def test_ready_window_yields_hold_still():
    observation = PerceptionObservation(
        visibility_score=0.7,
        readable_ratio=0.8,
        fill_ratio=0.5,
        center_offset=CenterOffset(dx=0.01, dy=-0.01),
        roi=BoundingBox(x1=0.2, y1=0.2, x2=0.8, y2=0.8),
    )

    decision = GuidancePolicyService().decide(observation)

    assert decision.action.value == "hold_still"


def test_oversized_target_yields_backward():
    observation = PerceptionObservation(
        visibility_score=0.7,
        readable_ratio=0.8,
        fill_ratio=0.9,
        center_offset=CenterOffset(dx=0.0, dy=0.0),
        roi=BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0),
    )

    decision = GuidancePolicyService().decide(observation)

    assert decision.action.value == "backward"
