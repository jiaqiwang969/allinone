from allinone.application.runtime.capture_evidence import capture_evidence
from allinone.application.runtime.request_guidance_decision import (
    request_guidance_decision,
)
from allinone.application.session.open_session import open_session
from allinone.domain.evidence.entities import EvidenceBundle, EvidenceItem
from allinone.domain.evidence.policies import EvidenceRequirementPolicy
from allinone.domain.guidance.entities import GuidanceDecision
from allinone.domain.perception.entities import PerceptionObservation
from allinone.domain.session.entities import WorkSession
from allinone.domain.shared.value_objects import BoundingBox, CenterOffset, SessionId, StageType


def test_open_session_usecase_returns_open_work_session():
    session = open_session(session_id="session-001", task_type="quality_inspection")

    assert isinstance(session, WorkSession)
    assert session.status == "open"
    assert session.task_type == "quality_inspection"


def test_request_guidance_decision_usecase_returns_domain_decision():
    observation = PerceptionObservation(
        visibility_score=0.7,
        readable_ratio=0.8,
        fill_ratio=0.5,
        center_offset=CenterOffset(dx=0.25, dy=0.0),
        roi=BoundingBox(x1=0.1, y1=0.1, x2=0.9, y2=0.9),
    )

    decision = request_guidance_decision(observation)

    assert isinstance(decision, GuidanceDecision)
    assert decision.action.value == "left"


def test_capture_evidence_usecase_adds_item_and_assesses_bundle():
    stage_type = StageType("capture")
    bundle = EvidenceBundle(
        session_id=SessionId("session-001"),
        stage_type=stage_type,
        required_types=EvidenceRequirementPolicy().required_types_for(stage_type),
    )

    first_assessment = capture_evidence(
        bundle=bundle,
        item=EvidenceItem(
            item_id="evidence-001",
            evidence_type="screenshot",
            uri="captures/frame-001.jpg",
        ),
    )
    assert first_assessment.acceptable is False

    second_assessment = capture_evidence(
        bundle=bundle,
        item=EvidenceItem(
            item_id="evidence-002",
            evidence_type="clip",
            uri="captures/clip-001.mp4",
        ),
    )

    assert second_assessment.acceptable is True
