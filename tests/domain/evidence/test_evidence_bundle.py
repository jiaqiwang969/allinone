import pytest

from allinone.domain.evidence.entities import EvidenceBundle, EvidenceItem
from allinone.domain.evidence.policies import EvidenceRequirementPolicy
from allinone.domain.evidence.services import EvidenceAssessmentService
from allinone.domain.shared.errors import DomainValidationError
from allinone.domain.shared.value_objects import SessionId, StageType


def test_bundle_initially_incomplete_for_capture_stage():
    stage_type = StageType("capture")
    bundle = EvidenceBundle(
        session_id=SessionId("session-001"),
        stage_type=stage_type,
        required_types=EvidenceRequirementPolicy().required_types_for(stage_type),
    )

    assessment = EvidenceAssessmentService().assess(bundle)

    assert bundle.is_complete is False
    assert assessment.acceptable is False
    assert assessment.missing_types == ("screenshot", "clip")


def test_bundle_becomes_acceptable_after_required_items_added():
    stage_type = StageType("capture")
    bundle = EvidenceBundle(
        session_id=SessionId("session-001"),
        stage_type=stage_type,
        required_types=EvidenceRequirementPolicy().required_types_for(stage_type),
    )

    bundle.add_item(
        EvidenceItem(
            item_id="evidence-001",
            evidence_type="screenshot",
            uri="captures/frame-001.jpg",
        )
    )
    assert bundle.is_complete is False

    bundle.add_item(
        EvidenceItem(
            item_id="evidence-002",
            evidence_type="clip",
            uri="captures/clip-001.mp4",
        )
    )

    assessment = EvidenceAssessmentService().assess(bundle)

    assert bundle.is_complete is True
    assert assessment.acceptable is True
    assert assessment.missing_types == ()


def test_invalid_evidence_type_is_rejected():
    with pytest.raises(DomainValidationError):
        EvidenceItem(
            item_id="evidence-003",
            evidence_type="waveform",
            uri="captures/audio.wav",
        )
