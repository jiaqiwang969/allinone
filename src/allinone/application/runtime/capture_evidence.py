"""Capture evidence from a running work session."""

from __future__ import annotations

from allinone.domain.evidence.entities import EvidenceBundle, EvidenceItem
from allinone.domain.evidence.services import EvidenceAssessment, EvidenceAssessmentService


def capture_evidence(
    *,
    bundle: EvidenceBundle,
    item: EvidenceItem,
    assessment_service: EvidenceAssessmentService | None = None,
) -> EvidenceAssessment:
    """Append a captured evidence item and assess whether the bundle is ready."""
    bundle.add_item(item)
    service = assessment_service or EvidenceAssessmentService()
    return service.assess(bundle)
