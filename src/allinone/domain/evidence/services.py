"""Evidence services."""

from __future__ import annotations

from dataclasses import dataclass

from allinone.domain.evidence.entities import EvidenceBundle


@dataclass(frozen=True)
class EvidenceAssessment:
    acceptable: bool
    missing_types: tuple[str, ...]


class EvidenceAssessmentService:
    def assess(self, bundle: EvidenceBundle) -> EvidenceAssessment:
        return EvidenceAssessment(
            acceptable=bundle.is_complete,
            missing_types=bundle.missing_types,
        )
