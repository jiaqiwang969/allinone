"""Evidence entities."""

from __future__ import annotations

from dataclasses import dataclass, field

from allinone.domain.shared.errors import DomainValidationError
from allinone.domain.shared.value_objects import SessionId, StageType


ALLOWED_EVIDENCE_TYPES = {
    "screenshot",
    "clip",
    "ocr_crop",
    "overview_photo",
}


@dataclass(frozen=True)
class EvidenceItem:
    item_id: str
    evidence_type: str
    uri: str

    def __post_init__(self) -> None:
        if not self.item_id or not self.item_id.strip():
            raise DomainValidationError("evidence item id must not be empty")
        if self.evidence_type not in ALLOWED_EVIDENCE_TYPES:
            raise DomainValidationError(f"unsupported evidence type: {self.evidence_type}")
        if not self.uri or not self.uri.strip():
            raise DomainValidationError("evidence uri must not be empty")


@dataclass
class EvidenceBundle:
    session_id: SessionId
    stage_type: StageType
    required_types: tuple[str, ...]
    items: list[EvidenceItem] = field(default_factory=list)

    def __post_init__(self) -> None:
        for evidence_type in self.required_types:
            if evidence_type not in ALLOWED_EVIDENCE_TYPES:
                raise DomainValidationError(
                    f"unsupported required evidence type: {evidence_type}"
                )

    def add_item(self, item: EvidenceItem) -> None:
        self.items.append(item)

    @property
    def collected_types(self) -> set[str]:
        return {item.evidence_type for item in self.items}

    @property
    def missing_types(self) -> tuple[str, ...]:
        return tuple(
            evidence_type
            for evidence_type in self.required_types
            if evidence_type not in self.collected_types
        )

    @property
    def is_complete(self) -> bool:
        return not self.missing_types
