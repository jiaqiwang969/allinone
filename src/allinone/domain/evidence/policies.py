"""Evidence policies."""

from __future__ import annotations

from dataclasses import dataclass

from allinone.domain.shared.value_objects import StageType


@dataclass(frozen=True)
class EvidenceRequirementPolicy:
    stage_requirements: dict[str, tuple[str, ...]] | None = None

    def required_types_for(self, stage_type: StageType) -> tuple[str, ...]:
        requirements = self.stage_requirements or {
            "capture": ("screenshot", "clip"),
            "inspection": ("screenshot", "clip"),
            "review": ("screenshot",),
        }
        return requirements.get(stage_type.value, ("screenshot",))
