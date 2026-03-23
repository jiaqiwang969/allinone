"""Perception entities."""

from __future__ import annotations

from dataclasses import dataclass

from allinone.domain.shared.value_objects import BoundingBox, CenterOffset


@dataclass(frozen=True)
class PerceptionObservation:
    visibility_score: float
    readable_ratio: float
    fill_ratio: float
    center_offset: CenterOffset
    roi: BoundingBox
