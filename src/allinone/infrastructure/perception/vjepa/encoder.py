"""V-JEPA encoder adapter boundary."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameQualitySignal:
    visibility_score: float
    readable_ratio: float


class VJEPAEncoderAdapter:
    """Normalize V-JEPA-side quality signals into project-facing scores."""

    def normalize_quality_signal(
        self, raw_signal: dict[str, object]
    ) -> FrameQualitySignal:
        return FrameQualitySignal(
            visibility_score=float(raw_signal["visibility_score"]),
            readable_ratio=float(raw_signal["readable_ratio"]),
        )
