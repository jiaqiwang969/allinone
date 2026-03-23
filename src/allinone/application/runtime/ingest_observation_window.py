"""Ingest runtime observation windows into the work session flow."""

from __future__ import annotations

from allinone.domain.perception.entities import PerceptionObservation
from allinone.infrastructure.perception.fusion.observation_builder import (
    ObservationBuilder,
)
from allinone.infrastructure.perception.yolo.detector import UltralyticsDetectorAdapter


def ingest_observation_window(
    *,
    prediction_rows: list[dict[str, object]],
    image_size: tuple[int, int],
    target_labels: tuple[str, ...],
    visibility_score: float,
    readable_ratio: float,
    detector_adapter: UltralyticsDetectorAdapter | None = None,
    observation_builder: ObservationBuilder | None = None,
) -> PerceptionObservation:
    """Translate normalized YOLO-like prediction rows into a domain observation."""
    detector = detector_adapter or UltralyticsDetectorAdapter()
    builder = observation_builder or ObservationBuilder()
    detections = detector.normalize_prediction_rows(
        prediction_rows=prediction_rows,
        image_size=image_size,
        target_labels=target_labels,
    )
    return builder.build_from_detections(
        detections,
        visibility_score=visibility_score,
        readable_ratio=readable_ratio,
    )
