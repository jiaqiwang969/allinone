"""Build domain-facing observations from raw perception outputs."""

from __future__ import annotations

from allinone.domain.perception.entities import PerceptionObservation
from allinone.domain.shared.value_objects import CenterOffset
from allinone.infrastructure.perception.yolo.detector import DetectionCandidate


class ObservationBuilder:
    """Build a domain observation from normalized target detections."""

    def build_from_detection(
        self,
        detection: DetectionCandidate,
        *,
        visibility_score: float,
        readable_ratio: float,
    ) -> PerceptionObservation:
        bbox = detection.bbox
        width = bbox.x2 - bbox.x1
        height = bbox.y2 - bbox.y1
        center_x = (bbox.x1 + bbox.x2) / 2
        center_y = (bbox.y1 + bbox.y2) / 2
        return PerceptionObservation(
            visibility_score=visibility_score,
            readable_ratio=readable_ratio,
            fill_ratio=width * height,
            center_offset=CenterOffset(dx=center_x - 0.5, dy=center_y - 0.5),
            roi=bbox,
        )

    def build_from_detections(
        self,
        detections: list[DetectionCandidate],
        *,
        visibility_score: float,
        readable_ratio: float,
    ) -> PerceptionObservation:
        if not detections:
            raise ValueError("at least one detection is required")
        return self.build_from_detection(
            detections[0],
            visibility_score=visibility_score,
            readable_ratio=readable_ratio,
        )
