import pytest

from allinone.domain.shared.value_objects import BoundingBox
from allinone.infrastructure.perception.fusion.observation_builder import (
    ObservationBuilder,
)
from allinone.infrastructure.perception.yolo.detector import (
    DetectionCandidate,
    UltralyticsDetectorAdapter,
)


def test_detector_adapter_normalizes_prediction_rows_and_filters_targets():
    adapter = UltralyticsDetectorAdapter()

    detections = adapter.normalize_prediction_rows(
        prediction_rows=[
            {"label": "person", "confidence": 0.99, "xyxy": [10, 20, 100, 200]},
            {"label": "meter", "confidence": 0.91, "xyxy": [600, 200, 900, 800]},
        ],
        image_size=(1000, 1000),
        target_labels=("meter", "roi_window"),
    )

    assert detections == [
        DetectionCandidate(
            label="meter",
            confidence=0.91,
            bbox=BoundingBox(x1=0.6, y1=0.2, x2=0.9, y2=0.8),
        )
    ]


def test_observation_builder_converts_detection_to_perception_observation():
    observation = ObservationBuilder().build_from_detection(
        DetectionCandidate(
            label="meter",
            confidence=0.91,
            bbox=BoundingBox(x1=0.6, y1=0.2, x2=0.9, y2=0.8),
        ),
        visibility_score=0.85,
        readable_ratio=0.8,
    )

    assert observation.visibility_score == pytest.approx(0.85)
    assert observation.readable_ratio == pytest.approx(0.8)
    assert observation.fill_ratio == pytest.approx(0.18)
    assert observation.center_offset.dx == pytest.approx(0.25)
    assert observation.center_offset.dy == pytest.approx(0.0)


def test_detection_candidate_exports_prediction_row():
    row = DetectionCandidate(
        label="meter",
        confidence=0.91,
        bbox=BoundingBox(x1=0.6, y1=0.2, x2=0.9, y2=0.8),
    ).to_prediction_row(image_size=(1000, 1000))

    assert row == {
        "label": "meter",
        "confidence": 0.91,
        "xyxy": [600.0, 200.0, 900.0, 800.0],
    }
