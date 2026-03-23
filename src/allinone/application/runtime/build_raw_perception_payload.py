"""Build raw perception payloads from live upstream image inputs."""

from __future__ import annotations

from PIL import Image

from allinone.infrastructure.perception.yolo.detector import (
    UltralyticsDetectorAdapter,
)


def build_raw_perception_payload_from_image(
    *,
    image_path: str,
    target_labels: tuple[str, ...],
    detector_adapter: UltralyticsDetectorAdapter | None = None,
    model_path: str | None = None,
    device: str | None = None,
    visibility_score: float = 1.0,
    readable_ratio: float = 1.0,
) -> dict[str, object]:
    with Image.open(image_path) as image:
        image_size = image.size

    detector = detector_adapter or UltralyticsDetectorAdapter(
        model_path=model_path,
        device=device,
    )
    detections = detector.predict(
        image_path=image_path,
        image_size=image_size,
        target_labels=target_labels,
    )
    prediction_rows = [
        detection.to_prediction_row(image_size=image_size) for detection in detections
    ]
    return {
        "detections": {
            "prediction_rows": prediction_rows,
            "image_size": list(image_size),
            "target_labels": list(target_labels),
        },
        "vjepa": {
            "visibility_score": visibility_score,
            "readable_ratio": readable_ratio,
        },
    }
