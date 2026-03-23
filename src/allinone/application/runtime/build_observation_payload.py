"""Build a standardized runtime payload from upstream perception outputs."""

from __future__ import annotations

from allinone.infrastructure.perception.vjepa.encoder import (
    FrameQualitySignal,
    VJEPAEncoderAdapter,
)


def build_observation_payload(
    *,
    prediction_rows: list[dict[str, object]],
    image_size: tuple[int, int],
    target_labels: tuple[str, ...],
    visibility_score: float,
    readable_ratio: float,
) -> dict[str, object]:
    return {
        "prediction_rows": prediction_rows,
        "image_size": list(image_size),
        "target_labels": list(target_labels),
        "visibility_score": visibility_score,
        "readable_ratio": readable_ratio,
    }


def build_observation_payload_from_raw(
    raw_payload: dict[str, object],
    *,
    quality_adapter: VJEPAEncoderAdapter | None = None,
) -> dict[str, object]:
    detections = raw_payload["detections"]
    signal = (quality_adapter or VJEPAEncoderAdapter()).normalize_quality_signal(
        raw_payload["vjepa"]
    )
    detection_payload = dict(detections)
    return build_observation_payload(
        prediction_rows=detection_payload["prediction_rows"],
        image_size=tuple(detection_payload["image_size"]),
        target_labels=tuple(detection_payload["target_labels"]),
        visibility_score=signal.visibility_score,
        readable_ratio=signal.readable_ratio,
    )
