"""Build raw perception payloads from upstream video clip inputs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def build_raw_perception_payload_from_clip(
    *,
    clip_path: str,
    target_labels: tuple[str, ...],
    sampler: Any,
    detector: Any,
    clip_scorer: Any,
) -> dict[str, object]:
    sampled_clip = sampler.sample(clip_path=clip_path)
    sampled_frames, frame_indices, image_size = _unpack_sampled_clip(sampled_clip)

    detection_result = detector.predict_sampled_frames(
        sampled_frames=sampled_frames,
        image_size=image_size,
        target_labels=target_labels,
    )
    vjepa_scores = clip_scorer.score_clip(
        sampled_frames=sampled_frames,
        frame_indices=frame_indices,
        image_size=image_size,
    )

    detections = {
        "prediction_rows": list(detection_result["prediction_rows"]),
        "image_size": list(image_size),
        "target_labels": list(target_labels),
    }
    if "best_frame_index" in detection_result:
        detections["best_frame_index"] = detection_result["best_frame_index"]

    return {
        "detections": detections,
        "vjepa": _serialize_quality_signal(vjepa_scores),
    }


def _unpack_sampled_clip(
    sampled_clip: object,
) -> tuple[list[object], list[int], tuple[int, int]]:
    if hasattr(sampled_clip, "frames"):
        return (
            list(getattr(sampled_clip, "frames")),
            list(getattr(sampled_clip, "frame_indices")),
            tuple(getattr(sampled_clip, "image_size")),
        )
    return (
        list(sampled_clip["frames"]),
        list(sampled_clip["frame_indices"]),
        tuple(sampled_clip["image_size"]),
    )


def _serialize_quality_signal(quality_signal: object) -> dict[str, object]:
    if is_dataclass(quality_signal):
        return asdict(quality_signal)
    return dict(quality_signal)
