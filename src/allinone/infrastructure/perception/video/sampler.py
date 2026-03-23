"""Clip frame sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SampledClip:
    frames: list[np.ndarray]
    frame_indices: list[int]
    image_size: tuple[int, int]


class ClipFrameSampler:
    def __init__(self, *, frame_count: int = 8) -> None:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        self.frame_count = frame_count

    def sample(self, *, clip_path: str) -> SampledClip:
        capture = cv2.VideoCapture(clip_path)
        if not capture.isOpened():
            raise RuntimeError(f"cannot open clip: {clip_path}")

        total_frames = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        frame_indices = self._build_frame_indices(total_frames=total_frames)
        frames: list[np.ndarray] = []
        image_size: tuple[int, int] | None = None

        try:
            for frame_index in frame_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"cannot read frame {frame_index} from clip: {clip_path}"
                    )
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
                if image_size is None:
                    image_size = (rgb_frame.shape[1], rgb_frame.shape[0])
        finally:
            capture.release()

        if image_size is None:
            raise RuntimeError(f"clip did not yield frames: {clip_path}")

        return SampledClip(
            frames=frames,
            frame_indices=frame_indices,
            image_size=image_size,
        )

    def _build_frame_indices(self, *, total_frames: int) -> list[int]:
        return np.linspace(
            0,
            total_frames - 1,
            num=self.frame_count,
            dtype=int,
        ).tolist()
