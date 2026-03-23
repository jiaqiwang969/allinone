import cv2
import numpy as np

from allinone.infrastructure.perception.video.sampler import (
    ClipFrameSampler,
    SampledClip,
)


def test_clip_frame_sampler_returns_evenly_spaced_frames(tmp_path):
    clip_path = tmp_path / "clip.mp4"
    height = 48
    width = 64
    writer = cv2.VideoWriter(
        str(clip_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (width, height),
    )
    for index in range(6):
        frame = np.full((height, width, 3), index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    sampled = ClipFrameSampler(frame_count=3).sample(clip_path=str(clip_path))

    assert isinstance(sampled, SampledClip)
    assert sampled.frame_indices == [0, 2, 5]
    assert sampled.image_size == (64, 48)
    assert len(sampled.frames) == 3
    assert sampled.frames[0].shape == (48, 64, 3)
