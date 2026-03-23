import numpy as np

from allinone.infrastructure.perception.vjepa.encoder import (
    ClipQualitySignal,
    FrameQualitySignal,
    VJEPAEncoderAdapter,
)


def test_vjepa_adapter_normalizes_quality_signal():
    signal = VJEPAEncoderAdapter().normalize_quality_signal(
        {
            "visibility_score": 0.85,
            "readable_ratio": 0.8,
        }
    )

    assert signal == FrameQualitySignal(
        visibility_score=0.85,
        readable_ratio=0.8,
    )


def test_vjepa_adapter_scores_clip_from_runtime_signal():
    class FakeRuntime:
        def score_clip(self, *, sampled_frames, frame_indices, image_size):
            assert len(sampled_frames) == 3
            assert frame_indices == [0, 4, 8]
            assert image_size == (1280, 720)
            return {
                "visibility_score": 0.86,
                "readable_ratio": 0.79,
                "stability_score": 0.91,
                "alignment_score": 0.84,
            }

    frames = [
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.full((8, 8, 3), 64, dtype=np.uint8),
        np.full((8, 8, 3), 128, dtype=np.uint8),
    ]

    signal = VJEPAEncoderAdapter(runtime=FakeRuntime()).score_clip(
        sampled_frames=frames,
        frame_indices=[0, 4, 8],
        image_size=(1280, 720),
    )

    assert signal == ClipQualitySignal(
        visibility_score=0.86,
        readable_ratio=0.79,
        stability_score=0.91,
        alignment_score=0.84,
    )


def test_vjepa_adapter_lazy_builds_runtime_from_repo_and_checkpoint():
    build_calls: list[tuple[str, str, str | None]] = []

    class FakeRuntime:
        def score_clip(self, *, sampled_frames, frame_indices, image_size):
            return {
                "visibility_score": 0.9,
                "readable_ratio": 0.82,
                "stability_score": 0.93,
                "alignment_score": 0.88,
            }

    def fake_runtime_factory(*, repo_path, checkpoint_path, device):
        build_calls.append((repo_path, checkpoint_path, device))
        return FakeRuntime()

    adapter = VJEPAEncoderAdapter(
        repo_path="/models/vjepa2",
        checkpoint_path="/models/vjepa2/model.pt",
        device="cuda:0",
        runtime_factory=fake_runtime_factory,
    )

    first = adapter.score_clip(
        sampled_frames=[np.zeros((8, 8, 3), dtype=np.uint8)],
        frame_indices=[0],
        image_size=(1280, 720),
    )
    second = adapter.score_clip(
        sampled_frames=[np.zeros((8, 8, 3), dtype=np.uint8)],
        frame_indices=[1],
        image_size=(1280, 720),
    )

    assert first.alignment_score == 0.88
    assert second.stability_score == 0.93
    assert build_calls == [
        ("/models/vjepa2", "/models/vjepa2/model.pt", "cuda:0")
    ]
