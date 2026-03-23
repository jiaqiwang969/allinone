"""V-JEPA encoder adapter boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FrameQualitySignal:
    visibility_score: float
    readable_ratio: float


@dataclass(frozen=True)
class ClipQualitySignal(FrameQualitySignal):
    stability_score: float
    alignment_score: float


class VJEPAEncoderAdapter:
    """Normalize V-JEPA-side quality signals into project-facing scores."""

    def __init__(
        self,
        *,
        runtime: Any | None = None,
        runtime_factory: Any | None = None,
        repo_path: str | None = None,
        checkpoint_path: str | None = None,
        device: str | None = None,
        model_name: str = "vjepa2_1_vit_base_384",
        num_frames: int = 8,
        crop_size: int = 384,
    ) -> None:
        self.runtime = runtime
        self.runtime_factory = runtime_factory
        self.repo_path = repo_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model_name = model_name
        self.num_frames = num_frames
        self.crop_size = crop_size

    def normalize_quality_signal(
        self, raw_signal: dict[str, object]
    ) -> FrameQualitySignal:
        return FrameQualitySignal(
            visibility_score=float(raw_signal["visibility_score"]),
            readable_ratio=float(raw_signal["readable_ratio"]),
        )

    def normalize_clip_quality_signal(
        self, raw_signal: dict[str, object]
    ) -> ClipQualitySignal:
        return ClipQualitySignal(
            visibility_score=float(raw_signal["visibility_score"]),
            readable_ratio=float(raw_signal["readable_ratio"]),
            stability_score=float(raw_signal["stability_score"]),
            alignment_score=float(raw_signal["alignment_score"]),
        )

    def score_clip(
        self,
        *,
        sampled_frames: list[object],
        frame_indices: list[int],
        image_size: tuple[int, int],
    ) -> ClipQualitySignal:
        runtime = self._resolve_runtime()
        raw_signal = (
            runtime.score_clip(
                sampled_frames=sampled_frames,
                frame_indices=frame_indices,
                image_size=image_size,
            )
            if runtime is not None
            else self._score_clip_heuristically(
                sampled_frames=sampled_frames,
                image_size=image_size,
            )
        )
        return self.normalize_clip_quality_signal(raw_signal)

    def _resolve_runtime(self) -> Any | None:
        if self.runtime is not None:
            return self.runtime
        if not self.repo_path or not self.checkpoint_path:
            return None
        if self.runtime_factory is not None:
            self.runtime = self.runtime_factory(
                repo_path=self.repo_path,
                checkpoint_path=self.checkpoint_path,
                device=self.device,
            )
            return self.runtime
        self.runtime = _TorchHubVJEPARuntime(
            repo_path=self.repo_path,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            model_name=self.model_name,
            num_frames=self.num_frames,
            crop_size=self.crop_size,
        )
        return self.runtime

    def _score_clip_heuristically(
        self,
        *,
        sampled_frames: list[object],
        image_size: tuple[int, int],
    ) -> dict[str, float]:
        if not sampled_frames:
            raise ValueError("sampled_frames must not be empty")

        frames = [np.asarray(frame, dtype=np.float32) for frame in sampled_frames]
        normalized_means = [float(frame.mean() / 255.0) for frame in frames]
        visibility_score = float(np.clip(np.mean(normalized_means), 0.0, 1.0))

        readable_components = [
            float(np.clip(frame.std() / 96.0, 0.0, 1.0)) for frame in frames
        ]
        readable_ratio = float(np.mean(readable_components))

        if len(frames) == 1:
            stability_score = 1.0
        else:
            frame_diffs = [
                float(np.mean(np.abs(current - following)) / 255.0)
                for current, following in zip(frames, frames[1:])
            ]
            stability_score = float(np.clip(1.0 - np.mean(frame_diffs), 0.0, 1.0))

        width, height = image_size
        aspect_bias = 1.0 if width and height else 0.0
        alignment_score = float(
            np.clip(0.6 * visibility_score + 0.4 * readable_ratio, 0.0, aspect_bias)
        )

        return {
            "visibility_score": visibility_score,
            "readable_ratio": readable_ratio,
            "stability_score": stability_score,
            "alignment_score": alignment_score,
        }


class _TorchHubVJEPARuntime:
    def __init__(
        self,
        *,
        repo_path: str,
        checkpoint_path: str,
        device: str | None,
        model_name: str,
        num_frames: int,
        crop_size: int,
    ) -> None:
        self.repo_path = Path(repo_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.model_name = model_name
        self.num_frames = num_frames
        self.crop_size = crop_size
        self._model = None
        self._preprocessor = None
        self._torch = None

    def score_clip(
        self,
        *,
        sampled_frames: list[object],
        frame_indices: list[int],
        image_size: tuple[int, int],
    ) -> dict[str, float]:
        torch = self._import_torch()
        clip_tensor = self._build_clip_tensor(sampled_frames=sampled_frames)
        model = self._ensure_model(torch)
        with torch.inference_mode():
            tokens = model(clip_tensor.to(self.device))
        pooled_features = self._pool_tokens(tokens=tokens, torch=torch)

        feature_energy = float(pooled_features.abs().mean().item())
        feature_spread = float(pooled_features.std(unbiased=False).item())
        vjepa_visibility = feature_energy / (1.0 + feature_energy)
        vjepa_readable = 1.0 / (1.0 + feature_spread)

        frame_signal = _heuristic_clip_signal(
            sampled_frames=sampled_frames,
            image_size=image_size,
        )
        return {
            "visibility_score": float(
                np.clip(0.55 * frame_signal["visibility_score"] + 0.45 * vjepa_visibility, 0.0, 1.0)
            ),
            "readable_ratio": float(
                np.clip(0.55 * frame_signal["readable_ratio"] + 0.45 * vjepa_readable, 0.0, 1.0)
            ),
            "stability_score": frame_signal["stability_score"],
            "alignment_score": float(
                np.clip(0.65 * frame_signal["alignment_score"] + 0.35 * vjepa_visibility, 0.0, 1.0)
            ),
        }

    def _build_clip_tensor(self, *, sampled_frames: list[object]):
        torch = self._import_torch()
        selected_frames = _select_evenly_spaced_frames(
            sampled_frames=sampled_frames,
            frame_count=self.num_frames,
        )
        buffer = np.stack([np.asarray(frame, dtype=np.uint8) for frame in selected_frames])
        transformed = self._ensure_preprocessor()(buffer)
        clip = transformed[0] if isinstance(transformed, list) else transformed
        if clip.ndim != 4:
            raise RuntimeError(f"unexpected V-JEPA clip tensor shape: {tuple(clip.shape)}")
        return clip.unsqueeze(0).to(torch.float32)

    def _ensure_preprocessor(self):
        if self._preprocessor is not None:
            return self._preprocessor
        torch = self._import_torch()
        self._preprocessor = torch.hub.load(
            str(self.repo_path),
            "vjepa2_preprocessor",
            source="local",
            pretrained=False,
            crop_size=self.crop_size,
        )
        return self._preprocessor

    def _ensure_model(self, torch):
        if self._model is not None:
            return self._model
        model = torch.hub.load(
            str(self.repo_path),
            self.model_name,
            source="local",
            pretrained=False,
            num_frames=self.num_frames,
        )
        if isinstance(model, tuple):
            model = model[0]
        state = self._load_checkpoint(torch=torch)
        checkpoint_key = "ema_encoder" if "ema_encoder" in state else "encoder"
        encoder_state = {
            key.replace("module.", "").replace("backbone.", ""): value
            for key, value in state[checkpoint_key].items()
        }
        model.load_state_dict(encoder_state, strict=False)
        target_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = target_device
        self._model = model.to(target_device)
        self._model.eval()
        return self._model

    def _load_checkpoint(self, *, torch):
        try:
            return torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            return torch.load(self.checkpoint_path, map_location="cpu")

    def _import_torch(self):
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    def _pool_tokens(self, *, tokens, torch):
        if isinstance(tokens, tuple):
            tokens = tokens[0]
        if tokens.ndim == 3:
            return tokens.mean(dim=1)
        if tokens.ndim == 2:
            return tokens
        raise RuntimeError(f"unexpected V-JEPA output shape: {tuple(tokens.shape)}")


def _heuristic_clip_signal(
    *,
    sampled_frames: list[object],
    image_size: tuple[int, int],
) -> dict[str, float]:
    if not sampled_frames:
        raise ValueError("sampled_frames must not be empty")

    frames = [np.asarray(frame, dtype=np.float32) for frame in sampled_frames]
    normalized_means = [float(frame.mean() / 255.0) for frame in frames]
    visibility_score = float(np.clip(np.mean(normalized_means), 0.0, 1.0))

    readable_components = [
        float(np.clip(frame.std() / 96.0, 0.0, 1.0)) for frame in frames
    ]
    readable_ratio = float(np.mean(readable_components))

    if len(frames) == 1:
        stability_score = 1.0
    else:
        frame_diffs = [
            float(np.mean(np.abs(current - following)) / 255.0)
            for current, following in zip(frames, frames[1:])
        ]
        stability_score = float(np.clip(1.0 - np.mean(frame_diffs), 0.0, 1.0))

    width, height = image_size
    aspect_bias = 1.0 if width and height else 0.0
    alignment_score = float(
        np.clip(0.6 * visibility_score + 0.4 * readable_ratio, 0.0, aspect_bias)
    )

    return {
        "visibility_score": visibility_score,
        "readable_ratio": readable_ratio,
        "stability_score": stability_score,
        "alignment_score": alignment_score,
    }


def _select_evenly_spaced_frames(
    *,
    sampled_frames: list[object],
    frame_count: int,
) -> list[object]:
    if len(sampled_frames) == frame_count:
        return sampled_frames
    indices = np.linspace(
        0,
        len(sampled_frames) - 1,
        num=frame_count,
        dtype=int,
    ).tolist()
    return [sampled_frames[index] for index in indices]
