"""Client boundary for Qwen runtime access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QwenGenerationRequest:
    prompt: str
    max_new_tokens: int
    temperature: float


class QwenClient:
    """Runtime wrapper for offline Qwen deployment."""

    def __init__(
        self,
        *,
        model_id: str,
        model_path: str,
        device: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._pipeline = None

    @classmethod
    def from_recipe(cls, recipe_path: str | Path) -> "QwenClient":
        data = cls._load_simple_yaml(recipe_path)
        return cls(
            model_id=data["model_id"],
            model_path=data["runtime_path"],
        )

    def build_generation_request(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> QwenGenerationRequest:
        return QwenGenerationRequest(
            prompt=prompt,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            temperature=self.temperature if temperature is None else temperature,
        )

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        request = self.build_generation_request(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        pipeline = self._ensure_pipeline()
        outputs = pipeline(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=request.temperature > 0,
            return_full_text=False,
        )
        return str(outputs[0]["generated_text"])

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "transformers is not installed; cannot run offline Qwen inference"
            ) from exc
        self._pipeline = pipeline(
            "text-generation",
            model=self.model_path,
            tokenizer=self.model_path,
            device_map=self.device,
        )
        return self._pipeline

    @staticmethod
    def _load_simple_yaml(recipe_path: str | Path) -> dict[str, str]:
        values: dict[str, str] = {}
        for raw_line in Path(recipe_path).read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition(":")
            values[key.strip()] = value.strip()
        return values
