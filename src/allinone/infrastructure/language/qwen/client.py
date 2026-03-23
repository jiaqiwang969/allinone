"""Client boundary for Qwen runtime access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
import re


@dataclass(frozen=True)
class QwenGenerationRequest:
    prompt: str
    max_new_tokens: int
    temperature: float


class QwenClient:
    """Runtime wrapper for offline Qwen deployment."""

    _client_cache: ClassVar[dict[tuple[str, str, str, int, float], "QwenClient"]] = {}

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
        self._runtime: tuple[object, object] | None = None

    def is_runtime_available(self) -> bool:
        return Path(self.model_path).exists()

    @classmethod
    def from_recipe(cls, recipe_path: str | Path) -> "QwenClient":
        data = cls._load_simple_yaml(recipe_path)
        model_id = data["model_id"]
        model_path = data["runtime_path"]
        device = data.get("device", "auto")
        max_new_tokens = int(data.get("max_new_tokens", "256"))
        temperature = float(data.get("temperature", "0.2"))
        cache_key = (
            model_id,
            model_path,
            device,
            max_new_tokens,
            temperature,
        )
        cached = cls._client_cache.get(cache_key)
        if cached is not None:
            return cached
        client = cls(
            model_id=model_id,
            model_path=model_path,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        cls._client_cache[cache_key] = client
        return client

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
        tokenizer, model = self._ensure_runtime()
        model_inputs = self._prepare_model_inputs(
            tokenizer=tokenizer,
            model=model,
            prompt=request.prompt,
        )
        generation_config = self._build_generation_config(request)
        outputs = model.generate(
            **model_inputs,
            generation_config=generation_config,
        )
        input_length = self._resolve_input_length(model_inputs["input_ids"])
        generated_token_ids = outputs[0][input_length:]
        raw_text = str(tokenizer.decode(generated_token_ids, skip_special_tokens=True))
        return self.sanitize_generated_text(raw_text)

    def _ensure_runtime(self) -> tuple[object, object]:
        if self._runtime is not None:
            return self._runtime
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is not installed; cannot run offline Qwen inference"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device,
            local_files_only=True,
            torch_dtype="auto",
        )
        self._normalize_padding_config(tokenizer=tokenizer, model=model)
        self._runtime = (tokenizer, model)
        return self._runtime

    def _build_generation_config(self, request: QwenGenerationRequest) -> object:
        try:
            from transformers import GenerationConfig
        except ImportError as exc:
            raise RuntimeError(
                "transformers is not installed; cannot run offline Qwen inference"
            ) from exc

        generation_kwargs: dict[str, object] = {
            "max_new_tokens": request.max_new_tokens,
            "do_sample": request.temperature > 0,
        }
        if request.temperature > 0:
            generation_kwargs["temperature"] = request.temperature
        return GenerationConfig(**generation_kwargs)

    @staticmethod
    def _prepare_model_inputs(
        *,
        tokenizer: object,
        model: object,
        prompt: str,
    ) -> dict[str, object]:
        encoded_inputs = tokenizer(prompt, return_tensors="pt")
        model_device = getattr(model, "device", None)
        if model_device is None:
            return dict(encoded_inputs)
        if hasattr(encoded_inputs, "to"):
            moved = encoded_inputs.to(model_device)
            return dict(moved)
        return {
            key: value.to(model_device) if hasattr(value, "to") else value
            for key, value in dict(encoded_inputs).items()
        }

    @staticmethod
    def _resolve_input_length(input_ids: object) -> int:
        shape = getattr(input_ids, "shape", None)
        if shape is not None:
            return int(shape[-1])
        if isinstance(input_ids, list):
            return len(input_ids[0])
        raise TypeError("unable to resolve input token length for generation output")

    @staticmethod
    def _normalize_padding_config(*, tokenizer: object, model: object) -> None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

        if pad_token_id is None and eos_token_id is not None:
            if getattr(tokenizer, "pad_token", None) is None and getattr(
                tokenizer, "eos_token", None
            ) is not None:
                setattr(tokenizer, "pad_token", getattr(tokenizer, "eos_token"))
            setattr(tokenizer, "pad_token_id", eos_token_id)
            pad_token_id = eos_token_id

        if pad_token_id is None:
            return

        config = getattr(model, "config", None)
        if config is not None and getattr(config, "pad_token_id", None) is None:
            setattr(config, "pad_token_id", pad_token_id)

        generation_config = getattr(model, "generation_config", None)
        if (
            generation_config is not None
            and getattr(generation_config, "pad_token_id", None) is None
        ):
            setattr(generation_config, "pad_token_id", pad_token_id)

    @classmethod
    def sanitize_generated_text(cls, raw_text: str) -> str:
        text = re.sub(
            r"<think\b[^>]*>.*?</think>",
            "",
            raw_text,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        structured_start = cls._find_structured_output_start(text)
        if structured_start is None:
            return text
        prefix = text[:structured_start].strip().lower()
        if prefix.startswith(("thinking process", "reasoning")):
            return text[structured_start:].strip()
        return text

    @staticmethod
    def _find_structured_output_start(text: str) -> int | None:
        positions = [
            position
            for position in (text.find("```"), text.find("{"), text.find("["))
            if position >= 0
        ]
        if not positions:
            return None
        return min(positions)

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
