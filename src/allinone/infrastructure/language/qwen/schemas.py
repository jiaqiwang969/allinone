"""Schemas for Qwen gateway config and service payloads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QwenServiceGenerateRequest:
    prompt: str
    max_new_tokens: int
    temperature: float

    def to_payload(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }


@dataclass(frozen=True)
class QwenServiceGenerateResponse:
    text: str
    model_id: str
    mode: str

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "QwenServiceGenerateResponse":
        return cls(
            text=str(payload["text"]),
            model_id=str(payload["model_id"]),
            mode=str(payload["mode"]),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "text": self.text,
            "model_id": self.model_id,
            "mode": self.mode,
        }


@dataclass(frozen=True)
class QwenGatewayConfig:
    mode: str
    service_url: str
    service_timeout_seconds: int
    model_id: str
    runtime_path: str
    device: str
    max_new_tokens: int
    temperature: float

    @classmethod
    def from_recipe(cls, recipe_path: str | Path) -> "QwenGatewayConfig":
        values = cls._load_simple_yaml(recipe_path)
        mode = values.get("mode", "auto")
        if mode == "offline":
            mode = "local"
        return cls(
            mode=mode,
            service_url=values.get("service_url", "http://127.0.0.1:8001"),
            service_timeout_seconds=int(values.get("service_timeout_seconds", "30")),
            model_id=values["model_id"],
            runtime_path=values["runtime_path"],
            device=values.get("device", "auto"),
            max_new_tokens=int(values.get("max_new_tokens", "256")),
            temperature=float(values.get("temperature", "0.2")),
        )

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
