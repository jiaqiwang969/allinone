"""Run the reusable runtime observation chain."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from allinone.application.runtime.ingest_observation_window import (
    ingest_observation_window,
)
from allinone.application.runtime.request_guidance_decision import (
    request_guidance_decision,
)
from allinone.domain.guidance.services import GuidanceThresholds
from allinone.infrastructure.language.qwen.gateway import QwenGateway
from allinone.infrastructure.language.qwen.prompt_builder import QwenPromptBuilder
from allinone.infrastructure.language.qwen.schemas import QwenGatewayConfig
from allinone.infrastructure.language.qwen.structured_output import (
    QwenStructuredOutputParser,
)

_DEFAULT_LANGUAGE_OUTPUT = """{
    "operator_message": "请向左移动，让仪表回到画面中央。",
    "suggested_action": "left",
    "confidence": 0.82,
    "evidence_focus": "确保整个表盘完整可见"
}"""


class RuntimeTextGenerator(Protocol):
    def generate(self, prompt: str) -> tuple[str, str]: ...


class QwenRuntimeTextGenerator:
    """Generate runtime explanations through the shared Qwen gateway."""

    def __init__(self, recipe_path: str | Path | None = None) -> None:
        self.recipe_path = Path(recipe_path) if recipe_path is not None else _default_recipe_path()
        self._gateway: QwenGateway | None = None

    def generate(self, prompt: str) -> tuple[str, str]:
        gateway = self._resolve_gateway()
        if gateway is not None:
            try:
                return gateway.generate_text(prompt)
            except RuntimeError:
                pass
        return _DEFAULT_LANGUAGE_OUTPUT, "mock"

    def _resolve_gateway(self) -> QwenGateway | None:
        if self._gateway is not None:
            return self._gateway
        if not self.recipe_path.exists():
            return None
        self._gateway = QwenGateway(
            config=QwenGatewayConfig.from_recipe(self.recipe_path)
        )
        return self._gateway


def run_runtime_observation(
    *,
    payload: dict[str, object],
    guidance_thresholds: GuidanceThresholds | None = None,
    prompt_builder: QwenPromptBuilder | None = None,
    output_parser: QwenStructuredOutputParser | None = None,
    text_generator: RuntimeTextGenerator | None = None,
) -> dict[str, object]:
    if not payload["prediction_rows"]:
        return _build_missing_target_result()
    observation = ingest_observation_window(
        prediction_rows=list(payload["prediction_rows"]),
        image_size=tuple(payload["image_size"]),
        target_labels=tuple(payload["target_labels"]),
        visibility_score=float(payload["visibility_score"]),
        readable_ratio=float(payload["readable_ratio"]),
    )
    decision = request_guidance_decision(
        observation,
        guidance_thresholds=guidance_thresholds,
    )
    prompt = (prompt_builder or QwenPromptBuilder()).build_guidance_explanation_prompt(
        observation=observation,
        decision=decision,
    )
    raw_text, source = (text_generator or QwenRuntimeTextGenerator()).generate(prompt)
    parsed = (output_parser or QwenStructuredOutputParser()).parse_guidance_explanation(
        raw_text
    )
    return {
        "guidance_action": decision.action.value,
        "reason": decision.reason,
        "language_action": parsed.suggested_action,
        "confidence": parsed.confidence,
        "operator_message": parsed.operator_message,
        "evidence_focus": parsed.evidence_focus,
        "language_source": source,
    }


def _default_recipe_path() -> Path:
    override = os.environ.get("ALLINONE_QWEN_GATEWAY_RECIPE")
    if override:
        return Path(override)
    override = os.environ.get("ALLINONE_QWEN_RECIPE")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[4] / "configs/model_recipes/qwen_gateway.yaml"


def _build_missing_target_result() -> dict[str, object]:
    return {
        "guidance_action": "hold_still",
        "reason": "target_not_detected",
        "language_action": "hold_still",
        "confidence": 0.0,
        "operator_message": "未检测到目标，请移动镜头搜索目标区域。",
        "evidence_focus": "先让目标进入画面，再继续判断取景质量",
        "language_source": "fallback",
    }
