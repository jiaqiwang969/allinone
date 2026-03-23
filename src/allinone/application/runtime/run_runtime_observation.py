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
from allinone.infrastructure.language.qwen.client import QwenClient
from allinone.infrastructure.language.qwen.prompt_builder import QwenPromptBuilder
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
    """Generate runtime explanations from an offline Qwen deployment."""

    def __init__(self, recipe_path: str | Path | None = None) -> None:
        self.recipe_path = Path(recipe_path) if recipe_path is not None else _default_recipe_path()

    def generate(self, prompt: str) -> tuple[str, str]:
        if self.recipe_path.exists():
            try:
                client = QwenClient.from_recipe(self.recipe_path)
                if client.is_runtime_available():
                    return client.generate_text(prompt), "qwen"
            except RuntimeError:
                pass
        return _DEFAULT_LANGUAGE_OUTPUT, "mock"


def run_runtime_observation(
    *,
    payload: dict[str, object],
    prompt_builder: QwenPromptBuilder | None = None,
    output_parser: QwenStructuredOutputParser | None = None,
    text_generator: RuntimeTextGenerator | None = None,
) -> dict[str, object]:
    observation = ingest_observation_window(
        prediction_rows=list(payload["prediction_rows"]),
        image_size=tuple(payload["image_size"]),
        target_labels=tuple(payload["target_labels"]),
        visibility_score=float(payload["visibility_score"]),
        readable_ratio=float(payload["readable_ratio"]),
    )
    decision = request_guidance_decision(observation)
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
    override = os.environ.get("ALLINONE_QWEN_RECIPE")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[4] / "configs/model_recipes/qwen35_9b.yaml"
