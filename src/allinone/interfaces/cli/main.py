"""CLI entrypoint for allinone."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from allinone.application.research.register_experiment import register_experiment
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
from allinone.infrastructure.research.autoresearch.replay_adapter import (
    AutoresearchReplayAdapter,
)

_DEFAULT_LANGUAGE_SMOKE_OUTPUT = """{
    "operator_message": "请向左移动，让仪表回到画面中央。",
    "suggested_action": "left",
    "confidence": 0.82,
    "evidence_focus": "确保整个表盘完整可见"
}"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="allinone")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True
    subparsers.add_parser("guidance-smoke")
    subparsers.add_parser("language-smoke")
    subparsers.add_parser("research-smoke")
    return parser


def _run_guidance_smoke() -> int:
    observation = ingest_observation_window(
        prediction_rows=[
            {"label": "meter", "confidence": 0.91, "xyxy": [600, 200, 900, 800]},
        ],
        image_size=(1000, 1000),
        target_labels=("meter",),
        visibility_score=0.85,
        readable_ratio=0.8,
    )
    decision = request_guidance_decision(observation)
    print(f"guidance_action={decision.action.value} reason={decision.reason}")
    return 0


def _run_research_smoke() -> int:
    run = register_experiment(
        experiment_id="exp-smoke-001",
        hypothesis="compare baseline and candidate guidance policies",
        target_metric="guidance_success_rate",
        candidate_names=["baseline", "candidate-a"],
    )
    payload = AutoresearchReplayAdapter().build_payload(run)
    candidate_names = ",".join(payload["candidate_names"])
    print(
        f"experiment_id={payload['experiment_id']} "
        f"target_metric={payload['target_metric']} "
        f"candidate_names={candidate_names}"
    )
    return 0


def _run_language_smoke() -> int:
    observation = ingest_observation_window(
        prediction_rows=[
            {"label": "meter", "confidence": 0.91, "xyxy": [600, 200, 900, 800]},
        ],
        image_size=(1000, 1000),
        target_labels=("meter",),
        visibility_score=0.85,
        readable_ratio=0.8,
    )
    decision = request_guidance_decision(observation)
    prompt = QwenPromptBuilder().build_guidance_explanation_prompt(
        observation=observation,
        decision=decision,
    )
    parsed, source = _generate_language_explanation(prompt)
    print(
        f"language_action={parsed.suggested_action} "
        f"confidence={parsed.confidence:.2f} "
        f"source={source} "
        f"message={parsed.operator_message}"
    )
    return 0


def _resolve_qwen_recipe_path() -> Path:
    override = os.environ.get("ALLINONE_QWEN_RECIPE")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[4] / "configs/model_recipes/qwen35_9b.yaml"


def _generate_language_explanation(prompt: str):
    parser = QwenStructuredOutputParser()
    recipe = _resolve_qwen_recipe_path()
    if recipe.exists():
        try:
            client = QwenClient.from_recipe(recipe)
            if client.is_runtime_available():
                raw_text = client.generate_text(prompt)
                return parser.parse_guidance_explanation(raw_text), "qwen"
        except RuntimeError:
            pass
    return (
        parser.parse_guidance_explanation(_DEFAULT_LANGUAGE_SMOKE_OUTPUT),
        "mock",
    )


def main(argv: list[str] | None = None) -> int:
    """Run the allinone CLI."""
    args = _build_parser().parse_args(argv)
    if args.command == "guidance-smoke":
        return _run_guidance_smoke()
    if args.command == "language-smoke":
        return _run_language_smoke()
    if args.command == "research-smoke":
        return _run_research_smoke()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
