"""CLI entrypoint for allinone."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from allinone.application.runtime.build_clip_perception_payload import (
    build_raw_perception_payload_from_clip,
)
from allinone.application.runtime.build_raw_perception_payload import (
    build_raw_perception_payload_from_image,
)
from allinone.application.runtime.build_observation_payload import (
    build_observation_payload_from_raw,
)
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
from allinone.infrastructure.perception.video.sampler import ClipFrameSampler
from allinone.infrastructure.perception.vjepa.encoder import VJEPAEncoderAdapter
from allinone.infrastructure.perception.yolo.detector import (
    UltralyticsDetectorAdapter,
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
    detect_image = subparsers.add_parser("detect-image")
    detect_image.add_argument("--image", required=True)
    detect_image.add_argument("--model", required=True)
    detect_image.add_argument("--targets", required=True)
    detect_image.add_argument("--output", required=True)
    detect_image.add_argument("--device", required=False)
    analyze_clip = subparsers.add_parser("analyze-clip")
    analyze_clip.add_argument("--clip", required=True)
    analyze_clip.add_argument("--yolo-model", required=True)
    analyze_clip.add_argument("--vjepa-repo", required=True)
    analyze_clip.add_argument("--vjepa-checkpoint", required=True)
    analyze_clip.add_argument("--targets", required=True)
    analyze_clip.add_argument("--output", required=True)
    analyze_clip.add_argument("--device", required=False)
    analyze_clip.add_argument("--sample-frames", required=False, type=int, default=8)
    build_observation_payload = subparsers.add_parser("build-observation-payload")
    build_observation_payload.add_argument("--input", required=True)
    build_observation_payload.add_argument("--output", required=True)
    subparsers.add_parser("guidance-smoke")
    subparsers.add_parser("language-smoke")
    subparsers.add_parser("research-smoke")
    runtime_observation = subparsers.add_parser("runtime-observation")
    runtime_observation.add_argument("--input", required=True)
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


def _run_build_observation_payload(input_path: str, output_path: str) -> int:
    raw_payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    payload = build_observation_payload_from_raw(raw_payload)
    Path(output_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


def _run_detect_image(
    image_path: str,
    model_path: str,
    targets: str,
    output_path: str,
    device: str | None,
) -> int:
    raw_payload = build_raw_perception_payload_from_image(
        image_path=image_path,
        target_labels=tuple(
            item.strip() for item in targets.split(",") if item.strip()
        ),
        model_path=model_path,
        device=device,
    )
    Path(output_path).write_text(
        json.dumps(raw_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


def _run_analyze_clip(
    clip_path: str,
    yolo_model_path: str,
    vjepa_repo: str,
    vjepa_checkpoint: str,
    targets: str,
    output_path: str,
    device: str | None,
    sample_frames: int,
) -> int:
    raw_payload = build_raw_perception_payload_from_clip(
        clip_path=clip_path,
        target_labels=tuple(
            item.strip() for item in targets.split(",") if item.strip()
        ),
        sampler=ClipFrameSampler(frame_count=sample_frames),
        detector=UltralyticsDetectorAdapter(
            model_path=yolo_model_path,
            device=device,
        ),
        clip_scorer=VJEPAEncoderAdapter(
            repo_path=vjepa_repo,
            checkpoint_path=vjepa_checkpoint,
            device=device,
        ),
    )
    Path(output_path).write_text(
        json.dumps(raw_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


def _run_language_smoke() -> int:
    observation = _build_sample_observation()
    return _print_guidance_and_language(observation)


def _run_runtime_observation(input_path: str) -> int:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    observation = ingest_observation_window(
        prediction_rows=payload["prediction_rows"],
        image_size=tuple(payload["image_size"]),
        target_labels=tuple(payload["target_labels"]),
        visibility_score=float(payload["visibility_score"]),
        readable_ratio=float(payload["readable_ratio"]),
    )
    return _print_guidance_and_language(observation)


def _build_sample_observation():
    return ingest_observation_window(
        prediction_rows=[
            {"label": "meter", "confidence": 0.91, "xyxy": [600, 200, 900, 800]},
        ],
        image_size=(1000, 1000),
        target_labels=("meter",),
        visibility_score=0.85,
        readable_ratio=0.8,
    )


def _print_guidance_and_language(observation) -> int:
    decision = request_guidance_decision(observation)
    prompt = QwenPromptBuilder().build_guidance_explanation_prompt(
        observation=observation,
        decision=decision,
    )
    parsed, source = _generate_language_explanation(prompt)
    print(
        f"guidance_action={decision.action.value} "
        f"reason={decision.reason} "
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
    if args.command == "detect-image":
        return _run_detect_image(
            image_path=args.image,
            model_path=args.model,
            targets=args.targets,
            output_path=args.output,
            device=args.device,
        )
    if args.command == "analyze-clip":
        return _run_analyze_clip(
            clip_path=args.clip,
            yolo_model_path=args.yolo_model,
            vjepa_repo=args.vjepa_repo,
            vjepa_checkpoint=args.vjepa_checkpoint,
            targets=args.targets,
            output_path=args.output,
            device=args.device,
            sample_frames=args.sample_frames,
        )
    if args.command == "build-observation-payload":
        return _run_build_observation_payload(args.input, args.output)
    if args.command == "guidance-smoke":
        return _run_guidance_smoke()
    if args.command == "language-smoke":
        return _run_language_smoke()
    if args.command == "runtime-observation":
        return _run_runtime_observation(args.input)
    if args.command == "research-smoke":
        return _run_research_smoke()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
