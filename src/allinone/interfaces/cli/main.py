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
from allinone.application.research.judge_experiment_candidates import (
    judge_experiment_candidates,
)
from allinone.application.research.run_experiment_batch import run_experiment_batch
from allinone.application.research.run_research_step import run_research_step
from allinone.application.research.register_experiment import register_experiment
from allinone.application.runtime.ingest_observation_window import (
    ingest_observation_window,
)
from allinone.application.runtime.run_runtime_observation import (
    run_runtime_observation as run_runtime_observation_usecase,
)
from allinone.application.runtime.request_guidance_decision import (
    request_guidance_decision,
)
from allinone.infrastructure.language.qwen.client import QwenClient
from allinone.infrastructure.guidance.policy_recipe import RuntimePolicyRecipeStore
from allinone.infrastructure.perception.video.sampler import ClipFrameSampler
from allinone.infrastructure.perception.vjepa.encoder import VJEPAEncoderAdapter
from allinone.infrastructure.perception.yolo.detector import (
    UltralyticsDetectorAdapter,
)
from allinone.infrastructure.research.autoresearch.policy_candidate_proposer import (
    RuleBasedPolicyCandidateProposer,
)
from allinone.infrastructure.research.autoresearch.replay_adapter import (
    AutoresearchReplayAdapter,
)
from allinone.infrastructure.research.autoresearch.judge_adapter import (
    AutoresearchJudgeAdapter,
)
from allinone.infrastructure.research.autoresearch.rule_based_judge import (
    RuleBasedAutoresearchJudge,
)
from allinone.infrastructure.research.autoresearch.run_writer import (
    AutoresearchRunWriter,
)
from allinone.domain.guidance.services import GuidanceThresholds
from allinone.domain.research.services import ExperimentSelectionService

_DEFAULT_LANGUAGE_OUTPUT = """{
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
    run_experiment = subparsers.add_parser("run-experiment")
    run_experiment.add_argument("--manifest", required=True)
    run_experiment.add_argument("--run-dir", required=True)
    run_experiment.add_argument("--candidate", required=True)
    run_experiment.add_argument("--yolo-model", required=True)
    run_experiment.add_argument("--vjepa-repo", required=True)
    run_experiment.add_argument("--vjepa-checkpoint", required=True)
    run_experiment.add_argument("--device", required=False)
    run_experiment.add_argument("--sample-frames", required=False, type=int, default=8)
    judge_experiment = subparsers.add_parser("judge-experiment")
    judge_experiment.add_argument("--experiment-id", required=True)
    judge_experiment.add_argument("--hypothesis", required=True)
    judge_experiment.add_argument("--target-metric", required=True)
    judge_experiment.add_argument(
        "--candidate-run",
        required=True,
        action="append",
    )
    judge_experiment.add_argument("--output", required=True)
    run_research_step_parser = subparsers.add_parser("run-research-step")
    run_research_step_parser.add_argument("--experiment-id", required=True)
    run_research_step_parser.add_argument("--hypothesis", required=True)
    run_research_step_parser.add_argument("--target-metric", required=True)
    run_research_step_parser.add_argument("--manifest", required=True)
    run_research_step_parser.add_argument("--base-policy", required=True)
    run_research_step_parser.add_argument("--candidate-count", required=True, type=int)
    run_research_step_parser.add_argument("--run-root", required=True)
    run_research_step_parser.add_argument("--output", required=True)
    run_research_step_parser.add_argument("--yolo-model", required=True)
    run_research_step_parser.add_argument("--vjepa-repo", required=True)
    run_research_step_parser.add_argument("--vjepa-checkpoint", required=True)
    run_research_step_parser.add_argument("--device", required=False)
    run_research_step_parser.add_argument(
        "--sample-frames",
        required=False,
        type=int,
        default=8,
    )
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
    result = run_runtime_observation_usecase(
        payload=_build_sample_payload(),
        text_generator=_CliRuntimeTextGenerator(),
    )
    return _print_runtime_result(result)


def _run_runtime_observation(input_path: str) -> int:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    result = run_runtime_observation_usecase(
        payload=payload,
        text_generator=_CliRuntimeTextGenerator(),
    )
    return _print_runtime_result(result)


def _run_experiment(
    manifest_path: str,
    run_dir: str,
    candidate_name: str,
    yolo_model_path: str,
    vjepa_repo: str,
    vjepa_checkpoint: str,
    device: str | None,
    sample_frames: int,
) -> int:
    batch = run_experiment_batch(
        manifest_rows=_load_manifest_rows(manifest_path),
        candidate_name=candidate_name,
        clip_analyzer=lambda *, clip_path, target_labels: build_raw_perception_payload_from_clip(
            clip_path=clip_path,
            target_labels=target_labels,
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
        ),
        runtime_runner=run_runtime_observation_usecase,
        run_writer=AutoresearchRunWriter(run_dir=run_dir),
    )
    summary = batch["run_artifacts"]["summary"]
    print(
        f"run_dir={batch['run_artifacts']['run_dir']} "
        f"candidate={candidate_name} "
        f"action_match_rate={float(summary['action_match_rate']):.2f} "
        f"target_detected_rate={float(summary['target_detected_rate']):.2f} "
        f"usable_clip_rate={float(summary['usable_clip_rate']):.2f}"
    )
    return 0


def _run_judge_experiment(
    experiment_id: str,
    hypothesis: str,
    target_metric: str,
    candidate_run_values: list[str],
    output_path: str,
) -> int:
    judgement = judge_experiment_candidates(
        experiment_id=experiment_id,
        hypothesis=hypothesis,
        target_metric=target_metric,
        candidate_runs=_parse_candidate_runs(candidate_run_values),
        replay_adapter=AutoresearchReplayAdapter(),
        candidate_judge=RuleBasedAutoresearchJudge(),
        judge_adapter=AutoresearchJudgeAdapter(),
        selection_service=ExperimentSelectionService(),
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(judgement, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"output={destination} "
        f"best_candidate_name={judgement['best_candidate_name']} "
        f"status={judgement['status']}"
    )
    return 0


def _run_research_step(
    *,
    experiment_id: str,
    hypothesis: str,
    target_metric: str,
    manifest_path: str,
    base_policy_path: str,
    candidate_count: int,
    run_root: str,
    output_path: str,
    yolo_model_path: str,
    vjepa_repo: str,
    vjepa_checkpoint: str,
    device: str | None,
    sample_frames: int,
) -> int:
    result = run_research_step(
        experiment_id=experiment_id,
        hypothesis=hypothesis,
        target_metric=target_metric,
        manifest_rows=_load_manifest_rows(manifest_path),
        base_policy_path=base_policy_path,
        candidate_count=candidate_count,
        run_root=run_root,
        policy_store=RuntimePolicyRecipeStore(),
        candidate_proposer=RuleBasedPolicyCandidateProposer(),
        candidate_runner=_CliExperimentBatchRunner(
            yolo_model_path=yolo_model_path,
            vjepa_repo=vjepa_repo,
            vjepa_checkpoint=vjepa_checkpoint,
            device=device,
            sample_frames=sample_frames,
        ),
        judge_usecase=_CliExperimentJudgeUseCase(),
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    judgement_path = Path(run_root) / "judgement.json"
    judgement_path.parent.mkdir(parents=True, exist_ok=True)
    judgement_path.write_text(
        json.dumps(result["judgement"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"output={destination} "
        f"judgement={judgement_path} "
        f"best_candidate_name={result['best_candidate_name']} "
        f"status={result['status']}"
    )
    return 0


def _build_sample_payload() -> dict[str, object]:
    return {
        "prediction_rows": [
            {"label": "meter", "confidence": 0.91, "xyxy": [600, 200, 900, 800]},
        ],
        "image_size": [1000, 1000],
        "target_labels": ["meter"],
        "visibility_score": 0.85,
        "readable_ratio": 0.8,
    }


def _print_runtime_result(result: dict[str, object]) -> int:
    print(
        f"guidance_action={result['guidance_action']} "
        f"reason={result['reason']} "
        f"language_action={result['language_action']} "
        f"confidence={float(result['confidence']):.2f} "
        f"source={result['language_source']} "
        f"message={result['operator_message']}"
    )
    return 0


def _load_manifest_rows(manifest_path: str) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in Path(manifest_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _parse_candidate_runs(values: list[str]) -> list[dict[str, str]]:
    candidate_runs: list[dict[str, str]] = []
    for value in values:
        candidate_name, separator, run_dir = value.partition("=")
        if not separator or not candidate_name.strip() or not run_dir.strip():
            raise ValueError("candidate-run must use the format <name>=<run_dir>")
        candidate_runs.append(
            {
                "candidate_name": candidate_name.strip(),
                "run_dir": run_dir.strip(),
            }
        )
    return candidate_runs


def _resolve_qwen_recipe_path() -> Path:
    override = os.environ.get("ALLINONE_QWEN_RECIPE")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[4] / "configs/model_recipes/qwen35_9b.yaml"


class _CliRuntimeTextGenerator:
    def generate(self, prompt: str) -> tuple[str, str]:
        recipe = _resolve_qwen_recipe_path()
        if recipe.exists():
            try:
                client = QwenClient.from_recipe(recipe)
                if client.is_runtime_available():
                    return client.generate_text(prompt), "qwen"
            except RuntimeError:
                pass
        return _DEFAULT_LANGUAGE_OUTPUT, "mock"


class _CliExperimentBatchRunner:
    def __init__(
        self,
        *,
        yolo_model_path: str,
        vjepa_repo: str,
        vjepa_checkpoint: str,
        device: str | None,
        sample_frames: int,
    ) -> None:
        self.yolo_model_path = yolo_model_path
        self.vjepa_repo = vjepa_repo
        self.vjepa_checkpoint = vjepa_checkpoint
        self.device = device
        self.sample_frames = sample_frames

    def run(
        self,
        *,
        manifest_rows: list[dict[str, object]],
        candidate_name: str,
        run_dir: str,
        guidance_thresholds: GuidanceThresholds,
        policy_path: str,
    ) -> dict[str, object]:
        return run_experiment_batch(
            manifest_rows=manifest_rows,
            candidate_name=candidate_name,
            clip_analyzer=lambda *, clip_path, target_labels: build_raw_perception_payload_from_clip(
                clip_path=clip_path,
                target_labels=target_labels,
                sampler=ClipFrameSampler(frame_count=self.sample_frames),
                detector=UltralyticsDetectorAdapter(
                    model_path=self.yolo_model_path,
                    device=self.device,
                ),
                clip_scorer=VJEPAEncoderAdapter(
                    repo_path=self.vjepa_repo,
                    checkpoint_path=self.vjepa_checkpoint,
                    device=self.device,
                ),
            ),
            runtime_runner=lambda *, payload: run_runtime_observation_usecase(
                payload=payload,
                guidance_thresholds=guidance_thresholds,
                text_generator=_CliRuntimeTextGenerator(),
            ),
            run_writer=AutoresearchRunWriter(run_dir=run_dir),
        )


class _CliExperimentJudgeUseCase:
    def __call__(
        self,
        *,
        experiment_id: str,
        hypothesis: str,
        target_metric: str,
        candidate_runs: list[dict[str, str]],
    ) -> dict[str, object]:
        return judge_experiment_candidates(
            experiment_id=experiment_id,
            hypothesis=hypothesis,
            target_metric=target_metric,
            candidate_runs=candidate_runs,
            replay_adapter=AutoresearchReplayAdapter(),
            candidate_judge=RuleBasedAutoresearchJudge(),
            judge_adapter=AutoresearchJudgeAdapter(),
            selection_service=ExperimentSelectionService(),
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
    if args.command == "run-experiment":
        return _run_experiment(
            manifest_path=args.manifest,
            run_dir=args.run_dir,
            candidate_name=args.candidate,
            yolo_model_path=args.yolo_model,
            vjepa_repo=args.vjepa_repo,
            vjepa_checkpoint=args.vjepa_checkpoint,
            device=args.device,
            sample_frames=args.sample_frames,
        )
    if args.command == "judge-experiment":
        return _run_judge_experiment(
            experiment_id=args.experiment_id,
            hypothesis=args.hypothesis,
            target_metric=args.target_metric,
            candidate_run_values=args.candidate_run,
            output_path=args.output,
        )
    if args.command == "run-research-step":
        return _run_research_step(
            experiment_id=args.experiment_id,
            hypothesis=args.hypothesis,
            target_metric=args.target_metric,
            manifest_path=args.manifest,
            base_policy_path=args.base_policy,
            candidate_count=args.candidate_count,
            run_root=args.run_root,
            output_path=args.output,
            yolo_model_path=args.yolo_model,
            vjepa_repo=args.vjepa_repo,
            vjepa_checkpoint=args.vjepa_checkpoint,
            device=args.device,
            sample_frames=args.sample_frames,
        )
    if args.command == "research-smoke":
        return _run_research_smoke()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
