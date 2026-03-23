import json
from pathlib import Path

from allinone.application.runtime.build_observation_payload import (
    build_observation_payload_from_raw,
)
from allinone.application.runtime.run_runtime_observation import (
    run_runtime_observation,
)
from allinone.domain.guidance.services import GuidanceThresholds
from allinone.domain.research.entities import CandidateConfig, ExperimentRun
from allinone.domain.research.value_objects import ExperimentId, MetricName
from allinone.infrastructure.research.autoresearch.judge_adapter import (
    AutoresearchJudgeAdapter,
)
from allinone.infrastructure.research.autoresearch.guidance_boundary_dataset import (
    GuidanceBoundaryDatasetBuilder,
)
from allinone.infrastructure.research.autoresearch.replay_adapter import (
    AutoresearchReplayAdapter,
)
from allinone.infrastructure.research.autoresearch.rule_based_judge import (
    RuleBasedAutoresearchJudge,
)
from allinone.infrastructure.research.autoresearch.policy_candidate_proposer import (
    RuleBasedPolicyCandidateProposer,
)
from allinone.infrastructure.research.autoresearch.run_writer import (
    AutoresearchRunWriter,
)


def _build_experiment_run() -> ExperimentRun:
    return ExperimentRun.register(
        experiment_id=ExperimentId("exp-001"),
        hypothesis="test better guidance thresholds",
        target_metric=MetricName("guidance_success_rate"),
        candidate_configs=[
            CandidateConfig(name="baseline", parameters={"policy": "v1"}),
            CandidateConfig(name="candidate-a", parameters={"policy": "v2"}),
        ],
    )


def test_replay_adapter_builds_runtime_payload_from_experiment_run():
    payload = AutoresearchReplayAdapter().build_payload(_build_experiment_run())

    assert payload["experiment_id"] == "exp-001"
    assert payload["target_metric"] == "guidance_success_rate"
    assert payload["candidate_names"] == ["baseline", "candidate-a"]


def test_run_writer_writes_run_artifacts_and_summary(tmp_path):
    run_dir = tmp_path / "run-2026-03-23-demo"
    writer = AutoresearchRunWriter(run_dir=run_dir)
    manifest_rows = [
        {
            "clip_id": "clip-001",
            "clip_path": "/data/clip-001.mp4",
            "target_labels": ["meter"],
            "task_type": "view_guidance",
            "expected_action": "left",
            "notes": "目标偏右",
        },
        {
            "clip_id": "clip-002",
            "clip_path": "/data/clip-002.mp4",
            "target_labels": ["meter"],
            "task_type": "view_guidance",
            "expected_action": "left",
            "notes": "目标丢失",
        },
    ]
    result_rows = [
        {
            "clip_id": "clip-001",
            "candidate_name": "baseline",
            "task_type": "view_guidance",
            "target_labels": ["meter"],
            "expected_action": "left",
            "guidance_action": "left",
            "guidance_reason": "target_shifted_right",
            "language_action": "left",
            "action_match": True,
            "target_detected": True,
            "best_frame_index": 2,
            "visibility_score": 0.85,
            "readable_ratio": 0.8,
            "stability_score": 0.91,
            "alignment_score": 0.84,
            "operator_message": "请向左移动",
            "evidence_focus": "表盘完整可见",
            "language_source": "fake-qwen",
            "error": None,
            "raw_payload": {
                "detections": {
                    "prediction_rows": [
                        {
                            "label": "meter",
                            "confidence": 0.91,
                            "xyxy": [600, 200, 900, 800],
                        }
                    ],
                    "image_size": [1000, 1000],
                    "target_labels": ["meter"],
                    "best_frame_index": 2,
                },
                "vjepa": {
                    "visibility_score": 0.85,
                    "readable_ratio": 0.8,
                    "stability_score": 0.91,
                    "alignment_score": 0.84,
                },
            },
            "payload": {
                "prediction_rows": [
                    {
                        "label": "meter",
                        "confidence": 0.91,
                        "xyxy": [600, 200, 900, 800],
                    }
                ],
                "image_size": [1000, 1000],
                "target_labels": ["meter"],
                "visibility_score": 0.85,
                "readable_ratio": 0.8,
            },
        },
        {
            "clip_id": "clip-002",
            "candidate_name": "baseline",
            "task_type": "view_guidance",
            "target_labels": ["meter"],
            "expected_action": "left",
            "guidance_action": "hold_still",
            "guidance_reason": "fully_centered",
            "language_action": "hold_still",
            "action_match": False,
            "target_detected": False,
            "best_frame_index": 0,
            "visibility_score": 0.4,
            "readable_ratio": 0.6,
            "stability_score": 0.51,
            "alignment_score": 0.45,
            "operator_message": "保持当前角度",
            "evidence_focus": "继续稳定画面",
            "language_source": "fake-qwen",
            "error": None,
            "raw_payload": {
                "detections": {
                    "prediction_rows": [],
                    "image_size": [1000, 1000],
                    "target_labels": ["meter"],
                    "best_frame_index": 0,
                },
                "vjepa": {
                    "visibility_score": 0.4,
                    "readable_ratio": 0.6,
                    "stability_score": 0.51,
                    "alignment_score": 0.45,
                },
            },
            "payload": {
                "prediction_rows": [],
                "image_size": [1000, 1000],
                "target_labels": ["meter"],
                "visibility_score": 0.4,
                "readable_ratio": 0.6,
            },
        },
    ]

    artifacts = writer.write(
        manifest_rows=manifest_rows,
        result_rows=result_rows,
        candidate_name="baseline",
    )

    assert artifacts == {
        "run_dir": str(run_dir),
        "manifest_path": str(run_dir / "manifest.jsonl"),
        "results_path": str(run_dir / "results.jsonl"),
        "summary_path": str(run_dir / "summary.json"),
        "summary": {
            "candidate_name": "baseline",
            "clip_count": 2,
            "action_match_rate": 0.5,
            "target_detected_rate": 0.5,
            "usable_clip_rate": 0.5,
        },
    }
    assert (run_dir / "manifest.jsonl").exists()
    assert (run_dir / "results.jsonl").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "raw" / "clip-001.json").exists()
    assert (run_dir / "payload" / "clip-001.json").exists()

    written_results = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert written_results == [
        {
            "clip_id": "clip-001",
            "candidate_name": "baseline",
            "task_type": "view_guidance",
            "target_labels": ["meter"],
            "expected_action": "left",
            "guidance_action": "left",
            "guidance_reason": "target_shifted_right",
            "language_action": "left",
            "action_match": True,
            "target_detected": True,
            "best_frame_index": 2,
            "visibility_score": 0.85,
            "readable_ratio": 0.8,
            "stability_score": 0.91,
            "alignment_score": 0.84,
            "operator_message": "请向左移动",
            "evidence_focus": "表盘完整可见",
            "language_source": "fake-qwen",
            "error": None,
        },
        {
            "clip_id": "clip-002",
            "candidate_name": "baseline",
            "task_type": "view_guidance",
            "target_labels": ["meter"],
            "expected_action": "left",
            "guidance_action": "hold_still",
            "guidance_reason": "fully_centered",
            "language_action": "hold_still",
            "action_match": False,
            "target_detected": False,
            "best_frame_index": 0,
            "visibility_score": 0.4,
            "readable_ratio": 0.6,
            "stability_score": 0.51,
            "alignment_score": 0.45,
            "operator_message": "保持当前角度",
            "evidence_focus": "继续稳定画面",
            "language_source": "fake-qwen",
            "error": None,
        },
    ]
    assert json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) == {
        "candidate_name": "baseline",
        "clip_count": 2,
        "action_match_rate": 0.5,
        "target_detected_rate": 0.5,
        "usable_clip_rate": 0.5,
    }


def test_run_writer_includes_reason_match_rate_when_result_rows_have_reason_labels(tmp_path):
    run_dir = tmp_path / "run-2026-03-23-reason-match"
    writer = AutoresearchRunWriter(run_dir=run_dir)

    artifacts = writer.write(
        manifest_rows=[
            {
                "clip_id": "clip-101",
                "raw_payload_path": "raw/clip-101.json",
                "target_labels": ["meter"],
                "task_type": "view_guidance",
                "expected_action": "hold_still",
                "expected_reason": "stabilize_before_capture",
            },
            {
                "clip_id": "clip-102",
                "raw_payload_path": "raw/clip-102.json",
                "target_labels": ["meter"],
                "task_type": "view_guidance",
                "expected_action": "left",
                "expected_reason": "target_shifted_right",
            },
        ],
        result_rows=[
            {
                "clip_id": "clip-101",
                "candidate_name": "candidate-1",
                "task_type": "view_guidance",
                "target_labels": ["meter"],
                "expected_action": "hold_still",
                "guidance_action": "hold_still",
                "guidance_reason": "stabilize_before_capture",
                "expected_reason": "stabilize_before_capture",
                "reason_match": True,
                "language_action": "hold_still",
                "action_match": True,
                "target_detected": True,
                "best_frame_index": 1,
                "visibility_score": 0.86,
                "readable_ratio": 0.79,
                "stability_score": 0.91,
                "alignment_score": 0.87,
                "operator_message": "保持稳定",
                "evidence_focus": "等待清晰抓拍",
                "language_source": "fake-qwen",
                "error": None,
                "raw_payload": {
                    "detections": {
                        "prediction_rows": [],
                        "image_size": [1000, 1000],
                        "target_labels": ["meter"],
                        "best_frame_index": 1,
                    },
                    "vjepa": {
                        "visibility_score": 0.86,
                        "readable_ratio": 0.79,
                        "stability_score": 0.91,
                        "alignment_score": 0.87,
                    },
                },
                "payload": {
                    "prediction_rows": [],
                    "image_size": [1000, 1000],
                    "target_labels": ["meter"],
                    "visibility_score": 0.86,
                    "readable_ratio": 0.79,
                },
            },
            {
                "clip_id": "clip-102",
                "candidate_name": "candidate-1",
                "task_type": "view_guidance",
                "target_labels": ["meter"],
                "expected_action": "left",
                "guidance_action": "left",
                "guidance_reason": "fully_centered",
                "expected_reason": "target_shifted_right",
                "reason_match": False,
                "language_action": "left",
                "action_match": True,
                "target_detected": True,
                "best_frame_index": 2,
                "visibility_score": 0.89,
                "readable_ratio": 0.83,
                "stability_score": 0.93,
                "alignment_score": 0.9,
                "operator_message": "请向左移动",
                "evidence_focus": "继续稳定画面",
                "language_source": "fake-qwen",
                "error": None,
                "raw_payload": {
                    "detections": {
                        "prediction_rows": [],
                        "image_size": [1000, 1000],
                        "target_labels": ["meter"],
                        "best_frame_index": 2,
                    },
                    "vjepa": {
                        "visibility_score": 0.89,
                        "readable_ratio": 0.83,
                        "stability_score": 0.93,
                        "alignment_score": 0.9,
                    },
                },
                "payload": {
                    "prediction_rows": [],
                    "image_size": [1000, 1000],
                    "target_labels": ["meter"],
                    "visibility_score": 0.89,
                    "readable_ratio": 0.83,
                },
            },
        ],
        candidate_name="candidate-1",
    )

    assert artifacts["summary"] == {
        "candidate_name": "candidate-1",
        "clip_count": 2,
        "action_match_rate": 1.0,
        "target_detected_rate": 1.0,
        "usable_clip_rate": 1.0,
        "reason_match_rate": 0.5,
    }
    assert json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) == {
        "candidate_name": "candidate-1",
        "clip_count": 2,
        "action_match_rate": 1.0,
        "target_detected_rate": 1.0,
        "usable_clip_rate": 1.0,
        "reason_match_rate": 0.5,
    }

    written_results = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert written_results == [
        {
            "clip_id": "clip-101",
            "candidate_name": "candidate-1",
            "task_type": "view_guidance",
            "target_labels": ["meter"],
            "expected_action": "hold_still",
            "guidance_action": "hold_still",
            "guidance_reason": "stabilize_before_capture",
            "expected_reason": "stabilize_before_capture",
            "reason_match": True,
            "language_action": "hold_still",
            "action_match": True,
            "target_detected": True,
            "best_frame_index": 1,
            "visibility_score": 0.86,
            "readable_ratio": 0.79,
            "stability_score": 0.91,
            "alignment_score": 0.87,
            "operator_message": "保持稳定",
            "evidence_focus": "等待清晰抓拍",
            "language_source": "fake-qwen",
            "error": None,
        },
        {
            "clip_id": "clip-102",
            "candidate_name": "candidate-1",
            "task_type": "view_guidance",
            "target_labels": ["meter"],
            "expected_action": "left",
            "guidance_action": "left",
            "guidance_reason": "fully_centered",
            "expected_reason": "target_shifted_right",
            "reason_match": False,
            "language_action": "left",
            "action_match": True,
            "target_detected": True,
            "best_frame_index": 2,
            "visibility_score": 0.89,
            "readable_ratio": 0.83,
            "stability_score": 0.93,
            "alignment_score": 0.9,
            "operator_message": "请向左移动",
            "evidence_focus": "继续稳定画面",
            "language_source": "fake-qwen",
            "error": None,
        },
    ]


def test_replay_adapter_builds_payload_from_run_directory(tmp_path):
    run_dir = tmp_path / "run-2026-03-23-demo"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "candidate_name": "baseline",
                "clip_count": 2,
                "action_match_rate": 0.5,
                "target_detected_rate": 0.5,
                "usable_clip_rate": 0.5,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "results.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"clip_id": "clip-001", "guidance_action": "left"}),
                json.dumps({"clip_id": "clip-002", "guidance_action": "hold_still"}),
            ]
        ),
        encoding="utf-8",
    )

    payload = AutoresearchReplayAdapter().build_run_payload(run_dir)

    assert payload == {
        "run_dir": str(run_dir),
        "candidate_name": "baseline",
        "summary": {
            "candidate_name": "baseline",
            "clip_count": 2,
            "action_match_rate": 0.5,
            "target_detected_rate": 0.5,
            "usable_clip_rate": 0.5,
        },
        "results_path": str(run_dir / "results.jsonl"),
        "result_count": 2,
    }


def test_judge_adapter_converts_score_row_to_domain_evaluation():
    evaluation = AutoresearchJudgeAdapter().to_candidate_evaluation(
        candidate_name="candidate-a",
        score=0.81,
        summary="better guidance alignment",
    )

    assert evaluation.candidate_name == "candidate-a"
    assert evaluation.score == 0.81


def test_rule_based_judge_scores_candidate_run_payload(tmp_path):
    run_dir = tmp_path / "run-2026-03-23-baseline"
    run_dir.mkdir(parents=True)
    results_path = run_dir / "results.jsonl"
    results_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "clip_id": "clip-001",
                        "target_detected": True,
                        "error": None,
                    }
                ),
                json.dumps(
                    {
                        "clip_id": "clip-002",
                        "target_detected": False,
                        "error": None,
                    }
                ),
                json.dumps(
                    {
                        "clip_id": "clip-003",
                        "target_detected": False,
                        "error": "runtime_error",
                    }
                ),
                json.dumps(
                    {
                        "clip_id": "clip-004",
                        "target_detected": True,
                        "error": None,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    payload = {
        "run_dir": str(run_dir),
        "candidate_name": "baseline",
        "summary": {
            "candidate_name": "baseline",
            "clip_count": 4,
            "action_match_rate": 0.75,
            "target_detected_rate": 0.5,
            "usable_clip_rate": 0.5,
        },
        "results_path": str(results_path),
        "result_count": 4,
    }

    judgement = RuleBasedAutoresearchJudge().score_candidate(payload)

    assert judgement == {
        "candidate_name": "baseline",
        "run_dir": str(run_dir),
        "score": 0.575,
        "summary": (
            "action_match_rate=0.75 target_detected_rate=0.50 "
            "usable_clip_rate=0.50 error_rate=0.25 "
            "target_not_detected_ratio=0.50"
        ),
        "metrics": {
            "action_match_rate": 0.75,
            "target_detected_rate": 0.5,
            "usable_clip_rate": 0.5,
            "error_rate": 0.25,
            "target_not_detected_ratio": 0.5,
            "result_count": 4,
        },
    }


def test_rule_based_judge_uses_reason_match_rate_when_present(tmp_path):
    run_dir = tmp_path / "run-2026-03-23-reason-aware"
    run_dir.mkdir(parents=True)
    results_path = run_dir / "results.jsonl"
    results_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "clip_id": "clip-101",
                        "target_detected": True,
                        "reason_match": True,
                        "error": None,
                    }
                ),
                json.dumps(
                    {
                        "clip_id": "clip-102",
                        "target_detected": True,
                        "reason_match": False,
                        "error": None,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    payload = {
        "run_dir": str(run_dir),
        "candidate_name": "candidate-1",
        "summary": {
            "candidate_name": "candidate-1",
            "clip_count": 2,
            "action_match_rate": 1.0,
            "target_detected_rate": 1.0,
            "usable_clip_rate": 1.0,
            "reason_match_rate": 0.5,
        },
        "results_path": str(results_path),
        "result_count": 2,
    }

    judgement = RuleBasedAutoresearchJudge().score_candidate(payload)

    assert judgement == {
        "candidate_name": "candidate-1",
        "run_dir": str(run_dir),
        "score": 0.9,
        "summary": (
            "action_match_rate=1.00 target_detected_rate=1.00 "
            "usable_clip_rate=1.00 reason_match_rate=0.50 "
            "error_rate=0.00 target_not_detected_ratio=0.00"
        ),
        "metrics": {
            "action_match_rate": 1.0,
            "target_detected_rate": 1.0,
            "usable_clip_rate": 1.0,
            "reason_match_rate": 0.5,
            "error_rate": 0.0,
            "target_not_detected_ratio": 0.0,
            "result_count": 2,
        },
    }


def test_rule_based_policy_candidate_proposer_generates_baseline_and_mutations():
    candidates = RuleBasedPolicyCandidateProposer().propose_candidates(
        base_thresholds={
            "centered_offset_max": 0.09,
            "directional_offset_min": 0.18,
            "ready_fill_ratio_max": 0.85,
        },
        candidate_count=4,
    )

    assert candidates == [
        {
            "candidate_name": "baseline",
            "mutation": "baseline",
            "guidance_thresholds": {
                "centered_offset_max": 0.09,
                "directional_offset_min": 0.18,
                "ready_fill_ratio_max": 0.85,
            },
        },
        {
            "candidate_name": "candidate-1",
            "mutation": "tighten_center_window",
            "guidance_thresholds": {
                "centered_offset_max": 0.072,
                "directional_offset_min": 0.18,
                "ready_fill_ratio_max": 0.85,
            },
        },
        {
            "candidate_name": "candidate-2",
            "mutation": "earlier_direction_trigger",
            "guidance_thresholds": {
                "centered_offset_max": 0.09,
                "directional_offset_min": 0.153,
                "ready_fill_ratio_max": 0.85,
            },
        },
        {
            "candidate_name": "candidate-3",
            "mutation": "allow_larger_target_before_backward",
            "guidance_thresholds": {
                "centered_offset_max": 0.09,
                "directional_offset_min": 0.18,
                "ready_fill_ratio_max": 0.8925,
            },
        },
    ]


def test_guidance_boundary_dataset_builder_creates_sensitive_boundary_dataset(tmp_path):
    output_dir = tmp_path / "guidance-boundary"
    builder = GuidanceBoundaryDatasetBuilder()
    base_raw_payload = {
        "detections": {
            "prediction_rows": [
                {
                    "label": "person",
                    "confidence": 0.97,
                    "xyxy": [320, 140, 680, 860],
                }
            ],
            "image_size": [1000, 1000],
            "target_labels": ["person"],
            "best_frame_index": 2,
        },
        "vjepa": {
            "visibility_score": 0.88,
            "readable_ratio": 0.81,
            "stability_score": 0.92,
            "alignment_score": 0.9,
        },
    }

    dataset = builder.build(
        base_raw_payload=base_raw_payload,
        output_dir=output_dir,
        target_label="person",
    )

    assert dataset == {
        "output_dir": str(output_dir),
        "manifest_path": str(output_dir / "manifest.jsonl"),
        "raw_dir": str(output_dir / "raw"),
        "case_count": 4,
    }
    assert (output_dir / "raw" / "tight_center_boundary.json").exists()
    assert (output_dir / "raw" / "direction_trigger_boundary.json").exists()
    assert (output_dir / "raw" / "reverse_direction_trigger_boundary.json").exists()
    assert (output_dir / "raw" / "oversize_boundary.json").exists()

    manifest_rows = [
        json.loads(line)
        for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest_rows == [
        {
            "clip_id": "tight_center_boundary",
            "raw_payload_path": str(output_dir / "raw" / "tight_center_boundary.json"),
            "target_labels": ["person"],
            "task_type": "view_guidance",
            "expected_action": "hold_still",
            "expected_reason": "stabilize_before_capture",
        },
        {
            "clip_id": "direction_trigger_boundary",
            "raw_payload_path": str(output_dir / "raw" / "direction_trigger_boundary.json"),
            "target_labels": ["person"],
            "task_type": "view_guidance",
            "expected_action": "left",
            "expected_reason": "target_shifted_right",
        },
        {
            "clip_id": "reverse_direction_trigger_boundary",
            "raw_payload_path": str(
                output_dir / "raw" / "reverse_direction_trigger_boundary.json"
            ),
            "target_labels": ["person"],
            "task_type": "view_guidance",
            "expected_action": "right",
            "expected_reason": "target_shifted_left",
        },
        {
            "clip_id": "oversize_boundary",
            "raw_payload_path": str(output_dir / "raw" / "oversize_boundary.json"),
            "target_labels": ["person"],
            "task_type": "view_guidance",
            "expected_action": "hold_still",
            "expected_reason": "fully_centered",
        },
    ]

    baseline = GuidanceThresholds()
    tighter_center = GuidanceThresholds(centered_offset_max=0.072)
    earlier_direction = GuidanceThresholds(directional_offset_min=0.153)
    relaxed_size = GuidanceThresholds(ready_fill_ratio_max=0.8925)

    tight_center = json.loads(
        (output_dir / "raw" / "tight_center_boundary.json").read_text(encoding="utf-8")
    )
    assert tight_center["vjepa"] == base_raw_payload["vjepa"]
    assert _run_guidance(tight_center, thresholds=baseline) == {
        "action": "hold_still",
        "reason": "fully_centered",
    }
    assert _run_guidance(tight_center, thresholds=tighter_center) == {
        "action": "hold_still",
        "reason": "stabilize_before_capture",
    }

    direction_trigger = json.loads(
        (output_dir / "raw" / "direction_trigger_boundary.json").read_text(
            encoding="utf-8"
        )
    )
    assert _run_guidance(direction_trigger, thresholds=baseline) == {
        "action": "hold_still",
        "reason": "stabilize_before_capture",
    }
    assert _run_guidance(direction_trigger, thresholds=earlier_direction) == {
        "action": "left",
        "reason": "target_shifted_right",
    }

    reverse_direction_trigger = json.loads(
        (output_dir / "raw" / "reverse_direction_trigger_boundary.json").read_text(
            encoding="utf-8"
        )
    )
    assert _run_guidance(reverse_direction_trigger, thresholds=baseline) == {
        "action": "hold_still",
        "reason": "stabilize_before_capture",
    }
    assert _run_guidance(reverse_direction_trigger, thresholds=earlier_direction) == {
        "action": "right",
        "reason": "target_shifted_left",
    }

    oversize = json.loads(
        (output_dir / "raw" / "oversize_boundary.json").read_text(encoding="utf-8")
    )
    assert _run_guidance(oversize, thresholds=baseline) == {
        "action": "backward",
        "reason": "target_too_large",
    }
    assert _run_guidance(oversize, thresholds=relaxed_size) == {
        "action": "hold_still",
        "reason": "fully_centered",
    }


def _run_guidance(
    raw_payload: dict[str, object],
    *,
    thresholds: GuidanceThresholds,
) -> dict[str, str]:
    runtime_result = run_runtime_observation(
        payload=build_observation_payload_from_raw(raw_payload),
        guidance_thresholds=thresholds,
    )
    return {
        "action": str(runtime_result["guidance_action"]),
        "reason": str(runtime_result["reason"]),
    }
