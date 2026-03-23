import json
from pathlib import Path

from allinone.domain.research.entities import CandidateConfig, ExperimentRun
from allinone.domain.research.value_objects import ExperimentId, MetricName
from allinone.infrastructure.research.autoresearch.judge_adapter import (
    AutoresearchJudgeAdapter,
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
