from allinone.application.research.register_experiment import register_experiment
from allinone.domain.research.entities import ExperimentRun


def test_register_experiment_usecase_returns_registered_run():
    run = register_experiment(
        experiment_id="exp-001",
        hypothesis="test better guidance thresholds",
        target_metric="guidance_success_rate",
        candidate_names=["baseline", "candidate-a"],
    )

    assert isinstance(run, ExperimentRun)
    assert run.status == "registered"
    assert [candidate.name for candidate in run.candidate_configs] == [
        "baseline",
        "candidate-a",
    ]


def test_run_experiment_batch_returns_clip_aggregated_results():
    from allinone.application.research.run_experiment_batch import (
        run_experiment_batch,
    )

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
            "expected_action": "hold_still",
            "notes": "目标已居中",
        },
    ]
    analyzer_calls = []
    runtime_calls = []
    writer_calls = []

    def fake_clip_analyzer(*, clip_path, target_labels):
        analyzer_calls.append((clip_path, target_labels))
        if clip_path.endswith("clip-001.mp4"):
            return {
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
            }
        return {
            "detections": {
                "prediction_rows": [],
                "image_size": [1000, 1000],
                "target_labels": ["meter"],
                "best_frame_index": 0,
            },
            "vjepa": {
                "visibility_score": 0.72,
                "readable_ratio": 0.77,
                "stability_score": 0.88,
                "alignment_score": 0.9,
            },
        }

    def fake_runtime_runner(*, payload):
        runtime_calls.append(payload)
        if payload["prediction_rows"]:
            return {
                "guidance_action": "left",
                "reason": "target_shifted_right",
                "language_action": "left",
                "confidence": 0.82,
                "operator_message": "请向左移动",
                "evidence_focus": "表盘完整可见",
                "language_source": "fake-qwen",
            }
        return {
            "guidance_action": "hold_still",
            "reason": "fully_centered",
            "language_action": "hold_still",
            "confidence": 0.77,
            "operator_message": "保持当前角度",
            "evidence_focus": "继续稳定画面",
            "language_source": "fake-qwen",
        }

    class FakeRunWriter:
        def write(self, *, manifest_rows, result_rows, candidate_name):
            writer_calls.append((manifest_rows, result_rows, candidate_name))
            return {
                "run_dir": "experiments/runs/demo-run",
                "result_count": len(result_rows),
            }

    batch = run_experiment_batch(
        manifest_rows=manifest_rows,
        candidate_name="baseline",
        clip_analyzer=fake_clip_analyzer,
        runtime_runner=fake_runtime_runner,
        run_writer=FakeRunWriter(),
    )

    assert analyzer_calls == [
        ("/data/clip-001.mp4", ("meter",)),
        ("/data/clip-002.mp4", ("meter",)),
    ]
    assert runtime_calls == [
        {
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
        {
            "prediction_rows": [],
            "image_size": [1000, 1000],
            "target_labels": ["meter"],
            "visibility_score": 0.72,
            "readable_ratio": 0.77,
        },
    ]
    assert batch["results"] == [
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
            "expected_action": "hold_still",
            "guidance_action": "hold_still",
            "language_action": "hold_still",
            "action_match": True,
            "target_detected": False,
            "best_frame_index": 0,
            "visibility_score": 0.72,
            "readable_ratio": 0.77,
            "stability_score": 0.88,
            "alignment_score": 0.9,
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
                    "visibility_score": 0.72,
                    "readable_ratio": 0.77,
                    "stability_score": 0.88,
                    "alignment_score": 0.9,
                },
            },
            "payload": {
                "prediction_rows": [],
                "image_size": [1000, 1000],
                "target_labels": ["meter"],
                "visibility_score": 0.72,
                "readable_ratio": 0.77,
            },
        },
    ]
    assert batch["run_artifacts"] == {
        "run_dir": "experiments/runs/demo-run",
        "result_count": 2,
    }
    assert writer_calls == [
        (
            manifest_rows,
            batch["results"],
            "baseline",
        )
    ]
