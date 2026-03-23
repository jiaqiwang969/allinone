import json

from allinone.interfaces.cli.main import main


def test_guidance_smoke_command_prints_decision(capsys):
    exit_code = main(["guidance-smoke"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "guidance_action=left" in captured.out


def test_research_smoke_command_prints_payload(capsys):
    exit_code = main(["research-smoke"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "experiment_id=exp-smoke-001" in captured.out
    assert "candidate_names=baseline,candidate-a" in captured.out


def test_runtime_observation_command_prints_guidance_and_language(
    capsys, monkeypatch, tmp_path
):
    payload = tmp_path / "observation.json"
    payload.write_text(
        json.dumps(
            {
                "prediction_rows": [
                    {"label": "meter", "confidence": 0.91, "xyxy": [600, 200, 900, 800]}
                ],
                "image_size": [1000, 1000],
                "target_labels": ["meter"],
                "visibility_score": 0.85,
                "readable_ratio": 0.8,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ALLINONE_QWEN_RECIPE", str(tmp_path / "missing-qwen.yaml"))

    exit_code = main(["runtime-observation", "--input", str(payload)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "guidance_action=left" in captured.out
    assert "reason=target_shifted_right" in captured.out
    assert "language_action=left" in captured.out
    assert "source=mock" in captured.out


def test_build_observation_payload_command_writes_standard_payload(tmp_path):
    raw_input = tmp_path / "raw-perception.json"
    output = tmp_path / "payload.json"
    raw_input.write_text(
        json.dumps(
            {
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
                },
                "vjepa": {
                    "visibility_score": 0.85,
                    "readable_ratio": 0.8,
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "build-observation-payload",
            "--input",
            str(raw_input),
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload == {
        "prediction_rows": [
            {"label": "meter", "confidence": 0.91, "xyxy": [600, 200, 900, 800]}
        ],
        "image_size": [1000, 1000],
        "target_labels": ["meter"],
        "visibility_score": 0.85,
        "readable_ratio": 0.8,
    }


def test_detect_image_command_writes_raw_perception_payload(tmp_path, monkeypatch):
    image_path = tmp_path / "frame.png"
    output = tmp_path / "raw.json"
    image_path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(
        "allinone.interfaces.cli.main.build_raw_perception_payload_from_image",
        lambda **kwargs: {
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
            },
            "vjepa": {
                "visibility_score": 1.0,
                "readable_ratio": 1.0,
            },
        },
    )

    exit_code = main(
        [
            "detect-image",
            "--image",
            str(image_path),
            "--model",
            "mock-yolo.pt",
            "--targets",
            "meter",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["detections"]["target_labels"] == ["meter"]
    assert payload["vjepa"]["visibility_score"] == 1.0


def test_analyze_clip_command_writes_clip_raw_perception_payload(
    tmp_path, monkeypatch
):
    clip_path = tmp_path / "clip.mp4"
    output = tmp_path / "clip-raw.json"
    clip_path.write_text("placeholder", encoding="utf-8")

    def fake_build_raw_perception_payload_from_clip(**kwargs):
        assert kwargs["clip_path"] == str(clip_path)
        assert kwargs["target_labels"] == ("meter",)
        assert kwargs["sampler"].frame_count == 8
        assert kwargs["detector"].model_path == "mock-yolo.pt"
        assert kwargs["clip_scorer"].repo_path == "/models/vjepa2"
        assert kwargs["clip_scorer"].checkpoint_path == "/models/vjepa2/model.pt"
        return {
            "detections": {
                "prediction_rows": [
                    {
                        "label": "meter",
                        "confidence": 0.97,
                        "xyxy": [300, 90, 1000, 650],
                    }
                ],
                "image_size": [1280, 720],
                "target_labels": ["meter"],
                "best_frame_index": 2,
            },
            "vjepa": {
                "visibility_score": 0.88,
                "readable_ratio": 0.81,
                "stability_score": 0.92,
                "alignment_score": 0.87,
            },
        }

    monkeypatch.setattr(
        "allinone.interfaces.cli.main.build_raw_perception_payload_from_clip",
        fake_build_raw_perception_payload_from_clip,
    )

    exit_code = main(
        [
            "analyze-clip",
            "--clip",
            str(clip_path),
            "--yolo-model",
            "mock-yolo.pt",
            "--vjepa-repo",
            "/models/vjepa2",
            "--vjepa-checkpoint",
            "/models/vjepa2/model.pt",
            "--targets",
            "meter",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["detections"]["best_frame_index"] == 2
    assert payload["vjepa"]["alignment_score"] == 0.87


def test_run_experiment_command_writes_run_directory(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.jsonl"
    run_dir = tmp_path / "runs" / "demo-run"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "clip_id": "clip-001",
                        "clip_path": "/data/clip-001.mp4",
                        "target_labels": ["meter"],
                        "task_type": "view_guidance",
                        "expected_action": "left",
                        "notes": "目标偏右",
                    }
                ),
                json.dumps(
                    {
                        "clip_id": "clip-002",
                        "clip_path": "/data/clip-002.mp4",
                        "target_labels": ["meter"],
                        "task_type": "view_guidance",
                        "expected_action": "hold_still",
                        "notes": "目标已居中",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    def fake_build_raw_perception_payload_from_clip(**kwargs):
        if kwargs["clip_path"].endswith("clip-001.mp4"):
            return {
                "detections": {
                    "prediction_rows": [
                        {
                            "label": "meter",
                            "confidence": 0.97,
                            "xyxy": [300, 90, 1000, 650],
                        }
                    ],
                    "image_size": [1280, 720],
                    "target_labels": ["meter"],
                    "best_frame_index": 2,
                },
                "vjepa": {
                    "visibility_score": 0.88,
                    "readable_ratio": 0.81,
                    "stability_score": 0.92,
                    "alignment_score": 0.87,
                },
            }
        return {
            "detections": {
                "prediction_rows": [],
                "image_size": [1280, 720],
                "target_labels": ["meter"],
                "best_frame_index": 0,
            },
            "vjepa": {
                "visibility_score": 0.74,
                "readable_ratio": 0.79,
                "stability_score": 0.83,
                "alignment_score": 0.85,
            },
        }

    def fake_run_runtime_observation_usecase(*, payload):
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

    monkeypatch.setattr(
        "allinone.interfaces.cli.main.build_raw_perception_payload_from_clip",
        fake_build_raw_perception_payload_from_clip,
    )
    monkeypatch.setattr(
        "allinone.interfaces.cli.main.run_runtime_observation_usecase",
        fake_run_runtime_observation_usecase,
    )

    exit_code = main(
        [
            "run-experiment",
            "--manifest",
            str(manifest),
            "--run-dir",
            str(run_dir),
            "--candidate",
            "baseline",
            "--yolo-model",
            "mock-yolo.pt",
            "--vjepa-repo",
            "/models/vjepa2",
            "--vjepa-checkpoint",
            "/models/vjepa2/model.pt",
        ]
    )

    assert exit_code == 0
    assert (run_dir / "manifest.jsonl").exists()
    assert (run_dir / "results.jsonl").exists()
    assert (run_dir / "summary.json").exists()
    assert json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) == {
        "candidate_name": "baseline",
        "clip_count": 2,
        "action_match_rate": 1.0,
        "target_detected_rate": 0.5,
        "usable_clip_rate": 1.0,
    }


def test_judge_experiment_command_writes_judgement_json(
    tmp_path, monkeypatch, capsys
):
    output = tmp_path / "judgements" / "exp-judge-001.json"

    def fake_judge_experiment_candidates(**kwargs):
        assert kwargs["experiment_id"] == "exp-judge-001"
        assert kwargs["hypothesis"] == "compare candidate runs"
        assert kwargs["target_metric"] == "guidance_success_rate"
        assert kwargs["candidate_runs"] == [
            {
                "candidate_name": "baseline",
                "run_dir": "/tmp/runs/baseline",
            },
            {
                "candidate_name": "candidate-a",
                "run_dir": "/tmp/runs/candidate-a",
            },
        ]
        return {
            "experiment_id": "exp-judge-001",
            "target_metric": "guidance_success_rate",
            "status": "completed",
            "candidate_scores": [
                {
                    "candidate_name": "baseline",
                    "run_dir": "/tmp/runs/baseline",
                    "score": 0.52,
                    "summary": "baseline summary",
                    "metrics": {"action_match_rate": 0.6},
                },
                {
                    "candidate_name": "candidate-a",
                    "run_dir": "/tmp/runs/candidate-a",
                    "score": 0.81,
                    "summary": "candidate-a summary",
                    "metrics": {"action_match_rate": 0.8},
                },
            ],
            "best_candidate_name": "candidate-a",
        }

    monkeypatch.setattr(
        "allinone.interfaces.cli.main.judge_experiment_candidates",
        fake_judge_experiment_candidates,
    )

    exit_code = main(
        [
            "judge-experiment",
            "--experiment-id",
            "exp-judge-001",
            "--hypothesis",
            "compare candidate runs",
            "--target-metric",
            "guidance_success_rate",
            "--candidate-run",
            "baseline=/tmp/runs/baseline",
            "--candidate-run",
            "candidate-a=/tmp/runs/candidate-a",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "experiment_id": "exp-judge-001",
        "target_metric": "guidance_success_rate",
        "status": "completed",
        "candidate_scores": [
            {
                "candidate_name": "baseline",
                "run_dir": "/tmp/runs/baseline",
                "score": 0.52,
                "summary": "baseline summary",
                "metrics": {"action_match_rate": 0.6},
            },
            {
                "candidate_name": "candidate-a",
                "run_dir": "/tmp/runs/candidate-a",
                "score": 0.81,
                "summary": "candidate-a summary",
                "metrics": {"action_match_rate": 0.8},
            },
        ],
        "best_candidate_name": "candidate-a",
    }
    assert "best_candidate_name=candidate-a" in captured.out


def test_run_research_step_command_writes_summary_json(
    tmp_path, monkeypatch, capsys
):
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "research" / "summary.json"
    manifest.write_text(
        json.dumps(
            {
                "clip_id": "clip-001",
                "clip_path": "/data/clip-001.mp4",
                "target_labels": ["meter"],
                "task_type": "view_guidance",
                "expected_action": "left",
            }
        ),
        encoding="utf-8",
    )

    def fake_run_research_step(**kwargs):
        assert kwargs["experiment_id"] == "exp-loop-001"
        assert kwargs["hypothesis"] == "tighten guidance thresholds"
        assert kwargs["target_metric"] == "guidance_success_rate"
        assert kwargs["candidate_count"] == 2
        assert kwargs["base_policy_path"] == "configs/runtime_policies/m400_default.json"
        assert kwargs["run_root"] == "experiments/research/exp-loop-001"
        assert kwargs["manifest_rows"] == [
            {
                "clip_id": "clip-001",
                "clip_path": "/data/clip-001.mp4",
                "target_labels": ["meter"],
                "task_type": "view_guidance",
                "expected_action": "left",
            }
        ]
        return {
            "experiment_id": "exp-loop-001",
            "target_metric": "guidance_success_rate",
            "status": "completed",
            "candidate_count": 2,
            "candidate_policies": [
                {
                    "candidate_name": "baseline",
                    "mutation": "baseline",
                    "policy_path": "experiments/research/exp-loop-001/candidate_policies/baseline.json",
                    "run_dir": "experiments/research/exp-loop-001/runs/baseline",
                },
                {
                    "candidate_name": "candidate-1",
                    "mutation": "tighten_center_window",
                    "policy_path": "experiments/research/exp-loop-001/candidate_policies/candidate-1.json",
                    "run_dir": "experiments/research/exp-loop-001/runs/candidate-1",
                },
            ],
            "best_candidate_name": "candidate-1",
            "best_policy_path": "experiments/research/exp-loop-001/candidate_policies/candidate-1.json",
            "judgement": {
                "experiment_id": "exp-loop-001",
                "target_metric": "guidance_success_rate",
                "status": "completed",
                "candidate_scores": [],
                "best_candidate_name": "candidate-1",
            },
        }

    monkeypatch.setattr(
        "allinone.interfaces.cli.main.run_research_step",
        fake_run_research_step,
    )

    exit_code = main(
        [
            "run-research-step",
            "--experiment-id",
            "exp-loop-001",
            "--hypothesis",
            "tighten guidance thresholds",
            "--target-metric",
            "guidance_success_rate",
            "--manifest",
            str(manifest),
            "--base-policy",
            "configs/runtime_policies/m400_default.json",
            "--candidate-count",
            "2",
            "--run-root",
            "experiments/research/exp-loop-001",
            "--output",
            str(output),
            "--yolo-model",
            "mock-yolo.pt",
            "--vjepa-repo",
            "/models/vjepa2",
            "--vjepa-checkpoint",
            "/models/vjepa2/model.pt",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "experiment_id": "exp-loop-001",
        "target_metric": "guidance_success_rate",
        "status": "completed",
        "candidate_count": 2,
        "candidate_policies": [
            {
                "candidate_name": "baseline",
                "mutation": "baseline",
                "policy_path": "experiments/research/exp-loop-001/candidate_policies/baseline.json",
                "run_dir": "experiments/research/exp-loop-001/runs/baseline",
            },
            {
                "candidate_name": "candidate-1",
                "mutation": "tighten_center_window",
                "policy_path": "experiments/research/exp-loop-001/candidate_policies/candidate-1.json",
                "run_dir": "experiments/research/exp-loop-001/runs/candidate-1",
            },
        ],
        "best_candidate_name": "candidate-1",
        "best_policy_path": "experiments/research/exp-loop-001/candidate_policies/candidate-1.json",
        "judgement": {
            "experiment_id": "exp-loop-001",
            "target_metric": "guidance_success_rate",
            "status": "completed",
            "candidate_scores": [],
            "best_candidate_name": "candidate-1",
        },
    }
    assert "best_candidate_name=candidate-1" in captured.out
