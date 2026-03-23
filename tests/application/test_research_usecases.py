import json

from allinone.application.research.register_experiment import register_experiment
from allinone.domain.research.entities import CandidateEvaluation, ExperimentRun
from allinone.domain.research.services import ExperimentSelectionService
from allinone.domain.guidance.services import GuidanceThresholds


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
            "expected_action": "hold_still",
            "guidance_action": "hold_still",
            "guidance_reason": "fully_centered",
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


def test_run_experiment_batch_can_replay_rows_from_raw_payload_path(tmp_path):
    from allinone.application.research.run_experiment_batch import (
        run_experiment_batch,
    )

    raw_payload_path = tmp_path / "raw" / "clip-raw.json"
    raw_payload_path.parent.mkdir(parents=True)
    raw_payload_path.write_text(
        json.dumps(
            {
                "detections": {
                    "prediction_rows": [
                        {
                            "label": "meter",
                            "confidence": 0.96,
                            "xyxy": [560, 180, 920, 820],
                        }
                    ],
                    "image_size": [1000, 1000],
                    "target_labels": ["meter"],
                    "best_frame_index": 1,
                },
                "vjepa": {
                    "visibility_score": 0.82,
                    "readable_ratio": 0.79,
                    "stability_score": 0.9,
                    "alignment_score": 0.86,
                },
            }
        ),
        encoding="utf-8",
    )

    analyzer_calls = []
    runtime_calls = []
    writer_calls = []

    def fake_clip_analyzer(*, clip_path, target_labels):
        analyzer_calls.append((clip_path, target_labels))
        raise AssertionError("clip_analyzer should not be called for raw payload replay")

    class FakeRawPayloadLoader:
        def load(self, path):
            assert path == str(raw_payload_path)
            return json.loads(raw_payload_path.read_text(encoding="utf-8"))

    def fake_runtime_runner(*, payload):
        runtime_calls.append(payload)
        return {
            "guidance_action": "hold_still",
            "reason": "fully_centered",
            "language_action": "hold_still",
            "confidence": 0.88,
            "operator_message": "保持当前角度",
            "evidence_focus": "继续稳定画面",
            "language_source": "fake-qwen",
        }

    class FakeRunWriter:
        def write(self, *, manifest_rows, result_rows, candidate_name):
            writer_calls.append((manifest_rows, result_rows, candidate_name))
            return {
                "run_dir": "experiments/runs/raw-replay-demo",
                "result_count": len(result_rows),
            }

    batch = run_experiment_batch(
        manifest_rows=[
            {
                "clip_id": "clip-raw-001",
                "raw_payload_path": str(raw_payload_path),
                "target_labels": ["meter"],
                "task_type": "view_guidance",
                "expected_action": "hold_still",
                "notes": "冻结 raw payload replay",
            }
        ],
        candidate_name="baseline",
        clip_analyzer=fake_clip_analyzer,
        raw_payload_loader=FakeRawPayloadLoader(),
        runtime_runner=fake_runtime_runner,
        run_writer=FakeRunWriter(),
    )

    assert analyzer_calls == []
    assert runtime_calls == [
        {
            "prediction_rows": [
                {
                    "label": "meter",
                    "confidence": 0.96,
                    "xyxy": [560, 180, 920, 820],
                }
            ],
            "image_size": [1000, 1000],
            "target_labels": ["meter"],
            "visibility_score": 0.82,
            "readable_ratio": 0.79,
        }
    ]
    assert batch["results"] == [
        {
            "clip_id": "clip-raw-001",
            "candidate_name": "baseline",
            "task_type": "view_guidance",
            "target_labels": ["meter"],
            "expected_action": "hold_still",
            "guidance_action": "hold_still",
            "guidance_reason": "fully_centered",
            "language_action": "hold_still",
            "action_match": True,
            "target_detected": True,
            "best_frame_index": 1,
            "visibility_score": 0.82,
            "readable_ratio": 0.79,
            "stability_score": 0.9,
            "alignment_score": 0.86,
            "operator_message": "保持当前角度",
            "evidence_focus": "继续稳定画面",
            "language_source": "fake-qwen",
            "error": None,
            "raw_payload": {
                "detections": {
                    "prediction_rows": [
                        {
                            "label": "meter",
                            "confidence": 0.96,
                            "xyxy": [560, 180, 920, 820],
                        }
                    ],
                    "image_size": [1000, 1000],
                    "target_labels": ["meter"],
                    "best_frame_index": 1,
                },
                "vjepa": {
                    "visibility_score": 0.82,
                    "readable_ratio": 0.79,
                    "stability_score": 0.9,
                    "alignment_score": 0.86,
                },
            },
            "payload": {
                "prediction_rows": [
                    {
                        "label": "meter",
                        "confidence": 0.96,
                        "xyxy": [560, 180, 920, 820],
                    }
                ],
                "image_size": [1000, 1000],
                "target_labels": ["meter"],
                "visibility_score": 0.82,
                "readable_ratio": 0.79,
            },
        }
    ]
    assert writer_calls == [
        (
            [
                {
                    "clip_id": "clip-raw-001",
                    "raw_payload_path": str(raw_payload_path),
                    "target_labels": ["meter"],
                    "task_type": "view_guidance",
                    "expected_action": "hold_still",
                    "notes": "冻结 raw payload replay",
                }
            ],
            batch["results"],
            "baseline",
        )
    ]


def test_run_experiment_batch_records_reason_match_when_manifest_provides_expected_reason():
    from allinone.application.research.run_experiment_batch import (
        run_experiment_batch,
    )

    def fake_clip_analyzer(*, clip_path, target_labels):
        assert clip_path == "/data/clip-003.mp4"
        assert target_labels == ("meter",)
        return {
            "detections": {
                "prediction_rows": [
                    {
                        "label": "meter",
                        "confidence": 0.93,
                        "xyxy": [580, 210, 910, 810],
                    }
                ],
                "image_size": [1000, 1000],
                "target_labels": ["meter"],
                "best_frame_index": 3,
            },
            "vjepa": {
                "visibility_score": 0.87,
                "readable_ratio": 0.81,
                "stability_score": 0.92,
                "alignment_score": 0.88,
            },
        }

    def fake_runtime_runner(*, payload):
        assert payload == {
            "prediction_rows": [
                {
                    "label": "meter",
                    "confidence": 0.93,
                    "xyxy": [580, 210, 910, 810],
                }
            ],
            "image_size": [1000, 1000],
            "target_labels": ["meter"],
            "visibility_score": 0.87,
            "readable_ratio": 0.81,
        }
        return {
            "guidance_action": "hold_still",
            "reason": "stabilize_before_capture",
            "language_action": "hold_still",
            "confidence": 0.9,
            "operator_message": "保持稳定",
            "evidence_focus": "等待清晰抓拍",
            "language_source": "fake-qwen",
        }

    class FakeRunWriter:
        def write(self, *, manifest_rows, result_rows, candidate_name):
            return {
                "run_dir": "experiments/runs/reason-match-demo",
                "result_count": len(result_rows),
            }

    batch = run_experiment_batch(
        manifest_rows=[
            {
                "clip_id": "clip-003",
                "clip_path": "/data/clip-003.mp4",
                "target_labels": ["meter"],
                "task_type": "view_guidance",
                "expected_action": "hold_still",
                "expected_reason": "stabilize_before_capture",
                "notes": "中心区域已接近阈值",
            }
        ],
        candidate_name="candidate-1",
        clip_analyzer=fake_clip_analyzer,
        runtime_runner=fake_runtime_runner,
        run_writer=FakeRunWriter(),
    )

    assert batch["results"] == [
        {
            "clip_id": "clip-003",
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
            "best_frame_index": 3,
            "visibility_score": 0.87,
            "readable_ratio": 0.81,
            "stability_score": 0.92,
            "alignment_score": 0.88,
            "operator_message": "保持稳定",
            "evidence_focus": "等待清晰抓拍",
            "language_source": "fake-qwen",
            "error": None,
            "raw_payload": {
                "detections": {
                    "prediction_rows": [
                        {
                            "label": "meter",
                            "confidence": 0.93,
                            "xyxy": [580, 210, 910, 810],
                        }
                    ],
                    "image_size": [1000, 1000],
                    "target_labels": ["meter"],
                    "best_frame_index": 3,
                },
                "vjepa": {
                    "visibility_score": 0.87,
                    "readable_ratio": 0.81,
                    "stability_score": 0.92,
                    "alignment_score": 0.88,
                },
            },
            "payload": {
                "prediction_rows": [
                    {
                        "label": "meter",
                        "confidence": 0.93,
                        "xyxy": [580, 210, 910, 810],
                    }
                ],
                "image_size": [1000, 1000],
                "target_labels": ["meter"],
                "visibility_score": 0.87,
                "readable_ratio": 0.81,
            },
        }
    ]


def test_judge_experiment_candidates_returns_best_candidate_summary():
    from allinone.application.research.judge_experiment_candidates import (
        judge_experiment_candidates,
    )

    replay_calls = []
    judge_calls = []
    adapter_calls = []

    class FakeReplayAdapter:
        def build_run_payload(self, run_dir):
            replay_calls.append(run_dir)
            candidate_name = "baseline" if run_dir.endswith("baseline") else "candidate-a"
            return {
                "run_dir": run_dir,
                "candidate_name": candidate_name,
                "summary": {
                    "candidate_name": candidate_name,
                    "action_match_rate": 0.6 if candidate_name == "baseline" else 0.8,
                    "target_detected_rate": 0.5 if candidate_name == "baseline" else 0.9,
                    "usable_clip_rate": 0.4 if candidate_name == "baseline" else 0.7,
                },
                "results_path": f"{run_dir}/results.jsonl",
                "result_count": 4,
            }

    class FakeRuleJudge:
        def score_candidate(self, run_payload):
            judge_calls.append(run_payload["candidate_name"])
            if run_payload["candidate_name"] == "baseline":
                return {
                    "candidate_name": "baseline",
                    "run_dir": run_payload["run_dir"],
                    "score": 0.52,
                    "summary": "baseline summary",
                    "metrics": {"action_match_rate": 0.6},
                }
            return {
                "candidate_name": "candidate-a",
                "run_dir": run_payload["run_dir"],
                "score": 0.81,
                "summary": "candidate-a summary",
                "metrics": {"action_match_rate": 0.8},
            }

    class FakeJudgeAdapter:
        def to_candidate_evaluation(self, *, candidate_name, score, summary):
            adapter_calls.append((candidate_name, score, summary))
            return CandidateEvaluation(
                candidate_name=candidate_name,
                score=score,
                summary=summary,
            )

    judgement = judge_experiment_candidates(
        experiment_id="exp-judge-001",
        hypothesis="candidate-a should improve guidance stability",
        target_metric="guidance_success_rate",
        candidate_runs=[
            {
                "candidate_name": "baseline",
                "run_dir": "experiments/runs/baseline",
            },
            {
                "candidate_name": "candidate-a",
                "run_dir": "experiments/runs/candidate-a",
            },
        ],
        replay_adapter=FakeReplayAdapter(),
        candidate_judge=FakeRuleJudge(),
        judge_adapter=FakeJudgeAdapter(),
        selection_service=ExperimentSelectionService(),
    )

    assert replay_calls == [
        "experiments/runs/baseline",
        "experiments/runs/candidate-a",
    ]
    assert judge_calls == ["baseline", "candidate-a"]
    assert adapter_calls == [
        ("baseline", 0.52, "baseline summary"),
        ("candidate-a", 0.81, "candidate-a summary"),
    ]
    assert judgement == {
        "experiment_id": "exp-judge-001",
        "target_metric": "guidance_success_rate",
        "status": "completed",
        "candidate_scores": [
            {
                "candidate_name": "baseline",
                "run_dir": "experiments/runs/baseline",
                "score": 0.52,
                "summary": "baseline summary",
                "metrics": {"action_match_rate": 0.6},
            },
            {
                "candidate_name": "candidate-a",
                "run_dir": "experiments/runs/candidate-a",
                "score": 0.81,
                "summary": "candidate-a summary",
                "metrics": {"action_match_rate": 0.8},
            },
        ],
        "best_candidate_name": "candidate-a",
    }


def test_run_research_step_materializes_candidates_runs_and_selects_best():
    from allinone.application.research.run_research_step import run_research_step

    write_calls = []
    runner_calls = []
    judge_calls = []

    class FakePolicyStore:
        def load_guidance_thresholds(self, recipe_path):
            assert recipe_path == "configs/runtime_policies/m400_default.json"
            return GuidanceThresholds(
                centered_offset_max=0.09,
                directional_offset_min=0.18,
                ready_fill_ratio_max=0.85,
            )

        def write_guidance_thresholds(self, *, recipe_path, thresholds):
            write_calls.append((str(recipe_path), thresholds))
            return recipe_path

    class FakeCandidateProposer:
        def propose_candidates(self, *, base_thresholds, candidate_count):
            assert base_thresholds == {
                "centered_offset_max": 0.09,
                "directional_offset_min": 0.18,
                "ready_fill_ratio_max": 0.85,
            }
            assert candidate_count == 2
            return [
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
            ]

    class FakeCandidateRunner:
        def run(
            self,
            *,
            manifest_rows,
            candidate_name,
            run_dir,
            guidance_thresholds,
            policy_path,
        ):
            runner_calls.append(
                (
                    manifest_rows,
                    candidate_name,
                    run_dir,
                    guidance_thresholds,
                    policy_path,
                )
            )
            return {
                "candidate_name": candidate_name,
                "run_dir": run_dir,
            }

    class FakeJudgeUseCase:
        def __call__(self, **kwargs):
            judge_calls.append(kwargs)
            return {
                "experiment_id": "exp-loop-001",
                "target_metric": "guidance_success_rate",
                "status": "completed",
                "candidate_scores": [
                    {
                        "candidate_name": "baseline",
                        "run_dir": "experiments/research/exp-loop-001/runs/baseline",
                        "score": 0.52,
                        "summary": "baseline summary",
                        "metrics": {"action_match_rate": 0.6},
                    },
                    {
                        "candidate_name": "candidate-1",
                        "run_dir": "experiments/research/exp-loop-001/runs/candidate-1",
                        "score": 0.81,
                        "summary": "candidate summary",
                        "metrics": {"action_match_rate": 0.8},
                    },
                ],
                "best_candidate_name": "candidate-1",
            }

    result = run_research_step(
        experiment_id="exp-loop-001",
        hypothesis="tighten guidance thresholds",
        target_metric="guidance_success_rate",
        manifest_rows=[
            {
                "clip_id": "clip-001",
                "clip_path": "/data/clip-001.mp4",
                "target_labels": ["meter"],
                "task_type": "view_guidance",
                "expected_action": "left",
            }
        ],
        base_policy_path="configs/runtime_policies/m400_default.json",
        candidate_count=2,
        run_root="experiments/research/exp-loop-001",
        policy_store=FakePolicyStore(),
        candidate_proposer=FakeCandidateProposer(),
        candidate_runner=FakeCandidateRunner(),
        judge_usecase=FakeJudgeUseCase(),
    )

    assert write_calls == [
        (
            "experiments/research/exp-loop-001/candidate_policies/baseline.json",
            GuidanceThresholds(
                centered_offset_max=0.09,
                directional_offset_min=0.18,
                ready_fill_ratio_max=0.85,
            ),
        ),
        (
            "experiments/research/exp-loop-001/candidate_policies/candidate-1.json",
            GuidanceThresholds(
                centered_offset_max=0.072,
                directional_offset_min=0.18,
                ready_fill_ratio_max=0.85,
            ),
        ),
    ]
    assert runner_calls == [
        (
            [
                {
                    "clip_id": "clip-001",
                    "clip_path": "/data/clip-001.mp4",
                    "target_labels": ["meter"],
                    "task_type": "view_guidance",
                    "expected_action": "left",
                }
            ],
            "baseline",
            "experiments/research/exp-loop-001/runs/baseline",
            GuidanceThresholds(
                centered_offset_max=0.09,
                directional_offset_min=0.18,
                ready_fill_ratio_max=0.85,
            ),
            "experiments/research/exp-loop-001/candidate_policies/baseline.json",
        ),
        (
            [
                {
                    "clip_id": "clip-001",
                    "clip_path": "/data/clip-001.mp4",
                    "target_labels": ["meter"],
                    "task_type": "view_guidance",
                    "expected_action": "left",
                }
            ],
            "candidate-1",
            "experiments/research/exp-loop-001/runs/candidate-1",
            GuidanceThresholds(
                centered_offset_max=0.072,
                directional_offset_min=0.18,
                ready_fill_ratio_max=0.85,
            ),
            "experiments/research/exp-loop-001/candidate_policies/candidate-1.json",
        ),
    ]
    assert judge_calls == [
        {
            "experiment_id": "exp-loop-001",
            "hypothesis": "tighten guidance thresholds",
            "target_metric": "guidance_success_rate",
            "candidate_runs": [
                {
                    "candidate_name": "baseline",
                    "run_dir": "experiments/research/exp-loop-001/runs/baseline",
                },
                {
                    "candidate_name": "candidate-1",
                    "run_dir": "experiments/research/exp-loop-001/runs/candidate-1",
                },
            ],
        }
    ]
    assert result == {
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
            "candidate_scores": [
                {
                    "candidate_name": "baseline",
                    "run_dir": "experiments/research/exp-loop-001/runs/baseline",
                    "score": 0.52,
                    "summary": "baseline summary",
                    "metrics": {"action_match_rate": 0.6},
                },
                {
                    "candidate_name": "candidate-1",
                    "run_dir": "experiments/research/exp-loop-001/runs/candidate-1",
                    "score": 0.81,
                    "summary": "candidate summary",
                    "metrics": {"action_match_rate": 0.8},
                },
            ],
            "best_candidate_name": "candidate-1",
        },
    }
