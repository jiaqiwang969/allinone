from allinone.domain.research.entities import CandidateConfig, ExperimentRun
from allinone.domain.research.value_objects import ExperimentId, MetricName
from allinone.infrastructure.research.autoresearch.judge_adapter import (
    AutoresearchJudgeAdapter,
)
from allinone.infrastructure.research.autoresearch.replay_adapter import (
    AutoresearchReplayAdapter,
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


def test_judge_adapter_converts_score_row_to_domain_evaluation():
    evaluation = AutoresearchJudgeAdapter().to_candidate_evaluation(
        candidate_name="candidate-a",
        score=0.81,
        summary="better guidance alignment",
    )

    assert evaluation.candidate_name == "candidate-a"
    assert evaluation.score == 0.81
