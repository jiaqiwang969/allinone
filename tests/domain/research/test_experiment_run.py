import pytest

from allinone.domain.research.entities import CandidateConfig, CandidateEvaluation, ExperimentRun
from allinone.domain.research.errors import ExperimentStateError
from allinone.domain.research.events import CandidateEvaluated, ExperimentCompleted, ExperimentRegistered
from allinone.domain.research.services import ExperimentSelectionService
from allinone.domain.research.value_objects import ExperimentId, MetricName


def test_register_experiment_requires_candidate_configs():
    with pytest.raises(ValueError):
        ExperimentRun.register(
            experiment_id=ExperimentId("exp-001"),
            hypothesis="test better guidance thresholds",
            target_metric=MetricName("guidance_success_rate"),
            candidate_configs=[],
        )


def test_register_experiment_emits_registered_event():
    run = ExperimentRun.register(
        experiment_id=ExperimentId("exp-001"),
        hypothesis="test better guidance thresholds",
        target_metric=MetricName("guidance_success_rate"),
        candidate_configs=[
            CandidateConfig(name="baseline", parameters={"policy": "v1"}),
            CandidateConfig(name="candidate-a", parameters={"policy": "v2"}),
        ],
    )

    assert run.status == "registered"
    assert isinstance(run.pending_events[-1], ExperimentRegistered)


def test_record_evaluation_and_select_best_candidate():
    run = ExperimentRun.register(
        experiment_id=ExperimentId("exp-001"),
        hypothesis="test better guidance thresholds",
        target_metric=MetricName("guidance_success_rate"),
        candidate_configs=[
            CandidateConfig(name="baseline", parameters={"policy": "v1"}),
            CandidateConfig(name="candidate-a", parameters={"policy": "v2"}),
        ],
    )

    run.record_evaluation(
        CandidateEvaluation(candidate_name="baseline", score=0.72, summary="stable")
    )
    run.record_evaluation(
        CandidateEvaluation(candidate_name="candidate-a", score=0.81, summary="better")
    )
    run.complete()

    assert run.status == "completed"
    assert isinstance(run.pending_events[-2], CandidateEvaluated)
    assert isinstance(run.pending_events[-1], ExperimentCompleted)
    assert ExperimentSelectionService().select_best(run).candidate_name == "candidate-a"


def test_record_evaluation_rejects_unknown_candidate():
    run = ExperimentRun.register(
        experiment_id=ExperimentId("exp-001"),
        hypothesis="test better guidance thresholds",
        target_metric=MetricName("guidance_success_rate"),
        candidate_configs=[CandidateConfig(name="baseline", parameters={"policy": "v1"})],
    )

    with pytest.raises(ExperimentStateError):
        run.record_evaluation(
            CandidateEvaluation(candidate_name="unknown", score=0.5, summary="bad")
        )
