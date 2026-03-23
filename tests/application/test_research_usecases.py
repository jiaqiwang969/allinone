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
