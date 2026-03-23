"""Register an experiment recipe into the research control plane."""

from __future__ import annotations

from allinone.domain.research.entities import CandidateConfig, ExperimentRun
from allinone.domain.research.value_objects import ExperimentId, MetricName


def register_experiment(
    *,
    experiment_id: str,
    hypothesis: str,
    target_metric: str,
    candidate_names: list[str],
) -> ExperimentRun:
    """Register a research experiment around comparable candidate policies."""
    return ExperimentRun.register(
        experiment_id=ExperimentId(experiment_id),
        hypothesis=hypothesis,
        target_metric=MetricName(target_metric),
        candidate_configs=[
            CandidateConfig(name=name, parameters={}) for name in candidate_names
        ],
    )
