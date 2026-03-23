"""Research events."""

from __future__ import annotations

from dataclasses import dataclass

from allinone.domain.research.value_objects import ExperimentId, MetricName


@dataclass(frozen=True)
class ExperimentRegistered:
    experiment_id: ExperimentId
    target_metric: MetricName


@dataclass(frozen=True)
class CandidateEvaluated:
    experiment_id: ExperimentId
    candidate_name: str
    score: float


@dataclass(frozen=True)
class ExperimentCompleted:
    experiment_id: ExperimentId
    best_candidate_name: str
