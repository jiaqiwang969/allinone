"""Research entities."""

from __future__ import annotations

from dataclasses import dataclass, field

from allinone.domain.research.errors import ExperimentStateError
from allinone.domain.research.events import (
    CandidateEvaluated,
    ExperimentCompleted,
    ExperimentRegistered,
)
from allinone.domain.research.value_objects import ExperimentId, MetricName


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    parameters: dict[str, object]

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("candidate name must not be empty")


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_name: str
    score: float
    summary: str

    def __post_init__(self) -> None:
        if not self.candidate_name or not self.candidate_name.strip():
            raise ValueError("candidate_name must not be empty")


@dataclass
class ExperimentRun:
    experiment_id: ExperimentId
    hypothesis: str
    target_metric: MetricName
    candidate_configs: list[CandidateConfig]
    status: str = "draft"
    evaluations: list[CandidateEvaluation] = field(default_factory=list)
    pending_events: list[object] = field(default_factory=list)

    @classmethod
    def register(
        cls,
        *,
        experiment_id: ExperimentId,
        hypothesis: str,
        target_metric: MetricName,
        candidate_configs: list[CandidateConfig],
    ) -> "ExperimentRun":
        if not hypothesis or not hypothesis.strip():
            raise ValueError("hypothesis must not be empty")
        if not candidate_configs:
            raise ValueError("candidate_configs must not be empty")
        run = cls(
            experiment_id=experiment_id,
            hypothesis=hypothesis.strip(),
            target_metric=target_metric,
            candidate_configs=candidate_configs,
            status="registered",
        )
        run.pending_events.append(
            ExperimentRegistered(
                experiment_id=run.experiment_id,
                target_metric=run.target_metric,
            )
        )
        return run

    def record_evaluation(self, evaluation: CandidateEvaluation) -> None:
        known_candidates = {candidate.name for candidate in self.candidate_configs}
        if evaluation.candidate_name not in known_candidates:
            raise ExperimentStateError("cannot evaluate unknown candidate")
        if any(
            existing.candidate_name == evaluation.candidate_name
            for existing in self.evaluations
        ):
            raise ExperimentStateError("candidate already evaluated")
        self.evaluations.append(evaluation)
        self.status = "running"
        self.pending_events.append(
            CandidateEvaluated(
                experiment_id=self.experiment_id,
                candidate_name=evaluation.candidate_name,
                score=evaluation.score,
            )
        )

    @property
    def best_evaluation(self) -> CandidateEvaluation | None:
        if not self.evaluations:
            return None
        return max(self.evaluations, key=lambda item: item.score)

    def complete(self) -> None:
        best = self.best_evaluation
        if best is None:
            raise ExperimentStateError("cannot complete experiment without evaluations")
        self.status = "completed"
        self.pending_events.append(
            ExperimentCompleted(
                experiment_id=self.experiment_id,
                best_candidate_name=best.candidate_name,
            )
        )
