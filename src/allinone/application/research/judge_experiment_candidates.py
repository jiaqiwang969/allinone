"""Judge multiple candidate runs and select the best one."""

from __future__ import annotations

from typing import Protocol

from allinone.domain.research.entities import CandidateConfig
from allinone.domain.research.services import ExperimentSelectionService
from allinone.domain.research.value_objects import ExperimentId, MetricName
from allinone.domain.research.entities import ExperimentRun


class ReplayAdapter(Protocol):
    def build_run_payload(self, run_dir: str) -> dict[str, object]: ...


class CandidateJudge(Protocol):
    def score_candidate(self, run_payload: dict[str, object]) -> dict[str, object]: ...


class JudgeAdapter(Protocol):
    def to_candidate_evaluation(
        self,
        *,
        candidate_name: str,
        score: float,
        summary: str,
    ) -> object: ...


def judge_experiment_candidates(
    *,
    experiment_id: str,
    hypothesis: str,
    target_metric: str,
    candidate_runs: list[dict[str, str]],
    replay_adapter: ReplayAdapter,
    candidate_judge: CandidateJudge,
    judge_adapter: JudgeAdapter,
    selection_service: ExperimentSelectionService,
) -> dict[str, object]:
    run = ExperimentRun.register(
        experiment_id=ExperimentId(experiment_id),
        hypothesis=hypothesis,
        target_metric=MetricName(target_metric),
        candidate_configs=[
            CandidateConfig(
                name=candidate_run["candidate_name"],
                parameters={"run_dir": candidate_run["run_dir"]},
            )
            for candidate_run in candidate_runs
        ],
    )

    candidate_scores: list[dict[str, object]] = []
    for candidate_run in candidate_runs:
        run_payload = replay_adapter.build_run_payload(candidate_run["run_dir"])
        judgement = candidate_judge.score_candidate(run_payload)
        evaluation = judge_adapter.to_candidate_evaluation(
            candidate_name=candidate_run["candidate_name"],
            score=float(judgement["score"]),
            summary=str(judgement["summary"]),
        )
        run.record_evaluation(evaluation)
        candidate_scores.append(
            {
                "candidate_name": candidate_run["candidate_name"],
                "run_dir": str(judgement["run_dir"]),
                "score": float(judgement["score"]),
                "summary": str(judgement["summary"]),
                "metrics": dict(judgement.get("metrics", {})),
            }
        )

    best = selection_service.select_best(run)
    run.complete()
    return {
        "experiment_id": run.experiment_id.value,
        "target_metric": run.target_metric.value,
        "status": run.status,
        "candidate_scores": candidate_scores,
        "best_candidate_name": best.candidate_name,
    }
