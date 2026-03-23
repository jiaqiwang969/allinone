"""Run one autoresearch step over runtime policy candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from allinone.domain.guidance.services import GuidanceThresholds


class PolicyStore(Protocol):
    def load_guidance_thresholds(
        self,
        recipe_path: str,
    ) -> GuidanceThresholds: ...

    def write_guidance_thresholds(
        self,
        *,
        recipe_path: str | Path,
        thresholds: GuidanceThresholds,
    ) -> Path | str: ...


class CandidateProposer(Protocol):
    def propose_candidates(
        self,
        *,
        base_thresholds: dict[str, float],
        candidate_count: int,
    ) -> list[dict[str, object]]: ...


class CandidateRunner(Protocol):
    def run(
        self,
        *,
        manifest_rows: list[dict[str, object]],
        candidate_name: str,
        run_dir: str,
        guidance_thresholds: GuidanceThresholds,
        policy_path: str,
    ) -> dict[str, object]: ...


class JudgeUseCase(Protocol):
    def __call__(
        self,
        *,
        experiment_id: str,
        hypothesis: str,
        target_metric: str,
        candidate_runs: list[dict[str, str]],
    ) -> dict[str, object]: ...


def run_research_step(
    *,
    experiment_id: str,
    hypothesis: str,
    target_metric: str,
    manifest_rows: list[dict[str, object]],
    base_policy_path: str,
    candidate_count: int,
    run_root: str,
    policy_store: PolicyStore,
    candidate_proposer: CandidateProposer,
    candidate_runner: CandidateRunner,
    judge_usecase: JudgeUseCase,
) -> dict[str, object]:
    base_thresholds = policy_store.load_guidance_thresholds(base_policy_path)
    candidate_rows = candidate_proposer.propose_candidates(
        base_thresholds={
            "centered_offset_max": base_thresholds.centered_offset_max,
            "directional_offset_min": base_thresholds.directional_offset_min,
            "ready_fill_ratio_max": base_thresholds.ready_fill_ratio_max,
        },
        candidate_count=candidate_count,
    )

    candidate_policies: list[dict[str, str]] = []
    candidate_runs: list[dict[str, str]] = []
    for candidate_row in candidate_rows:
        candidate_name = str(candidate_row["candidate_name"])
        thresholds = GuidanceThresholds(**dict(candidate_row["guidance_thresholds"]))
        policy_path = str(
            policy_store.write_guidance_thresholds(
                recipe_path=Path(run_root) / "candidate_policies" / f"{candidate_name}.json",
                thresholds=thresholds,
            )
        )
        run_dir = str(Path(run_root) / "runs" / candidate_name)
        candidate_runner.run(
            manifest_rows=manifest_rows,
            candidate_name=candidate_name,
            run_dir=run_dir,
            guidance_thresholds=thresholds,
            policy_path=policy_path,
        )
        candidate_policies.append(
            {
                "candidate_name": candidate_name,
                "mutation": str(candidate_row["mutation"]),
                "policy_path": policy_path,
                "run_dir": run_dir,
            }
        )
        candidate_runs.append(
            {
                "candidate_name": candidate_name,
                "run_dir": run_dir,
            }
        )

    judgement = judge_usecase(
        experiment_id=experiment_id,
        hypothesis=hypothesis,
        target_metric=target_metric,
        candidate_runs=candidate_runs,
    )
    best_candidate_name = str(judgement["best_candidate_name"])
    best_policy_path = next(
        candidate["policy_path"]
        for candidate in candidate_policies
        if candidate["candidate_name"] == best_candidate_name
    )
    return {
        "experiment_id": experiment_id,
        "target_metric": target_metric,
        "status": str(judgement["status"]),
        "candidate_count": len(candidate_policies),
        "candidate_policies": candidate_policies,
        "best_candidate_name": best_candidate_name,
        "best_policy_path": best_policy_path,
        "judgement": judgement,
    }
