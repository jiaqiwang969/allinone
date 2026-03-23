"""Adapter boundary for runtime replay integration."""

from __future__ import annotations

from allinone.domain.research.entities import ExperimentRun


class AutoresearchReplayAdapter:
    """Translate an experiment run into an autoresearch replay payload."""

    def build_payload(self, run: ExperimentRun) -> dict[str, object]:
        return {
            "experiment_id": run.experiment_id.value,
            "hypothesis": run.hypothesis,
            "target_metric": run.target_metric.value,
            "candidate_names": [candidate.name for candidate in run.candidate_configs],
        }
