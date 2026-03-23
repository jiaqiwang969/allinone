"""Adapter boundary for runtime replay integration."""

from __future__ import annotations

import json
from pathlib import Path

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

    def build_run_payload(self, run_dir: str | Path) -> dict[str, object]:
        run_path = Path(run_dir)
        summary = json.loads((run_path / "summary.json").read_text(encoding="utf-8"))
        results_path = run_path / "results.jsonl"
        result_count = len(
            [
                line
                for line in results_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        )
        return {
            "run_dir": str(run_path),
            "candidate_name": summary["candidate_name"],
            "summary": summary,
            "result_count": result_count,
        }
