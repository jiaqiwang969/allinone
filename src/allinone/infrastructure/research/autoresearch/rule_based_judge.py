"""Rule-based judge for offline candidate run comparison."""

from __future__ import annotations

import json
from pathlib import Path


class RuleBasedAutoresearchJudge:
    """Score a candidate run from persisted summary and result artifacts."""

    ACTION_MATCH_WEIGHT = 0.5
    TARGET_DETECTED_WEIGHT = 0.3
    USABLE_CLIP_WEIGHT = 0.2
    ACTION_MATCH_WITH_REASON_WEIGHT = 0.4
    TARGET_DETECTED_WITH_REASON_WEIGHT = 0.25
    USABLE_CLIP_WITH_REASON_WEIGHT = 0.15
    REASON_MATCH_WEIGHT = 0.2
    ERROR_PENALTY_WEIGHT = 0.1
    MISSING_TARGET_PENALTY_WEIGHT = 0.05

    def score_candidate(self, run_payload: dict[str, object]) -> dict[str, object]:
        summary = dict(run_payload["summary"])
        result_rows = self._load_results(Path(str(run_payload["results_path"])))
        result_count = int(run_payload["result_count"]) or len(result_rows)
        if result_count <= 0:
            result_count = len(result_rows) or 1

        error_rate = self._compute_error_rate(result_rows, result_count)
        target_not_detected_ratio = self._compute_missing_target_ratio(
            result_rows,
            result_count,
        )
        score = max(
            0.0,
            self._compute_main_score(summary)
            - (self.ERROR_PENALTY_WEIGHT * error_rate)
            - (self.MISSING_TARGET_PENALTY_WEIGHT * target_not_detected_ratio),
        )
        metrics = {
            "action_match_rate": float(summary.get("action_match_rate", 0.0)),
            "target_detected_rate": float(summary.get("target_detected_rate", 0.0)),
            "usable_clip_rate": float(summary.get("usable_clip_rate", 0.0)),
            "error_rate": error_rate,
            "target_not_detected_ratio": target_not_detected_ratio,
            "result_count": result_count,
        }
        if "reason_match_rate" in summary:
            metrics["reason_match_rate"] = float(summary["reason_match_rate"])
        return {
            "candidate_name": str(run_payload["candidate_name"]),
            "run_dir": str(run_payload["run_dir"]),
            "score": round(score, 4),
            "summary": self._build_summary(metrics),
            "metrics": metrics,
        }

    def _load_results(self, results_path: Path) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _compute_main_score(self, summary: dict[str, object]) -> float:
        if "reason_match_rate" in summary:
            return (
                self.ACTION_MATCH_WITH_REASON_WEIGHT
                * float(summary.get("action_match_rate", 0.0))
                + self.TARGET_DETECTED_WITH_REASON_WEIGHT
                * float(summary.get("target_detected_rate", 0.0))
                + self.USABLE_CLIP_WITH_REASON_WEIGHT
                * float(summary.get("usable_clip_rate", 0.0))
                + self.REASON_MATCH_WEIGHT
                * float(summary.get("reason_match_rate", 0.0))
            )
        return (
            self.ACTION_MATCH_WEIGHT * float(summary.get("action_match_rate", 0.0))
            + self.TARGET_DETECTED_WEIGHT
            * float(summary.get("target_detected_rate", 0.0))
            + self.USABLE_CLIP_WEIGHT * float(summary.get("usable_clip_rate", 0.0))
        )

    def _compute_error_rate(
        self,
        result_rows: list[dict[str, object]],
        result_count: int,
    ) -> float:
        error_count = sum(1 for row in result_rows if row.get("error"))
        return error_count / result_count

    def _compute_missing_target_ratio(
        self,
        result_rows: list[dict[str, object]],
        result_count: int,
    ) -> float:
        missing_target_count = sum(
            1 for row in result_rows if not bool(row.get("target_detected"))
        )
        return missing_target_count / result_count

    def _build_summary(self, metrics: dict[str, object]) -> str:
        reason_summary = ""
        if "reason_match_rate" in metrics:
            reason_summary = (
                f"reason_match_rate={float(metrics['reason_match_rate']):.2f} "
            )
        return (
            f"action_match_rate={float(metrics['action_match_rate']):.2f} "
            f"target_detected_rate={float(metrics['target_detected_rate']):.2f} "
            f"usable_clip_rate={float(metrics['usable_clip_rate']):.2f} "
            f"{reason_summary}"
            f"error_rate={float(metrics['error_rate']):.2f} "
            f"target_not_detected_ratio="
            f"{float(metrics['target_not_detected_ratio']):.2f}"
        )
