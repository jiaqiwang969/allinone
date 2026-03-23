"""Write experiment replay artifacts into an autoresearch-style run directory."""

from __future__ import annotations

import json
from pathlib import Path


class AutoresearchRunWriter:
    """Persist manifest, per-clip artifacts, results, and summary for a run."""

    def __init__(self, *, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)

    def write(
        self,
        *,
        manifest_rows: list[dict[str, object]],
        result_rows: list[dict[str, object]],
        candidate_name: str,
    ) -> dict[str, object]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = self.run_dir / "raw"
        payload_dir = self.run_dir / "payload"
        raw_dir.mkdir(parents=True, exist_ok=True)
        payload_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = self.run_dir / "manifest.jsonl"
        results_path = self.run_dir / "results.jsonl"
        summary_path = self.run_dir / "summary.json"

        self._write_jsonl(manifest_path, manifest_rows)
        self._write_artifact_rows(result_rows, raw_dir=raw_dir, payload_dir=payload_dir)

        public_result_rows = [self._to_public_result_row(row) for row in result_rows]
        self._write_jsonl(results_path, public_result_rows)

        summary = self._build_summary(
            candidate_name=candidate_name,
            result_rows=public_result_rows,
        )
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {
            "run_dir": str(self.run_dir),
            "manifest_path": str(manifest_path),
            "results_path": str(results_path),
            "summary_path": str(summary_path),
            "summary": summary,
        }

    def _write_artifact_rows(
        self,
        result_rows: list[dict[str, object]],
        *,
        raw_dir: Path,
        payload_dir: Path,
    ) -> None:
        for row in result_rows:
            clip_id = str(row["clip_id"])
            (raw_dir / f"{clip_id}.json").write_text(
                json.dumps(row["raw_payload"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (payload_dir / f"{clip_id}.json").write_text(
                json.dumps(row["payload"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _build_summary(
        self,
        *,
        candidate_name: str,
        result_rows: list[dict[str, object]],
    ) -> dict[str, object]:
        clip_count = len(result_rows)
        return {
            "candidate_name": candidate_name,
            "clip_count": clip_count,
            "action_match_rate": _safe_rate(
                sum(1 for row in result_rows if bool(row["action_match"])),
                clip_count,
            ),
            "target_detected_rate": _safe_rate(
                sum(1 for row in result_rows if bool(row["target_detected"])),
                clip_count,
            ),
            "usable_clip_rate": _safe_rate(
                sum(
                    1
                    for row in result_rows
                    if float(row["visibility_score"]) >= 0.5
                    and float(row["readable_ratio"]) >= 0.5
                ),
                clip_count,
            ),
        }

    def _to_public_result_row(self, row: dict[str, object]) -> dict[str, object]:
        return {
            key: value
            for key, value in row.items()
            if key not in {"raw_payload", "payload"}
        }

    def _write_jsonl(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "\n".join(
                json.dumps(row, ensure_ascii=False)
                for row in rows
            ),
            encoding="utf-8",
        )


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
