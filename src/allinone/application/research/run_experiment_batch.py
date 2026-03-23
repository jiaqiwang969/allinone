"""Batch experiment orchestration for offline replay runs."""

from __future__ import annotations

from typing import Protocol

from allinone.application.runtime.build_observation_payload import (
    build_observation_payload_from_raw,
)


class ClipAnalyzer(Protocol):
    def __call__(
        self,
        *,
        clip_path: str,
        target_labels: tuple[str, ...],
    ) -> dict[str, object]: ...


class RuntimeRunner(Protocol):
    def __call__(self, *, payload: dict[str, object]) -> dict[str, object]: ...


class RawPayloadLoader(Protocol):
    def load(self, path: str) -> dict[str, object]: ...


class RunWriter(Protocol):
    def write(
        self,
        *,
        manifest_rows: list[dict[str, object]],
        result_rows: list[dict[str, object]],
        candidate_name: str,
    ) -> dict[str, object]: ...


def run_experiment_batch(
    *,
    manifest_rows: list[dict[str, object]],
    candidate_name: str,
    clip_analyzer: ClipAnalyzer,
    raw_payload_loader: RawPayloadLoader | None = None,
    runtime_runner: RuntimeRunner,
    run_writer: RunWriter,
) -> dict[str, object]:
    result_rows: list[dict[str, object]] = []
    for manifest_row in manifest_rows:
        raw_payload = _resolve_raw_payload(
            manifest_row=manifest_row,
            clip_analyzer=clip_analyzer,
            raw_payload_loader=raw_payload_loader,
        )
        payload = build_observation_payload_from_raw(raw_payload)
        runtime_result = runtime_runner(payload=payload)
        result_rows.append(
            _build_result_row(
                manifest_row=manifest_row,
                candidate_name=candidate_name,
                raw_payload=raw_payload,
                payload=payload,
                runtime_result=runtime_result,
            )
        )
    return {
        "candidate_name": candidate_name,
        "results": result_rows,
        "run_artifacts": run_writer.write(
            manifest_rows=manifest_rows,
            result_rows=result_rows,
            candidate_name=candidate_name,
        ),
    }


def _resolve_raw_payload(
    *,
    manifest_row: dict[str, object],
    clip_analyzer: ClipAnalyzer,
    raw_payload_loader: RawPayloadLoader | None,
) -> dict[str, object]:
    raw_payload_path = manifest_row.get("raw_payload_path")
    if raw_payload_path is not None:
        if raw_payload_loader is None:
            raise ValueError("raw_payload_loader is required for raw payload replay")
        return raw_payload_loader.load(str(raw_payload_path))
    return clip_analyzer(
        clip_path=str(manifest_row["clip_path"]),
        target_labels=tuple(manifest_row["target_labels"]),
    )


def _build_result_row(
    *,
    manifest_row: dict[str, object],
    candidate_name: str,
    raw_payload: dict[str, object],
    payload: dict[str, object],
    runtime_result: dict[str, object],
) -> dict[str, object]:
    detections = raw_payload["detections"]
    signal = raw_payload["vjepa"]
    prediction_rows = detections["prediction_rows"]
    guidance_action = runtime_result["guidance_action"]
    expected_action = manifest_row.get("expected_action")
    return {
        "clip_id": manifest_row["clip_id"],
        "candidate_name": candidate_name,
        "task_type": manifest_row["task_type"],
        "target_labels": list(manifest_row["target_labels"]),
        "expected_action": expected_action,
        "guidance_action": guidance_action,
        "language_action": runtime_result["language_action"],
        "action_match": guidance_action == expected_action,
        "target_detected": bool(prediction_rows),
        "best_frame_index": detections.get("best_frame_index"),
        "visibility_score": signal["visibility_score"],
        "readable_ratio": signal["readable_ratio"],
        "stability_score": signal.get("stability_score"),
        "alignment_score": signal.get("alignment_score"),
        "operator_message": runtime_result["operator_message"],
        "evidence_focus": runtime_result.get("evidence_focus"),
        "language_source": runtime_result["language_source"],
        "error": None,
        "raw_payload": raw_payload,
        "payload": payload,
    }
