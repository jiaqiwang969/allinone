"""Generate deterministic guidance replay datasets from a base raw payload."""

from __future__ import annotations

import copy
import json
from pathlib import Path


class GuidanceBoundaryDatasetBuilder:
    """Materialize boundary raw payload cases that expose policy threshold changes."""

    def build(
        self,
        *,
        base_raw_payload: dict[str, object],
        output_dir: str | Path,
        target_label: str,
    ) -> dict[str, object]:
        output_path = Path(output_dir)
        raw_dir = output_path / "raw"
        output_path.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)

        detections = dict(base_raw_payload["detections"])
        image_size = tuple(detections["image_size"])
        target_row = self._find_target_row(
            prediction_rows=list(detections["prediction_rows"]),
            target_label=target_label,
        )
        x1, y1, x2, y2 = self._normalize_bbox(
            xyxy=target_row["xyxy"],
            image_size=image_size,
        )
        base_width = x2 - x1
        base_height = y2 - y1

        cases = [
            {
                "clip_id": "tight_center_boundary",
                "bbox": self._shift_bbox(
                    width=base_width,
                    height=base_height,
                    center_x=0.581,
                    center_y=0.5,
                ),
                "expected_action": "hold_still",
                "expected_reason": "stabilize_before_capture",
            },
            {
                "clip_id": "direction_trigger_boundary",
                "bbox": self._shift_bbox(
                    width=base_width,
                    height=base_height,
                    center_x=0.6665,
                    center_y=0.5,
                ),
                "expected_action": "left",
                "expected_reason": "target_shifted_right",
            },
            {
                "clip_id": "oversize_boundary",
                "bbox": self._centered_square(area=0.87125),
                "expected_action": "hold_still",
                "expected_reason": "fully_centered",
            },
        ]

        manifest_rows: list[dict[str, object]] = []
        for case in cases:
            raw_payload = self._build_case_payload(
                base_raw_payload=base_raw_payload,
                target_label=target_label,
                bbox=case["bbox"],
            )
            raw_payload_path = raw_dir / f"{case['clip_id']}.json"
            raw_payload_path.write_text(
                json.dumps(raw_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            manifest_rows.append(
                {
                    "clip_id": case["clip_id"],
                    "raw_payload_path": str(raw_payload_path),
                    "target_labels": list(detections["target_labels"]),
                    "task_type": "view_guidance",
                    "expected_action": case["expected_action"],
                    "expected_reason": case["expected_reason"],
                }
            )

        manifest_path = output_path / "manifest.jsonl"
        manifest_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in manifest_rows),
            encoding="utf-8",
        )
        return {
            "output_dir": str(output_path),
            "manifest_path": str(manifest_path),
            "raw_dir": str(raw_dir),
            "case_count": len(manifest_rows),
        }

    def _build_case_payload(
        self,
        *,
        base_raw_payload: dict[str, object],
        target_label: str,
        bbox: tuple[float, float, float, float],
    ) -> dict[str, object]:
        raw_payload = copy.deepcopy(base_raw_payload)
        detections = raw_payload["detections"]
        image_size = tuple(detections["image_size"])
        prediction_rows = []
        for row in detections["prediction_rows"]:
            if str(row["label"]) == target_label:
                updated_row = dict(row)
                updated_row["xyxy"] = self._denormalize_bbox(
                    bbox=bbox,
                    image_size=image_size,
                )
                prediction_rows.append(updated_row)
                continue
            prediction_rows.append(dict(row))
        detections["prediction_rows"] = prediction_rows
        return raw_payload

    def _find_target_row(
        self,
        *,
        prediction_rows: list[dict[str, object]],
        target_label: str,
    ) -> dict[str, object]:
        for row in prediction_rows:
            if str(row["label"]) == target_label:
                return dict(row)
        raise ValueError(f"target label not found in raw payload: {target_label}")

    def _normalize_bbox(
        self,
        *,
        xyxy: object,
        image_size: tuple[object, object],
    ) -> tuple[float, float, float, float]:
        width = float(image_size[0])
        height = float(image_size[1])
        x1, y1, x2, y2 = xyxy  # type: ignore[misc]
        return (
            float(x1) / width,
            float(y1) / height,
            float(x2) / width,
            float(y2) / height,
        )

    def _denormalize_bbox(
        self,
        *,
        bbox: tuple[float, float, float, float],
        image_size: tuple[object, object],
    ) -> list[float]:
        width = float(image_size[0])
        height = float(image_size[1])
        x1, y1, x2, y2 = bbox
        return [
            round(x1 * width, 4),
            round(y1 * height, 4),
            round(x2 * width, 4),
            round(y2 * height, 4),
        ]

    def _shift_bbox(
        self,
        *,
        width: float,
        height: float,
        center_x: float,
        center_y: float,
    ) -> tuple[float, float, float, float]:
        half_width = width / 2
        half_height = height / 2
        safe_center_x = min(max(center_x, half_width), 1.0 - half_width)
        safe_center_y = min(max(center_y, half_height), 1.0 - half_height)
        return (
            round(safe_center_x - half_width, 6),
            round(safe_center_y - half_height, 6),
            round(safe_center_x + half_width, 6),
            round(safe_center_y + half_height, 6),
        )

    def _centered_square(self, *, area: float) -> tuple[float, float, float, float]:
        side = min(0.98, area ** 0.5)
        half_side = side / 2
        return (
            round(0.5 - half_side, 6),
            round(0.5 - half_side, 6),
            round(0.5 + half_side, 6),
            round(0.5 + half_side, 6),
        )
