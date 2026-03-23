"""Ultralytics YOLO detector adapter boundary."""

from __future__ import annotations

from dataclasses import dataclass

from allinone.domain.shared.value_objects import BoundingBox


@dataclass(frozen=True)
class DetectionCandidate:
    label: str
    confidence: float
    bbox: BoundingBox

    def to_prediction_row(self, *, image_size: tuple[int, int]) -> dict[str, object]:
        width, height = image_size
        return {
            "label": self.label,
            "confidence": self.confidence,
            "xyxy": [
                self.bbox.x1 * width,
                self.bbox.y1 * height,
                self.bbox.x2 * width,
                self.bbox.y2 * height,
            ],
        }


class UltralyticsDetectorAdapter:
    """Normalize upstream Ultralytics results into project-facing detections."""

    def __init__(self, model_path: str | None = None, device: str | None = None) -> None:
        self.model_path = model_path
        self.device = device
        self._model = None

    def normalize_prediction_rows(
        self,
        *,
        prediction_rows: list[dict[str, object]],
        image_size: tuple[int, int],
        target_labels: tuple[str, ...] | None = None,
    ) -> list[DetectionCandidate]:
        width, height = image_size
        target_set = set(target_labels or ())
        detections: list[DetectionCandidate] = []
        for row in prediction_rows:
            label = str(row["label"])
            if target_set and label not in target_set:
                continue
            x1, y1, x2, y2 = row["xyxy"]  # type: ignore[index]
            detections.append(
                DetectionCandidate(
                    label=label,
                    confidence=float(row["confidence"]),
                    bbox=BoundingBox(
                        x1=float(x1) / width,
                        y1=float(y1) / height,
                        x2=float(x2) / width,
                        y2=float(y2) / height,
                    ),
                )
            )
        return sorted(detections, key=lambda item: item.confidence, reverse=True)

    def predict(
        self,
        *,
        image_path: str,
        image_size: tuple[int, int],
        target_labels: tuple[str, ...] | None = None,
    ) -> list[DetectionCandidate]:
        model = self._ensure_model()
        results = model(image_path, device=self.device, verbose=False)
        rows = self._collect_prediction_rows(results)
        return self.normalize_prediction_rows(
            prediction_rows=rows,
            image_size=image_size,
            target_labels=target_labels,
        )

    def predict_sampled_frames(
        self,
        *,
        sampled_frames: list[object],
        image_size: tuple[int, int],
        target_labels: tuple[str, ...] | None = None,
    ) -> dict[str, object]:
        frame_detections = [
            self._predict_frame(
                frame=frame,
                image_size=image_size,
                target_labels=target_labels,
            )
            for frame in sampled_frames
        ]
        best_frame_index = self.select_best_frame_index(
            frame_detections=frame_detections
        )
        prediction_rows: list[dict[str, object]] = []
        if best_frame_index is not None:
            prediction_rows = [
                detection.to_prediction_row(image_size=image_size)
                for detection in frame_detections[best_frame_index]
            ]
        return {
            "prediction_rows": prediction_rows,
            "best_frame_index": best_frame_index,
        }

    def select_best_frame_index(
        self,
        *,
        frame_detections: list[list[DetectionCandidate]],
    ) -> int | None:
        best_index: int | None = None
        best_score = -1.0
        for index, detections in enumerate(frame_detections):
            frame_score = max(
                (self._score_detection_candidate(item) for item in detections),
                default=-1.0,
            )
            if frame_score > best_score:
                best_index = index
                best_score = frame_score
        return best_index

    def _predict_frame(
        self,
        *,
        frame: object,
        image_size: tuple[int, int],
        target_labels: tuple[str, ...] | None = None,
    ) -> list[DetectionCandidate]:
        model = self._ensure_model()
        results = model(frame, device=self.device, verbose=False)
        rows = self._collect_prediction_rows(results)
        return self.normalize_prediction_rows(
            prediction_rows=rows,
            image_size=image_size,
            target_labels=target_labels,
        )

    def _collect_prediction_rows(self, results) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for result in results:
            names = result.names
            for index in range(len(result.boxes)):
                cls_id = int(result.boxes.cls[index].item())
                rows.append(
                    {
                        "label": names[cls_id],
                        "confidence": float(result.boxes.conf[index].item()),
                        "xyxy": result.boxes.xyxy[index].tolist(),
                    }
                )
        return rows

    def _score_detection_candidate(self, detection: DetectionCandidate) -> float:
        bbox = detection.bbox
        width = max(0.0, bbox.x2 - bbox.x1)
        height = max(0.0, bbox.y2 - bbox.y1)
        area = width * height
        center_x = (bbox.x1 + bbox.x2) / 2
        center_y = (bbox.y1 + bbox.y2) / 2
        center_bonus = max(0.0, 1.0 - (abs(center_x - 0.5) + abs(center_y - 0.5)))
        return detection.confidence * area * center_bonus

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        if not self.model_path:
            raise RuntimeError("model_path is required for live YOLO inference")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed; cannot run live YOLO inference"
            ) from exc
        self._model = YOLO(self.model_path)
        return self._model
