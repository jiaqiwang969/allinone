"""Ultralytics YOLO detector adapter boundary."""

from __future__ import annotations

from dataclasses import dataclass

from allinone.domain.shared.value_objects import BoundingBox


@dataclass(frozen=True)
class DetectionCandidate:
    label: str
    confidence: float
    bbox: BoundingBox


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
        return self.normalize_prediction_rows(
            prediction_rows=rows,
            image_size=image_size,
            target_labels=target_labels,
        )

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
