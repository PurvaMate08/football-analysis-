"""YOLO adapter. Accepts an existing Ultralytics model or loads one from disk."""

from pathlib import Path
from typing import Any, Iterable

from config import YOLO_CONFIDENCE, YOLO_IMAGE_SIZE, YOLO_MODEL_PATH
from detection.results import Detection


class YOLODetector:
    """Thin inference wrapper that leaves the underlying model reusable."""

    def __init__(
        self,
        model: Any | None = None,
        model_path: str | Path = YOLO_MODEL_PATH,
        confidence: float = YOLO_CONFIDENCE,
        image_size: int = YOLO_IMAGE_SIZE,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        if image_size <= 0:
            raise ValueError("image_size must be greater than zero.")
        if model is None:
            path = Path(model_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(
                    f"YOLO model not found: {path}. Set YOLO_MODEL_PATH or supply a model."
                )
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise ImportError("Install the 'ultralytics' package to load a YOLO model.") from exc
            model = YOLO(str(path))
        self.model = model
        self.confidence = confidence
        self.image_size = image_size

    def detect(
        self,
        frame: Any,
        class_ids: Iterable[int] | None = None,
    ) -> list[Detection]:
        """Run inference on one OpenCV BGR frame and return normalized records."""
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        names = result.names
        selected_ids = set(class_ids) if class_ids is not None else None
        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is None:
            return detections
        for box in boxes:
            class_id = int(box.cls[0].item())
            if selected_ids is not None and class_id not in selected_ids:
                continue
            coordinates = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=str(names[class_id]),
                    confidence=float(box.conf[0].item()),
                    x1=float(coordinates[0]),
                    y1=float(coordinates[1]),
                    x2=float(coordinates[2]),
                    y2=float(coordinates[3]),
                )
            )
        return detections
