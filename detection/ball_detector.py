"""Football-only inference using the project's reusable YOLO adapter."""

from typing import Any

from config import BALL_CLASS_IDS
from detection.detector import YOLODetector
from detection.results import Detection


def detect_football(frame: Any, detector: YOLODetector) -> list[Detection]:
    """Return configured football-class detections for one frame."""
    return detector.detect(frame, class_ids=BALL_CLASS_IDS)
