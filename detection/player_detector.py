"""Player-only inference using the project's reusable YOLO adapter."""

from typing import Any

from config import PLAYER_CLASS_IDS
from detection.detector import YOLODetector
from detection.results import Detection


def detect_players(frame: Any, detector: YOLODetector) -> list[Detection]:
    """Return configured player-class detections for one frame."""
    return detector.detect(frame, class_ids=PLAYER_CLASS_IDS)
