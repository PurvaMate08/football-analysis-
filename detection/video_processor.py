"""Orchestrates video frame extraction and reusable YOLO inference."""

from pathlib import Path

import pandas as pd

from config import BALL_CLASS_IDS, PLAYER_CLASS_IDS, PROCESSED_DIR, ensure_project_directories
from detection.detector import YOLODetector
from detection.frame_extractor import iter_video_frames
from utils.file_handler import validate_file, write_csv
from utils.logging_config import get_logger

LOGGER = get_logger(__name__)
DETECTION_COLUMNS = [
    "video_name", "frame_index", "timestamp_seconds", "class_id", "class_name",
    "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y",
]


def process_video(
    video_path: str | Path,
    detector: YOLODetector,
    output_csv: str | Path | None = None,
    sample_every: int = 1,
    max_frames: int | None = None,
) -> pd.DataFrame:
    """Detect configured player and ball classes and save frame-level detections."""
    path = validate_file(video_path, (".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"))
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be at least 1 when provided.")
    records: list[dict[str, int | float | str]] = []
    for processed_count, (frame_index, timestamp, frame) in enumerate(
        iter_video_frames(path, sample_every), start=1
    ):
        frame_detections = detector.detect(
            frame, class_ids=PLAYER_CLASS_IDS + BALL_CLASS_IDS
        )
        for detection in frame_detections:
            row = detection.to_dict()
            row.update(
                video_name=path.name,
                frame_index=frame_index,
                timestamp_seconds=timestamp,
                center_x=(detection.x1 + detection.x2) / 2,
                center_y=(detection.y1 + detection.y2) / 2,
            )
            records.append(row)
        if not frame_detections:
            records.append(
                {
                    "video_name": path.name,
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp,
                    "class_id": -1,
                    "class_name": "no_detection",
                    "confidence": float("nan"),
                    "x1": float("nan"),
                    "y1": float("nan"),
                    "x2": float("nan"),
                    "y2": float("nan"),
                    "center_x": float("nan"),
                    "center_y": float("nan"),
                }
            )
        if max_frames is not None and processed_count >= max_frames:
            break
    if not records:
        raise ValueError(f"No readable video frames were found in: {path}")
    result = pd.DataFrame.from_records(records, columns=DETECTION_COLUMNS)
    if output_csv is None:
        ensure_project_directories()
        output_csv = PROCESSED_DIR / f"{path.stem}_detections.csv"
    saved_path = write_csv(result, output_csv)
    LOGGER.info("Processed %s; saved %d detections to %s", path, len(result), saved_path)
    return result
