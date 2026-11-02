"""Frame extraction utilities for videos readable by OpenCV."""

from collections.abc import Iterator
from pathlib import Path

import cv2

from utils.file_handler import validate_file


def iter_video_frames(
    video_path: str | Path,
    sample_every: int = 1,
) -> Iterator[tuple[int, float, object]]:
    """Yield (frame index, timestamp seconds, frame) at a configurable interval."""
    path = validate_file(video_path, (".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"))
    if sample_every < 1:
        raise ValueError("sample_every must be at least 1.")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"OpenCV could not open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise ValueError(f"Video has an invalid frame rate: {path}")
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % sample_every == 0:
                yield index, index / fps, frame
            index += 1
    finally:
        capture.release()
