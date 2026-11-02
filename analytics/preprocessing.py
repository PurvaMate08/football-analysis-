"""Validation and cleaning for detection exports and generated datasets."""

import pandas as pd

DETECTION_REQUIRED_COLUMNS = {
    "video_name", "frame_index", "timestamp_seconds", "class_id", "class_name",
    "confidence", "center_x", "center_y",
}


def validate_detections(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of detections with validated types and nonnegative frame indices."""
    missing = DETECTION_REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Detection data is missing columns: {', '.join(sorted(missing))}")
    cleaned = frame.copy()
    cleaned["frame_index"] = pd.to_numeric(cleaned["frame_index"], errors="raise").astype(int)
    cleaned["class_id"] = pd.to_numeric(cleaned["class_id"], errors="raise").astype(int)
    cleaned["timestamp_seconds"] = pd.to_numeric(
        cleaned["timestamp_seconds"], errors="raise"
    )
    if (cleaned["frame_index"] < 0).any() or (cleaned["timestamp_seconds"] < 0).any():
        raise ValueError("Frame indices and timestamps cannot be negative.")
    if cleaned["video_name"].isna().any():
        raise ValueError("video_name cannot be missing.")
    return cleaned


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate frame-level features and remove rows with unusable target values."""
    required = {"video_name", "frame_index", "timestamp_seconds", "player_count", "ball_detected"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")
    result = frame.copy()
    numeric = ["frame_index", "timestamp_seconds", "player_count", "ball_detected"]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=numeric)
    if result.empty:
        raise ValueError("No valid dataset rows remain after preprocessing.")
    if (result["player_count"] < 0).any():
        raise ValueError("player_count cannot be negative.")
    return result.sort_values(["video_name", "frame_index"]).reset_index(drop=True)
