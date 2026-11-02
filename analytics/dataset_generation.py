"""Convert per-object detection rows into observed frame-level analytics datasets."""

from pathlib import Path

import numpy as np
import pandas as pd

from analytics.preprocessing import validate_detections
from config import BALL_CLASS_IDS, DATASETS_DIR, PLAYER_CLASS_IDS, ensure_project_directories
from utils.database import save_dataframe
from utils.file_handler import read_csv, write_csv
from reports.report_generator import write_analytical_reports


def build_frame_dataset(detections: pd.DataFrame) -> pd.DataFrame:
    """Aggregate detections by observed frame; no synthetic video data is generated."""
    data = validate_detections(detections)
    frame_keys = data[["video_name", "frame_index", "timestamp_seconds"]].drop_duplicates()
    data = data.loc[data["class_id"] >= 0].copy()

    player_mask = data["class_id"].isin(PLAYER_CLASS_IDS) | data["class_name"].str.lower().isin(
        {"player", "person", "football player"}
    )
    ball_mask = data["class_id"].isin(BALL_CLASS_IDS) | data["class_name"].str.lower().str.contains(
        "ball", na=False
    )
    data["is_player"] = player_mask.astype(int)
    data["is_ball"] = ball_mask.astype(int)
    if data.empty:
        result = frame_keys.copy()
        result["player_count"] = 0
        result["ball_detected"] = 0
        result["mean_confidence"] = 0.0
        result["mean_center_x"] = np.nan
        result["mean_center_y"] = np.nan
    else:
        group = data.groupby(["video_name", "frame_index", "timestamp_seconds"], as_index=False)
        aggregates = group.agg(
            player_count=("is_player", "sum"),
            ball_detected=("is_ball", "max"),
            mean_confidence=("confidence", "mean"),
            mean_center_x=("center_x", "mean"),
            mean_center_y=("center_y", "mean"),
        )
        result = frame_keys.merge(
            aggregates,
            on=["video_name", "frame_index", "timestamp_seconds"],
            how="left",
        )
        result["player_count"] = result["player_count"].fillna(0).astype(int)
        result["ball_detected"] = result["ball_detected"].fillna(0).astype(int)
        result["mean_confidence"] = result["mean_confidence"].fillna(0.0)
    return result.sort_values(["video_name", "frame_index"]).reset_index(drop=True)


def generate_dataset(
    detections_csv: str | Path,
    output_csv: str | Path | None = None,
    sqlite_table: str | None = None,
) -> pd.DataFrame:
    """Read an actual detection CSV, save its aggregate, and optionally persist to SQLite."""
    detection_path = Path(detections_csv)
    frame_dataset = build_frame_dataset(read_csv(detection_path))
    if output_csv is None:
        ensure_project_directories()
        output_csv = DATASETS_DIR / f"{detection_path.stem}_frames.csv"
    write_csv(frame_dataset, output_csv)
    if sqlite_table:
        save_dataframe(frame_dataset, sqlite_table)
    write_analytical_reports(frame_dataset, Path(output_csv).stem)
    return frame_dataset
