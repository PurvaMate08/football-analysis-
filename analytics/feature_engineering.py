"""Build leakage-aware next-observed-frame prediction features."""

import pandas as pd

from analytics.preprocessing import clean_dataset

MODEL_FEATURES = [
    "player_count",
    "ball_detected",
    "mean_confidence",
    "mean_center_x",
    "mean_center_y",
    "frame_delta",
    "time_delta",
]
TARGET_COLUMN = "target_player_count"


def create_prediction_features(frame_dataset: pd.DataFrame) -> pd.DataFrame:
    """Use current-frame observations to predict player count in the next observed frame."""
    data = clean_dataset(frame_dataset)
    if "mean_confidence" not in data:
        data["mean_confidence"] = 0.0
    for column in ("mean_center_x", "mean_center_y"):
        if column not in data:
            data[column] = 0.0
    data["frame_delta"] = data.groupby("video_name")["frame_index"].diff().fillna(0)
    data["time_delta"] = data.groupby("video_name")["timestamp_seconds"].diff().fillna(0)
    data[TARGET_COLUMN] = data.groupby("video_name")["player_count"].shift(-1)
    data = data.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    if data.empty:
        raise ValueError("At least two observed frames in a video are required for prediction.")
    data[MODEL_FEATURES] = data[MODEL_FEATURES].replace([float("inf"), float("-inf")], 0).fillna(0)
    return data
