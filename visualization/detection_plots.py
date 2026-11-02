"""Interactive detection-count and confidence charts."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import PLAYER_CLASS_IDS


def player_detection_trend(frame_dataset: pd.DataFrame) -> go.Figure:
    """Plot observed detected-player counts over video time."""
    required = {"video_name", "timestamp_seconds", "player_count"}
    if not required.issubset(frame_dataset.columns):
        raise ValueError("Frame dataset must include video_name, timestamp_seconds, and player_count.")
    return px.line(
        frame_dataset.sort_values(["video_name", "timestamp_seconds"]),
        x="timestamp_seconds",
        y="player_count",
        color="video_name",
        title="Detected players by observed frame",
        labels={"timestamp_seconds": "Time (seconds)", "player_count": "Detected players"},
    )


def ball_detection_analysis(frame_dataset: pd.DataFrame) -> go.Figure:
    """Plot observed ball-detection counts over time."""
    required = {"video_name", "timestamp_seconds", "ball_detected"}
    if not required.issubset(frame_dataset.columns):
        raise ValueError("Frame dataset must include video_name, timestamp_seconds, and ball_detected.")
    return px.line(
        frame_dataset.sort_values(["video_name", "timestamp_seconds"]),
        x="timestamp_seconds",
        y="ball_detected",
        color="video_name",
        title="Football detection by observed frame",
        labels={"timestamp_seconds": "Time (seconds)", "ball_detected": "Ball detected"},
    )


def confidence_distribution(detections: pd.DataFrame) -> go.Figure:
    """Plot confidence values present in an actual detection export."""
    if "confidence" not in detections:
        raise ValueError("Detection data must include confidence.")
    values = pd.to_numeric(detections["confidence"], errors="coerce").dropna()
    if values.empty:
        raise ValueError("No confidence observations are available.")
    return px.histogram(values, x="confidence", nbins=25, title="Detection confidence distribution")


def player_spatial_distribution(detections: pd.DataFrame) -> go.Figure:
    """Plot detected player centers in image-pixel coordinates."""
    required = {"center_x", "center_y", "class_id"}
    if not required.issubset(detections.columns):
        raise ValueError("Detection data must include center_x, center_y, and class_id.")
    players = detections.loc[
        pd.to_numeric(detections["class_id"], errors="coerce").isin(PLAYER_CLASS_IDS)
    ]
    players = players.dropna(subset=["center_x", "center_y"])
    if players.empty:
        raise ValueError("No player center coordinates are available.")
    return px.scatter(
        players, x="center_x", y="center_y", color="video_name" if "video_name" in players else None,
        title="Detected player spatial distribution",
        labels={"center_x": "Image x (pixels)", "center_y": "Image y (pixels)"},
    ).update_yaxes(autorange="reversed")
