"""Static Matplotlib/Seaborn chart exports for analytical reports."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_player_trend(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """Export a player-count timeline from frame-level observations."""
    required = {"video_name", "timestamp_seconds", "player_count"}
    if not required.issubset(frame.columns):
        raise ValueError("Frame dataset must include video_name, timestamp_seconds, and player_count.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=frame.sort_values(["video_name", "timestamp_seconds"]),
        x="timestamp_seconds",
        y="player_count",
        hue="video_name",
        ax=axis,
    )
    axis.set(title="Detected players by observed frame", xlabel="Time (seconds)", ylabel="Players")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def save_correlation_heatmap(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """Export a heatmap computed from numeric columns in the supplied dataset."""
    numeric = frame.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        raise ValueError("At least two numeric columns are required for a correlation heatmap.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axis)
    axis.set_title("Feature correlation heatmap")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output
