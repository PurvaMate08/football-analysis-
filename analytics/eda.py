"""Descriptive statistics for actual generated frame-level data."""

import pandas as pd


def summarize_dataset(frame: pd.DataFrame) -> dict[str, object]:
    """Return schema, coverage, and numeric descriptive statistics."""
    if frame.empty:
        raise ValueError("Cannot summarize an empty dataset.")
    numeric = frame.select_dtypes(include="number")
    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "videos": int(frame["video_name"].nunique()) if "video_name" in frame else None,
        "frame_span": (
            int(frame["frame_index"].min()),
            int(frame["frame_index"].max()),
        ) if "frame_index" in frame else None,
        "missing_values": frame.isna().sum().to_dict(),
        "numeric_describe": numeric.describe().to_dict(),
    }
