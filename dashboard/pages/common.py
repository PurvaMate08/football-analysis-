"""Shared dataset discovery and loading for Streamlit page renderers."""

from pathlib import Path

import pandas as pd

from config import DATASETS_DIR, PROCESSED_DIR


def available_datasets() -> list[Path]:
    """Return generated CSV datasets, newest first."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(DATASETS_DIR.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)


def available_detection_exports() -> list[Path]:
    """Return per-object detection exports, newest first."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(PROCESSED_DIR.glob("*_detections.csv"), key=lambda path: path.stat().st_mtime, reverse=True)


def select_dataset(st, key: str = "dataset") -> pd.DataFrame | None:
    """Render a dataset selector and return the selected CSV, if available."""
    choices = available_datasets()
    if not choices:
        st.info("No generated datasets yet. Process a video and generate a dataset first.")
        return None
    selected = st.selectbox("Generated dataset", choices, format_func=lambda path: path.name, key=key)
    try:
        frame = pd.read_csv(selected)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        st.error(f"Could not read {selected.name}: {exc}")
        return None
    if frame.empty:
        st.warning(f"{selected.name} has no data rows.")
        return None
    return frame
