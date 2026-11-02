"""Streamlit workflow for uploaded football videos."""

import tempfile
from pathlib import Path

import streamlit as st

from config import PROCESSED_DIR, YOLO_MODEL_PATH, ensure_project_directories
from analytics.dataset_generation import generate_dataset
from detection.detector import YOLODetector
from detection.video_processor import process_video


def render() -> None:
    st.header("Video processing")
    st.write("Run the configured YOLO model on an uploaded match video.")
    model_path = st.text_input("YOLO model file", value=str(YOLO_MODEL_PATH))
    sample_every = st.number_input("Process every Nth frame", min_value=1, value=5, step=1)
    uploaded = st.file_uploader("Football video", type=["mp4", "avi", "mov", "mkv", "mpeg", "mpg"])
    if uploaded is None or not st.button("Process video", type="primary"):
        return

    ensure_project_directories()
    suffix = Path(uploaded.name).suffix.lower()
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
            temporary_file.write(uploaded.getvalue())
            temporary_path = Path(temporary_file.name)
        output_path = PROCESSED_DIR / f"{Path(uploaded.name).stem}_detections.csv"
        detector = YOLODetector(model_path=model_path)
        result = process_video(
            temporary_path,
            detector,
            output_csv=output_path,
            sample_every=int(sample_every),
        )
        st.success(f"Processed video. Saved {len(result)} detection rows to {output_path}.")
        st.dataframe(result.head(100), use_container_width=True)
        frame_dataset = generate_dataset(output_path)
        st.success(f"Generated {len(frame_dataset)} observed frame rows in data/datasets/.")
        st.dataframe(frame_dataset.head(100), use_container_width=True)
    except (FileNotFoundError, ImportError, ValueError, OSError, RuntimeError) as exc:
        st.error(f"Video processing failed: {exc}")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
