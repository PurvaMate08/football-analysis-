"""Single-page Streamlit interface for video upload and basic match statistics."""

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from analytics.dataset_generation import generate_dataset
from config import PROCESSED_DIR, YOLO_MODEL_PATH, ensure_project_directories
from detection.detector import YOLODetector
from detection.video_processor import process_video
from visualization.detection_plots import player_detection_trend


def run() -> None:
    st.set_page_config(
        page_title="Football Video Analytics",
        page_icon="⚽",
        layout="wide",
    )
    st.title("⚽ Football Video Analytics")
    st.caption("Upload a match clip to detect players and the ball, then view basic statistics.")

    uploaded_video = st.file_uploader(
        "Choose a football video",
        type=["mp4", "avi", "mov", "mkv", "mpeg", "mpg"],
        help="Select a video file readable by OpenCV.",
    )
    sample_every = st.slider(
        "Analyze every Nth frame",
        min_value=1,
        max_value=30,
        value=5,
        help="Larger values process fewer frames and finish faster.",
    )

    if uploaded_video is None:
        st.info("Upload a video to get started.")
        return

    st.video(uploaded_video.getvalue())
    if not st.button("Analyze video", type="primary"):
        return

    ensure_project_directories()
    temporary_path: Path | None = None
    try:
        suffix = Path(uploaded_video.name).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
            temporary_file.write(uploaded_video.getvalue())
            temporary_path = Path(temporary_file.name)

        output_path = PROCESSED_DIR / f"{Path(uploaded_video.name).stem}_detections.csv"
        detector = YOLODetector(model_path=YOLO_MODEL_PATH)
        detections = process_video(
            temporary_path,
            detector,
            output_csv=output_path,
            sample_every=sample_every,
        )
        dataset = generate_dataset(output_path)
    except (FileNotFoundError, ImportError, ValueError, OSError, RuntimeError) as exc:
        st.error(f"Could not analyze this video: {exc}")
        return
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    st.success("Video analysis complete. Statistics below are based on detected sampled frames.")
    _show_statistics(detections, dataset)
    st.subheader("Player detections over time")
    st.plotly_chart(player_detection_trend(dataset), use_container_width=True)
    with st.expander("View generated frame-level dataset"):
        st.dataframe(dataset, use_container_width=True)
    with st.expander("View raw detections"):
        st.dataframe(detections, use_container_width=True)
    st.download_button(
        "Download frame statistics (CSV)",
        data=dataset.to_csv(index=False).encode("utf-8"),
        file_name=f"{Path(uploaded_video.name).stem}_frame_statistics.csv",
        mime="text/csv",
    )


def _show_statistics(detections: pd.DataFrame, dataset: pd.DataFrame) -> None:
    """Display summary metrics calculated only from real detection outputs."""
    valid_detections = detections.loc[detections["class_id"] >= 0]
    confidence = pd.to_numeric(valid_detections["confidence"], errors="coerce").dropna()
    frame_count = len(dataset)
    player_total = int(dataset["player_count"].sum())
    average_players = float(dataset["player_count"].mean()) if frame_count else 0.0
    max_players = int(dataset["player_count"].max()) if frame_count else 0
    ball_frames = int(dataset["ball_detected"].sum())
    ball_rate = (ball_frames / frame_count * 100) if frame_count else 0.0

    metrics = st.columns(5)
    metrics[0].metric("Frames analyzed", f"{frame_count:,}")
    metrics[1].metric("Player detections", f"{player_total:,}")
    metrics[2].metric("Average players / frame", f"{average_players:.1f}")
    metrics[3].metric("Maximum players / frame", f"{max_players:,}")
    metrics[4].metric("Frames with ball", f"{ball_rate:.1f}%")
    if not confidence.empty:
        st.caption(f"Mean YOLO confidence: {confidence.mean():.3f}")


if __name__ == "__main__":
    run()
