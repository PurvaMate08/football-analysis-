"""Streamlit interface for uploading football clips and exploring basic stats."""

import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from analytics.dataset_generation import generate_dataset
from config import PROCESSED_DIR, YOLO_MODEL_PATH, ensure_project_directories
from detection.detector import YOLODetector
from detection.video_processor import process_video
from visualization.detection_plots import player_detection_trend

VIDEO_TYPES = ["mp4", "avi", "mov", "mkv", "mpeg", "mpg"]


def run() -> None:
    st.set_page_config(
        page_title="MatchLens | Football Analytics",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .hero { padding: 1.2rem 1.5rem; border-radius: 18px; background:
          linear-gradient(120deg, #102a43, #147d64); color: white; margin-bottom: 1rem; }
        .hero h1 { margin: 0; font-size: 2.2rem; }
        .hero p { margin: .35rem 0 0; color: #d9f3ea; }
        [data-testid="stMetric"] { background: #f7fafc; border-radius: 12px; padding: .8rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero"><h1>⚽ MatchLens</h1>'
        "<p>Upload a football highlight clip and turn it into a quick match snapshot.</p></div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Analysis settings")
        sample_every = st.slider(
            "Sample every Nth frame",
            min_value=1,
            max_value=30,
            value=5,
            help="Higher values make analysis faster by looking at fewer frames.",
        )
        st.caption("Motion and video metadata work without model weights.")
        if YOLO_MODEL_PATH.is_file():
            st.success("Player/ball detector ready")
        else:
            st.info("Optional YOLO weights not found")

    uploaded_video = st.file_uploader(
        "Add a football highlight video",
        type=VIDEO_TYPES,
        help="MP4 is recommended. The clip is analyzed locally and is not uploaded anywhere.",
    )
    if uploaded_video is None:
        st.info("Choose a clip above to preview it and start analysis.")
        return

    video_bytes = uploaded_video.getvalue()
    st.video(video_bytes)
    if not st.button("Analyze highlight", type="primary", use_container_width=True):
        return

    ensure_project_directories()
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=Path(uploaded_video.name).suffix.lower(), delete=False
        ) as temporary_file:
            temporary_file.write(video_bytes)
            temporary_path = Path(temporary_file.name)

        with st.spinner("Reading the clip and calculating match indicators..."):
            motion_stats = _analyze_motion(temporary_path, sample_every)
            detections, dataset = _analyze_detections(
                temporary_path, uploaded_video.name, sample_every
            )
    except (FileNotFoundError, ImportError, ValueError, OSError, RuntimeError) as exc:
        st.error(f"Could not analyze this video: {exc}")
        return
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    st.success("Analysis complete.")
    _show_video_stats(motion_stats)
    if detections is None or dataset is None:
        st.warning(
            "Player and ball counts are unavailable because no YOLO weights are configured. "
            "The indicators above are calculated directly from the uploaded video."
        )
        _show_motion_chart(motion_stats)
        return

    st.subheader("Detected football statistics")
    _show_statistics(detections, dataset)
    st.plotly_chart(player_detection_trend(dataset), use_container_width=True)
    with st.expander("View frame-level statistics"):
        st.dataframe(dataset, use_container_width=True)
    st.download_button(
        "Download frame statistics (CSV)",
        data=dataset.to_csv(index=False).encode("utf-8"),
        file_name=f"{Path(uploaded_video.name).stem}_frame_statistics.csv",
        mime="text/csv",
    )


def _analyze_motion(video_path: Path, sample_every: int) -> pd.DataFrame:
    """Measure observable frame-to-frame activity and brightness."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("The uploaded file could not be opened as a video.")

    rows: list[dict[str, float | int]] = []
    previous_gray = None
    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % sample_every == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                motion = 0.0 if previous_gray is None else float(
                    cv2.absdiff(gray, previous_gray).mean()
                )
                rows.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": float(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000),
                        "motion_score": motion,
                        "brightness": float(gray.mean()),
                    }
                )
                previous_gray = gray
            frame_index += 1
    finally:
        capture.release()

    if not rows:
        raise ValueError("No readable video frames were found.")
    return pd.DataFrame(rows)


def _analyze_detections(
    video_path: Path, original_name: str, sample_every: int
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Run optional YOLO analysis without making video metadata depend on it."""
    if not YOLO_MODEL_PATH.is_file():
        return None, None
    output_path = PROCESSED_DIR / f"{Path(original_name).stem}_detections.csv"
    detector = YOLODetector(model_path=YOLO_MODEL_PATH)
    detections = process_video(
        video_path, detector, output_csv=output_path, sample_every=sample_every
    )
    return detections, generate_dataset(output_path)


def _show_video_stats(motion_stats: pd.DataFrame) -> None:
    duration = float(motion_stats["timestamp_seconds"].max())
    active = motion_stats["motion_score"] > motion_stats["motion_score"].median()
    metrics = st.columns(4)
    metrics[0].metric("Sampled frames", f"{len(motion_stats):,}")
    metrics[1].metric("Approx. duration", f"{duration:.1f} s")
    metrics[2].metric("High-activity frames", f"{int(active.sum()):,}")
    metrics[3].metric("Avg brightness", f"{motion_stats['brightness'].mean():.0f}/255")


def _show_motion_chart(motion_stats: pd.DataFrame) -> None:
    st.subheader("Clip activity")
    chart = motion_stats.set_index("timestamp_seconds")[["motion_score", "brightness"]]
    st.line_chart(chart, y_label="Score", x_label="Time (seconds)")


def _show_statistics(detections: pd.DataFrame, dataset: pd.DataFrame) -> None:
    """Display summary metrics calculated only from real detection outputs."""
    valid = detections.loc[detections["class_id"] >= 0]
    confidence = pd.to_numeric(valid["confidence"], errors="coerce").dropna()
    frame_count = len(dataset)
    player_total = int(dataset["player_count"].sum())
    average_players = float(dataset["player_count"].mean()) if frame_count else 0.0
    max_players = int(dataset["player_count"].max()) if frame_count else 0
    ball_rate = float(dataset["ball_detected"].mean() * 100) if frame_count else 0.0
    metrics = st.columns(5)
    metrics[0].metric("Frames analyzed", f"{frame_count:,}")
    metrics[1].metric("Player detections", f"{player_total:,}")
    metrics[2].metric("Avg players / frame", f"{average_players:.1f}")
    metrics[3].metric("Peak players / frame", f"{max_players:,}")
    metrics[4].metric("Ball visibility", f"{ball_rate:.1f}%")
    if not confidence.empty:
        st.caption(f"Mean detector confidence: {confidence.mean():.3f}")


if __name__ == "__main__":
    run()
