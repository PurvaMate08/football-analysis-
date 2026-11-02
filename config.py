"""Project-wide paths and environment-backed settings."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DATASETS_DIR = DATA_DIR / "datasets"
GENERATED_CSV_DIR = DATA_DIR / "generated_csv"
SAMPLE_VIDEOS_DIR = DATA_DIR / "sample_videos"
REPORTS_DIR = PROJECT_ROOT / "reports"
SAVED_MODELS_DIR = PROJECT_ROOT / "models" / "saved"

YOLO_MODEL_PATH = Path(os.getenv("YOLO_MODEL_PATH", str(PROJECT_ROOT / "yolov8n.pt")))
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.25"))
YOLO_IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", "640"))
PLAYER_CLASS_IDS = (0,)
BALL_CLASS_IDS = (32,)
DATABASE_PATH = Path(os.getenv("FOOTBALL_DB_PATH", str(DATA_DIR / "football_analysis.sqlite3")))


def ensure_project_directories() -> None:
    """Create writable output directories; inputs such as sample videos stay user-provided."""
    for path in (
        RAW_DIR,
        PROCESSED_DIR,
        DATASETS_DIR,
        GENERATED_CSV_DIR,
        REPORTS_DIR / "generated",
        REPORTS_DIR / "model_performance",
        REPORTS_DIR / "charts",
        SAVED_MODELS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
