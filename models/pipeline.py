"""End-to-end feature preparation, model comparison, persistence, and report export."""

from pathlib import Path

import pandas as pd

from analytics.feature_engineering import create_prediction_features
from config import REPORTS_DIR, ensure_project_directories
from models.model_comparison import rank_models
from models.persistence import save_model
from models.train import train_models


def train_and_compare(frame_dataset: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Train all supported estimators, persist bundles, and export measured metrics."""
    features = create_prediction_features(frame_dataset)
    metrics, fitted = train_models(features)
    ensure_project_directories()
    metrics = rank_models(metrics)
    for name, bundle in fitted.items():
        save_model(bundle, name)
    report_path = REPORTS_DIR / "model_performance" / "model_comparison.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(report_path, index=False)
    return metrics, fitted


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read an existing generated dataset for training."""
    dataset_path = Path(path)
    if not dataset_path.is_file() or dataset_path.suffix.lower() != ".csv":
        raise FileNotFoundError(f"Generated CSV dataset not found: {dataset_path}")
    return pd.read_csv(dataset_path)
