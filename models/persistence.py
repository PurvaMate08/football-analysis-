"""Joblib persistence for trained estimator bundles."""

from pathlib import Path
from typing import Any

import joblib

from config import SAVED_MODELS_DIR, ensure_project_directories


def save_model(bundle: dict[str, Any], name: str, output_dir: str | Path = SAVED_MODELS_DIR) -> Path:
    """Persist an estimator bundle and return its path."""
    if not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Model name may contain only letters, numbers, underscores, and hyphens.")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{name}.joblib"
    joblib.dump(bundle, output)
    return output


def load_model(path: str | Path) -> dict[str, Any]:
    """Load a trusted local Joblib bundle; never load model files from untrusted sources."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved model not found: {model_path}")
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "estimator" not in bundle:
        raise ValueError(f"Invalid model bundle: {model_path}")
    return bundle
