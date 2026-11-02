"""Regression metrics shared by training and reporting."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate standard regression metrics for a non-empty evaluation set."""
    if len(actual) == 0 or len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same non-zero length.")
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan"),
    }
