"""Model comparison helpers."""

import pandas as pd


def rank_models(metrics: pd.DataFrame, metric: str = "rmse") -> pd.DataFrame:
    """Rank models by a known evaluation metric (lower is better for errors)."""
    if "model" not in metrics or metric not in metrics:
        raise ValueError(f"Metrics must contain 'model' and '{metric}' columns.")
    return metrics.sort_values(metric, ascending=True, na_position="last").reset_index(drop=True)
