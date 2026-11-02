"""Correlation calculations on numeric dataset columns."""

import pandas as pd


def correlation_matrix(frame: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Compute correlations without including non-numeric identifiers."""
    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be pearson, spearman, or kendall.")
    numeric = frame.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        raise ValueError("At least two numeric columns are required for correlation.")
    return numeric.corr(method=method)
