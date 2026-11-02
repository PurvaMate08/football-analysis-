"""IQR-based outlier flagging for numeric columns."""

import pandas as pd


def flag_iqr_outliers(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Return a copy with one boolean '<column>_outlier' flag per selected column."""
    result = frame.copy()
    selected = columns or list(result.select_dtypes(include="number").columns)
    if not selected:
        raise ValueError("No numeric columns available for outlier detection.")
    for column in selected:
        if column not in result or not pd.api.types.is_numeric_dtype(result[column]):
            raise ValueError(f"'{column}' is missing or is not numeric.")
        q1, q3 = result[column].quantile([0.25, 0.75])
        spread = q3 - q1
        result[f"{column}_outlier"] = (
            (result[column] < q1 - 1.5 * spread) |
            (result[column] > q3 + 1.5 * spread)
        )
    return result
