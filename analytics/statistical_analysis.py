"""Statistical tests and confidence intervals for frame-level observations."""

import pandas as pd
from scipy import stats


def compare_player_counts(
    frame: pd.DataFrame,
    first_video: str,
    second_video: str,
) -> dict[str, float | int]:
    """Welch's t-test for player counts in two videos, using observed data only."""
    required = {"video_name", "player_count"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")
    first = frame.loc[frame["video_name"] == first_video, "player_count"].dropna()
    second = frame.loc[frame["video_name"] == second_video, "player_count"].dropna()
    if len(first) < 2 or len(second) < 2:
        raise ValueError("Each selected video needs at least two observed player-count frames.")
    statistic, p_value = stats.ttest_ind(first, second, equal_var=False)
    return {
        "first_n": int(len(first)),
        "second_n": int(len(second)),
        "first_mean": float(first.mean()),
        "second_mean": float(second.mean()),
        "t_statistic": float(statistic),
        "p_value": float(p_value),
    }


def mean_confidence_interval(values: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    """Calculate a Student-t confidence interval for the sample mean."""
    observed = pd.to_numeric(values, errors="coerce").dropna()
    if len(observed) < 2:
        raise ValueError("At least two numeric observations are required.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")
    mean = float(observed.mean())
    margin = float(stats.sem(observed) * stats.t.ppf((1 + confidence) / 2, len(observed) - 1))
    return mean - margin, mean + margin


def player_count_trend(frame: pd.DataFrame, video_name: str) -> dict[str, float]:
    """Fit an OLS linear trend over observed player counts for one video."""
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise ImportError("Install the 'statsmodels' dependency for OLS trend analysis.") from exc
    required = {"video_name", "frame_index", "player_count"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")
    sample = frame.loc[frame["video_name"] == video_name].dropna(
        subset=["frame_index", "player_count"]
    )
    if len(sample) < 3:
        raise ValueError("At least three observed frames are required for trend analysis.")
    x = sm.add_constant(sample["frame_index"].astype(float))
    result = sm.OLS(sample["player_count"].astype(float), x).fit()
    return {
        "slope_per_frame": float(result.params["frame_index"]),
        "intercept": float(result.params["const"]),
        "r_squared": float(result.rsquared),
        "p_value": float(result.pvalues["frame_index"]),
    }
