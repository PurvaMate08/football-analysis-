"""Train supported regressors using a chronological holdout split."""

from typing import Any

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from analytics.feature_engineering import MODEL_FEATURES, TARGET_COLUMN
from models.evaluation import regression_metrics

MODEL_FACTORIES = {
    "linear_regression": LinearRegression,
    "decision_tree": lambda: DecisionTreeRegressor(random_state=42, max_depth=8),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1, min_samples_leaf=2
    ),
}


def train_models(feature_data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Fit all project regressors and compare holdout performance.

    The final 20% of ordered observations is held out to avoid random frame leakage.
    """
    required = set(MODEL_FEATURES + [TARGET_COLUMN])
    missing = required.difference(feature_data.columns)
    if missing:
        raise ValueError(f"Training data is missing columns: {', '.join(sorted(missing))}")
    data = feature_data.dropna(subset=list(required)).reset_index(drop=True)
    if len(data) < 5:
        raise ValueError("At least five model-ready observations are required for training.")
    split = max(2, int(len(data) * 0.8))
    if split >= len(data):
        split = len(data) - 1
    x_train = data.loc[: split - 1, MODEL_FEATURES]
    y_train = data.loc[: split - 1, TARGET_COLUMN]
    x_test = data.loc[split:, MODEL_FEATURES]
    y_test = data.loc[split:, TARGET_COLUMN]

    results: list[dict[str, str | float]] = []
    fitted: dict[str, dict[str, Any]] = {}
    for name, factory in MODEL_FACTORIES.items():
        estimator = factory()
        estimator.fit(x_train, y_train)
        predictions = estimator.predict(x_test)
        metrics = regression_metrics(y_test.to_numpy(), predictions)
        results.append({"model": name, **metrics})
        fitted[name] = {
            "estimator": estimator,
            "feature_columns": list(MODEL_FEATURES),
            "target_column": TARGET_COLUMN,
            "actual": y_test.to_numpy(),
            "predicted": predictions,
        }
    return pd.DataFrame(results).sort_values("rmse").reset_index(drop=True), fitted
