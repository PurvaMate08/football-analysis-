"""Correlation and model-evaluation charts."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def correlation_heatmap(correlations: pd.DataFrame) -> go.Figure:
    """Render a correlation matrix as an annotated interactive heatmap."""
    if correlations.empty:
        raise ValueError("Correlation matrix is empty.")
    return px.imshow(
        correlations, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
        title="Feature correlation heatmap",
    )


def actual_vs_predicted(actual, predicted, model_name: str = "Model") -> go.Figure:
    """Compare held-out model predictions with observed target values."""
    if len(actual) == 0 or len(actual) != len(predicted):
        raise ValueError("actual and predicted values must have the same non-zero length.")
    figure = px.scatter(
        x=actual, y=predicted,
        labels={"x": "Actual next-frame player count", "y": "Predicted player count"},
        title=f"Actual vs predicted — {model_name}",
    )
    lower = float(min(min(actual), min(predicted)))
    upper = float(max(max(actual), max(predicted)))
    figure.add_trace(go.Scatter(x=[lower, upper], y=[lower, upper], mode="lines", name="Ideal"))
    return figure
