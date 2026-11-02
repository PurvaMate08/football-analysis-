"""Train, compare, and visualize predictive models."""

import streamlit as st

from dashboard.pages.common import select_dataset
from models.pipeline import train_and_compare
from visualization.analytics_plots import actual_vs_predicted


def render() -> None:
    st.header("Prediction")
    st.caption("Target: detected player count in the next observed frame of the same video.")
    data = select_dataset(st, "prediction_dataset")
    if data is None:
        return
    if not st.button("Train and compare models", type="primary"):
        return
    try:
        metrics, fitted = train_and_compare(data)
    except (ValueError, OSError, ImportError) as exc:
        st.error(f"Model training failed: {exc}")
        return
    st.subheader("Chronological holdout performance")
    st.dataframe(metrics, use_container_width=True)
    best_name = metrics.iloc[0]["model"]
    st.success(f"Lowest holdout RMSE: {best_name}")
    st.plotly_chart(
        actual_vs_predicted(
            fitted[best_name]["actual"],
            fitted[best_name]["predicted"],
            best_name,
        ),
        use_container_width=True,
    )
    st.caption("Metrics and fitted model bundles are saved under reports/ and models/saved/.")
