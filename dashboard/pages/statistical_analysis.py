"""EDA, statistical summaries, correlations, and outlier inspection."""

import streamlit as st
import pandas as pd

from analytics.correlation import correlation_matrix
from analytics.eda import summarize_dataset
from analytics.outlier_detection import flag_iqr_outliers
from analytics.statistical_analysis import mean_confidence_interval, player_count_trend
from dashboard.pages.common import available_detection_exports, select_dataset
from visualization.analytics_plots import correlation_heatmap
from visualization.detection_plots import (
    ball_detection_analysis,
    confidence_distribution,
    player_detection_trend,
    player_spatial_distribution,
)


def render() -> None:
    st.header("Statistical analysis")
    data = select_dataset(st, "statistics_dataset")
    if data is None:
        return

    summary = summarize_dataset(data)
    left, right = st.columns(2)
    left.metric("Observed rows", summary["rows"])
    right.metric("Videos", summary["videos"] if summary["videos"] is not None else "—")
    st.subheader("Numeric descriptive statistics")
    st.dataframe(data.describe(include="number").T, use_container_width=True)

    if {"timestamp_seconds", "player_count", "video_name"}.issubset(data.columns):
        st.plotly_chart(player_detection_trend(data), use_container_width=True)
        st.plotly_chart(ball_detection_analysis(data), use_container_width=True)
        video = st.selectbox("Video for trend statistics", sorted(data["video_name"].unique()))
        if st.button("Calculate 95% mean confidence interval"):
            try:
                lower, upper = mean_confidence_interval(data.loc[data.video_name == video, "player_count"])
                st.write(f"95% CI for mean player count: **{lower:.3f} to {upper:.3f}**")
            except ValueError as exc:
                st.warning(str(exc))
        if st.button("Fit OLS player-count trend"):
            try:
                st.json(player_count_trend(data, video))
            except (ImportError, ValueError) as exc:
                st.warning(str(exc))

    numeric = data.select_dtypes(include="number")
    if numeric.shape[1] >= 2:
        correlations = correlation_matrix(data)
        st.plotly_chart(correlation_heatmap(correlations), use_container_width=True)
        st.subheader("IQR outlier flags")
        st.dataframe(flag_iqr_outliers(data), use_container_width=True)
    else:
        st.info("At least two numeric columns are required for correlations.")

    exports = available_detection_exports()
    if exports:
        st.subheader("Per-object detection analysis")
        selected = st.selectbox(
            "Detection export",
            exports,
            format_func=lambda path: path.name,
            key="statistics_detection_export",
        )
        try:
            detections = pd.read_csv(selected)
            st.plotly_chart(confidence_distribution(detections), use_container_width=True)
            st.plotly_chart(player_spatial_distribution(detections), use_container_width=True)
        except (OSError, pd.errors.ParserError, ValueError) as exc:
            st.warning(f"Could not plot detection export: {exc}")
