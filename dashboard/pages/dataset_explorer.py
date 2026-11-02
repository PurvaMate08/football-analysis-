"""Dataset inspection and download."""

import streamlit as st

from dashboard.pages.common import available_datasets, select_dataset


def render() -> None:
    st.header("Dataset explorer")
    dataset = select_dataset(st, "explorer_dataset")
    if dataset is None:
        return
    st.metric("Observed frame rows", len(dataset))
    st.metric("Video clips", dataset["video_name"].nunique() if "video_name" in dataset else "—")
    st.dataframe(dataset, use_container_width=True)
    choices = available_datasets()
    if choices:
        selected = st.selectbox(
            "Download CSV",
            choices,
            format_func=lambda path: path.name,
            key="download_dataset",
        )
        st.download_button(
            "Download selected dataset",
            data=selected.read_bytes(),
            file_name=selected.name,
            mime="text/csv",
        )
