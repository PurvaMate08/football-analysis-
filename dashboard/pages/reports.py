"""Browse generated analytical and model reports."""

from pathlib import Path

import streamlit as st

from config import REPORTS_DIR


def render() -> None:
    st.header("Reports")
    files = sorted(
        path for path in REPORTS_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".png", ".html"}
    )
    if not files:
        st.info("No reports have been generated yet.")
        return
    selected = st.selectbox("Report file", files, format_func=lambda path: str(path.relative_to(REPORTS_DIR)))
    if selected.suffix.lower() == ".csv":
        import pandas as pd
        st.dataframe(pd.read_csv(selected), use_container_width=True)
    elif selected.suffix.lower() == ".json":
        st.json(selected.read_text(encoding="utf-8"))
    elif selected.suffix.lower() == ".png":
        st.image(str(selected))
    elif selected.suffix.lower() == ".html":
        st.download_button("Download HTML report", selected.read_bytes(), file_name=selected.name)
