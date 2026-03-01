"""Quant Stack Dashboard — unified Streamlit entry point.

Launch with:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="Quant Stack", layout="wide")


def overview_page() -> None:
    """Overview page wrapper."""
    from src.dashboard.pages.overview import render_overview
    render_overview()


def execution_page() -> None:
    """Execution page wrapper."""
    from src.dashboard.pages.execution import render_execution_page
    render_execution_page()


pages = {
    "Overview": st.Page(overview_page, title="Overview", icon=":material/dashboard:"),
    "Execution": st.Page(execution_page, title="Execution", icon=":material/swap_horiz:"),
}

nav = st.navigation(list(pages.values()))
nav.run()
