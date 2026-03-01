"""Overview page -- data status, latest prices, and portfolio summary.

Provides a high-level view of available data, current prices,
portfolio risk metrics, and pipeline health.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from src.services.data_service import DataService
from src.services.portfolio_service import PortfolioService

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PIPELINE_STATUS_PATH = _PROJECT_ROOT / "data" / "processed" / "pipeline_status.json"


def _get_services() -> tuple[DataService, PortfolioService]:
    """Initialise services once per Streamlit session."""
    if "data_service" not in st.session_state:
        try:
            from src.utils.config import load_config

            config = load_config()
        except FileNotFoundError:
            config = None
        st.session_state.data_service = DataService(config=config)
        st.session_state.portfolio_service = PortfolioService(
            st.session_state.data_service,
            config=config,
        )
    return st.session_state.data_service, st.session_state.portfolio_service


def _load_pipeline_status() -> dict[str, Any] | None:
    """Read pipeline_status.json if it exists."""
    if _PIPELINE_STATUS_PATH.exists():
        try:
            with open(_PIPELINE_STATUS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def render_overview() -> None:
    """Render the overview page."""
    st.title("Overview")

    data_svc, portfolio_svc = _get_services()

    # ── Data Status ───────────────────────────────────────────
    st.header("Data Status")
    status = data_svc.get_data_status()

    if not status["tickers_available"]:
        st.warning(
            "No data available yet. Run: "
            "`python -m scripts.fetch_data --source synthetic`"
        )
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Tickers Available", len(status["tickers_available"]))
        col2.metric("Data Source", status["data_source"].title())
        date_range = status.get("date_range")
        if date_range:
            col3.metric("Date Range", f"{date_range['start']} to {date_range['end']}")

    # ── Latest Prices ─────────────────────────────────────────
    st.header("Latest Prices")
    latest = data_svc.get_latest_prices()
    if latest.empty:
        st.info("No price data loaded.")
    else:
        st.dataframe(
            latest.to_frame("Price").style.format("£{:.2f}"),
            use_container_width=True,
        )

    # ── Portfolio Risk Metrics ────────────────────────────────
    st.header("Risk Metrics")
    metrics = portfolio_svc.get_risk_metrics()
    col1, col2, col3 = st.columns(3)

    def _fmt_pct(val: float) -> str:
        return f"{val:.2%}" if not np.isnan(val) else "N/A"

    def _fmt_dec(val: float) -> str:
        return f"{val:.2f}" if not np.isnan(val) else "N/A"

    col1.metric("Annual Return", _fmt_pct(metrics["annual_return"]))
    col2.metric("Sharpe Ratio", _fmt_dec(metrics["sharpe_ratio"]))
    col3.metric("Max Drawdown", _fmt_pct(metrics["max_drawdown"]))

    # ── Allocation ────────────────────────────────────────────
    st.header("Current Allocation")
    alloc = portfolio_svc.get_allocation_chart_data()
    if alloc["labels"]:
        try:
            import plotly.express as px

            fig = px.pie(
                names=alloc["labels"],
                values=alloc["values"],
                title="Portfolio Weights",
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.bar_chart(
                data=dict(zip(alloc["labels"], alloc["values"])),
            )
    else:
        st.info("No allocation data available.")

    # ── Pipeline Status ───────────────────────────────────────
    st.header("Pipeline Status")
    pipeline = _load_pipeline_status()
    if pipeline is None:
        st.info("Pipeline has not been run yet.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Status", pipeline.get("status", "unknown").upper())
        col2.metric("Tickers Updated", len(pipeline.get("tickers_updated", [])))
        col3.metric("Duration", f"{pipeline.get('duration_seconds', 0):.1f}s")
        if pipeline.get("errors"):
            with st.expander("Errors"):
                for err in pipeline["errors"]:
                    st.text(f"  * {err}")


if __name__ == "__main__":
    st.set_page_config(page_title="Quant Stack — Overview", layout="wide")
    render_overview()
