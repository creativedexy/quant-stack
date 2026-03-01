"""Overview page — live price display and portfolio summary.

This is the main dashboard page. It shows live prices for all configured
tickers, with source badges, auto-refresh, and source status indicators.

Run with: streamlit run src/dashboard/pages/overview.py
"""Overview dashboard page — pipeline status and portfolio summary.

Displays pipeline health, last run details, and provides manual
trigger controls. Designed for Streamlit but can be imported
independently for status rendering.

Usage (Streamlit):
    streamlit run src/dashboard/pages/overview.py
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import streamlit as st

from src.dashboard.components import (
    auto_refresh_toggle,
    last_updated_indicator,
    price_with_source,
    source_status_badges,
)
from src.services.data_service import DataService


def _get_data_service() -> DataService:
    """Initialise DataService once per Streamlit session."""
    if "data_service" not in st.session_state:
        try:
            from src.utils.config import load_config
            config = load_config()
        except FileNotFoundError:
            config = None
        st.session_state.data_service = DataService(config=config)
    return st.session_state.data_service


def render_overview() -> None:
    """Render the overview page with live prices and controls."""
    st.header("Market Overview")

    svc = _get_data_service()
    tickers = svc.get_tickers()

    # ── Controls row ──────────────────────────────────────
    col_refresh, col_toggle = st.columns([1, 2])
    with col_refresh:
        manual_refresh = st.button("Refresh Now")
    with col_toggle:
        auto_enabled, auto_interval = auto_refresh_toggle()

    # ── Source status ─────────────────────────────────────
    with st.expander("Data source status"):
        status = svc.live_price_service.get_price_source_status()
        source_status_badges(status)

    # ── Fetch prices ──────────────────────────────────────
    st.subheader("Live Prices")

    if not tickers:
        st.warning("No tickers configured. Check config/settings.yaml.")
        return

    prices_df = svc.get_live_prices(tickers)
    last_update_time = datetime.now(tz=timezone.utc)
    st.session_state["last_update_time"] = last_update_time

    # ── Display prices ────────────────────────────────────
    for ticker in tickers:
        if ticker in prices_df.index:
            row = prices_df.loc[ticker]
            col_name, col_price = st.columns([1, 3])
            with col_name:
                st.markdown(f"**{ticker}**")
            with col_price:
                price_with_source(
                    price=row["price"],
                    source=row["source"],
                    delayed=row.get("delayed", True),
                )

    # ── Last updated indicator ────────────────────────────
    last_updated_indicator(st.session_state.get("last_update_time"))

    # ── Auto-refresh loop ─────────────────────────────────
    if auto_enabled:
        time.sleep(auto_interval)
        st.rerun()


if __name__ == "__main__":
    st.set_page_config(page_title="Quant Stack — Overview", layout="wide")
    render_overview()
from datetime import datetime
from typing import Any

from src.services.data_service import DataService
from src.utils.logging import get_logger

logger = get_logger(__name__)


def render_pipeline_status(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Render pipeline status information for the dashboard.

    Reads the last pipeline result and formats it for display.
    Can be used by Streamlit or any other front-end consumer.

    Args:
        config: Project configuration dict.

    Returns:
        Dictionary with formatted status information:
        - last_run_time: human-readable last run timestamp
        - next_scheduled_run: next scheduled time (if scheduler running)
        - last_status: "success" | "partial" | "failed" | "never_run"
        - tickers_updated: count of successfully updated tickers
        - tickers_failed: count of failed tickers
        - duration: formatted duration string
        - errors: list of error messages
    """
    service = DataService(config)
    status = service.get_pipeline_status()

    if status is None:
        return {
            "last_run_time": "Never",
            "next_scheduled_run": "Not scheduled",
            "last_status": "never_run",
            "tickers_updated": 0,
            "tickers_failed": 0,
            "duration": "N/A",
            "errors": [],
        }

    # Format timestamp
    try:
        ts = datetime.fromisoformat(status["timestamp"])
        last_run = ts.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (KeyError, ValueError):
        last_run = "Unknown"

    # Format duration
    duration_s = status.get("duration_seconds", 0)
    if duration_s < 60:
        duration = f"{duration_s:.1f}s"
    else:
        duration = f"{duration_s / 60:.1f}min"

    return {
        "last_run_time": last_run,
        "next_scheduled_run": "Not scheduled",
        "last_status": status.get("status", "unknown"),
        "tickers_updated": len(status.get("tickers_updated", [])),
        "tickers_failed": len(status.get("tickers_failed", [])),
        "duration": duration,
        "errors": status.get("errors", []),
    }


def run_pipeline_now(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Trigger a manual pipeline run (for dashboard button).

    Args:
        config: Project configuration dict.

    Returns:
        Pipeline result dictionary.
    """
    from src.scheduler.pipeline import PipelineRunner

    runner = PipelineRunner(config)
    return runner.run_daily()


# ── Streamlit page (only runs when executed as a Streamlit page) ──

def _streamlit_page() -> None:
    """Render the overview page in Streamlit."""
    try:
        import streamlit as st
    except ImportError:
        logger.info("Streamlit not installed — skipping dashboard render")
        return

    st.set_page_config(page_title="Quant Stack — Overview", layout="wide")
    st.title("Quant Stack — Overview")

    # ── Pipeline Status Expander ───────────────────────────────
    with st.expander("Pipeline Status", expanded=True):
        status = render_pipeline_status()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Last Run", status["last_run_time"])
        with col2:
            st.metric("Status", status["last_status"].upper())
        with col3:
            st.metric("Duration", status["duration"])

        col4, col5 = st.columns(2)
        with col4:
            st.metric("Tickers Updated", status["tickers_updated"])
        with col5:
            st.metric("Tickers Failed", status["tickers_failed"])

        if status["errors"]:
            st.error("Errors from last run:")
            for err in status["errors"]:
                st.text(f"  • {err}")

        if st.button("Run Pipeline Now"):
            with st.spinner("Running pipeline..."):
                result = run_pipeline_now()
            if result["status"] == "success":
                st.success(f"Pipeline completed successfully in {result['duration_seconds']}s")
            elif result["status"] == "partial":
                st.warning(
                    f"Pipeline partially succeeded. "
                    f"Failed: {result['tickers_failed']}"
                )
            else:
                st.error(f"Pipeline failed: {result['errors']}")
            st.rerun()


if __name__ == "__main__":
    _streamlit_page()
