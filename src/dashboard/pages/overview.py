"""Overview page — live price display and portfolio summary.

This is the main dashboard page. It shows live prices for all configured
tickers, with source badges, auto-refresh, and source status indicators.

Run with: streamlit run src/dashboard/pages/overview.py
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
