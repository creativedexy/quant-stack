"""Reusable Streamlit components for the dashboard.

Provides small, composable UI elements that can be used across pages.
All components assume Streamlit is available in the calling context.

Usage:
    from src.dashboard.components import price_with_source, auto_refresh_toggle
    price_with_source(187.42, "yfinance", delayed=True)
"""

from __future__ import annotations

from datetime import datetime


def price_with_source(
    price: float,
    source: str,
    delayed: bool = False,
) -> None:
    """Display a price with a source badge next to it.

    Renders the price value with a coloured badge indicating the data source.
    If the price is delayed, a small indicator is shown.

    Args:
        price: Current price value.
        source: Data source name ('ib', 'yfinance', 'alpha_vantage', 'cached').
        delayed: Whether the price is delayed (not real-time).
    """
    import streamlit as st
    import math

    source_colours = {
        "ib": "green",
        "yfinance": "blue",
        "alpha_vantage": "orange",
        "cached": "red",
        "unavailable": "grey",
    }
    colour = source_colours.get(source, "grey")

    source_labels = {
        "ib": "IB Real-time",
        "yfinance": "Yahoo",
        "alpha_vantage": "Alpha Vantage",
        "cached": "Cached",
        "unavailable": "N/A",
    }
    label = source_labels.get(source, source)

    delay_marker = " ~" if delayed else ""

    if math.isnan(price):
        st.markdown(f"**--** :{colour}[{label}]{delay_marker}")
    else:
        st.markdown(f"**{price:,.2f}** :{colour}[{label}]{delay_marker}")


def auto_refresh_toggle(key: str = "auto_refresh") -> tuple[bool, int]:
    """Render an auto-refresh toggle and interval selector.

    Args:
        key: Streamlit widget key prefix to avoid conflicts.

    Returns:
        Tuple of (auto_refresh_enabled, interval_seconds).
    """
    import streamlit as st

    col1, col2 = st.columns([1, 2])
    with col1:
        enabled = st.toggle("Auto-refresh", value=False, key=f"{key}_toggle")
    with col2:
        interval = st.select_slider(
            "Interval (s)",
            options=[10, 30, 60, 120, 300],
            value=60,
            key=f"{key}_interval",
            disabled=not enabled,
        )

    return enabled, int(interval)


def last_updated_indicator(timestamp: datetime | None) -> None:
    """Show a 'Last updated: X seconds ago' indicator.

    Args:
        timestamp: When the data was last fetched. If None, shows 'Never'.
    """
    import streamlit as st
    from datetime import timezone

    if timestamp is None:
        st.caption("Last updated: Never")
        return

    now = datetime.now(tz=timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    delta = now - timestamp
    seconds = int(delta.total_seconds())

    if seconds < 60:
        text = f"{seconds}s ago"
    elif seconds < 3600:
        text = f"{seconds // 60}m ago"
    else:
        text = f"{seconds // 3600}h ago"

    st.caption(f"Last updated: {text}")


def source_status_badges(status: dict[str, dict]) -> None:
    """Display availability badges for all configured price sources.

    Args:
        status: Dict from LivePriceService.get_price_source_status().
    """
    import streamlit as st

    cols = st.columns(len(status))
    for col, (source, info) in zip(cols, status.items()):
        with col:
            icon = "🟢" if info["available"] else "🔴"
            st.markdown(f"{icon} **{source}**")
            st.caption(info["detail"])
