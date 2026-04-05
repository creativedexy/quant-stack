"""Shared utility functions for web route handlers."""

from __future__ import annotations

from typing import Any


def get_tab_statuses(
    settings: dict[str, Any],
    signal_count: int = 0,
    unread_news: int = 0,
) -> dict[str, str]:
    """Compute status dot states for the tab bar.

    Returns a dict mapping tab names to "live", "action", or "idle".
    """
    exec_mode = settings.get("execution", {}).get("mode", "paper")
    return {
        "watchlist": "live",
        "chart": "live",
        "portfolio": "action" if signal_count > 0 else "idle",
        "research": "idle",
        "news": "live" if unread_news > 0 else "idle",
        "execution": "live" if exec_mode == "live" else "idle",
    }
