"""DCA (Dollar Cost Averaging) planning service.

Manages purchase history, computes entry quality metrics, and provides
chart data for the production UI DCA planner.

Purchase records are stored in SQLite via :class:`DCAStorage`.
Prices are in pence; monetary amounts are in GBP.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from src.services.dca_storage import DCAStorage
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Entry quality thresholds based on RSI at purchase date
_QUALITY_THRESHOLDS: list[tuple[float, str]] = [
    (35.0, "best"),
    (45.0, "good"),
    (60.0, "neutral"),
]
_QUALITY_DEFAULT = "late"

# Dot colours keyed by quality label
QUALITY_COLOURS: dict[str, str] = {
    "best": "#22c55e",
    "good": "rgba(34,197,94,0.6)",
    "neutral": "#f59e0b",
    "late": "#ef4444",
}


def _classify_entry_quality(rsi: float | None) -> str:
    """Return quality label from RSI value.

    Thresholds (inclusive upper bound):
    RSI < 35   -> best
    35 <= RSI < 45  -> good
    45 <= RSI <= 60 -> neutral
    RSI > 60   -> late
    """
    if rsi is None:
        return "neutral"
    for threshold, label in _QUALITY_THRESHOLDS:
        if rsi < threshold:
            return label
    # Check inclusive upper bound for last band
    if rsi <= 60.0:
        return "neutral"
    return _QUALITY_DEFAULT


class DCAService:
    """DCA planning, history analysis, and AI-limit recommendation.

    Args:
        config: Application configuration dict.
        chart_service: ChartService instance (for RSI look-up).
        data_service: DataService instance (for historical prices).
        market_data_service: MarketDataService instance (for current prices).
        storage: Optional DCAStorage instance.  If ``None``, a default
            SQLite-backed storage is created in the data directory.
    """

    def __init__(
        self,
        config: dict[str, Any],
        chart_service: Any,
        data_service: Any,
        market_data_service: Any,
        storage: DCAStorage | None = None,
    ) -> None:
        self._config = config
        self._chart_service = chart_service
        self._data_service = data_service
        self._market_data_service = market_data_service

        data_dir = config.get("general", {}).get("data_dir", "data")
        self._dca_dir = Path(data_dir) / "dca"
        self._dca_dir.mkdir(parents=True, exist_ok=True)

        self._storage = storage or DCAStorage(self._dca_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_purchase(
        self,
        ticker: str,
        date: str,
        price: float,
        amount_gbp: float,
        note: str = "",
    ) -> dict[str, Any]:
        """Record a new DCA purchase.

        Args:
            ticker: Instrument ticker (e.g. ``"SHEL.L"``).
            date: Purchase date as ``YYYY-MM-DD``.
            price: Purchase price in pence.
            amount_gbp: Amount invested in GBP.
            note: Optional free-text note.

        Returns:
            The new purchase record dict.

        Raises:
            ValueError: If a purchase already exists for this date.
        """
        # Compute shares: price is in pence, amount in GBP
        shares = math.floor(amount_gbp / (price / 100.0))

        record: dict[str, Any] = {
            "date": date,
            "price": round(price, 2),
            "amount_gbp": round(amount_gbp, 2),
            "shares": shares,
            "note": note,
        }

        import sqlite3

        try:
            self._storage.add(ticker, record)
        except sqlite3.IntegrityError:
            raise ValueError(
                f"Purchase already recorded for {ticker} on {date}"
            )

        return record

    def get_purchase_history(self, ticker: str) -> list[dict[str, Any]]:
        """Load purchase history with enrichment.

        Each purchase is enriched with ``current_value``, ``return_pct``,
        ``return_gbp``, ``entry_quality``, and ``rsi_at_purchase``.

        Args:
            ticker: Instrument ticker.

        Returns:
            List of enriched purchase dicts, sorted by date ascending.
        """
        purchases = self._storage.get_all(ticker)
        if not purchases:
            return []

        current_price = self._get_current_price(ticker)

        # Get RSI series for quality assessment
        rsi_lookup = self._build_rsi_lookup(ticker)

        enriched: list[dict[str, Any]] = []
        for p in purchases:
            shares = p["shares"]
            entry_price_gbp = p["price"] / 100.0
            current_price_gbp = current_price / 100.0

            current_value = round(shares * current_price_gbp, 2)
            cost_basis = round(shares * entry_price_gbp, 2)
            return_gbp = round(current_value - cost_basis, 2)
            return_pct = round(
                (current_value / cost_basis - 1.0) if cost_basis > 0 else 0.0,
                4,
            )

            rsi_val = rsi_lookup.get(p["date"])
            quality = _classify_entry_quality(rsi_val)

            enriched.append({
                **p,
                "current_value": current_value,
                "return_pct": return_pct,
                "return_gbp": return_gbp,
                "entry_quality": quality,
                "rsi_at_purchase": round(rsi_val, 2) if rsi_val is not None else None,
            })

        # Sort by date ascending
        enriched.sort(key=lambda x: x["date"])
        return enriched

    def get_dca_summary(self, ticker: str) -> dict[str, Any]:
        """Aggregated DCA statistics.

        Args:
            ticker: Instrument ticker.

        Returns:
            Dict with ``total_invested``, ``current_value``,
            ``total_return_pct``, ``total_return_gbp``,
            ``avg_entry_price``, ``total_shares``,
            ``best_entry``, ``worst_entry``.
        """
        history = self.get_purchase_history(ticker)
        if not history:
            return self._empty_summary(ticker)

        total_invested = sum(p["amount_gbp"] for p in history)
        current_value = sum(p["current_value"] for p in history)
        total_shares = sum(p["shares"] for p in history)
        total_return_gbp = round(current_value - total_invested, 2)
        total_return_pct = round(
            (current_value / total_invested - 1.0) if total_invested > 0 else 0.0,
            4,
        )

        # Weighted average entry price (weighted by shares)
        if total_shares > 0:
            avg_entry_price = round(
                sum(p["price"] * p["shares"] for p in history) / total_shares,
                2,
            )
        else:
            avg_entry_price = 0.0

        # Best / worst entries by return_pct
        best_entry = max(history, key=lambda p: p["return_pct"])
        worst_entry = min(history, key=lambda p: p["return_pct"])

        return {
            "ticker": ticker,
            "total_invested": round(total_invested, 2),
            "current_value": round(current_value, 2),
            "total_return_pct": total_return_pct,
            "total_return_gbp": total_return_gbp,
            "avg_entry_price": avg_entry_price,
            "total_shares": total_shares,
            "best_entry": {
                "date": best_entry["date"],
                "price": best_entry["price"],
                "return_pct": best_entry["return_pct"],
            },
            "worst_entry": {
                "date": worst_entry["date"],
                "price": worst_entry["price"],
                "return_pct": worst_entry["return_pct"],
            },
        }

    def get_recommended_limit(self, ticker: str) -> dict[str, Any]:
        """Compute recommended limit price for next DCA purchase.

        Logic: ``min(52w_high * 0.95, avg_cost_basis * 1.08)``.

        Args:
            ticker: Instrument ticker.

        Returns:
            Dict with ``recommended_limit_pence``,
            ``recommended_limit_display``, ``rationale``,
            ``current_vs_limit``.
        """
        summary = self.get_dca_summary(ticker)
        avg_cost = summary["avg_entry_price"]
        high_52w = self._get_52w_high(ticker)
        current_price = self._get_current_price(ticker)

        # Compute two candidates
        candidate_high = round(high_52w * 0.95, 2)
        candidate_avg = round(avg_cost * 1.08, 2) if avg_cost > 0 else candidate_high

        recommended = min(candidate_high, candidate_avg)

        # Determine rationale
        if candidate_avg <= candidate_high:
            rationale = (
                f"Based on avg cost basis ({avg_cost:.0f}p) + 8% buffer"
            )
        else:
            rationale = (
                f"Based on 52-week high ({high_52w:.0f}p) - 5% discount"
            )

        # Current price vs limit
        if recommended > 0:
            vs_limit = round((current_price / recommended - 1.0), 4)
        else:
            vs_limit = 0.0

        return {
            "recommended_limit_pence": recommended,
            "recommended_limit_display": f"{recommended:.0f}p",
            "rationale": rationale,
            "current_vs_limit": vs_limit,
        }

    def compute_dca_chart_data(self, ticker: str) -> dict[str, Any]:
        """SVG chart data for the DCA planner.

        Args:
            ticker: Instrument ticker.

        Returns:
            Dict with ``price_history``, ``purchase_dots``,
            ``avg_basis_y``, ``svg_price_line``, ``svg_avg_line_y``.
        """
        history = self.get_purchase_history(ticker)

        # Get price history for chart
        try:
            price_data = self._chart_service.get_price_history(ticker, "1Y")
        except Exception:
            price_data = {"dates": [], "close": []}

        dates = price_data.get("dates", [])
        closes = price_data.get("close", [])

        if not dates or not closes:
            return self._empty_chart_data()

        # Build SVG price line
        svg_price_line = _to_svg_points(list(range(len(closes))), closes)

        # Compute average cost basis line
        summary = self.get_dca_summary(ticker)
        avg_price_pence = summary["avg_entry_price"]
        avg_price_gbp = avg_price_pence / 100.0 if avg_price_pence > 0 else None

        # Map avg cost to SVG y coordinate
        svg_avg_line_y = None
        if avg_price_gbp is not None and closes:
            min_v = min(closes)
            max_v = max(closes)
            v_range = max_v - min_v if max_v != min_v else 1.0
            pad_y = 10
            height = 230
            svg_avg_line_y = round(
                pad_y + (1 - (avg_price_gbp - min_v) / v_range) * (height - 2 * pad_y),
                1,
            )

        # Build purchase dots
        purchase_dots: list[dict[str, Any]] = []
        for p in history:
            p_date = p["date"]
            if p_date in dates:
                idx = dates.index(p_date)
                # Map to SVG coordinates
                n = len(closes)
                pad_x = 40
                width = 900
                x = pad_x + (idx / max(n - 1, 1)) * (width - 2 * pad_x)

                price_gbp = p["price"] / 100.0
                min_v = min(closes)
                max_v = max(closes)
                v_range = max_v - min_v if max_v != min_v else 1.0
                pad_y = 10
                height = 230
                y = pad_y + (1 - (price_gbp - min_v) / v_range) * (height - 2 * pad_y)

                purchase_dots.append({
                    "x": round(x, 1),
                    "y": round(y, 1),
                    "date": p_date,
                    "price": p["price"],
                    "quality": p["entry_quality"],
                    "colour": QUALITY_COLOURS.get(p["entry_quality"], "#6b7280"),
                })

        return {
            "price_history": {"dates": dates, "close": closes},
            "purchase_dots": purchase_dots,
            "avg_basis_y": svg_avg_line_y,
            "svg_price_line": svg_price_line,
            "svg_avg_line_y": svg_avg_line_y,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_current_price(self, ticker: str) -> float:
        """Get current price in pence from MarketDataService."""
        try:
            intraday = self._market_data_service.get_intraday_prices(ticker)
            # prev_close is already in the correct unit (pence for LSE)
            return float(intraday.get("current", 0.0))
        except Exception as exc:
            logger.warning(
                "Current price unavailable for %s: %s", ticker, exc,
            )
            return 0.0

    def _get_52w_high(self, ticker: str) -> float:
        """Get 52-week high price in pence."""
        try:
            prices = self._data_service.get_prices(tickers=[ticker])
            if prices.empty or ticker not in prices.columns:
                return 0.0
            series = prices[ticker].dropna().tail(252)
            if series.empty:
                return 0.0
            # Prices from data_service are in the same unit as chart
            # Convert to pence: multiply by 100 if values look like GBP
            high = float(series.max())
            # Synthetic data uses values around 100-200, treat as pence
            return round(high * 100, 2) if high < 50 else round(high, 2)
        except Exception as exc:
            logger.warning("52w high unavailable for %s: %s", ticker, exc)
            return 0.0

    def _build_rsi_lookup(self, ticker: str) -> dict[str, float]:
        """Build date -> RSI mapping from ChartService."""
        try:
            rsi_data = self._chart_service.get_rsi_series(ticker, "All")
            dates = rsi_data.get("dates", [])
            rsi_vals = rsi_data.get("rsi", [])
            lookup: dict[str, float] = {}
            for d, r in zip(dates, rsi_vals):
                if r is not None:
                    lookup[d] = float(r)
            return lookup
        except Exception as exc:
            logger.warning("RSI lookup failed for %s: %s", ticker, exc)
            return {}

    def _empty_summary(self, ticker: str) -> dict[str, Any]:
        """Return empty summary when no purchases exist."""
        return {
            "ticker": ticker,
            "total_invested": 0.0,
            "current_value": 0.0,
            "total_return_pct": 0.0,
            "total_return_gbp": 0.0,
            "avg_entry_price": 0.0,
            "total_shares": 0,
            "best_entry": None,
            "worst_entry": None,
        }

    def _empty_chart_data(self) -> dict[str, Any]:
        """Return empty chart data."""
        return {
            "price_history": {"dates": [], "close": []},
            "purchase_dots": [],
            "avg_basis_y": None,
            "svg_price_line": "",
            "svg_avg_line_y": None,
        }


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _to_svg_points(
    indices: list,
    values: list,
    width: int = 900,
    height: int = 230,
    pad_x: int = 40,
    pad_y: int = 10,
) -> str:
    """Convert index/value pairs into an SVG polyline points string."""
    pairs: list[tuple[int, float]] = []
    for i, v in enumerate(values):
        if v is not None:
            pairs.append((i, float(v)))

    if len(pairs) < 2:
        return ""

    n = len(values)
    vals = [p[1] for p in pairs]
    min_v = min(vals)
    max_v = max(vals)
    v_range = max_v - min_v if max_v != min_v else 1.0

    pts: list[str] = []
    for idx, val in pairs:
        x = pad_x + (idx / max(n - 1, 1)) * (width - 2 * pad_x)
        y = pad_y + (1 - (val - min_v) / v_range) * (height - 2 * pad_y)
        pts.append(f"{x:.1f},{y:.1f}")

    return " ".join(pts)
