"""Performance logging for smart-dca fill tracking and analysis.

Records every executed fill in a SQLite database and provides
per-asset and per-week performance summaries comparing actual fill
prices against naive baseline prices (Monday 09:00 UTC).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL,
    fill_price REAL NOT NULL,
    amount_gbp REAL NOT NULL,
    quantity REAL NOT NULL,
    naive_price REAL NOT NULL,
    score_at_buy REAL NOT NULL,
    timestamp TEXT NOT NULL
)
"""


@dataclass
class FillRecord:
    """A single executed fill for performance tracking.

    Attributes:
        asset: Trading pair symbol (e.g. BTCGBP).
        fill_price: Actual price achieved for the fill.
        amount_gbp: GBP value of the purchase.
        quantity: Amount of the asset purchased.
        naive_price: Monday 09:00 UTC price for baseline comparison.
        score_at_buy: Composite score at the time of purchase.
        timestamp: ISO format UTC string of the fill time.
    """

    asset: str
    fill_price: float
    amount_gbp: float
    quantity: float
    naive_price: float
    score_at_buy: float
    timestamp: str


class PerformanceLog:
    """Append-only performance log backed by SQLite.

    Stores fill records and provides aggregated performance summaries
    to evaluate whether the smart timing strategy beats a naive
    weekly DCA approach.

    Args:
        db_path: Path to the SQLite database file. Created if it
            does not exist.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_table()
        logger.info("performance_log_initialised", db_path=str(db_path))

    def _get_connection(self) -> sqlite3.Connection:
        """Open a new connection to the SQLite database.

        Returns:
            A sqlite3 Connection with row_factory set to sqlite3.Row.
        """
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self) -> None:
        """Create the fills table if it does not already exist."""
        conn = self._get_connection()
        try:
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()
        finally:
            conn.close()

    def record_fill(self, fill: FillRecord) -> None:
        """Append a fill record to the database.

        Args:
            fill: The fill to record.
        """
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO fills
                    (asset, fill_price, amount_gbp, quantity, naive_price,
                     score_at_buy, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fill.asset,
                    fill.fill_price,
                    fill.amount_gbp,
                    fill.quantity,
                    fill.naive_price,
                    fill.score_at_buy,
                    fill.timestamp,
                ),
            )
            conn.commit()
            logger.info(
                "fill_recorded",
                asset=fill.asset,
                fill_price=fill.fill_price,
                amount_gbp=fill.amount_gbp,
            )
        finally:
            conn.close()

    def get_all_fills(self, asset: str | None = None) -> list[FillRecord]:
        """Return all fills, optionally filtered by asset.

        Results are ordered by timestamp ascending.

        Args:
            asset: If provided, only return fills for this asset.

        Returns:
            List of FillRecord instances ordered by timestamp.
        """
        conn = self._get_connection()
        try:
            if asset is not None:
                rows = conn.execute(
                    "SELECT asset, fill_price, amount_gbp, quantity, "
                    "naive_price, score_at_buy, timestamp "
                    "FROM fills WHERE asset = ? ORDER BY timestamp ASC",
                    (asset,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT asset, fill_price, amount_gbp, quantity, "
                    "naive_price, score_at_buy, timestamp "
                    "FROM fills ORDER BY timestamp ASC",
                ).fetchall()

            return [
                FillRecord(
                    asset=row["asset"],
                    fill_price=row["fill_price"],
                    amount_gbp=row["amount_gbp"],
                    quantity=row["quantity"],
                    naive_price=row["naive_price"],
                    score_at_buy=row["score_at_buy"],
                    timestamp=row["timestamp"],
                )
                for row in rows
            ]
        finally:
            conn.close()

    def compute_performance_summary(self) -> list[dict]:
        """Compute per-asset performance summary.

        Returns a list of dicts, one per asset, containing:
            - asset: The trading pair symbol.
            - total_spent_gbp: Sum of amount_gbp across all fills.
            - total_quantity: Sum of quantity across all fills.
            - avg_fill_price: Weighted average fill price.
            - avg_naive_price: Weighted average naive price.
            - improvement_pct: Percentage improvement over naive price.
            - fill_count: Number of fills for this asset.

        improvement_pct = ((avg_naive - avg_fill) / avg_naive) * 100

        Returns:
            List of per-asset summary dicts.  Empty list if no fills
            are recorded.
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                """
                SELECT
                    asset,
                    SUM(amount_gbp)  AS total_spent_gbp,
                    SUM(quantity)     AS total_quantity,
                    COUNT(*)          AS fill_count
                FROM fills
                GROUP BY asset
                ORDER BY asset ASC
                """,
            ).fetchall()

            if not rows:
                return []

            summaries: list[dict] = []
            for row in rows:
                asset_name = row["asset"]
                total_spent = row["total_spent_gbp"]
                total_qty = row["total_quantity"]
                fill_count = row["fill_count"]

                # Compute quantity-weighted average prices
                price_row = conn.execute(
                    """
                    SELECT
                        SUM(fill_price * quantity) / SUM(quantity) AS avg_fill,
                        SUM(naive_price * quantity) / SUM(quantity) AS avg_naive
                    FROM fills
                    WHERE asset = ?
                    """,
                    (asset_name,),
                ).fetchone()

                avg_fill = price_row["avg_fill"]
                avg_naive = price_row["avg_naive"]

                if avg_naive != 0:
                    improvement_pct = ((avg_naive - avg_fill) / avg_naive) * 100
                else:
                    improvement_pct = 0.0

                summaries.append(
                    {
                        "asset": asset_name,
                        "total_spent_gbp": total_spent,
                        "total_quantity": total_qty,
                        "avg_fill_price": avg_fill,
                        "avg_naive_price": avg_naive,
                        "improvement_pct": improvement_pct,
                        "fill_count": fill_count,
                    }
                )

            return summaries
        finally:
            conn.close()

    def get_weekly_performance(self, weeks: int = 8) -> list[dict]:
        """Return per-week performance comparison for the last N weeks.

        Each entry contains:
            - week_start: ISO date string for the Monday of that week.
            - total_spent_gbp: Sum of amount_gbp in the week.
            - fill_count: Number of fills in the week.
            - avg_fill_price: Quantity-weighted average fill price.
            - avg_naive_price: Quantity-weighted average naive price.
            - improvement_pct: Percentage improvement over naive.

        Args:
            weeks: Number of recent weeks to include (default 8).

        Returns:
            List of per-week summary dicts, ordered oldest first.
            Empty list if no fills are recorded.
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(weeks=weeks)
        cutoff_str = cutoff.isoformat()

        conn = self._get_connection()
        try:
            # Fetch all fills within the window
            rows = conn.execute(
                """
                SELECT asset, fill_price, amount_gbp, quantity,
                       naive_price, score_at_buy, timestamp
                FROM fills
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
                """,
                (cutoff_str,),
            ).fetchall()

            if not rows:
                return []

            # Bucket fills by ISO week
            buckets: dict[str, list[sqlite3.Row]] = {}
            for row in rows:
                ts = datetime.fromisoformat(row["timestamp"])
                # Find the Monday of the week containing this timestamp
                monday = ts - timedelta(days=ts.weekday())
                week_key = monday.strftime("%Y-%m-%d")
                if week_key not in buckets:
                    buckets[week_key] = []
                buckets[week_key].append(row)

            results: list[dict] = []
            for week_start in sorted(buckets.keys()):
                week_rows = buckets[week_start]
                total_spent = sum(r["amount_gbp"] for r in week_rows)
                total_qty = sum(r["quantity"] for r in week_rows)
                fill_count = len(week_rows)

                if total_qty > 0:
                    avg_fill = (
                        sum(r["fill_price"] * r["quantity"] for r in week_rows)
                        / total_qty
                    )
                    avg_naive = (
                        sum(r["naive_price"] * r["quantity"] for r in week_rows)
                        / total_qty
                    )
                else:
                    avg_fill = 0.0
                    avg_naive = 0.0

                if avg_naive != 0:
                    improvement_pct = ((avg_naive - avg_fill) / avg_naive) * 100
                else:
                    improvement_pct = 0.0

                results.append(
                    {
                        "week_start": week_start,
                        "total_spent_gbp": total_spent,
                        "fill_count": fill_count,
                        "avg_fill_price": avg_fill,
                        "avg_naive_price": avg_naive,
                        "improvement_pct": improvement_pct,
                    }
                )

            return results
        finally:
            conn.close()
