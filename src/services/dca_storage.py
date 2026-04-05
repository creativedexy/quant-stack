"""SQLite storage backend for DCA purchase history.

Replaces the flat JSON file approach with proper ACID transactions,
indexing, and thread safety.  The database is created automatically
on first use and migrates existing JSON files if present.

Usage:
    from src.services.dca_storage import DCAStorage
    storage = DCAStorage(Path("data/dca"))
    storage.add("SHEL.L", {"date": "2025-03-01", "price": 2500.0, ...})
    purchases = storage.get_all("SHEL.L")
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS purchases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    price REAL NOT NULL,
    amount_gbp REAL NOT NULL,
    shares REAL NOT NULL,
    note TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE(ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_purchases_ticker ON purchases(ticker);
"""


class DCAStorage:
    """SQLite-backed storage for DCA purchase records.

    Thread-safe: each method opens its own connection.

    Args:
        data_dir: Directory where ``purchases.db`` will be created.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._data_dir / "purchases.db"

        # Create tables
        self._execute_script(_SCHEMA)

        # Auto-migrate from JSON if DB is new but JSON files exist
        if self._is_empty():
            json_files = list(self._data_dir.glob("*.json"))
            if json_files:
                count = self.migrate_from_json(self._data_dir)
                if count > 0:
                    logger.info(
                        "Migrated %d purchases from JSON to SQLite", count,
                    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, ticker: str, record: dict[str, Any]) -> None:
        """Insert a purchase record.

        Args:
            ticker: Instrument ticker.
            record: Dict with ``date``, ``price``, ``amount_gbp``,
                ``shares``, and optionally ``note``.

        Raises:
            sqlite3.IntegrityError: If a purchase already exists for
                this ticker and date.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO purchases "
                "(ticker, date, price, amount_gbp, shares, note, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    ticker,
                    record["date"],
                    record["price"],
                    record["amount_gbp"],
                    record["shares"],
                    record.get("note", ""),
                    now,
                ),
            )

    def get_all(self, ticker: str) -> list[dict[str, Any]]:
        """Return all purchases for a ticker, ordered by date ascending.

        Args:
            ticker: Instrument ticker.

        Returns:
            List of purchase dicts.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT date, price, amount_gbp, shares, note "
                "FROM purchases WHERE ticker = ? ORDER BY date ASC",
                (ticker,),
            ).fetchall()

        return [
            {
                "date": row[0],
                "price": row[1],
                "amount_gbp": row[2],
                "shares": row[3],
                "note": row[4],
            }
            for row in rows
        ]

    def get_all_tickers(self) -> list[str]:
        """Return distinct tickers with at least one purchase.

        Returns:
            Sorted list of ticker strings.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT ticker FROM purchases ORDER BY ticker",
            ).fetchall()
        return [row[0] for row in rows]

    def delete(self, ticker: str, date: str) -> bool:
        """Remove a purchase record.

        Args:
            ticker: Instrument ticker.
            date: Purchase date (``YYYY-MM-DD``).

        Returns:
            ``True`` if a row was deleted, ``False`` if not found.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM purchases WHERE ticker = ? AND date = ?",
                (ticker, date),
            )
            return cursor.rowcount > 0

    def migrate_from_json(self, json_dir: Path) -> int:
        """Import purchases from JSON files into SQLite.

        Reads ``{safe_ticker}.json`` files (as written by the old
        JSON storage) and inserts them.  Duplicates are silently
        skipped (``INSERT OR IGNORE``).

        Args:
            json_dir: Directory containing the JSON files.

        Returns:
            Number of records inserted.
        """
        count = 0
        for json_file in sorted(json_dir.glob("*.json")):
            # Reverse the safe-name transform: _ -> . and idx_ -> ^
            stem = json_file.stem
            ticker = stem.replace("idx_", "^").replace("_", ".")

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Skipping %s during migration: %s", json_file, exc,
                )
                continue

            if not isinstance(records, list):
                continue

            now = datetime.now(timezone.utc).isoformat()
            with self._connect() as conn:
                for rec in records:
                    try:
                        conn.execute(
                            "INSERT OR IGNORE INTO purchases "
                            "(ticker, date, price, amount_gbp, shares, "
                            "note, created_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (
                                ticker,
                                rec["date"],
                                rec["price"],
                                rec["amount_gbp"],
                                rec["shares"],
                                rec.get("note", ""),
                                now,
                            ),
                        )
                        count += 1
                    except (KeyError, sqlite3.Error) as exc:
                        logger.warning(
                            "Skipping record in %s: %s", json_file, exc,
                        )

        return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open a new connection (thread-safe pattern)."""
        conn = sqlite3.connect(
            str(self._db_path), check_same_thread=False,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _execute_script(self, script: str) -> None:
        """Run a multi-statement script for schema creation."""
        with self._connect() as conn:
            conn.executescript(script)

    def _is_empty(self) -> bool:
        """Return True if the purchases table has no rows."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM purchases",
            ).fetchone()
            return row[0] == 0
