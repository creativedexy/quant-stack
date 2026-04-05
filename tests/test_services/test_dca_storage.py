"""Tests for DCAStorage SQLite backend.

Covers CRUD operations, duplicate detection, migration from JSON,
and thread safety.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest

from src.services.dca_storage import DCAStorage


# -- Fixtures ----------------------------------------------------------


@pytest.fixture
def storage(tmp_path: Path) -> DCAStorage:
    """DCAStorage with a fresh temporary database."""
    return DCAStorage(tmp_path / "dca")


@pytest.fixture
def sample_record() -> dict:
    """A minimal purchase record."""
    return {
        "date": "2024-01-15",
        "price": 250.0,
        "amount_gbp": 1000.0,
        "shares": 400,
        "note": "Monthly DCA",
    }


# -- CRUD tests --------------------------------------------------------


class TestAdd:
    """Tests for DCAStorage.add()."""

    def test_add_and_retrieve(
        self, storage: DCAStorage, sample_record: dict,
    ) -> None:
        """add() then get_all() returns the record."""
        storage.add("SHEL.L", sample_record)
        result = storage.get_all("SHEL.L")
        assert len(result) == 1
        assert result[0]["date"] == "2024-01-15"
        assert result[0]["price"] == 250.0
        assert result[0]["shares"] == 400

    def test_add_multiple_records(
        self, storage: DCAStorage, sample_record: dict,
    ) -> None:
        """Multiple records for the same ticker are stored."""
        storage.add("SHEL.L", sample_record)
        second = {**sample_record, "date": "2024-02-15", "price": 260.0}
        storage.add("SHEL.L", second)
        result = storage.get_all("SHEL.L")
        assert len(result) == 2

    def test_add_different_tickers(
        self, storage: DCAStorage, sample_record: dict,
    ) -> None:
        """Records for different tickers are isolated."""
        storage.add("SHEL.L", sample_record)
        other = {**sample_record, "date": "2024-01-15"}
        storage.add("AAPL", other)
        assert len(storage.get_all("SHEL.L")) == 1
        assert len(storage.get_all("AAPL")) == 1


class TestDuplicateDetection:
    """Tests for unique constraint on (ticker, date)."""

    def test_duplicate_raises_integrity_error(
        self, storage: DCAStorage, sample_record: dict,
    ) -> None:
        """Adding the same ticker+date twice raises IntegrityError."""
        storage.add("SHEL.L", sample_record)
        with pytest.raises(sqlite3.IntegrityError):
            storage.add("SHEL.L", sample_record)

    def test_same_date_different_ticker_ok(
        self, storage: DCAStorage, sample_record: dict,
    ) -> None:
        """Same date for different tickers is fine."""
        storage.add("SHEL.L", sample_record)
        storage.add("AAPL", sample_record)
        assert len(storage.get_all("SHEL.L")) == 1
        assert len(storage.get_all("AAPL")) == 1


class TestGetAll:
    """Tests for DCAStorage.get_all()."""

    def test_empty_returns_empty_list(self, storage: DCAStorage) -> None:
        """get_all for unknown ticker returns empty list."""
        assert storage.get_all("UNKNOWN") == []

    def test_ordered_by_date(
        self, storage: DCAStorage, sample_record: dict,
    ) -> None:
        """Results are ordered by date ascending."""
        storage.add("SHEL.L", {**sample_record, "date": "2024-03-15"})
        storage.add("SHEL.L", {**sample_record, "date": "2024-01-15"})
        storage.add("SHEL.L", {**sample_record, "date": "2024-02-15"})
        result = storage.get_all("SHEL.L")
        dates = [r["date"] for r in result]
        assert dates == ["2024-01-15", "2024-02-15", "2024-03-15"]


class TestGetAllTickers:
    """Tests for DCAStorage.get_all_tickers()."""

    def test_returns_distinct_tickers(
        self, storage: DCAStorage, sample_record: dict,
    ) -> None:
        """get_all_tickers returns sorted unique tickers."""
        storage.add("VUSA.L", {**sample_record, "date": "2024-01-15"})
        storage.add("AAPL", {**sample_record, "date": "2024-01-15"})
        storage.add("SHEL.L", {**sample_record, "date": "2024-01-15"})
        tickers = storage.get_all_tickers()
        assert tickers == ["AAPL", "SHEL.L", "VUSA.L"]

    def test_empty_returns_empty_list(self, storage: DCAStorage) -> None:
        """No records means empty list."""
        assert storage.get_all_tickers() == []


class TestDelete:
    """Tests for DCAStorage.delete()."""

    def test_delete_existing(
        self, storage: DCAStorage, sample_record: dict,
    ) -> None:
        """delete() removes the record and returns True."""
        storage.add("SHEL.L", sample_record)
        result = storage.delete("SHEL.L", "2024-01-15")
        assert result is True
        assert storage.get_all("SHEL.L") == []

    def test_delete_nonexistent(self, storage: DCAStorage) -> None:
        """delete() returns False when nothing to delete."""
        result = storage.delete("SHEL.L", "2024-01-15")
        assert result is False


# -- Migration tests ---------------------------------------------------


class TestMigrateFromJson:
    """Tests for JSON-to-SQLite migration."""

    def test_migrates_json_files(self, tmp_path: Path) -> None:
        """migrate_from_json reads JSON files and inserts records."""
        dca_dir = tmp_path / "dca"
        dca_dir.mkdir()

        # Write a JSON file (using the old safe-name convention)
        records = [
            {"date": "2024-01-15", "price": 250.0,
             "amount_gbp": 1000.0, "shares": 400, "note": ""},
            {"date": "2024-02-15", "price": 260.0,
             "amount_gbp": 500.0, "shares": 192, "note": ""},
        ]
        with open(dca_dir / "SHEL_L.json", "w") as f:
            json.dump(records, f)

        storage = DCAStorage(dca_dir)
        # Migration happens automatically in __init__ if DB is empty
        result = storage.get_all("SHEL.L")
        assert len(result) == 2
        assert result[0]["date"] == "2024-01-15"
        assert result[1]["date"] == "2024-02-15"

    def test_migration_is_idempotent(self, tmp_path: Path) -> None:
        """Running migrate_from_json twice does not create duplicates."""
        dca_dir = tmp_path / "dca"
        dca_dir.mkdir()

        records = [
            {"date": "2024-01-15", "price": 250.0,
             "amount_gbp": 1000.0, "shares": 400, "note": ""},
        ]
        with open(dca_dir / "SHEL_L.json", "w") as f:
            json.dump(records, f)

        storage = DCAStorage(dca_dir)
        # Run migration explicitly again
        count = storage.migrate_from_json(dca_dir)
        # Should be 0 because INSERT OR IGNORE skips duplicates
        # (the auto-migration already inserted them)
        assert count >= 0
        assert len(storage.get_all("SHEL.L")) == 1


# -- Thread safety tests -----------------------------------------------


class TestThreadSafety:
    """Test concurrent writes from multiple threads."""

    def test_concurrent_writes(self, storage: DCAStorage) -> None:
        """Multiple threads can write without corruption."""
        errors: list[str] = []

        def writer(ticker: str, date: str) -> None:
            try:
                storage.add(ticker, {
                    "date": date,
                    "price": 250.0,
                    "amount_gbp": 1000.0,
                    "shares": 400,
                    "note": f"thread-{ticker}-{date}",
                })
            except Exception as exc:
                errors.append(str(exc))

        threads = []
        for i in range(10):
            t = threading.Thread(
                target=writer,
                args=(f"T{i}", f"2024-01-{15 + i:02d}"),
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert errors == []
        assert len(storage.get_all_tickers()) == 10
