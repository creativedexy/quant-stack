"""Tests for the data fetching, cleaning, and synthetic generation modules."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import generate_synthetic_ohlcv, generate_multi_asset_data
from src.data.fetcher import create_fetcher, SyntheticFetcher
from src.data.cleaner import DataCleaner, compute_returns
from src.utils.validators import validate_ohlcv, DataValidationError


# ─────────────────────────────────────────────
# Synthetic Data Tests
# ─────────────────────────────────────────────

class TestSyntheticData:
    """Tests for synthetic OHLCV generation."""

    def test_generates_correct_shape(self):
        df = generate_synthetic_ohlcv(days=100)
        assert len(df) == 100
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_has_datetime_index(self):
        df = generate_synthetic_ohlcv(days=50)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "Date"

    def test_ohlc_relationships_hold(self):
        """High >= max(Open, Close) and Low <= min(Open, Close)."""
        df = generate_synthetic_ohlcv(days=1000, seed=123)
        assert (df["High"] >= df["Open"]).all()
        assert (df["High"] >= df["Close"]).all()
        assert (df["Low"] <= df["Open"]).all()
        assert (df["Low"] <= df["Close"]).all()

    def test_no_negative_prices(self):
        df = generate_synthetic_ohlcv(days=2000, seed=7)
        for col in ["Open", "High", "Low", "Close"]:
            assert (df[col] > 0).all(), f"Negative values in {col}"

    def test_volume_is_positive_integer(self):
        df = generate_synthetic_ohlcv(days=500)
        assert (df["Volume"] > 0).all()
        assert df["Volume"].dtype in [np.int64, np.int32]

    def test_reproducibility(self):
        df1 = generate_synthetic_ohlcv(days=100, seed=42)
        df2 = generate_synthetic_ohlcv(days=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = generate_synthetic_ohlcv(days=100, seed=42)
        df2 = generate_synthetic_ohlcv(days=100, seed=99)
        assert not df1["Close"].equals(df2["Close"])

    def test_multi_asset_generation(self):
        tickers = ["A", "B", "C"]
        data = generate_multi_asset_data(tickers, days=500)
        assert set(data.keys()) == set(tickers)
        for ticker, df in data.items():
            assert len(df) == 500
            assert isinstance(df.index, pd.DatetimeIndex)


# ─────────────────────────────────────────────
# Fetcher Tests
# ─────────────────────────────────────────────

class TestFetcher:
    """Tests for the data fetcher interface."""

    def test_create_synthetic_fetcher(self):
        fetcher = create_fetcher("synthetic")
        assert isinstance(fetcher, SyntheticFetcher)

    def test_create_unknown_fetcher_raises(self):
        with pytest.raises(ValueError, match="Unknown data source"):
            create_fetcher("bloomberg")

    def test_synthetic_fetch_returns_valid_ohlcv(self):
        fetcher = create_fetcher("synthetic")
        df = fetcher.fetch("TEST.L", start="2020-01-01", end="2023-12-31")
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert all(c in df.columns for c in ["Open", "High", "Low", "Close", "Volume"])

    def test_synthetic_fetch_multiple(self):
        fetcher = create_fetcher("synthetic")
        data = fetcher.fetch_multiple(
            ["A", "B", "C"], start="2020-01-01", end="2022-12-31"
        )
        assert len(data) == 3
        assert all(isinstance(df, pd.DataFrame) for df in data.values())

    def test_save_parquet(self, sample_ohlcv):
        fetcher = create_fetcher("synthetic")
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = fetcher.save(
                {"TEST": sample_ohlcv}, Path(tmpdir), fmt="parquet"
            )
            assert len(paths) == 1
            assert paths[0].suffix == ".parquet"
            # Verify we can read it back
            loaded = pd.read_parquet(paths[0])
            assert len(loaded) == len(sample_ohlcv)

    def test_save_csv(self, sample_ohlcv):
        fetcher = create_fetcher("synthetic")
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = fetcher.save(
                {"TEST": sample_ohlcv}, Path(tmpdir), fmt="csv"
            )
            assert len(paths) == 1
            assert paths[0].suffix == ".csv"


# ─────────────────────────────────────────────
# Cleaner Tests
# ─────────────────────────────────────────────

class TestDataCleaner:
    """Tests for the data cleaning pipeline."""

    def test_clean_returns_valid_dataframe(self, sample_ohlcv):
        cleaner = DataCleaner()
        result = cleaner.clean(sample_ohlcv, ticker="TEST")
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_handles_missing_data(self, ohlcv_with_missing):
        cleaner = DataCleaner(fill_method="ffill")
        result = cleaner.clean(ohlcv_with_missing, ticker="TEST")
        assert not result.isnull().any().any(), "Should have no NaN after cleaning"

    def test_handles_missing_data_interpolation(self, ohlcv_with_missing):
        cleaner = DataCleaner(fill_method="interpolate")
        result = cleaner.clean(ohlcv_with_missing, ticker="TEST")
        assert not result.isnull().any().any()

    def test_removes_duplicate_dates(self, sample_ohlcv):
        # Create duplicates
        df = pd.concat([sample_ohlcv, sample_ohlcv.iloc[:5]])
        cleaner = DataCleaner()
        result = cleaner.clean(df, ticker="TEST")
        assert not result.index.duplicated().any()

    def test_sorts_chronologically(self, sample_ohlcv):
        # Reverse the order
        df = sample_ohlcv.iloc[::-1]
        cleaner = DataCleaner()
        result = cleaner.clean(df, ticker="TEST")
        assert result.index.is_monotonic_increasing

    def test_clean_multiple(self, multi_asset_data):
        cleaner = DataCleaner()
        results = cleaner.clean_multiple(multi_asset_data)
        assert len(results) == len(multi_asset_data)
        for ticker, df in results.items():
            assert isinstance(df.index, pd.DatetimeIndex)

    def test_ensures_positive_prices(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        df.loc[df.index[0], "Close"] = -5.0
        df.loc[df.index[0], "Low"] = -10.0
        cleaner = DataCleaner()
        result = cleaner.clean(df, ticker="TEST")
        assert (result["Close"] > 0).all()
        assert (result["Low"] > 0).all()


# ─────────────────────────────────────────────
# Returns Computation Tests
# ─────────────────────────────────────────────

class TestReturns:
    """Tests for return calculations."""

    def test_compute_log_returns(self, sample_ohlcv):
        returns = compute_returns(sample_ohlcv, windows=[1, 5], log_returns=True)
        assert "ret_1d" in returns.columns
        assert "ret_5d" in returns.columns
        # First row should be NaN (no prior data for return calc)
        assert pd.isna(returns["ret_1d"].iloc[0])

    def test_compute_simple_returns(self, sample_ohlcv):
        returns = compute_returns(sample_ohlcv, windows=[1], log_returns=False)
        assert "ret_1d" in returns.columns

    def test_returns_approximately_correct(self):
        """Verify return calculation against manual calculation."""
        dates = pd.bdate_range("2020-01-01", periods=5)
        prices = pd.DataFrame(
            {
                "Open": [100, 102, 104, 103, 105],
                "High": [103, 105, 106, 105, 107],
                "Low": [99, 101, 103, 102, 104],
                "Close": [102, 104, 103, 105, 106],
                "Volume": [1000] * 5,
            },
            index=dates,
        )
        returns = compute_returns(prices, windows=[1], log_returns=False)
        # Day 2 return: (104 - 102) / 102 ≈ 0.0196
        assert abs(returns["ret_1d"].iloc[1] - (104 - 102) / 102) < 1e-10


# ─────────────────────────────────────────────
# Validator Tests
# ─────────────────────────────────────────────

class TestValidators:
    """Tests for data validation utilities."""

    def test_valid_ohlcv_passes(self, sample_ohlcv):
        result = validate_ohlcv(sample_ohlcv, "TEST")
        assert result is not None

    def test_missing_columns_raises(self):
        df = pd.DataFrame(
            {"Close": [100, 101]},
            index=pd.bdate_range("2020-01-01", periods=2),
        )
        with pytest.raises(DataValidationError, match="Missing required columns"):
            validate_ohlcv(df, "BAD")

    def test_non_datetime_index_raises(self):
        df = pd.DataFrame(
            {
                "Open": [100], "High": [105], "Low": [95],
                "Close": [102], "Volume": [1000],
            },
            index=[0],
        )
        with pytest.raises(DataValidationError, match="DatetimeIndex"):
            validate_ohlcv(df, "BAD")

    def test_high_less_than_low_raises(self):
        dates = pd.bdate_range("2020-01-01", periods=2)
        df = pd.DataFrame(
            {
                "Open": [100, 100], "High": [90, 105],  # First row: High < Low
                "Low": [95, 95], "Close": [98, 102], "Volume": [1000, 1000],
            },
            index=dates,
        )
        with pytest.raises(DataValidationError, match="High < Low"):
            validate_ohlcv(df, "BAD")
