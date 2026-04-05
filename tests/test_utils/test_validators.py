"""Tests for data validation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.validators import (
    DataValidationError,
    check_missing_data,
    validate_no_lookahead,
    validate_ohlcv,
)


# ─────────────────────────────────────────────────────────────
# validate_ohlcv
# ─────────────────────────────────────────────────────────────

class TestValidateOhlcv:
    """OHLCV DataFrame validation."""

    @pytest.fixture
    def valid_ohlcv(self) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        return pd.DataFrame(
            {
                "Open": np.random.uniform(90, 110, 10),
                "High": np.random.uniform(100, 120, 10),
                "Low": np.random.uniform(80, 100, 10),
                "Close": np.random.uniform(90, 110, 10),
                "Volume": np.random.randint(1000, 10000, 10),
            },
            index=dates,
        )

    def test_valid_data_passes(self, valid_ohlcv: pd.DataFrame) -> None:
        # Ensure High >= Low for validity
        valid_ohlcv["High"] = valid_ohlcv[["Open", "High", "Low", "Close"]].max(axis=1)
        valid_ohlcv["Low"] = valid_ohlcv[["Open", "High", "Low", "Close"]].min(axis=1)
        result = validate_ohlcv(valid_ohlcv, ticker="TEST")
        assert result is valid_ohlcv

    def test_missing_column_raises(self) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {"Open": [1] * 5, "High": [2] * 5, "Low": [0.5] * 5},
            index=dates,
        )
        with pytest.raises(DataValidationError, match="Missing required"):
            validate_ohlcv(df)

    def test_non_datetime_index_raises(self) -> None:
        df = pd.DataFrame(
            {
                "Open": [1], "High": [2], "Low": [0.5],
                "Close": [1.5], "Volume": [100],
            },
            index=[0],
        )
        with pytest.raises(DataValidationError, match="DatetimeIndex"):
            validate_ohlcv(df)

    def test_unsorted_index_raises(self) -> None:
        dates = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"])
        df = pd.DataFrame(
            {
                "Open": [1, 2, 3], "High": [2, 3, 4],
                "Low": [0.5, 1, 2], "Close": [1.5, 2.5, 3.5],
                "Volume": [100, 200, 300],
            },
            index=dates,
        )
        with pytest.raises(DataValidationError, match="not sorted"):
            validate_ohlcv(df)

    def test_negative_prices_raises(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        df = pd.DataFrame(
            {
                "Open": [1, -1, 3], "High": [2, 3, 4],
                "Low": [0.5, 1, 2], "Close": [1.5, 2.5, 3.5],
                "Volume": [100, 200, 300],
            },
            index=dates,
        )
        with pytest.raises(DataValidationError, match="Negative"):
            validate_ohlcv(df)

    def test_high_less_than_low_raises(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        df = pd.DataFrame(
            {
                "Open": [1, 2, 3], "High": [2, 1, 4],  # row 1: High < Low
                "Low": [0.5, 2, 2], "Close": [1.5, 2.5, 3.5],
                "Volume": [100, 200, 300],
            },
            index=dates,
        )
        with pytest.raises(DataValidationError, match="High < Low"):
            validate_ohlcv(df)


# ─────────────────────────────────────────────────────────────
# validate_no_lookahead
# ─────────────────────────────────────────────────────────────

class TestValidateNoLookahead:
    """Lookahead bias detection."""

    def test_passes_with_valid_data(self) -> None:
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        features = pd.DataFrame({"f1": range(10)}, index=dates)
        target = pd.Series(range(10), index=dates)
        # Should not raise
        validate_no_lookahead(features, target)

    def test_raises_on_future_features(self) -> None:
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        features = pd.DataFrame({"f1": range(10)}, index=dates)
        target = pd.Series(range(10), index=dates)
        max_date = pd.Timestamp("2024-01-07")
        with pytest.raises(DataValidationError, match="rows after"):
            validate_no_lookahead(features, target, max_date=max_date)

    def test_raises_non_datetime_features(self) -> None:
        features = pd.DataFrame({"f1": [1, 2, 3]}, index=[0, 1, 2])
        target = pd.Series([1, 2, 3], index=pd.date_range("2024-01-01", periods=3))
        with pytest.raises(DataValidationError, match="Features must have"):
            validate_no_lookahead(features, target)

    def test_raises_non_datetime_target(self) -> None:
        features = pd.DataFrame(
            {"f1": [1, 2, 3]},
            index=pd.date_range("2024-01-01", periods=3),
        )
        target = pd.Series([1, 2, 3], index=[0, 1, 2])
        with pytest.raises(DataValidationError, match="Target must have"):
            validate_no_lookahead(features, target)


# ─────────────────────────────────────────────────────────────
# check_missing_data
# ─────────────────────────────────────────────────────────────

class TestCheckMissingData:
    """Missing data reporting."""

    def test_no_missing_data(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = check_missing_data(df, ticker="TEST")
        assert result["a"] == 0.0
        assert result["b"] == 0.0

    def test_reports_missing_percentage(self) -> None:
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, 6]})
        result = check_missing_data(df, ticker="TEST")
        assert abs(result["a"] - 1 / 3) < 1e-6
        assert result["b"] == 0.0

    def test_warns_above_threshold(self) -> None:
        df = pd.DataFrame({"a": [1, np.nan, np.nan], "b": [4, 5, 6]})
        with pytest.warns(UserWarning, match="missing data"):
            check_missing_data(df, max_missing_pct=0.05, ticker="TEST")

    def test_no_warning_below_threshold(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        # Should not warn
        result = check_missing_data(df, max_missing_pct=0.05, ticker="TEST")
        assert all(v == 0.0 for v in result.values())
