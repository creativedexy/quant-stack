"""Tests for target variable construction (src.models.targets).

Covers:
- Direction target is binary (only 0 and 1)
- Return target has NaN only at the end (last horizon rows)
- align_features_and_target drops NaN rows and preserves alignment
- Aligned output has identical DatetimeIndex in both X and y
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.targets import (
    create_direction_target,
    create_return_target,
    align_features_and_target,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def prices() -> pd.Series:
    """200 rows of synthetic prices with DatetimeIndex."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=200, freq="B")
    returns = 1.0 + rng.standard_normal(200) * 0.01
    price = 100.0 * np.cumprod(returns)
    return pd.Series(price, index=dates, name="Close")


@pytest.fixture
def features(prices: pd.Series) -> pd.DataFrame:
    """Simple feature DataFrame aligned to prices (some initial NaN)."""
    rng = np.random.default_rng(99)
    df = pd.DataFrame(
        {
            "feat_a": rng.standard_normal(len(prices)),
            "feat_b": rng.standard_normal(len(prices)),
        },
        index=prices.index,
    )
    # Introduce NaN in first 10 rows (simulate warm-up period)
    df.iloc[:10, 0] = np.nan
    return df


# ------------------------------------------------------------------
# create_direction_target
# ------------------------------------------------------------------

class TestCreateDirectionTarget:

    def test_is_binary(self, prices):
        target = create_direction_target(prices, horizon=5)
        valid = target.dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})

    def test_nan_at_end(self, prices):
        horizon = 5
        target = create_direction_target(prices, horizon=horizon)
        # Last `horizon` rows must be NaN
        assert target.iloc[-horizon:].isna().all()
        # All rows before that should be non-NaN
        assert target.iloc[:-horizon].notna().all()

    def test_length_matches_input(self, prices):
        target = create_direction_target(prices, horizon=5)
        assert len(target) == len(prices)

    def test_index_matches_input(self, prices):
        target = create_direction_target(prices, horizon=5)
        assert (target.index == prices.index).all()

    def test_named_correctly(self, prices):
        target = create_direction_target(prices, horizon=10)
        assert target.name == "direction_10d"

    def test_horizon_1(self, prices):
        target = create_direction_target(prices, horizon=1)
        assert target.iloc[-1:].isna().all()
        assert target.iloc[:-1].notna().all()

    def test_known_values(self):
        """Verify direction against manually computed values."""
        dates = pd.bdate_range("2024-01-02", periods=6, freq="B")
        prices = pd.Series([100, 102, 98, 105, 101, 99], index=dates)
        target = create_direction_target(prices, horizon=2)
        # day 0: future = 98, current = 100 → 0 (lower)
        assert target.iloc[0] == 0.0
        # day 1: future = 105, current = 102 → 1 (higher)
        assert target.iloc[1] == 1.0
        # day 2: future = 101, current = 98 → 1 (higher)
        assert target.iloc[2] == 1.0
        # day 3: future = 99, current = 105 → 0 (lower)
        assert target.iloc[3] == 0.0
        # last 2 are NaN
        assert target.iloc[-2:].isna().all()


# ------------------------------------------------------------------
# create_return_target
# ------------------------------------------------------------------

class TestCreateReturnTarget:

    def test_nan_at_end(self, prices):
        horizon = 5
        target = create_return_target(prices, horizon=horizon)
        assert target.iloc[-horizon:].isna().all()
        assert target.iloc[:-horizon].notna().all()

    def test_length_matches_input(self, prices):
        target = create_return_target(prices, horizon=5)
        assert len(target) == len(prices)

    def test_log_returns(self, prices):
        target = create_return_target(prices, horizon=5, log=True)
        valid = target.dropna()
        # Log returns should be roughly in a reasonable range
        assert valid.abs().max() < 1.0  # Less than 100% for daily-ish data

    def test_simple_returns(self, prices):
        target = create_return_target(prices, horizon=5, log=False)
        valid = target.dropna()
        assert valid.abs().max() < 1.0

    def test_named_correctly(self, prices):
        target = create_return_target(prices, horizon=10)
        assert target.name == "fwd_ret_10d"

    def test_known_values_log(self):
        dates = pd.bdate_range("2024-01-02", periods=4, freq="B")
        prices = pd.Series([100.0, 110.0, 105.0, 120.0], index=dates)
        target = create_return_target(prices, horizon=2, log=True)
        expected_0 = np.log(105.0 / 100.0)
        assert target.iloc[0] == pytest.approx(expected_0)
        expected_1 = np.log(120.0 / 110.0)
        assert target.iloc[1] == pytest.approx(expected_1)
        assert target.iloc[-2:].isna().all()

    def test_known_values_simple(self):
        dates = pd.bdate_range("2024-01-02", periods=4, freq="B")
        prices = pd.Series([100.0, 110.0, 105.0, 120.0], index=dates)
        target = create_return_target(prices, horizon=2, log=False)
        expected_0 = (105.0 / 100.0) - 1.0
        assert target.iloc[0] == pytest.approx(expected_0)

    def test_continuous_values(self, prices):
        """Return target should be continuous, not binary."""
        target = create_return_target(prices, horizon=5)
        valid = target.dropna()
        assert valid.nunique() > 2


# ------------------------------------------------------------------
# align_features_and_target
# ------------------------------------------------------------------

class TestAlignFeaturesAndTarget:

    def test_drops_nan_rows(self, features, prices):
        target = create_direction_target(prices, horizon=5)
        X, y = align_features_and_target(features, target)
        assert X.isna().sum().sum() == 0
        assert y.isna().sum() == 0

    def test_identical_index(self, features, prices):
        target = create_direction_target(prices, horizon=5)
        X, y = align_features_and_target(features, target)
        assert (X.index == y.index).all()

    def test_fewer_rows_than_input(self, features, prices):
        target = create_direction_target(prices, horizon=5)
        X, y = align_features_and_target(features, target)
        # Should have dropped: 10 warm-up NaN rows + 5 trailing NaN
        assert len(X) < len(features)

    def test_preserves_feature_columns(self, features, prices):
        target = create_direction_target(prices, horizon=5)
        X, y = align_features_and_target(features, target)
        assert list(X.columns) == list(features.columns)

    def test_target_name_preserved(self, features, prices):
        target = create_direction_target(prices, horizon=5)
        _, y = align_features_and_target(features, target)
        assert y.name == target.name

    def test_inner_join_semantics(self):
        """Only dates present in BOTH features and target survive."""
        dates_a = pd.bdate_range("2024-01-02", periods=10, freq="B")
        dates_b = pd.bdate_range("2024-01-05", periods=10, freq="B")
        feat = pd.DataFrame({"f1": range(10)}, index=dates_a)
        target = pd.Series(range(10), index=dates_b, name="target")
        X, y = align_features_and_target(feat, target)
        # Overlap is the intersection of the two date ranges
        expected_dates = dates_a.intersection(dates_b)
        assert len(X) == len(expected_dates)
        assert (X.index == expected_dates).all()
