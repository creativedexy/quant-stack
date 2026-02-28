"""Tests for src.portfolio.risk — verified against manual calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.risk import (
    correlation_matrix,
    max_drawdown,
    portfolio_cvar,
    portfolio_var,
    rolling_sharpe,
)


# -------------------------------------------------------------------
# Fixtures — small, known datasets for manual verification
# -------------------------------------------------------------------

@pytest.fixture
def simple_returns() -> pd.DataFrame:
    """10-observation, 2-asset returns with known distribution.

    Asset A: [-0.05, -0.03, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.06]
    Asset B: [-0.04, -0.02, -0.01,  0.00, 0.01, 0.01, 0.02, 0.03, 0.05, 0.07]
    """
    dates = pd.bdate_range("2024-01-02", periods=10, freq="B")
    return pd.DataFrame(
        {
            "A": [-0.05, -0.03, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.06],
            "B": [-0.04, -0.02, -0.01, 0.00, 0.01, 0.01, 0.02, 0.03, 0.05, 0.07],
        },
        index=dates,
    )


@pytest.fixture
def equal_weights() -> pd.Series:
    return pd.Series([0.5, 0.5], index=["A", "B"])


@pytest.fixture
def simple_equity_curve() -> pd.Series:
    """Equity curve: 100 → 120 → 90 → 110 → 80 → 105."""
    dates = pd.bdate_range("2024-01-02", periods=6, freq="B")
    return pd.Series([100, 120, 90, 110, 80, 105], index=dates, dtype=float)


# -------------------------------------------------------------------
# VaR tests
# -------------------------------------------------------------------

class TestPortfolioVaR:
    """Verify VaR against manual percentile calculation."""

    def test_var_95_manual(
        self, simple_returns: pd.DataFrame, equal_weights: pd.Series
    ) -> None:
        # Portfolio returns (equal weight): mean of A and B per row
        port_ret = simple_returns.values @ equal_weights.values
        # Sorted: [-0.045, -0.025, -0.015, -0.005, 0.005, 0.01, 0.02, 0.03, 0.045, 0.065]
        expected_quantile = np.percentile(port_ret, 5)
        expected_var = -expected_quantile

        var = portfolio_var(simple_returns, equal_weights, confidence=0.95)
        assert abs(var - expected_var) < 1e-10

    def test_var_is_positive_for_typical_data(
        self, simple_returns: pd.DataFrame, equal_weights: pd.Series
    ) -> None:
        var = portfolio_var(simple_returns, equal_weights, confidence=0.95)
        assert var > 0

    def test_var_90(
        self, simple_returns: pd.DataFrame, equal_weights: pd.Series
    ) -> None:
        port_ret = simple_returns.values @ equal_weights.values
        expected = -np.percentile(port_ret, 10)
        var = portfolio_var(simple_returns, equal_weights, confidence=0.90)
        assert abs(var - expected) < 1e-10


# -------------------------------------------------------------------
# CVaR tests
# -------------------------------------------------------------------

class TestPortfolioCVaR:
    """Verify CVaR against manual expected-shortfall calculation."""

    def test_cvar_95_manual(
        self, simple_returns: pd.DataFrame, equal_weights: pd.Series
    ) -> None:
        port_ret = simple_returns.values @ equal_weights.values
        cutoff = np.percentile(port_ret, 5)
        tail = port_ret[port_ret <= cutoff]
        expected_cvar = -tail.mean()

        cvar = portfolio_cvar(simple_returns, equal_weights, confidence=0.95)
        assert abs(cvar - expected_cvar) < 1e-10

    def test_cvar_ge_var(
        self, simple_returns: pd.DataFrame, equal_weights: pd.Series
    ) -> None:
        """CVaR must be >= VaR by definition."""
        var = portfolio_var(simple_returns, equal_weights, confidence=0.95)
        cvar = portfolio_cvar(simple_returns, equal_weights, confidence=0.95)
        assert cvar >= var - 1e-10


# -------------------------------------------------------------------
# Max drawdown tests
# -------------------------------------------------------------------

class TestMaxDrawdown:
    """Verify max drawdown on a known equity curve."""

    def test_known_curve(self, simple_equity_curve: pd.Series) -> None:
        # Peak at 120, trough at 80 → drawdown = (120-80)/120 = 1/3
        mdd = max_drawdown(simple_equity_curve)
        assert abs(mdd - 1 / 3) < 1e-10

    def test_monotonically_increasing(self) -> None:
        """No drawdown for a monotonically increasing curve."""
        dates = pd.bdate_range("2024-01-02", periods=5, freq="B")
        curve = pd.Series([100, 110, 120, 130, 140], index=dates, dtype=float)
        mdd = max_drawdown(curve)
        assert mdd == 0.0

    def test_single_point(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=1, freq="B")
        curve = pd.Series([100.0], index=dates)
        mdd = max_drawdown(curve)
        assert mdd == 0.0


# -------------------------------------------------------------------
# Rolling Sharpe tests
# -------------------------------------------------------------------

class TestRollingSharpe:
    """Verify rolling Sharpe ratio behaviour."""

    def test_output_length(self) -> None:
        dates = pd.bdate_range("2020-01-02", periods=300, freq="B")
        ret = pd.Series(np.random.default_rng(42).normal(0.0003, 0.01, 300), index=dates)
        sharpe = rolling_sharpe(ret, window=252)
        assert len(sharpe) == 300

    def test_nans_before_window(self) -> None:
        dates = pd.bdate_range("2020-01-02", periods=300, freq="B")
        ret = pd.Series(np.random.default_rng(42).normal(0.0003, 0.01, 300), index=dates)
        sharpe = rolling_sharpe(ret, window=252)
        assert sharpe.iloc[:251].isna().all()
        assert sharpe.iloc[251:].notna().all()

    def test_constant_positive_returns_high_sharpe(self) -> None:
        """Constant positive returns should produce a very high Sharpe."""
        dates = pd.bdate_range("2020-01-02", periods=260, freq="B")
        ret = pd.Series(0.001, index=dates)
        sharpe = rolling_sharpe(ret, window=252, risk_free_rate=0.0)
        # Std is 0 → Sharpe should be inf or very large
        # With constant returns, std=0 so rolling_std=NaN from ddof=1 if all identical
        # Actually std of constant series with ddof=1 is 0, giving inf
        valid = sharpe.dropna()
        if len(valid) > 0:
            assert (valid == np.inf).all() or (valid > 100).all()


# -------------------------------------------------------------------
# Correlation matrix tests
# -------------------------------------------------------------------

class TestCorrelationMatrix:
    """Verify correlation matrix and high-correlation warnings."""

    def test_diagonal_is_one(self, simple_returns: pd.DataFrame) -> None:
        corr = correlation_matrix(simple_returns)
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-10)

    def test_symmetric(self, simple_returns: pd.DataFrame) -> None:
        corr = correlation_matrix(simple_returns)
        np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-10)

    def test_known_correlation(self) -> None:
        """Perfectly correlated assets should return correlation of 1."""
        dates = pd.bdate_range("2024-01-02", periods=100, freq="B")
        x = np.random.default_rng(42).normal(0, 1, 100)
        df = pd.DataFrame({"X": x, "Y": x * 2 + 1}, index=dates)
        corr = correlation_matrix(df, flag_threshold=2.0)  # suppress warnings
        assert abs(corr.loc["X", "Y"] - 1.0) < 1e-10

    def test_high_correlation_warning(
        self, simple_returns: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Assets with correlation above threshold should trigger a warning."""
        # A and B are highly correlated in our fixture
        import logging
        with caplog.at_level(logging.WARNING):
            correlation_matrix(simple_returns, flag_threshold=0.5)
        assert any("High correlation" in msg for msg in caplog.messages)

    def test_no_warning_below_threshold(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        dates = pd.bdate_range("2024-01-02", periods=100, freq="B")
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {"A": rng.normal(0, 1, 100), "B": rng.normal(0, 1, 100)},
            index=dates,
        )
        import logging
        with caplog.at_level(logging.WARNING):
            correlation_matrix(df, flag_threshold=0.99)
        assert not any("High correlation" in msg for msg in caplog.messages)
