"""Tests for src.portfolio.risk — verified against manual calculations.

Covers all 9 public functions:
- portfolio_returns
- sharpe_ratio
- sortino_ratio
- max_drawdown
- value_at_risk
- conditional_var
- rolling_sharpe
- correlation_report
- risk_summary
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.risk import (
    conditional_var,
    correlation_report,
    max_drawdown,
    portfolio_returns,
    risk_summary,
    rolling_sharpe,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
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
def daily_returns() -> pd.Series:
    """500 days of synthetic daily returns."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-02", periods=500, freq="B")
    return pd.Series(
        rng.normal(0.0003, 0.015, 500),
        index=dates,
        name="strategy",
    )


# -------------------------------------------------------------------
# portfolio_returns tests
# -------------------------------------------------------------------

class TestPortfolioReturns:

    def test_basic_equal_weight(
        self, simple_returns: pd.DataFrame, equal_weights: pd.Series
    ) -> None:
        port_ret = portfolio_returns(simple_returns, equal_weights)
        expected = simple_returns.values @ equal_weights.values
        np.testing.assert_allclose(port_ret.values, expected)

    def test_returns_series(
        self, simple_returns: pd.DataFrame, equal_weights: pd.Series
    ) -> None:
        port_ret = portfolio_returns(simple_returns, equal_weights)
        assert isinstance(port_ret, pd.Series)
        assert len(port_ret) == len(simple_returns)
        assert (port_ret.index == simple_returns.index).all()

    def test_weights_must_sum_to_one(self, simple_returns: pd.DataFrame) -> None:
        bad_weights = pd.Series([0.3, 0.3], index=["A", "B"])
        with pytest.raises(ValueError, match="sum to ~1.0"):
            portfolio_returns(simple_returns, bad_weights)

    def test_weights_tolerance(self, simple_returns: pd.DataFrame) -> None:
        """Weights summing to 1.005 should be accepted."""
        weights = pd.Series([0.505, 0.500], index=["A", "B"])
        port_ret = portfolio_returns(simple_returns, weights)
        assert len(port_ret) == len(simple_returns)

    def test_missing_ticker_raises(self, simple_returns: pd.DataFrame) -> None:
        weights = pd.Series([0.5, 0.5], index=["A", "C"])
        with pytest.raises(ValueError, match="not found"):
            portfolio_returns(simple_returns, weights)


# -------------------------------------------------------------------
# sharpe_ratio tests
# -------------------------------------------------------------------

class TestSharpeRatio:

    def test_positive_excess_returns(self, daily_returns: pd.Series) -> None:
        sr = sharpe_ratio(daily_returns, risk_free_rate=0.0)
        assert isinstance(sr, float)
        assert np.isfinite(sr)

    def test_zero_returns(self) -> None:
        dates = pd.bdate_range("2020-01-02", periods=100, freq="B")
        ret = pd.Series(0.0, index=dates)
        sr = sharpe_ratio(ret, risk_free_rate=0.0)
        # Zero mean, zero std → should handle gracefully
        assert np.isnan(sr) or sr == 0.0 or np.isinf(sr)

    def test_constant_positive_returns(self) -> None:
        dates = pd.bdate_range("2020-01-02", periods=100, freq="B")
        ret = pd.Series(0.001, index=dates)
        sr = sharpe_ratio(ret, risk_free_rate=0.0)
        assert sr == float("inf") or sr > 100

    def test_higher_returns_higher_sharpe(self) -> None:
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-02", periods=500, freq="B")
        low = pd.Series(rng.normal(0.0001, 0.01, 500), index=dates)
        high = pd.Series(rng.normal(0.001, 0.01, 500), index=dates)
        assert sharpe_ratio(high) > sharpe_ratio(low)


# -------------------------------------------------------------------
# sortino_ratio tests
# -------------------------------------------------------------------

class TestSortinoRatio:

    def test_returns_finite(self, daily_returns: pd.Series) -> None:
        sr = sortino_ratio(daily_returns, risk_free_rate=0.0)
        assert isinstance(sr, float)
        assert np.isfinite(sr)

    def test_all_positive_returns(self) -> None:
        """With no downside, Sortino should be inf."""
        dates = pd.bdate_range("2020-01-02", periods=100, freq="B")
        ret = pd.Series(0.001, index=dates)
        sr = sortino_ratio(ret, risk_free_rate=0.0)
        assert sr == float("inf") or sr > 100

    def test_sortino_ge_sharpe_for_skewed(self) -> None:
        """For positively skewed returns, Sortino >= Sharpe."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-02", periods=500, freq="B")
        # Log-normal-ish returns (right-skewed)
        ret = pd.Series(np.abs(rng.normal(0.001, 0.01, 500)), index=dates)
        sr = sortino_ratio(ret, risk_free_rate=0.0)
        sh = sharpe_ratio(ret, risk_free_rate=0.0)
        assert sr >= sh


# -------------------------------------------------------------------
# max_drawdown tests
# -------------------------------------------------------------------

class TestMaxDrawdown:

    def test_known_returns(self) -> None:
        """Returns that produce a known equity curve."""
        dates = pd.bdate_range("2024-01-02", periods=5, freq="B")
        # Wealth: 1.0 → 1.20 → 0.90 → 1.10 → 0.80
        # Returns: 0.20, -0.25, 0.222..., -0.2727...
        ret = pd.Series([0.20, -0.25, 0.22222222, -0.27272727, 0.0], index=dates)
        result = max_drawdown(ret)
        assert isinstance(result, dict)
        assert "max_drawdown" in result
        assert "peak_date" in result
        assert "trough_date" in result
        assert "recovery_date" in result
        assert result["max_drawdown"] < 0

    def test_monotonically_increasing(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=5, freq="B")
        ret = pd.Series([0.01, 0.02, 0.01, 0.03, 0.02], index=dates)
        result = max_drawdown(ret)
        assert result["max_drawdown"] == 0.0
        assert result["peak_date"] is None
        assert result["trough_date"] is None
        assert result["recovery_date"] is None

    def test_max_drawdown_is_negative(self, daily_returns: pd.Series) -> None:
        result = max_drawdown(daily_returns)
        assert result["max_drawdown"] <= 0

    def test_peak_before_trough(self, daily_returns: pd.Series) -> None:
        result = max_drawdown(daily_returns)
        if result["peak_date"] is not None:
            assert result["peak_date"] <= result["trough_date"]

    def test_recovery_after_trough(self, daily_returns: pd.Series) -> None:
        result = max_drawdown(daily_returns)
        if result["recovery_date"] is not None:
            assert result["recovery_date"] > result["trough_date"]

    def test_empty_returns(self) -> None:
        ret = pd.Series([], dtype=float)
        result = max_drawdown(ret)
        assert result["max_drawdown"] == 0.0


# -------------------------------------------------------------------
# value_at_risk tests
# -------------------------------------------------------------------

class TestValueAtRisk:

    def test_var_95_manual(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=10, freq="B")
        ret = pd.Series(
            [-0.05, -0.03, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.06],
            index=dates,
        )
        expected = -np.percentile(ret.values, 5)
        var = value_at_risk(ret, confidence=0.95)
        assert abs(var - expected) < 1e-10

    def test_var_is_positive(self, daily_returns: pd.Series) -> None:
        var = value_at_risk(daily_returns, confidence=0.95)
        assert var > 0

    def test_higher_confidence_higher_var(self, daily_returns: pd.Series) -> None:
        var_90 = value_at_risk(daily_returns, confidence=0.90)
        var_99 = value_at_risk(daily_returns, confidence=0.99)
        assert var_99 >= var_90


# -------------------------------------------------------------------
# conditional_var tests
# -------------------------------------------------------------------

class TestConditionalVaR:

    def test_cvar_ge_var(self, daily_returns: pd.Series) -> None:
        """CVaR must be >= VaR by definition."""
        var = value_at_risk(daily_returns, confidence=0.95)
        cvar = conditional_var(daily_returns, confidence=0.95)
        assert cvar >= var - 1e-10

    def test_cvar_is_positive(self, daily_returns: pd.Series) -> None:
        cvar = conditional_var(daily_returns, confidence=0.95)
        assert cvar > 0

    def test_cvar_manual(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=10, freq="B")
        ret = pd.Series(
            [-0.05, -0.03, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.06],
            index=dates,
        )
        cutoff = np.percentile(ret.values, 5)
        tail = ret.values[ret.values <= cutoff]
        expected = -tail.mean()
        cvar = conditional_var(ret, confidence=0.95)
        assert abs(cvar - expected) < 1e-10


# -------------------------------------------------------------------
# rolling_sharpe tests
# -------------------------------------------------------------------

class TestRollingSharpe:

    def test_output_length(self) -> None:
        dates = pd.bdate_range("2020-01-02", periods=300, freq="B")
        ret = pd.Series(np.random.default_rng(42).normal(0.0003, 0.01, 300), index=dates)
        rs = rolling_sharpe(ret, window=252)
        assert len(rs) == 300

    def test_nans_before_window(self) -> None:
        dates = pd.bdate_range("2020-01-02", periods=300, freq="B")
        ret = pd.Series(np.random.default_rng(42).normal(0.0003, 0.01, 300), index=dates)
        rs = rolling_sharpe(ret, window=252)
        assert rs.iloc[:251].isna().all()
        assert rs.iloc[251:].notna().all()

    def test_name_is_rolling_sharpe(self) -> None:
        dates = pd.bdate_range("2020-01-02", periods=260, freq="B")
        ret = pd.Series(0.001, index=dates)
        rs = rolling_sharpe(ret, window=252, risk_free_rate=0.0)
        assert rs.name == "rolling_sharpe"


# -------------------------------------------------------------------
# correlation_report tests
# -------------------------------------------------------------------

class TestCorrelationReport:

    def test_returns_dict(self, simple_returns: pd.DataFrame) -> None:
        result = correlation_report(simple_returns)
        assert "correlation_matrix" in result
        assert "high_pairs" in result

    def test_diagonal_is_one(self, simple_returns: pd.DataFrame) -> None:
        result = correlation_report(simple_returns)
        corr = result["correlation_matrix"]
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-10)

    def test_symmetric(self, simple_returns: pd.DataFrame) -> None:
        result = correlation_report(simple_returns)
        corr = result["correlation_matrix"]
        np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-10)

    def test_high_pairs_above_threshold(self, simple_returns: pd.DataFrame) -> None:
        result = correlation_report(simple_returns, threshold=0.5)
        assert len(result["high_pairs"]) > 0
        for pair in result["high_pairs"]:
            assert abs(pair[2]) >= 0.5

    def test_no_pairs_below_threshold(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=100, freq="B")
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {"A": rng.normal(0, 1, 100), "B": rng.normal(0, 1, 100)},
            index=dates,
        )
        result = correlation_report(df, threshold=0.99)
        assert len(result["high_pairs"]) == 0

    def test_perfectly_correlated(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=100, freq="B")
        x = np.random.default_rng(42).normal(0, 1, 100)
        df = pd.DataFrame({"X": x, "Y": x * 2 + 1}, index=dates)
        result = correlation_report(df, threshold=0.5)
        assert len(result["high_pairs"]) == 1
        assert abs(result["high_pairs"][0][2] - 1.0) < 1e-10

    def test_high_correlation_warning(
        self, simple_returns: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging
        with caplog.at_level(logging.WARNING):
            correlation_report(simple_returns, threshold=0.5)
        assert any("High correlation" in msg for msg in caplog.messages)


# -------------------------------------------------------------------
# risk_summary tests
# -------------------------------------------------------------------

class TestRiskSummary:

    def test_returns_all_keys(self, daily_returns: pd.Series) -> None:
        result = risk_summary(daily_returns)
        expected_keys = {
            "sharpe", "sortino", "max_drawdown", "var_95", "cvar_95",
            "annualised_return", "annualised_volatility", "calmar_ratio",
        }
        assert expected_keys == set(result.keys())

    def test_all_values_finite(self, daily_returns: pd.Series) -> None:
        result = risk_summary(daily_returns)
        for key, val in result.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"

    def test_max_drawdown_is_negative(self, daily_returns: pd.Series) -> None:
        result = risk_summary(daily_returns)
        assert result["max_drawdown"] <= 0

    def test_var_le_cvar(self, daily_returns: pd.Series) -> None:
        result = risk_summary(daily_returns)
        assert result["var_95"] <= result["cvar_95"] + 1e-10

    def test_annualised_volatility_positive(self, daily_returns: pd.Series) -> None:
        result = risk_summary(daily_returns)
        assert result["annualised_volatility"] > 0

    def test_consistency_with_individual_functions(
        self, daily_returns: pd.Series
    ) -> None:
        result = risk_summary(daily_returns)
        assert abs(result["sharpe"] - sharpe_ratio(daily_returns)) < 1e-10
        assert abs(result["sortino"] - sortino_ratio(daily_returns)) < 1e-10
        assert abs(result["var_95"] - value_at_risk(daily_returns)) < 1e-10
        assert abs(result["cvar_95"] - conditional_var(daily_returns)) < 1e-10
