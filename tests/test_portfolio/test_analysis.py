"""Tests for src.portfolio.analysis — factor evaluation and performance reporting.

Tests are designed to work regardless of whether alphalens/pyfolio are
installed, by testing the public API contract rather than backend-specific
behaviour.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.portfolio.analysis import (
    _overall_verdict,
    _signal_quality_label,
    compare_strategies,
    evaluate_factor,
    generate_tearsheet,
)


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def prices() -> pd.DataFrame:
    """Synthetic close prices for 3 assets over ~2 years."""
    rng = np.random.default_rng(42)
    days = 504
    dates = pd.bdate_range("2020-01-02", periods=days, freq="B")
    tickers = ["ALPHA", "BETA", "GAMMA"]
    data = {}
    for ticker in tickers:
        log_ret = rng.normal(0.0003, 0.015, days)
        data[ticker] = 100.0 * np.exp(np.cumsum(log_ret))
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def factor_series(prices: pd.DataFrame) -> pd.Series:
    """Synthetic alpha factor (momentum-like signal, MultiIndex: date x asset)."""
    returns = prices.pct_change()
    momentum = returns.rolling(window=21, min_periods=21).mean()
    momentum = momentum.dropna()
    factor = momentum.stack()
    factor.index.names = ["date", "asset"]
    factor.name = "factor"
    return factor


@pytest.fixture
def backtest_returns(prices: pd.DataFrame) -> pd.Series:
    """Simple equal-weight daily portfolio returns."""
    ret = prices.pct_change().dropna()
    port_ret = ret.mean(axis=1)
    port_ret.name = "strategy"
    return port_ret


@pytest.fixture
def benchmark_returns(prices: pd.DataFrame) -> pd.Series:
    """Benchmark returns (first asset as proxy)."""
    ret = prices["ALPHA"].pct_change().dropna()
    ret.name = "benchmark"
    return ret


# -------------------------------------------------------------------
# evaluate_factor tests
# -------------------------------------------------------------------

class TestEvaluateFactor:

    def test_returns_ic_and_summary(
        self, factor_series: pd.Series, prices: pd.DataFrame
    ) -> None:
        result = evaluate_factor(factor_series, prices, periods=[1, 5])
        assert "ic" in result
        assert "summary" in result

    def test_summary_contains_quality_labels(
        self, factor_series: pd.Series, prices: pd.DataFrame
    ) -> None:
        result = evaluate_factor(factor_series, prices, periods=[1])
        summary = result["summary"]
        assert "1D" in summary
        entry = summary["1D"]
        assert "mean_ic" in entry
        assert "signal_quality" in entry
        assert entry["signal_quality"] in ("strong", "moderate", "weak", "none")

    def test_ic_is_series(
        self, factor_series: pd.Series, prices: pd.DataFrame
    ) -> None:
        result = evaluate_factor(factor_series, prices, periods=[1, 5])
        ic = result["ic"]
        assert isinstance(ic, (pd.Series, pd.DataFrame))
        assert not ic.empty

    def test_multiple_periods(
        self, factor_series: pd.Series, prices: pd.DataFrame
    ) -> None:
        periods = [1, 5, 21]
        result = evaluate_factor(factor_series, prices, periods=periods)
        summary = result["summary"]
        for p in periods:
            assert f"{p}D" in summary


# -------------------------------------------------------------------
# generate_tearsheet tests
# -------------------------------------------------------------------

class TestGenerateTearsheet:

    def test_returns_metrics_dict(self, backtest_returns: pd.Series) -> None:
        result = generate_tearsheet(backtest_returns)
        metrics = result["metrics"]
        assert isinstance(metrics, dict)
        expected = {
            "annual_return", "annual_volatility", "sharpe_ratio",
            "sortino_ratio", "max_drawdown", "calmar_ratio",
        }
        assert expected.issubset(metrics.keys())

    def test_all_metrics_are_finite(self, backtest_returns: pd.Series) -> None:
        result = generate_tearsheet(backtest_returns)
        for key, val in result["metrics"].items():
            assert np.isfinite(val), f"{key} is not finite: {val}"

    def test_sharpe_is_reasonable(self, backtest_returns: pd.Series) -> None:
        result = generate_tearsheet(backtest_returns)
        sharpe = result["metrics"]["sharpe_ratio"]
        assert -10.0 < sharpe < 20.0

    def test_max_drawdown_is_non_positive(self, backtest_returns: pd.Series) -> None:
        result = generate_tearsheet(backtest_returns)
        assert result["metrics"]["max_drawdown"] <= 0.0

    def test_benchmark_adds_tracking_metrics(
        self, backtest_returns: pd.Series, benchmark_returns: pd.Series
    ) -> None:
        result = generate_tearsheet(
            backtest_returns, benchmark_returns=benchmark_returns,
        )
        metrics = result["metrics"]
        assert "excess_annual_return" in metrics
        assert "tracking_error" in metrics
        assert "information_ratio" in metrics

    def test_save_dir_creates_figures(self, backtest_returns: pd.Series) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_tearsheet(backtest_returns, save_dir=tmpdir)
            figs = result["figures"]
            assert len(figs) == 3
            png_files = list(Path(tmpdir).glob("*.png"))
            assert len(png_files) == 3

    def test_no_figures_without_save_dir(self, backtest_returns: pd.Series) -> None:
        result = generate_tearsheet(backtest_returns)
        assert result["figures"] == []


# -------------------------------------------------------------------
# compare_strategies tests
# -------------------------------------------------------------------

class TestCompareStrategies:

    def test_returns_dataframe(self, backtest_returns: pd.Series) -> None:
        rng = np.random.default_rng(99)
        dates = backtest_returns.index
        strat_b = pd.Series(rng.normal(0.0002, 0.01, len(dates)), index=dates)
        result = compare_strategies(
            {"momentum": backtest_returns, "mean_reversion": strat_b},
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_sorted_by_sharpe(self, backtest_returns: pd.Series) -> None:
        rng = np.random.default_rng(99)
        dates = backtest_returns.index
        strat_b = pd.Series(rng.normal(-0.001, 0.02, len(dates)), index=dates)
        result = compare_strategies(
            {"good": backtest_returns, "bad": strat_b},
        )
        assert result.index[0] == "good"  # Higher Sharpe first

    def test_all_metrics_present(self, backtest_returns: pd.Series) -> None:
        result = compare_strategies({"strat": backtest_returns})
        expected = {
            "sharpe", "sortino", "max_drawdown", "var_95", "cvar_95",
            "annualised_return", "annualised_volatility", "calmar_ratio",
        }
        assert expected.issubset(result.columns)

    def test_index_is_strategy_names(self, backtest_returns: pd.Series) -> None:
        result = compare_strategies({"alpha_strat": backtest_returns})
        assert result.index.name == "strategy"
        assert "alpha_strat" in result.index


# -------------------------------------------------------------------
# Helper function tests
# -------------------------------------------------------------------

class TestSignalQualityLabel:

    def test_strong(self) -> None:
        assert _signal_quality_label(0.06) == "strong"
        assert _signal_quality_label(-0.05) == "strong"

    def test_moderate(self) -> None:
        assert _signal_quality_label(0.03) == "moderate"
        assert _signal_quality_label(-0.02) == "moderate"

    def test_weak(self) -> None:
        assert _signal_quality_label(0.01) == "weak"

    def test_none(self) -> None:
        assert _signal_quality_label(0.0) == "none"

    def test_nan(self) -> None:
        assert _signal_quality_label(float("nan")) == "none"


class TestOverallVerdict:

    def test_go(self) -> None:
        alpha = {"1D": {"signal_quality": "strong"}}
        perf = {"sharpe_ratio": 1.5, "max_drawdown": -0.10}
        verdict = _overall_verdict(alpha, perf)
        assert "GO" in verdict and "NO-GO" not in verdict

    def test_no_go(self) -> None:
        alpha = {"1D": {"signal_quality": "none"}}
        perf = {"sharpe_ratio": -0.5, "max_drawdown": -0.50}
        verdict = _overall_verdict(alpha, perf)
        assert "NO-GO" in verdict

    def test_review(self) -> None:
        alpha = {"1D": {"signal_quality": "moderate"}}
        perf = {"sharpe_ratio": 0.3, "max_drawdown": -0.40}
        verdict = _overall_verdict(alpha, perf)
        assert "REVIEW" in verdict
