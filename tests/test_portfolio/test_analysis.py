"""Tests for src.portfolio.analysis — factor evaluation and performance reporting."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.portfolio.analysis import (
    _signal_quality_label,
    _overall_verdict,
    evaluate_alpha,
    full_strategy_report,
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
    for i, ticker in enumerate(tickers):
        log_ret = rng.normal(0.0003, 0.015, days)
        data[ticker] = 100.0 * np.exp(np.cumsum(log_ret))
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def factor_series(prices: pd.DataFrame) -> pd.Series:
    """Synthetic alpha factor — momentum-like signal (MultiIndex: date × asset).

    Factor is computed from lagged 21-day returns so that we deliberately
    have data available across the price history.
    """
    returns = prices.pct_change()
    momentum = returns.rolling(window=21, min_periods=21).mean()
    # Drop initial NaN rows
    momentum = momentum.dropna()
    # Stack into MultiIndex Series (date, asset) → factor value
    factor = momentum.stack()
    factor.index.names = ["date", "asset"]
    factor.name = "factor"
    return factor


@pytest.fixture
def backtest_returns(prices: pd.DataFrame) -> pd.Series:
    """Simple equal-weight daily portfolio returns for the same period."""
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
# evaluate_alpha tests
# -------------------------------------------------------------------

class TestEvaluateAlpha:
    """Tests for the Alphalens-based factor evaluation."""

    def test_returns_expected_keys(
        self, factor_series: pd.Series, prices: pd.DataFrame
    ) -> None:
        result = evaluate_alpha(factor_series, prices, periods=[1, 5])
        expected_keys = {
            "factor_data", "ic", "ic_by_period", "factor_returns",
            "turnover", "summary",
        }
        assert expected_keys == set(result.keys())

    def test_ic_is_numeric(
        self, factor_series: pd.Series, prices: pd.DataFrame
    ) -> None:
        result = evaluate_alpha(factor_series, prices, periods=[1, 5])
        ic = result["ic"]
        assert isinstance(ic, (pd.Series, pd.DataFrame))
        assert not ic.empty

    def test_factor_returns_has_period_columns(
        self, factor_series: pd.Series, prices: pd.DataFrame
    ) -> None:
        periods = [1, 5]
        result = evaluate_alpha(factor_series, prices, periods=periods)
        fr = result["factor_returns"]
        for p in periods:
            assert f"{p}D" in fr.columns

    def test_summary_contains_quality_labels(
        self, factor_series: pd.Series, prices: pd.DataFrame
    ) -> None:
        result = evaluate_alpha(factor_series, prices, periods=[1])
        summary = result["summary"]
        assert "1D" in summary
        entry = summary["1D"]
        assert "mean_ic" in entry
        assert "signal_quality" in entry
        assert entry["signal_quality"] in ("strong", "moderate", "weak", "none")

    def test_turnover_has_top_and_bottom(
        self, factor_series: pd.Series, prices: pd.DataFrame
    ) -> None:
        result = evaluate_alpha(factor_series, prices, periods=[5])
        turnover = result["turnover"]
        assert "5D" in turnover
        assert "top_quantile" in turnover["5D"]
        assert "bottom_quantile" in turnover["5D"]


# -------------------------------------------------------------------
# generate_tearsheet tests
# -------------------------------------------------------------------

class TestGenerateTearsheet:
    """Tests for the Pyfolio-based performance tear sheet."""

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
        """Sharpe should be a finite number, not wildly out of range."""
        result = generate_tearsheet(backtest_returns)
        sharpe = result["metrics"]["sharpe_ratio"]
        assert -5.0 < sharpe < 10.0

    def test_max_drawdown_is_negative(self, backtest_returns: pd.Series) -> None:
        """Pyfolio reports max drawdown as a negative number."""
        result = generate_tearsheet(backtest_returns)
        assert result["metrics"]["max_drawdown"] <= 0.0

    def test_benchmark_adds_tracking_metrics(
        self, backtest_returns: pd.Series, benchmark_returns: pd.Series
    ) -> None:
        result = generate_tearsheet(
            backtest_returns, benchmark_returns=benchmark_returns
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
            # Verify PNG files written
            png_files = list(Path(tmpdir).glob("*.png"))
            assert len(png_files) == 3

    def test_no_figures_without_save_dir(self, backtest_returns: pd.Series) -> None:
        result = generate_tearsheet(backtest_returns)
        assert result["figures"] == []


# -------------------------------------------------------------------
# full_strategy_report tests
# -------------------------------------------------------------------

class TestFullStrategyReport:
    """Tests for the combined report function."""

    def test_returns_all_sections(
        self,
        factor_series: pd.Series,
        prices: pd.DataFrame,
        backtest_returns: pd.Series,
    ) -> None:
        report = full_strategy_report(
            "test_momentum",
            factor_series,
            prices,
            backtest_returns,
            periods=[1, 5],
        )
        assert report["strategy_name"] == "test_momentum"
        assert "alpha" in report
        assert "performance" in report
        assert "verdict" in report

    def test_verdict_is_string(
        self,
        factor_series: pd.Series,
        prices: pd.DataFrame,
        backtest_returns: pd.Series,
    ) -> None:
        report = full_strategy_report(
            "test_strat",
            factor_series,
            prices,
            backtest_returns,
            periods=[1],
        )
        assert isinstance(report["verdict"], str)
        assert any(
            label in report["verdict"]
            for label in ("GO", "REVIEW", "NO-GO")
        )

    def test_save_dir_creates_subfolder(
        self,
        factor_series: pd.Series,
        prices: pd.DataFrame,
        backtest_returns: pd.Series,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = full_strategy_report(
                "test_strat",
                factor_series,
                prices,
                backtest_returns,
                periods=[1],
                save_dir=tmpdir,
            )
            strat_dir = Path(tmpdir) / "test_strat"
            assert strat_dir.exists()
            assert len(list(strat_dir.glob("*.png"))) == 3


# -------------------------------------------------------------------
# Helper function tests
# -------------------------------------------------------------------

class TestSignalQualityLabel:
    """Tests for _signal_quality_label."""

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


class TestOverallVerdict:
    """Tests for _overall_verdict."""

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
