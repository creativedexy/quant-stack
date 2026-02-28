"""Factor evaluation and performance reporting via Alphalens and Pyfolio.

Wraps alphalens-reloaded and pyfolio-reloaded to provide:
- Factor quality assessment (information coefficient, factor returns, turnover)
- Strategy performance tear sheets (Sharpe, Sortino, drawdown, Calmar)
- A combined report function that chains both into a single summary

Usage:
    from src.portfolio.analysis import evaluate_alpha, generate_tearsheet
    alpha_report = evaluate_alpha(factor_data, prices)
    perf_report = generate_tearsheet(returns)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------
try:
    import alphalens.performance as al_perf
    import alphalens.utils as al_utils

    _HAS_ALPHALENS = True
except ImportError:  # pragma: no cover
    _HAS_ALPHALENS = False

try:
    import pyfolio.timeseries as pf_ts

    _HAS_PYFOLIO = True
except ImportError:  # pragma: no cover
    _HAS_PYFOLIO = False


# ===================================================================
# 1.  Factor evaluation (Alphalens)
# ===================================================================

def evaluate_alpha(
    factor: pd.Series,
    prices: pd.DataFrame,
    periods: tuple[int, ...] | list[int] = (1, 5, 21),
    quantiles: int = 5,
    max_loss: float = 0.5,
) -> dict[str, Any]:
    """Evaluate an alpha factor using alphalens-reloaded.

    Args:
        factor: A Series with a MultiIndex of (date, asset) mapping to a
            scalar factor value.  Dates must be a subset of the *prices*
            index; assets must be a subset of *prices* columns.
        prices: Pricing DataFrame (DatetimeIndex, tickers as columns).
            Used by alphalens to compute forward returns.
        periods: Forward-return horizons in trading days.
        quantiles: Number of quantile buckets for the factor.
        max_loss: Maximum fraction of factor observations allowed to be
            dropped during cleaning before raising an error.

    Returns:
        Dictionary containing:
        - ``factor_data``: Cleaned factor + forward-returns DataFrame.
        - ``ic``: Information coefficient per period (Series of IC means).
        - ``ic_by_period``: Full IC time series per period.
        - ``factor_returns``: Long-short factor return per period.
        - ``turnover``: Top/bottom quantile turnover per period.
        - ``summary``: Human-readable quality metrics dict with a
          go/no-go signal for each period.

    Raises:
        ImportError: If alphalens-reloaded is not installed.
    """
    if not _HAS_ALPHALENS:  # pragma: no cover
        raise ImportError(
            "alphalens-reloaded is required for factor evaluation. "
            "Install with: pip install 'quant-stack[portfolio]'"
        )

    logger.info(
        "Evaluating alpha factor",
        extra={"periods": list(periods), "quantiles": quantiles},
    )

    # Alphalens expects periods as a tuple of ints
    periods = tuple(int(p) for p in periods)

    factor_data = al_utils.get_clean_factor_and_forward_returns(
        factor,
        prices,
        quantiles=quantiles,
        periods=periods,
        max_loss=max_loss,
    )

    # Information coefficient (Spearman rank correlation)
    ic_by_period = al_perf.factor_information_coefficient(factor_data)
    ic_means = al_perf.mean_information_coefficient(factor_data)

    # Factor returns (mean return per quantile spread, long-short)
    factor_returns = al_perf.factor_returns(factor_data)

    # Turnover: quantile membership turnover for top and bottom buckets
    turnover: dict[str, pd.Series] = {}
    for period in periods:
        col = f"{period}D"
        top_q = al_perf.quantile_turnover(factor_data, quantiles, period)
        bottom_q = al_perf.quantile_turnover(factor_data, 1, period)
        turnover[col] = pd.Series(
            {"top_quantile": float(top_q.mean()), "bottom_quantile": float(bottom_q.mean())},
        )

    # Build summary with go/no-go assessment
    summary: dict[str, dict[str, Any]] = {}
    for period in periods:
        col = f"{period}D"
        mean_ic = float(ic_means.loc[col]) if col in ic_means.index else float("nan")
        ann_factor_ret = float(factor_returns[col].mean()) * 252
        summary[col] = {
            "mean_ic": round(mean_ic, 4),
            "annualised_factor_return": round(ann_factor_ret, 6),
            "top_quantile_turnover": round(turnover[col]["top_quantile"], 4),
            "signal_quality": _signal_quality_label(mean_ic),
        }

    logger.info("Alpha evaluation complete", extra={"summary": summary})

    return {
        "factor_data": factor_data,
        "ic": ic_means,
        "ic_by_period": ic_by_period,
        "factor_returns": factor_returns,
        "turnover": turnover,
        "summary": summary,
    }


# ===================================================================
# 2.  Performance tear sheet (Pyfolio)
# ===================================================================

def generate_tearsheet(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.045,
    save_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a performance tear sheet using pyfolio-reloaded.

    Args:
        returns: Daily strategy returns with DatetimeIndex.
        benchmark_returns: Optional benchmark daily returns for comparison.
        risk_free_rate: Annualised risk-free rate for Sharpe/Sortino.
        save_dir: If provided, save matplotlib figures to this directory.
            Created automatically if it does not exist.

    Returns:
        Dictionary containing:
        - ``metrics``: Dict of scalar performance metrics.
        - ``figures``: List of matplotlib Figure objects (empty when
          *save_dir* is ``None``).

    Raises:
        ImportError: If pyfolio-reloaded is not installed.
    """
    if not _HAS_PYFOLIO:  # pragma: no cover
        raise ImportError(
            "pyfolio-reloaded is required for tear-sheet generation. "
            "Install with: pip install 'quant-stack[portfolio]'"
        )

    logger.info("Generating performance tear sheet")

    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1

    metrics: dict[str, float] = {
        "annual_return": float(pf_ts.annual_return(returns)),
        "annual_volatility": float(pf_ts.annual_volatility(returns)),
        "sharpe_ratio": float(pf_ts.sharpe_ratio(returns, risk_free=daily_rf)),
        "sortino_ratio": float(pf_ts.sortino_ratio(returns, required_return=daily_rf)),
        "max_drawdown": float(pf_ts.max_drawdown(returns)),
        "calmar_ratio": float(pf_ts.calmar_ratio(returns)),
    }

    if benchmark_returns is not None:
        aligned_ret, aligned_bench = returns.align(benchmark_returns, join="inner")
        excess = aligned_ret - aligned_bench
        metrics["excess_annual_return"] = float(pf_ts.annual_return(excess))
        metrics["tracking_error"] = float(pf_ts.annual_volatility(excess))
        if metrics["tracking_error"] != 0:
            metrics["information_ratio"] = (
                metrics["excess_annual_return"] / metrics["tracking_error"]
            )
        else:
            metrics["information_ratio"] = float("nan")

    figures: list[plt.Figure] = []
    if save_dir is not None:
        figures = _save_tearsheet_figures(returns, benchmark_returns, save_dir)

    logger.info("Tear sheet complete", extra={"metrics": metrics})

    return {"metrics": metrics, "figures": figures}


# ===================================================================
# 3.  Combined strategy report
# ===================================================================

def full_strategy_report(
    strategy_name: str,
    factor_data: pd.Series,
    prices: pd.DataFrame,
    backtest_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    periods: tuple[int, ...] | list[int] = (1, 5, 21),
    risk_free_rate: float = 0.045,
    save_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Chain factor evaluation and performance tear sheet into one report.

    Args:
        strategy_name: Human-readable strategy label.
        factor_data: Alpha factor Series (MultiIndex: date × asset).
        prices: Pricing DataFrame for alphalens forward returns.
        backtest_returns: Daily backtest return series.
        benchmark_returns: Optional benchmark returns.
        periods: Forward-return horizons for factor evaluation.
        risk_free_rate: Annualised risk-free rate.
        save_dir: If provided, save figures to *save_dir/strategy_name/*.

    Returns:
        Dictionary with keys ``strategy_name``, ``alpha``, ``performance``,
        and ``verdict`` (overall go/no-go string).
    """
    logger.info("Running full strategy report for '%s'", strategy_name)

    # Factor evaluation
    alpha_result = evaluate_alpha(factor_data, prices, periods=periods)

    # Performance tear sheet
    figure_dir = None
    if save_dir is not None:
        figure_dir = Path(save_dir) / strategy_name
    perf_result = generate_tearsheet(
        backtest_returns,
        benchmark_returns=benchmark_returns,
        risk_free_rate=risk_free_rate,
        save_dir=figure_dir,
    )

    # Overall verdict
    verdict = _overall_verdict(alpha_result["summary"], perf_result["metrics"])

    report = {
        "strategy_name": strategy_name,
        "alpha": alpha_result,
        "performance": perf_result,
        "verdict": verdict,
    }

    logger.info(
        "Strategy report for '%s': %s",
        strategy_name,
        verdict,
    )

    return report


# ===================================================================
# Internal helpers
# ===================================================================

def _signal_quality_label(mean_ic: float) -> str:
    """Map mean IC to a human-readable quality label."""
    abs_ic = abs(mean_ic)
    if abs_ic >= 0.05:
        return "strong"
    elif abs_ic >= 0.02:
        return "moderate"
    elif abs_ic > 0:
        return "weak"
    return "none"


def _overall_verdict(
    alpha_summary: dict[str, dict[str, Any]],
    perf_metrics: dict[str, float],
) -> str:
    """Produce a go/no-go verdict from alpha and performance results."""
    # Check if any period has at least moderate signal quality
    has_signal = any(
        entry["signal_quality"] in ("strong", "moderate")
        for entry in alpha_summary.values()
    )

    sharpe = perf_metrics.get("sharpe_ratio", 0.0)
    mdd = abs(perf_metrics.get("max_drawdown", 1.0))

    if has_signal and sharpe > 0.5 and mdd < 0.30:
        return "GO — signal quality and risk metrics acceptable"
    elif has_signal or sharpe > 0:
        return "REVIEW — mixed signals, further investigation recommended"
    return "NO-GO — insufficient signal quality or poor risk-adjusted returns"


def _save_tearsheet_figures(
    returns: pd.Series,
    benchmark_returns: pd.Series | None,
    save_dir: str | Path,
) -> list[plt.Figure]:
    """Generate and save standard tear-sheet figures."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    figures: list[plt.Figure] = []

    # 1. Cumulative returns
    fig, ax = plt.subplots(figsize=(12, 5))
    cum_ret = (1 + returns).cumprod()
    ax.plot(cum_ret.index, cum_ret.values, label="Strategy")
    if benchmark_returns is not None:
        cum_bench = (1 + benchmark_returns).cumprod()
        ax.plot(cum_bench.index, cum_bench.values, label="Benchmark", alpha=0.7)
    ax.set_title("Cumulative Returns")
    ax.set_ylabel("Growth of £1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "cumulative_returns.png", dpi=100)
    figures.append(fig)
    plt.close(fig)

    # 2. Drawdown
    fig, ax = plt.subplots(figsize=(12, 4))
    cum_max = cum_ret.cummax()
    drawdown = (cum_ret - cum_max) / cum_max
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color="red")
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "drawdown.png", dpi=100)
    figures.append(fig)
    plt.close(fig)

    # 3. Rolling Sharpe
    fig, ax = plt.subplots(figsize=(12, 4))
    rolling_mean = returns.rolling(window=126, min_periods=126).mean()
    rolling_std = returns.rolling(window=126, min_periods=126).std(ddof=1)
    rolling_sr = (rolling_mean / rolling_std) * np.sqrt(252)
    ax.plot(rolling_sr.index, rolling_sr.values)
    ax.axhline(y=0, color="grey", linestyle="--", alpha=0.5)
    ax.set_title("Rolling Sharpe Ratio (6-month)")
    ax.set_ylabel("Sharpe Ratio")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "rolling_sharpe.png", dpi=100)
    figures.append(fig)
    plt.close(fig)

    logger.info("Saved %d figures to %s", len(figures), save_dir)
    return figures
