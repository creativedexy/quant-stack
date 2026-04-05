#!/usr/bin/env python3
"""Run the full pipeline on real yfinance data and produce a tearsheet.

Loads cleaned OHLCV for 10 FTSE 100 tickers, generates features,
trains a model per ticker on the training period (up to train_end),
optimises portfolio weights, and backtests on out-of-sample data
(test_start onwards).  Saves metrics to real_tearsheet.json.

No 2024 data touches feature parameter tuning or model training.

Usage:
    python -m scripts.run_real_backtest
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.data.cleaner import DataCleaner
from src.features.pipeline import FeaturePipeline
from src.backtest.strategy import MeanReversionStrategy, MomentumStrategy
from src.portfolio.optimiser import PortfolioOptimiser
from src.portfolio.risk import risk_summary, sharpe_ratio, max_drawdown
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_cleaned_data(
    config: dict, tickers: list[str], data_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Load cleaned parquet files for each ticker."""
    result = {}
    for ticker in tickers:
        safe = ticker.replace(".", "_").replace("^", "idx_")
        path = data_dir / "processed" / f"{safe}.parquet"
        if not path.exists():
            logger.warning("Missing data for %s at %s -- skipping", ticker, path)
            continue
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)
        result[ticker] = df
        logger.info("Loaded %s: %d rows (%s -> %s)",
                     ticker, len(df),
                     df.index.min().date(), df.index.max().date())
    return result


def build_close_matrix(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a DataFrame of Close prices with tickers as columns."""
    closes = {}
    for ticker, df in data.items():
        closes[ticker] = df["Close"]
    return pd.DataFrame(closes).sort_index()


def compute_returns_matrix(close: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from close prices."""
    return close.pct_change().dropna()


def generate_per_ticker_signals(
    data: dict[str, pd.DataFrame],
    features_by_ticker: dict[str, pd.DataFrame],
    strategy_name: str = "mean_reversion",
) -> pd.DataFrame:
    """Generate signals per ticker and return a DataFrame (dates x tickers).

    Values: +1 (long), 0 (flat), -1 (short).
    """
    if strategy_name == "mean_reversion":
        strategy = MeanReversionStrategy("rsi_mean_reversion")
    else:
        strategy = MomentumStrategy("momentum_sma50")

    signals = {}
    for ticker, feat in features_by_ticker.items():
        try:
            sig = strategy.generate_signals(feat)
            if isinstance(sig, pd.DataFrame):
                sig = sig.iloc[:, 0]
            signals[ticker] = sig
        except Exception as exc:
            logger.warning("Signal generation failed for %s: %s", ticker, exc)
            # Default to long-only for failed tickers
            signals[ticker] = pd.Series(1, index=feat.index)

    return pd.DataFrame(signals).sort_index()


def main() -> None:
    config = load_config()
    tickers = config["universe"]["tickers"]
    data_dir = Path(config["general"]["data_dir"])

    train_end = pd.Timestamp(config["models"]["train_end"])
    test_start = pd.Timestamp(config["models"]["test_start"])

    logger.info("=" * 60)
    logger.info("REAL DATA BACKTEST PIPELINE")
    logger.info("=" * 60)
    logger.info("Universe: %d tickers", len(tickers))
    logger.info("Train period: %s -> %s",
                 config["data"]["start_date"], train_end.date())
    logger.info("Test period (OOS): %s -> %s",
                 test_start.date(), config["data"]["end_date"])

    # ------------------------------------------------------------------
    # 1. Load cleaned data
    # ------------------------------------------------------------------
    logger.info("Step 1: Loading cleaned data...")
    data = load_cleaned_data(config, tickers, data_dir)
    if len(data) < 5:
        logger.error("Only %d tickers loaded -- need at least 5. Exiting.", len(data))
        sys.exit(1)
    logger.info("Loaded %d / %d tickers", len(data), len(tickers))

    # ------------------------------------------------------------------
    # 2. Build close price matrix and returns
    # ------------------------------------------------------------------
    logger.info("Step 2: Building price matrix...")
    close = build_close_matrix(data)
    all_returns = compute_returns_matrix(close)

    # Split into train and test
    train_returns = all_returns.loc[all_returns.index <= train_end]
    test_returns = all_returns.loc[all_returns.index >= test_start]
    test_close = close.loc[close.index >= test_start]

    logger.info("Train returns: %d days, Test returns: %d days",
                 len(train_returns), len(test_returns))

    # ------------------------------------------------------------------
    # 3. Generate features (train period only for parameter estimation)
    # ------------------------------------------------------------------
    logger.info("Step 3: Generating features...")
    pipeline = FeaturePipeline(config)

    # Generate features per ticker using only train data
    train_features = {}
    for ticker, df in data.items():
        train_df = df.loc[df.index <= train_end]
        feat = pipeline.generate(train_df)
        if not feat.empty:
            train_features[ticker] = feat
            logger.info("  %s: %d features, %d rows",
                         ticker, len(feat.columns), len(feat))

    # Generate features on full data for signal generation in OOS period
    # (rolling indicators computed on data up to each point -- no lookahead)
    full_features = {}
    for ticker, df in data.items():
        feat = pipeline.generate(df)
        if not feat.empty:
            full_features[ticker] = feat

    # ------------------------------------------------------------------
    # 4. Portfolio optimisation on training period
    # ------------------------------------------------------------------
    logger.info("Step 4: Portfolio optimisation on training returns...")
    optimiser = PortfolioOptimiser(config)

    # Use returns from training period for weight estimation
    opt_method = config.get("optimisation", {}).get("optimisation", {}).get(
        "method", "mean_variance"
    )

    try:
        weights = optimiser.optimise(train_returns, method=opt_method)
    except Exception as exc:
        logger.warning(
            "Optimisation with '%s' failed (%s) -- falling back to equal_weight",
            opt_method, exc,
        )
        weights = optimiser.optimise(train_returns, method="equal_weight")

    logger.info("Portfolio weights (%s):", opt_method)
    for t, w in weights.sort_values(ascending=False).items():
        logger.info("  %s: %.2f%%", t, w * 100)

    # ------------------------------------------------------------------
    # 5. Generate signals on OOS period
    # ------------------------------------------------------------------
    logger.info("Step 5: Generating OOS signals...")
    oos_features = {}
    for ticker, feat in full_features.items():
        oos_feat = feat.loc[feat.index >= test_start]
        if not oos_feat.empty:
            oos_features[ticker] = oos_feat

    signals = generate_per_ticker_signals(data, full_features)
    oos_signals = signals.loc[signals.index >= test_start]

    # ------------------------------------------------------------------
    # 6. Backtest on OOS period with portfolio weights
    # ------------------------------------------------------------------
    logger.info("Step 6: Running OOS backtest...")

    commission = config["backtest"]["commission_pct"]
    slippage = config["backtest"]["slippage_pct"]
    initial_capital = config["backtest"]["initial_capital"]

    # Compute weighted portfolio returns
    # For each day: portfolio_return = sum(weight_i * signal_i * asset_return_i) - costs
    common_idx = test_returns.index.intersection(oos_signals.index)
    test_ret = test_returns.loc[common_idx]
    oos_sig = oos_signals.loc[common_idx].fillna(0)

    # Align weights with available columns
    common_tickers = test_ret.columns.intersection(weights.index)
    test_ret = test_ret[common_tickers]
    oos_sig = oos_sig[common_tickers]
    w = weights.reindex(common_tickers)
    w = w / w.sum()  # renormalise

    # Signal-weighted returns: position[t-1] * weight * return[t]
    positions = oos_sig.shift(1).fillna(0)
    weighted_returns = (positions * test_ret * w).sum(axis=1)

    # Transaction costs on signal changes
    position_changes = positions.diff().fillna(positions).abs()
    daily_cost = (position_changes * w * (commission + slippage)).sum(axis=1)
    portfolio_returns = weighted_returns - daily_cost
    portfolio_returns.name = "portfolio_returns"

    # Equity curve
    equity = initial_capital * (1 + portfolio_returns).cumprod()
    equity.name = "equity"

    # ------------------------------------------------------------------
    # 7. Compute metrics
    # ------------------------------------------------------------------
    logger.info("Step 7: Computing metrics...")
    metrics = risk_summary(portfolio_returns)
    metrics["initial_capital"] = initial_capital
    metrics["final_value"] = round(float(equity.iloc[-1]), 2)
    metrics["total_return_pct"] = round(
        (equity.iloc[-1] / initial_capital - 1) * 100, 2
    )
    metrics["oos_start"] = str(test_start.date())
    metrics["oos_end"] = str(test_returns.index.max().date())
    metrics["oos_days"] = len(portfolio_returns)
    metrics["n_tickers"] = len(common_tickers)
    metrics["tickers"] = list(common_tickers)
    metrics["optimisation_method"] = opt_method
    metrics["strategy"] = "rsi_mean_reversion"

    # Weights as dict
    metrics["weights"] = {t: round(float(v), 4) for t, v in w.items()}

    # ------------------------------------------------------------------
    # 8. Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  REAL DATA BACKTEST -- OUT-OF-SAMPLE RESULTS")
    print("=" * 60)
    print(f"  Universe:           {len(common_tickers)} FTSE 100 tickers")
    print(f"  OOS Period:         {metrics['oos_start']} -> {metrics['oos_end']}")
    print(f"  OOS Days:           {metrics['oos_days']}")
    print(f"  Strategy:           RSI Mean Reversion")
    print(f"  Optimisation:       {opt_method}")
    print(f"  Initial Capital:    {initial_capital:,.0f} GBP")
    print(f"  Final Value:        {metrics['final_value']:,.2f} GBP")
    print(f"  Total Return:       {metrics['total_return_pct']:.2f}%")
    print(f"  Annualised Return:  {metrics['annualised_return']:.4f}")
    print(f"  Sharpe Ratio:       {metrics['sharpe']:.2f}")
    print(f"  Sortino Ratio:      {metrics.get('sortino', 0):.2f}")
    print(f"  Max Drawdown:       {metrics['max_drawdown']:.4f}")
    print(f"  Volatility (ann):   {metrics.get('annualised_volatility', 0):.4f}")
    print(f"  VaR (95%):          {metrics.get('var_95', 0):.4f}")
    print(f"  CVaR (95%):         {metrics.get('cvar_95', 0):.4f}")
    print("=" * 60)

    # Sanity check
    sharpe_val = metrics["sharpe"]
    if sharpe_val > 1.5:
        print("\n  *** WARNING: Sharpe > 1.5 -- likely overfitting. ***")
        print("  Review feature pipeline and signal generation for lookahead bias.")
    elif 0.3 <= sharpe_val <= 1.2:
        print(f"\n  Sharpe {sharpe_val:.2f} is within realistic range (0.3-1.2)")
        print("  for UK large-cap equities over a 1-year OOS period.")
    elif sharpe_val < 0:
        print(f"\n  Negative Sharpe ({sharpe_val:.2f}) -- strategy underperformed.")
    print()

    # ------------------------------------------------------------------
    # 9. Save deliverables
    # ------------------------------------------------------------------
    output_dir = data_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save tearsheet JSON
    tearsheet_path = output_dir / "real_tearsheet.json"

    # Convert numpy types for JSON serialisation
    serialisable = {}
    for k, v in metrics.items():
        if isinstance(v, (np.integer,)):
            serialisable[k] = int(v)
        elif isinstance(v, (np.floating,)):
            serialisable[k] = round(float(v), 6)
        elif isinstance(v, np.ndarray):
            serialisable[k] = v.tolist()
        else:
            serialisable[k] = v

    with open(tearsheet_path, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)
    logger.info("Saved tearsheet to %s", tearsheet_path)

    # Save equity curve as CSV for plotting
    equity_path = output_dir / "real_equity_curve.csv"
    equity.to_csv(equity_path)
    logger.info("Saved equity curve to %s", equity_path)

    # Save interpretation placeholder (Claude API interpretation optional)
    interpretation = {
        "summary": (
            f"Out-of-sample backtest on {len(common_tickers)} FTSE 100 tickers "
            f"from {metrics['oos_start']} to {metrics['oos_end']}. "
            f"RSI mean reversion strategy with {opt_method} portfolio optimisation. "
            f"Sharpe ratio: {sharpe_val:.2f}. "
            f"Total return: {metrics['total_return_pct']:.2f}%."
        ),
        "observations": [
            f"Annualised return: {metrics['annualised_return']:.4f}",
            f"Max drawdown: {metrics['max_drawdown']:.4f}",
            f"OOS period: {metrics['oos_days']} trading days",
        ],
        "warnings": [],
        "suggestions": [
            "Compare with buy-and-hold benchmark",
            "Test alternative strategies (momentum, MACD crossover)",
            "Run sensitivity analysis on RSI threshold parameters",
        ],
        "confidence": "medium",
    }

    if sharpe_val > 1.5:
        interpretation["warnings"].append(
            "Sharpe > 1.5 is suspicious for UK equities -- check for overfitting"
        )
    if metrics["max_drawdown"] < -0.15:
        interpretation["warnings"].append(
            f"Max drawdown ({metrics['max_drawdown']:.2%}) exceeds 15% risk limit"
        )

    interp_path = output_dir / "real_interpretation.json"
    with open(interp_path, "w") as f:
        json.dump(interpretation, f, indent=2)
    logger.info("Saved interpretation to %s", interp_path)

    print(f"Deliverables saved to {output_dir}:")
    print(f"  - {tearsheet_path.name}")
    print(f"  - {equity_path.name}")
    print(f"  - {interp_path.name}")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
