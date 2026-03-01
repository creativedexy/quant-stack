#!/usr/bin/env python3
"""Generate a portfolio risk report from synthetic or historical data.

Usage::

    python -m scripts.generate_report
    python -m scripts.generate_report --tickers SHEL.L BP.L HSBA.L
    python -m scripts.generate_report --output reports/my_report
    python -m scripts.generate_report --risk-free-rate 0.05

The script:
1. Loads or generates multi-asset return data.
2. Computes a risk summary for the equal-weight portfolio.
3. Generates a performance tear sheet (with plots saved to disk).
4. Prints the summary to stdout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src is importable when running as ``python -m scripts.generate_report``
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthetic import generate_synthetic_ohlcv
from src.portfolio.analysis import compare_strategies, generate_tearsheet
from src.portfolio.optimiser import equal_weight, inverse_volatility
from src.portfolio.risk import (
    correlation_report,
    portfolio_returns,
    risk_summary,
)
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a portfolio risk report.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Ticker symbols. Defaults to universe from config/settings.yaml.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1260,
        help="Number of trading days for synthetic data (default: 1260 ≈ 5 years).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports",
        help="Directory to save the report (default: reports/).",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=None,
        help="Annualised risk-free rate (default: from config).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data generation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the report generator."""
    args = _parse_args(argv)

    config = load_config()
    tickers = args.tickers or config.get("universe", {}).get(
        "tickers", ["ASSET_A", "ASSET_B", "ASSET_C"]
    )
    risk_free_rate = args.risk_free_rate or config.get("portfolio", {}).get(
        "risk_free_rate", 0.045
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Generate synthetic close prices and compute returns
    # ------------------------------------------------------------------
    logger.info("Generating synthetic data for %s", tickers)
    price_frames = {}
    for i, ticker in enumerate(tickers):
        ohlcv = generate_synthetic_ohlcv(ticker, days=args.days, seed=args.seed + i)
        price_frames[ticker] = ohlcv["Close"]

    prices = pd.DataFrame(price_frames)
    asset_returns = prices.pct_change().dropna()

    # ------------------------------------------------------------------
    # 2. Compute portfolio weights (equal-weight and inverse-vol)
    # ------------------------------------------------------------------
    ew_weights = equal_weight(asset_returns)
    print("\n=== Equal-Weight Portfolio ===")
    print(ew_weights.to_string())

    try:
        iv_weights = inverse_volatility(asset_returns)
        print("\n=== Inverse-Volatility Portfolio ===")
        print(iv_weights.to_string())
    except ValueError:
        iv_weights = None
        print("\nInverse-volatility weights not available (zero-vol asset).")

    # ------------------------------------------------------------------
    # 3. Compute portfolio returns
    # ------------------------------------------------------------------
    ew_returns = portfolio_returns(asset_returns, ew_weights)

    strategies = {"equal_weight": ew_returns}
    if iv_weights is not None:
        iv_returns = portfolio_returns(asset_returns, iv_weights)
        strategies["inverse_volatility"] = iv_returns

    # ------------------------------------------------------------------
    # 4. Risk summary
    # ------------------------------------------------------------------
    print("\n=== Risk Summary (Equal-Weight) ===")
    summary = risk_summary(ew_returns, risk_free_rate=risk_free_rate)
    for key, val in summary.items():
        print(f"  {key:25s}: {val:>12.6f}")

    # ------------------------------------------------------------------
    # 5. Correlation report
    # ------------------------------------------------------------------
    corr_result = correlation_report(
        asset_returns,
        threshold=config.get("portfolio", {}).get("risk", {}).get(
            "correlation_flag_threshold", 0.85,
        ),
    )
    print("\n=== Correlation Matrix ===")
    print(corr_result["correlation_matrix"].round(3).to_string())
    if corr_result["high_pairs"]:
        print("\nHighly correlated pairs:")
        for a, b, val in corr_result["high_pairs"]:
            print(f"  {a} — {b}: {val:.3f}")

    # ------------------------------------------------------------------
    # 6. Strategy comparison
    # ------------------------------------------------------------------
    if len(strategies) > 1:
        print("\n=== Strategy Comparison ===")
        comparison = compare_strategies(strategies, risk_free_rate=risk_free_rate)
        print(comparison.round(4).to_string())

    # ------------------------------------------------------------------
    # 7. Tear sheet with figures
    # ------------------------------------------------------------------
    print(f"\n=== Generating tear sheet → {output_dir} ===")
    tearsheet = generate_tearsheet(
        ew_returns,
        risk_free_rate=risk_free_rate,
        save_dir=output_dir,
    )
    print("Tear sheet metrics:")
    for key, val in tearsheet["metrics"].items():
        print(f"  {key:25s}: {val:>12.6f}")

    n_figs = len(tearsheet["figures"])
    print(f"\nSaved {n_figs} figure(s) to {output_dir}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
