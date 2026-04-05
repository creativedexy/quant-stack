#!/usr/bin/env python3
"""Execution pipeline smoke test (paper mode, no IB required).

Validates the full signal-to-execution pipeline using synthetic data
and the PaperBroker.  Seven steps, each with a PASS/FAIL verdict:

1. Generate synthetic data
2. Compute features via FeaturePipeline
3. Generate signals via strategy
4. Convert signals to target weights via SignalBridge
5. Create PaperBroker, compute rebalance orders via OMS
6. Execute plan (paper mode)
7. Reconcile positions vs targets

Exit code 0 if all pass, 1 if any fail.

Usage::

    python -m scripts.smoke_test_execution
    python -m scripts.smoke_test_execution --strategy momentum
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import get_logger

logger = get_logger(__name__)

_PASS = "PASS"
_FAIL = "FAIL"


def _step(number: int, desc: str) -> None:
    """Print step header."""
    print(f"\n  Step {number}: {desc}", end=" ... ", flush=True)


def main() -> None:
    """Run 7-step execution smoke test."""
    parser = argparse.ArgumentParser(description="Execution pipeline smoke test")
    parser.add_argument(
        "--strategy",
        type=str,
        default="momentum",
        help="Strategy name to use (default: momentum)",
    )
    args = parser.parse_args()

    results: list[tuple[str, str]] = []
    config = None

    print("=" * 60)
    print("  Execution Pipeline Smoke Test (paper mode)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Generate synthetic data
    # ------------------------------------------------------------------
    _step(1, "Generate synthetic data")
    try:
        from src.data.synthetic import generate_multi_asset_data
        from src.utils.config import load_config

        config = load_config()
        tickers = config["universe"]["tickers"][:4]  # Use subset for speed

        multi_data = generate_multi_asset_data(tickers, days=252, seed=42)
        assert len(multi_data) == len(tickers)
        print(f"{_PASS} ({len(tickers)} tickers, 252 days)")
        results.append(("Generate synthetic data", _PASS))
    except Exception as exc:
        print(f"{_FAIL} ({exc})")
        results.append(("Generate synthetic data", _FAIL))
        _print_summary(results)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Compute features
    # ------------------------------------------------------------------
    _step(2, "Compute features via FeaturePipeline")
    try:
        from src.features.pipeline import FeaturePipeline

        import pandas as pd

        fp = FeaturePipeline(config=config)
        features_map: dict[str, pd.DataFrame] = {}
        for ticker, df in multi_data.items():
            feat = fp.generate(df)
            # Strategies need Close for signal generation; merge it back
            feat["Close"] = df["Close"].reindex(feat.index)
            features_map[ticker] = feat

        assert all(len(f) > 0 for f in features_map.values())
        sample = list(features_map.values())[0]
        print(f"{_PASS} ({len(features_map)} tickers, {len(sample.columns)} features)")
        results.append(("Compute features", _PASS))
    except Exception as exc:
        print(f"{_FAIL} ({exc})")
        results.append(("Compute features", _FAIL))

    # ------------------------------------------------------------------
    # Step 3: Generate signals
    # ------------------------------------------------------------------
    _step(3, f"Generate signals via '{args.strategy}'")
    try:
        from src.backtest.strategy import strategy_registry

        strategy = strategy_registry.create(args.strategy)
        signals: dict[str, int] = {}
        for ticker, feat in features_map.items():
            sig_df = strategy.generate_signals(feat)
            if not sig_df.empty and "signal" in sig_df.columns:
                signals[ticker] = int(sig_df["signal"].iloc[-1])

        assert len(signals) > 0
        longs = sum(1 for s in signals.values() if s == 1)
        print(f"{_PASS} ({len(signals)} tickers: {longs} long)")
        results.append(("Generate signals", _PASS))
    except Exception as exc:
        print(f"{_FAIL} ({exc})")
        results.append(("Generate signals", _FAIL))
        signals = {t: 1 for t in tickers}  # Fallback: all long

    # ------------------------------------------------------------------
    # Step 4: Convert signals to target weights
    # ------------------------------------------------------------------
    _step(4, "Convert signals to target weights via SignalBridge")
    try:
        from src.execution.signal_bridge import SignalBridge

        closes = pd.DataFrame({t: df["Close"] for t, df in multi_data.items()})
        returns = closes.pct_change().dropna()

        bridge = SignalBridge(config=config)
        target_weights = bridge.signals_to_weights(signals, returns)

        assert abs(target_weights.sum() - 1.0) < 0.01 or target_weights.sum() == 0.0
        non_zero = (target_weights > 0).sum()
        print(f"{_PASS} ({non_zero} positions, sum={target_weights.sum():.4f})")
        results.append(("Convert to weights", _PASS))
    except Exception as exc:
        print(f"{_FAIL} ({exc})")
        results.append(("Convert to weights", _FAIL))

    # ------------------------------------------------------------------
    # Step 5: Compute rebalance orders via OMS
    # ------------------------------------------------------------------
    _step(5, "Compute rebalance orders via OMS")
    try:
        from src.execution.broker import PaperBroker
        from src.execution.oms import OrderManagementSystem

        broker = PaperBroker(config=config)
        broker.connect()

        oms = OrderManagementSystem(broker, config=config)
        positions = broker.get_positions()
        account_value = broker.get_account_value()
        prices = {t: float(df["Close"].iloc[-1]) for t, df in multi_data.items()}

        orders = oms.compute_rebalance_orders(
            target_weights, positions, account_value, prices,
        )
        print(f"{_PASS} ({len(orders)} orders, account={account_value:,.0f})")
        results.append(("Compute orders", _PASS))
    except Exception as exc:
        print(f"{_FAIL} ({exc})")
        results.append(("Compute orders", _FAIL))

    # ------------------------------------------------------------------
    # Step 6: Execute plan (paper mode)
    # ------------------------------------------------------------------
    _step(6, "Execute plan (paper mode)")
    try:
        # Set prices on paper broker so fills work
        for ticker, price in prices.items():
            broker.set_price(ticker, price)

        report = oms.execute_plan(orders, dry_run=False)
        executed = len(report.orders_executed)
        failed = len(report.orders_failed)
        assert failed == 0, f"{failed} orders failed"
        print(f"{_PASS} ({executed} executed, {failed} failed)")
        results.append(("Execute plan", _PASS))
    except Exception as exc:
        print(f"{_FAIL} ({exc})")
        results.append(("Execute plan", _FAIL))

    # ------------------------------------------------------------------
    # Step 7: Reconcile
    # ------------------------------------------------------------------
    _step(7, "Reconcile positions vs targets")
    try:
        recon = oms.reconcile(target_weights, account_value, prices)
        drift = recon.get("total_drift", 0.0)
        aligned = recon.get("aligned", False)
        status = "aligned" if aligned else f"drift={drift:.4f}"
        print(f"{_PASS} ({status})")
        results.append(("Reconcile", _PASS))
    except Exception as exc:
        print(f"{_FAIL} ({exc})")
        results.append(("Reconcile", _FAIL))

    broker.disconnect()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _print_summary(results)

    if any(r == _FAIL for _, r in results):
        sys.exit(1)
    sys.exit(0)


def _print_summary(results: list[tuple[str, str]]) -> None:
    """Print results summary."""
    print(f"\n{'=' * 60}")
    print("  Results Summary")
    print(f"{'=' * 60}")
    for desc, status in results:
        marker = "+" if status == _PASS else "X"
        print(f"  [{marker}] {desc}: {status}")
    passed = sum(1 for _, s in results if s == _PASS)
    total = len(results)
    print(f"\n  {passed}/{total} passed")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
