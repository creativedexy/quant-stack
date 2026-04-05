#!/usr/bin/env python3
"""Minimum viable loop: one ticker, real data, one paper trade.

Fetches real market data via yfinance, generates features and a trading
signal, then executes a single paper trade via the PaperBroker.

Usage::

    python -m scripts.mvp_loop
    python -m scripts.mvp_loop --ticker AAPL
    python -m scripts.mvp_loop --ticker VUSA.L --strategy mean_reversion
    python -m scripts.mvp_loop --ticker SMH --capital 50000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.backtest.strategy import strategy_registry
from src.data.cleaner import DataCleaner
from src.data.fetcher import create_fetcher
from src.execution.broker import PaperBroker
from src.execution.oms import OrderManagementSystem
from src.execution.signal_bridge import SignalBridge
from src.features.pipeline import FeaturePipeline
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

_PASS = "PASS"
_FAIL = "FAIL"


def _step(number: int, desc: str) -> None:
    """Print step header."""
    print(f"\n  Step {number}: {desc}", end=" ... ", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimum viable loop")
    parser.add_argument(
        "--ticker", type=str, default="VUSA.L",
        help="Ticker to trade (default: VUSA.L)",
    )
    parser.add_argument(
        "--strategy", type=str, default="momentum",
        help="Strategy name (default: momentum)",
    )
    parser.add_argument(
        "--capital", type=float, default=100_000,
        help="Starting capital (default: 100000)",
    )
    parser.add_argument(
        "--start", type=str, default="2023-01-01",
        help="Data start date (default: 2023-01-01)",
    )
    args = parser.parse_args()

    ticker = args.ticker
    config = load_config()

    # Force paper mode regardless of settings.yaml
    config.setdefault("execution", {})["mode"] = "paper"
    config["execution"].setdefault("broker", {})["type"] = "paper"
    config["backtest"]["initial_capital"] = args.capital

    results: list[tuple[str, str]] = []

    print("=" * 60)
    print(f"  MVP Loop: {ticker} via {args.strategy}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Fetch real data via yfinance
    # ------------------------------------------------------------------
    _step(1, f"Fetch {ticker} from yfinance")
    try:
        fetcher = create_fetcher("yfinance")
        raw_df = fetcher.fetch(ticker, start=args.start)

        if raw_df.empty:
            raise RuntimeError(f"No data returned for {ticker}")

        print(f"{_PASS} ({len(raw_df)} rows, "
              f"{raw_df.index.min().date()} -> {raw_df.index.max().date()})")
        results.append(("Fetch data", _PASS))
    except Exception as exc:
        print(f"{_FAIL} ({exc})")
        results.append(("Fetch data", _FAIL))
        _print_summary(results)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Clean data
    # ------------------------------------------------------------------
    _step(2, "Clean data")
    try:
        cleaner = DataCleaner()
        clean_df = cleaner.clean(raw_df, ticker=ticker)
        last_close = float(clean_df["Close"].iloc[-1])
        print(f"{_PASS} ({len(clean_df)} rows, last close={last_close:.2f})")
        results.append(("Clean data", _PASS))
    except Exception as exc:
        print(f"{_FAIL} ({exc})")
        results.append(("Clean data", _FAIL))
        _print_summary(results)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3: Generate features
    # ------------------------------------------------------------------
    _step(3, "Generate features")
    try:
        pipeline = FeaturePipeline(config=config)
        features = pipeline.generate(clean_df)

        feature_names = pipeline.get_feature_names()
        # Strategies need Close for signal generation; pipeline returns
        # features-only by design, so we merge Close back here.
        features["Close"] = clean_df["Close"].reindex(features.index)
        has_sma50 = "sma_50" in features.columns
        print(f"{_PASS} ({len(feature_names)} features, {len(features)} rows, "
              f"sma_50={'yes' if has_sma50 else 'NO'})")
        results.append(("Generate features", _PASS))
    except Exception as exc:
        print(f"{_FAIL} ({exc})")
        results.append(("Generate features", _FAIL))
        _print_summary(results)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 4: Generate signal
    # ------------------------------------------------------------------
    _step(4, f"Generate signal via '{args.strategy}'")
    try:
        strategy = strategy_registry.create(args.strategy)
        sig_df = strategy.generate_signals(features)
        signal_value = int(sig_df["signal"].iloc[-1])

        signal_label = {1: "LONG", 0: "FLAT", -1: "SHORT"}.get(signal_value, "?")
        print(f"{_PASS} (signal={signal_value} [{signal_label}])")
        results.append(("Generate signal", _PASS))

        if signal_value == 0:
            print(f"\n  Signal is FLAT -- no trade needed. This is a valid outcome.")
            _print_summary(results)
            sys.exit(0)
    except Exception as exc:
        print(f"{_FAIL} ({exc})")
        results.append(("Generate signal", _FAIL))
        _print_summary(results)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 5: Convert signal to target weight
    # ------------------------------------------------------------------
    _step(5, "Signal -> target weight via SignalBridge")
    try:
        signals = {ticker: signal_value}
        returns_df = clean_df["Close"].pct_change().dropna().to_frame(name=ticker)

        bridge = SignalBridge(config=config)
        target_weights = bridge.signals_to_weights(signals, returns_df)

        print(f"{_PASS} (weight={target_weights.get(ticker, 0):.4f})")
        results.append(("Signal to weight", _PASS))
    except Exception as exc:
        print(f"{_FAIL} ({exc})")
        results.append(("Signal to weight", _FAIL))
        _print_summary(results)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 6: Compute and execute paper trade
    # ------------------------------------------------------------------
    _step(6, "Paper trade via OMS + PaperBroker")
    try:
        broker = PaperBroker(config=config)
        broker.connect()
        broker.set_price(ticker, last_close)

        oms = OrderManagementSystem(broker, config=config)
        positions = broker.get_positions()
        account_value = broker.get_account_value()
        prices = {ticker: last_close}

        orders = oms.compute_rebalance_orders(
            target_weights, positions, account_value, prices,
        )

        if not orders:
            print(f"{_PASS} (no orders needed -- already at target)")
            results.append(("Paper trade", _PASS))
        else:
            # Set est_price on orders for proper fills
            for order in orders:
                order.est_price = prices[order.ticker]

            report = oms.execute_plan(orders, dry_run=False)
            executed = len(report.orders_executed)
            failed = len(report.orders_failed)

            if failed > 0:
                raise RuntimeError(f"{failed} orders failed")

            order = orders[0]
            print(f"{_PASS} ({order.side.upper()} {order.quantity} {order.ticker} "
                  f"@ {last_close:.2f})")
            results.append(("Paper trade", _PASS))
    except Exception as exc:
        print(f"{_FAIL} ({exc})")
        results.append(("Paper trade", _FAIL))
        _print_summary(results)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 7: Reconcile
    # ------------------------------------------------------------------
    _step(7, "Reconcile positions")
    try:
        recon = oms.reconcile(target_weights, account_value, prices)
        aligned = recon.get("aligned", False)
        drift = recon.get("total_drift", 0.0)
        final_positions = broker.get_positions()
        final_value = broker.get_account_value()

        status = "aligned" if aligned else f"drift={drift:.4f}"
        print(f"{_PASS} ({status})")

        print(f"\n  Final state:")
        print(f"    Account value: {final_value:,.2f}")
        print(f"    Cash: {broker.cash:,.2f}")
        for pos_ticker, pos in final_positions.items():
            qty = pos.get("quantity", pos) if isinstance(pos, dict) else pos
            print(f"    {pos_ticker}: {qty} shares")

        results.append(("Reconcile", _PASS))
    except Exception as exc:
        print(f"{_FAIL} ({exc})")
        results.append(("Reconcile", _FAIL))

    broker.disconnect()
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
