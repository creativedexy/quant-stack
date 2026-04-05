"""Standalone diagnostic script to verify IB Gateway/TWS connectivity.

Runs five sequential checks and reports PASS/FAIL for each.  Designed
for quick validation after installing ibapi or configuring a new
TWS/Gateway instance.

Usage::

    py -3 -m scripts.test_ib_connection
    py -3 -m scripts.test_ib_connection --test-order --ticker MSFT
    py -3 -m scripts.test_ib_connection --host 192.168.1.50 --port 4002
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Step names for the summary table.
_STEPS: list[str] = [
    "Connect to TWS/Gateway",
    "Query account value",
    "List positions",
    "Place test order",
    "Validate contracts",
    "Disconnect cleanly",
]


def _banner(step_num: int, total: int, description: str) -> None:
    """Print a clear step header."""
    print(f"\n{'=' * 60}")
    print(f"  Step {step_num}/{total}: {description}")
    print(f"{'=' * 60}")


def _result(passed: bool, detail: str = "") -> None:
    """Print PASS or FAIL with optional detail."""
    tag = "PASS" if passed else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  -> {tag}{suffix}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Diagnose IB Gateway/TWS connectivity end-to-end.",
    )
    parser.add_argument(
        "--test-order",
        action="store_true",
        default=False,
        help="Actually place a 1-share market order (default: skip).",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Ticker symbol for the test order (default: AAPL).",
    )
    parser.add_argument(
        "--validate-contracts",
        action="store_true",
        default=False,
        help="Validate contract resolution for representative tickers.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Override the TWS/Gateway host from config.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override the TWS/Gateway port (default from config, usually 7497).",
    )
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> dict[str, Any]:
    """Load settings.yaml and apply any CLI overrides.

    Returns:
        Merged configuration dict ready for ``IBBroker``.
    """
    from src.utils.config import load_config

    config = load_config()

    # Apply host/port overrides into the nested execution.broker dict.
    exec_cfg = config.setdefault("execution", {})
    broker_cfg = exec_cfg.get("broker", {})
    if isinstance(broker_cfg, str):
        broker_cfg = {}
    exec_cfg.setdefault("broker", broker_cfg)

    if args.host is not None:
        broker_cfg["host"] = args.host
    if args.port is not None:
        broker_cfg["port"] = args.port

    return config


def run_diagnostics(args: argparse.Namespace) -> int:
    """Execute all diagnostic steps and return an exit code.

    Returns:
        0 if every executed step passed, 1 otherwise.
    """
    # Import guard -- ibapi is an optional dependency.
    try:
        from src.execution.broker import IBBroker  # noqa: F401
    except ImportError:
        print(
            "\nERROR: ibapi is not installed.\n"
            "\n"
            "Install the Interactive Brokers Python API:\n"
            "  pip install ibapi\n"
            "\n"
            "Or download from:\n"
            "  https://interactivebrokers.github.io/\n"
        )
        return 1

    config = _build_config(args)
    results: dict[int, bool] = {}
    broker: IBBroker | None = None
    total_steps = len(_STEPS)

    try:
        # ----------------------------------------------------------
        # Step 1: Connect
        # ----------------------------------------------------------
        _banner(1, total_steps, _STEPS[0])
        try:
            broker = IBBroker(config)
            connected = broker.connect()
            if connected and broker.is_connected():
                mode = broker.get_mode()
                _result(True, f"mode={mode}, host={broker.host}, port={broker.port}")
                results[1] = True
            else:
                _result(False, "connect() returned False or is_connected() is False")
                results[1] = False
        except Exception as exc:
            _result(False, str(exc))
            results[1] = False
            logger.error("Step 1 failed: %s", exc, exc_info=True)

        # If connection failed there is no point running the rest.
        if not results.get(1, False):
            print("\nConnection failed -> skipping remaining steps.")
            for step in range(2, total_steps + 1):
                results[step] = False
            return _print_summary(results, args)

        # ----------------------------------------------------------
        # Step 2: Query account value
        # ----------------------------------------------------------
        _banner(2, total_steps, _STEPS[1])
        try:
            account_value = broker.get_account_value()
            _result(True, f"account value = {account_value:,.2f}")
            results[2] = True
        except Exception as exc:
            _result(False, str(exc))
            results[2] = False
            logger.error("Step 2 failed: %s", exc, exc_info=True)

        # ----------------------------------------------------------
        # Step 3: List positions
        # ----------------------------------------------------------
        _banner(3, total_steps, _STEPS[2])
        try:
            positions = broker.get_positions()
            if positions:
                for ticker, qty in positions.items():
                    print(f"    {ticker}: {qty}")
            else:
                print("    (no open positions)")
            _result(True, f"{len(positions)} position(s)")
            results[3] = True
        except Exception as exc:
            _result(False, str(exc))
            results[3] = False
            logger.error("Step 3 failed: %s", exc, exc_info=True)

        # ----------------------------------------------------------
        # Step 4: Place test order (optional)
        # ----------------------------------------------------------
        _banner(4, total_steps, _STEPS[3])
        if not args.test_order:
            print("    Skipped (pass --test-order to enable).")
            results[4] = True  # Skipped counts as pass.
        else:
            try:
                receipt = broker.place_order(
                    ticker=args.ticker,
                    quantity=1,
                    side="buy",
                    order_type="market",
                )
                status = receipt.get("status", "unknown")
                order_id = receipt.get("order_id", "n/a")
                _result(True, f"order_id={order_id}, status={status}")
                results[4] = True
            except Exception as exc:
                _result(False, str(exc))
                results[4] = False
                logger.error("Step 4 failed: %s", exc, exc_info=True)

        # ----------------------------------------------------------
        # Step 5: Validate contracts (optional)
        # ----------------------------------------------------------
        _banner(5, total_steps, _STEPS[4])
        if not args.validate_contracts:
            print("    Skipped (pass --validate-contracts to enable).")
            results[5] = True  # Skipped counts as pass.
        else:
            try:
                tickers = ["SHEL.L", "AAPL", "BTC.CFD"]
                all_ok = True
                for ticker in tickers:
                    result = broker.validate_contract(ticker)
                    if "error" in result:
                        print(f"    {ticker}: FAIL - {result['error']}")
                        all_ok = False
                    else:
                        name = result.get("longName", "resolved")
                        print(f"    {ticker}: OK - {name}")
                _result(all_ok, f"{len(tickers)} tickers checked")
                results[5] = all_ok
            except Exception as exc:
                _result(False, str(exc))
                results[5] = False
                logger.error("Step 5 failed: %s", exc, exc_info=True)

        # ----------------------------------------------------------
        # Step 6: Disconnect
        # ----------------------------------------------------------
        _banner(6, total_steps, _STEPS[5])
        try:
            broker.disconnect()
            still_connected = broker.is_connected()
            if not still_connected:
                _result(True, "disconnected successfully")
                results[6] = True
            else:
                _result(False, "is_connected() still True after disconnect()")
                results[6] = False
        except Exception as exc:
            _result(False, str(exc))
            results[6] = False
            logger.error("Step 6 failed: %s", exc, exc_info=True)

    finally:
        # Guarantee we try to disconnect even on unexpected errors.
        if broker is not None:
            try:
                if broker.is_connected():
                    broker.disconnect()
            except Exception:
                pass

    return _print_summary(results, args)


def _print_summary(results: dict[int, bool], args: argparse.Namespace) -> int:
    """Print a summary table and return the appropriate exit code.

    Returns:
        0 if all steps passed, 1 otherwise.
    """
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    all_passed = True
    for i, name in enumerate(_STEPS, start=1):
        passed = results.get(i, False)
        tag = "PASS" if passed else "FAIL"
        skip_note = ""
        if i == 4 and not args.test_order:
            skip_note = " (skipped)"
        if i == 5 and not args.validate_contracts:
            skip_note = " (skipped)"
        print(f"  Step {i}: {tag}{skip_note} - {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n  All steps passed. IB connectivity is working.\n")
    else:
        failed = [str(i) for i, p in results.items() if not p]
        print(f"\n  Steps failed: {', '.join(failed)}\n")

    return 0 if all_passed else 1


def main() -> None:
    """Entry point for ``py -3 -m scripts.test_ib_connection``."""
    args = _parse_args()
    exit_code = run_diagnostics(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
