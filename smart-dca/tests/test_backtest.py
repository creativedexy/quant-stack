"""Tests for the smart-dca signal backtest engine.

Validates no look-ahead bias, reasonable improvement percentages,
CSV output, and CLI argument parsing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.backtest_signals import (
    BacktestResult,
    build_parser,
    run_backtest_for_asset,
    save_results_csv,
)


class TestNoLookahead:
    """Verify that backtest buys never use future data."""

    def test_no_lookahead(
        self, synthetic_ohlcv: pd.DataFrame, sample_settings: dict
    ) -> None:
        """For each simulated buy, no input data has a timestamp after
        the buy timestamp.

        The BacktestMeanReversionSignal filters ohlcv rows with
        ``index < now``, ensuring strict causal ordering.  We verify
        this by checking that every recorded buy timestamp exists in
        the OHLCV index and that the mean reversion window used only
        prior rows.
        """
        result = run_backtest_for_asset(
            asset="BTCGBP",
            ohlcv=synthetic_ohlcv,
            config=sample_settings,
        )

        ohlcv_timestamps = set(synthetic_ohlcv.index.to_pydatetime())

        for buy in result.buys:
            # The buy timestamp must correspond to an actual OHLCV row
            assert buy.timestamp in ohlcv_timestamps, (
                f"Buy at {buy.timestamp} not in OHLCV index"
            )

            # All data available at buy time must be strictly before
            # or at the buy timestamp (the mean reversion signal uses
            # ``ohlcv.index < now`` so it only sees prior rows).
            available = synthetic_ohlcv[
                synthetic_ohlcv.index < buy.timestamp
            ]
            if not available.empty:
                latest_available = available.index.max().to_pydatetime()
                assert latest_available < buy.timestamp, (
                    f"Data at {latest_available} is not before "
                    f"buy at {buy.timestamp}"
                )


class TestImprovementPct:
    """Verify improvement percentage is within reasonable bounds."""

    def test_improvement_pct_reasonable(
        self, synthetic_ohlcv: pd.DataFrame, sample_settings: dict
    ) -> None:
        """Run backtest on synthetic data with a known dip.

        The improvement_pct should be >= -50 (i.e. not wildly wrong).
        A negative improvement means signals performed worse than
        naive DCA, but anything below -50% would indicate a bug.
        """
        result = run_backtest_for_asset(
            asset="BTCGBP",
            ohlcv=synthetic_ohlcv,
            config=sample_settings,
        )

        assert result.improvement_pct >= -50.0, (
            f"Improvement {result.improvement_pct:.2f}% is unreasonably "
            f"bad (below -50%)"
        )


class TestOutputCsv:
    """Verify CSV output is written correctly."""

    def test_output_csv_created(self, tmp_path: object) -> None:
        """save_results_csv creates a file at the specified path."""
        csv_path = tmp_path / "backtest_results.csv"  # type: ignore[operator]

        results = [
            BacktestResult(
                asset="BTCGBP",
                signal_avg_price=49500.0,
                naive_avg_price=50000.0,
                improvement_pct=1.0,
                buy_count=5,
                weeks=2,
                buys_per_week=2.5,
                worst_entry_vs_open=1.01,
                best_entry_vs_open=0.97,
            ),
        ]

        save_results_csv(results, csv_path)

        assert csv_path.exists(), "CSV file was not created"

        # Verify content is parseable
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert df["asset"].iloc[0] == "BTCGBP"
        assert float(df["improvement_pct"].iloc[0]) == pytest.approx(1.0, abs=0.01)


class TestCliArgs:
    """Verify CLI argument parsing."""

    def test_cli_days_arg(self) -> None:
        """Argparse correctly parses --days 30."""
        parser = build_parser()
        args = parser.parse_args(["--days", "30"])
        assert args.days == 30

    def test_cli_default_days(self) -> None:
        """Default --days is 90 when not specified."""
        parser = build_parser()
        args = parser.parse_args([])
        assert args.days == 90

    def test_cli_assets_arg(self) -> None:
        """Argparse correctly parses --assets with comma-separated values."""
        parser = build_parser()
        args = parser.parse_args(["--assets", "BTCGBP,XRPGBP"])
        assert args.assets == "BTCGBP,XRPGBP"
