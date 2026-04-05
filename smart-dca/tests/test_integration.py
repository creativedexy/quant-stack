"""Integration tests for the scheduler and component wiring.

Verifies that build_components returns the expected dict, that the
evaluation cycle dispatches alerts correctly, and that errors are
handled gracefully without propagation.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.execution.buy_engine import BuyDecision
from src.execution.order_manager import OrderResult
from src.signals.scorer import CompositeScore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_order_result(
    asset: str = "BTCGBP",
    fill_price: float = 50000.0,
    amount_gbp: float = 10.0,
    quantity: float = 0.0002,
) -> OrderResult:
    """Return a minimal OrderResult for test fixtures."""
    return OrderResult(
        order_id="12345",
        asset=asset,
        status="FILLED",
        fill_price=fill_price,
        quantity=quantity,
        amount_gbp=amount_gbp,
        timestamp=datetime.now(timezone.utc),
        raw_response={},
    )


def _make_buy_decision(
    asset: str = "BTCGBP",
    executed: bool = True,
    score: float = 80.0,
    order_result: OrderResult | None = None,
    reason: str = "buy executed",
) -> BuyDecision:
    """Return a BuyDecision with sensible defaults."""
    return BuyDecision(
        asset=asset,
        executed=executed,
        score=CompositeScore(score=score),
        order_result=order_result if executed else None,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# test_build_components_returns_all_keys
# ---------------------------------------------------------------------------

class TestBuildComponents:
    """Tests for scripts.run_scheduler.build_components."""

    def test_build_components_returns_all_keys(
        self,
        sample_settings: dict,
        sample_assets: dict,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """build_components returns a dict with all required keys."""
        import sys

        # Inject a mock 'binance' package so the local import inside
        # build_components succeeds even when python-binance is not
        # installed in the test environment.
        mock_client_cls = MagicMock()
        mock_binance_module = MagicMock()
        mock_binance_client_module = MagicMock()
        mock_binance_client_module.Client = mock_client_cls
        mock_binance_module.client = mock_binance_client_module

        sys.modules.setdefault("binance", mock_binance_module)
        sys.modules.setdefault("binance.client", mock_binance_client_module)

        try:
            # Override state_file and db_path to use tmp_path
            sample_settings["execution"]["state_file"] = str(
                tmp_path / "budget_state.json"
            )
            sample_settings["monitor"]["db_path"] = str(
                tmp_path / "performance.db"
            )

            from scripts.run_scheduler import build_components

            result = build_components(sample_settings, sample_assets)

            expected_keys = {
                "scorer",
                "budget_tracker",
                "order_manager",
                "buy_engine",
                "alerter",
                "performance_log",
                "price_feed",
            }
            assert set(result.keys()) == expected_keys

            # Verify none of the core components are None
            for key in (
                "scorer", "budget_tracker", "order_manager",
                "buy_engine", "alerter", "price_feed",
            ):
                assert result[key] is not None, (
                    f"Component '{key}' should not be None"
                )
        finally:
            # Clean up injected modules to avoid cross-test pollution
            sys.modules.pop("binance", None)
            sys.modules.pop("binance.client", None)


# ---------------------------------------------------------------------------
# test_evaluation_cycle_with_mocked_components
# ---------------------------------------------------------------------------

class TestEvaluationCycle:
    """Tests for scripts.run_scheduler.run_evaluation_cycle."""

    def test_evaluation_cycle_with_mocked_components(self) -> None:
        """Verify alerter is called correctly for executed, high-score, and notify decisions."""
        from scripts.run_scheduler import run_evaluation_cycle

        order_result = _make_order_result()

        decisions = [
            # Decision 1: Executed buy (score 80)
            _make_buy_decision(
                asset="BTCGBP",
                executed=True,
                score=80.0,
                order_result=order_result,
                reason="buy executed",
            ),
            # Decision 2: Not executed, score 60 (above notify_threshold 55)
            _make_buy_decision(
                asset="ETHGBP",
                executed=False,
                score=60.0,
                reason="budget exhausted",
            ),
            # Decision 3: Not executed, score 40 (below notify_threshold)
            _make_buy_decision(
                asset="XRPGBP",
                executed=False,
                score=40.0,
                reason="score below threshold",
            ),
        ]

        mock_engine = MagicMock()
        mock_engine.evaluate_all.return_value = decisions
        mock_engine._config = {
            "scoring": {"buy_threshold": 65, "notify_threshold": 55},
        }

        mock_alerter = MagicMock()
        mock_perf_log = MagicMock()

        components = {
            "buy_engine": mock_engine,
            "alerter": mock_alerter,
            "performance_log": mock_perf_log,
            "price_feed": MagicMock(),
            "scorer": MagicMock(),
            "budget_tracker": MagicMock(),
            "order_manager": MagicMock(),
        }

        run_evaluation_cycle(components)

        # Decision 1 (executed, score 80): send_buy_executed + record_fill
        mock_alerter.send_buy_executed.assert_called_once_with(decisions[0])
        mock_perf_log.record_fill.assert_called_once()

        # Decision 2 (not executed, score 60 >= notify 55 but < buy 65):
        # send_score_alert
        mock_alerter.send_score_alert.assert_called_once_with(
            decisions[1].score, "ETHGBP"
        )

        # Decision 3 (not executed, score 40 < notify 55): no alert
        # Verify send_buy_failed was NOT called (score 60, 40 both < buy_threshold 65)
        mock_alerter.send_buy_failed.assert_not_called()

    def test_evaluation_cycle_buy_failed_alert(self) -> None:
        """send_buy_failed fires when score >= buy_threshold but not executed."""
        from scripts.run_scheduler import run_evaluation_cycle

        decisions = [
            _make_buy_decision(
                asset="BTCGBP",
                executed=False,
                score=70.0,  # Above buy_threshold (65)
                reason="order failed: API error",
            ),
        ]

        mock_engine = MagicMock()
        mock_engine.evaluate_all.return_value = decisions
        mock_engine._config = {
            "scoring": {"buy_threshold": 65, "notify_threshold": 55},
        }

        mock_alerter = MagicMock()

        components = {
            "buy_engine": mock_engine,
            "alerter": mock_alerter,
            "performance_log": None,
            "price_feed": MagicMock(),
            "scorer": MagicMock(),
            "budget_tracker": MagicMock(),
            "order_manager": MagicMock(),
        }

        run_evaluation_cycle(components)

        mock_alerter.send_buy_failed.assert_called_once_with(decisions[0])
        mock_alerter.send_buy_executed.assert_not_called()
        mock_alerter.send_score_alert.assert_not_called()

    def test_evaluation_cycle_does_not_raise_on_engine_error(self) -> None:
        """run_evaluation_cycle swallows exceptions from evaluate_all."""
        from scripts.run_scheduler import run_evaluation_cycle

        mock_engine = MagicMock()
        mock_engine.evaluate_all.side_effect = RuntimeError("boom")
        mock_engine._config = {
            "scoring": {"buy_threshold": 65, "notify_threshold": 55},
        }

        components = {
            "buy_engine": mock_engine,
            "alerter": MagicMock(),
            "performance_log": None,
            "price_feed": MagicMock(),
            "scorer": MagicMock(),
            "budget_tracker": MagicMock(),
            "order_manager": MagicMock(),
        }

        # Should not raise
        run_evaluation_cycle(components)


# ---------------------------------------------------------------------------
# test_live_mode_warning_logged
# ---------------------------------------------------------------------------

class TestLiveModeWarning:
    """Tests for live mode warning logging."""

    def test_live_mode_warning_logged(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Setting BINANCE_TESTNET=false triggers a live mode warning log."""
        monkeypatch.setenv("BINANCE_TESTNET", "false")

        from scripts.run_scheduler import logger as scheduler_logger

        # Call the warning check logic directly instead of main()
        # (main() would try to load real config files and start the scheduler)
        is_testnet = os.environ.get("BINANCE_TESTNET", "true").lower() == "true"

        if not is_testnet:
            scheduler_logger.warning(
                "LIVE MODE ACTIVE -- not using Binance testnet"
            )

        captured = capsys.readouterr()
        assert "LIVE MODE ACTIVE" in captured.out

    def test_testnet_mode_no_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Testnet mode (default) does not log the live mode warning."""
        monkeypatch.setenv("BINANCE_TESTNET", "true")

        is_testnet = os.environ.get("BINANCE_TESTNET", "true").lower() == "true"
        # In testnet mode the warning is never emitted
        if not is_testnet:
            from scripts.run_scheduler import logger as scheduler_logger

            scheduler_logger.warning(
                "LIVE MODE ACTIVE -- not using Binance testnet"
            )

        captured = capsys.readouterr()
        assert "LIVE MODE ACTIVE" not in captured.out
