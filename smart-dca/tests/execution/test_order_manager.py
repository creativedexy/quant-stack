"""Tests for BinanceOrderManager."""

from __future__ import annotations

import pytest

from src.execution.order_manager import BinanceOrderManager, OrderResult
from src.utils.exceptions import OrderError


@pytest.fixture
def config(sample_settings: dict) -> dict:
    return sample_settings


@pytest.fixture
def manager(mock_binance_client: object, config: dict) -> BinanceOrderManager:
    return BinanceOrderManager(mock_binance_client, config)


class TestBinanceOrderManager:
    def test_place_limit_buy_calls_create_order(
        self, manager: BinanceOrderManager, mock_binance_client: object
    ) -> None:
        """Verify client.create_order called with correct parameters."""
        manager.place_limit_buy("BTCGBP", 10.0, 50000.0)
        mock_binance_client.create_order.assert_called_once()
        call_kwargs = mock_binance_client.create_order.call_args[1]
        assert call_kwargs["side"] == "BUY"
        assert call_kwargs["type"] == "LIMIT"
        assert call_kwargs["timeInForce"] == "GTC"
        assert call_kwargs["symbol"] == "BTCGBP"

    def test_place_limit_buy_amount_above_max_raises(
        self, manager: BinanceOrderManager
    ) -> None:
        """amount_gbp=100, max=50 -> OrderError."""
        with pytest.raises(OrderError):
            manager.place_limit_buy("BTCGBP", 100.0, 50000.0)

    def test_place_limit_buy_returns_order_result(
        self, manager: BinanceOrderManager
    ) -> None:
        """Mock response -> OrderResult with correct status."""
        result = manager.place_limit_buy("BTCGBP", 10.0, 50000.0)
        assert isinstance(result, OrderResult)
        assert result.order_id == "12345"
        assert result.asset == "BTCGBP"
        assert result.status == "NEW"

    def test_get_order_status_maps_correctly(
        self, manager: BinanceOrderManager
    ) -> None:
        """Mock FILLED response -> status 'FILLED'."""
        result = manager.get_order_status("BTCGBP", "12345")
        assert result.status == "FILLED"
        assert result.order_id == "12345"

    def test_cancel_order_returns_cancelled(
        self, manager: BinanceOrderManager
    ) -> None:
        """Mock cancel response -> status 'CANCELLED'."""
        result = manager.cancel_order("BTCGBP", "12345")
        assert result.status == "CANCELLED"

    def test_get_account_balance_returns_free(
        self, manager: BinanceOrderManager
    ) -> None:
        """Mock balance response -> float."""
        balance = manager.get_account_balance("GBP")
        assert balance == pytest.approx(100.0)

    def test_binance_exception_wrapped_in_order_error(
        self, manager: BinanceOrderManager, mock_binance_client: object
    ) -> None:
        """Mock raises BinanceAPIException -> OrderError raised."""
        mock_binance_client.create_order.side_effect = Exception("API error")
        with pytest.raises(OrderError):
            manager.place_limit_buy("BTCGBP", 10.0, 50000.0)
