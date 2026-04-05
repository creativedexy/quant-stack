"""Integration tests for IBBroker with real IB Gateway.

All tests require a running IB Gateway session.  They are marked with
``@pytest.mark.integration`` and skip automatically if the gateway is
not reachable.
"""

from __future__ import annotations

import pytest

from src.utils.config import load_config

try:
    from src.execution.broker import IBBroker
    HAS_IBAPI = True
except ImportError:
    HAS_IBAPI = False


def _can_connect() -> bool:
    """Check if IB Gateway is reachable."""
    if not HAS_IBAPI:
        return False
    config = load_config()
    broker = IBBroker(config=config)
    try:
        connected = broker.connect()
        if connected:
            broker.disconnect()
        return connected
    except Exception:
        return False


# Skip entire module if gateway not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_IBAPI, reason="ibapi not installed"),
]

_gateway_available = None


@pytest.fixture(scope="module")
def ib_broker():
    """Create and connect an IBBroker for the test session."""
    global _gateway_available  # noqa: PLW0603
    if _gateway_available is None:
        _gateway_available = _can_connect()
    if not _gateway_available:
        pytest.skip("IB Gateway not reachable")

    config = load_config()
    broker = IBBroker(config=config)
    broker.connect()
    yield broker
    broker.disconnect()


class TestIBConnection:
    """Basic connection lifecycle."""

    def test_connect_and_disconnect(self, ib_broker: IBBroker) -> None:
        assert ib_broker.is_connected()

    def test_get_account_value(self, ib_broker: IBBroker) -> None:
        value = ib_broker.get_account_value()
        assert isinstance(value, float)

    def test_get_account_summary(self, ib_broker: IBBroker) -> None:
        summary = ib_broker.get_account_summary()
        assert "account_value" in summary
        assert "cash" in summary
        assert "invested" in summary

    def test_get_positions(self, ib_broker: IBBroker) -> None:
        positions = ib_broker.get_positions()
        assert isinstance(positions, dict)


class TestIBContractValidation:
    """Contract resolution against real IB."""

    def test_us_stock_resolves(self, ib_broker: IBBroker) -> None:
        result = ib_broker.validate_contract("AAPL")
        assert "error" not in result
        assert result.get("secType") == "STK"

    def test_uk_stock_resolves(self, ib_broker: IBBroker) -> None:
        result = ib_broker.validate_contract("SHEL.L")
        assert "error" not in result

    def test_invalid_ticker_returns_error(self, ib_broker: IBBroker) -> None:
        result = ib_broker.validate_contract("ZZZZZZZ.FAKE")
        assert "error" in result
