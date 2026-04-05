# IB Paper Trading Integration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Connect the quant-stack pipeline to Interactive Brokers paper trading, enabling end-to-end signal-to-execution with real IB Gateway.

**Architecture:** The IBBroker, _IBConnection, SignalBridge, and OMS are already built. This plan installs dependencies, fixes placeholder values, validates contracts against real IB, and wires the full pipeline (signals -> rebalance -> paper fills).

**Tech Stack:** ibapi (IB API), existing IBBroker/_IBConnection, FastAPI execution service

---

### Task 1: Install IB Gateway & ibapi (Manual/Interactive)

This task requires user interaction — IB Gateway installer is in the Downloads folder.

**Step 1: Install IB Gateway**

The installer is in the user's Downloads folder. Run:
```bash
ls /c/Users/dexte/Downloads/*gateway* /c/Users/dexte/Downloads/*IB* 2>/dev/null
```

Launch the installer and follow the GUI. Key settings during setup:
- Use paper trading account credentials
- Accept default installation directory

**Step 2: Configure IB Gateway for API access**

After installation, launch IB Gateway and log in with paper trading credentials.
Then in the IB Gateway GUI:
- Go to Configure -> Settings -> API -> Settings
- Check "Enable ActiveX and Socket Clients"
- Set "Socket port" to `7497` (paper trading port)
- Set "Trusted IPs" to `127.0.0.1`
- Uncheck "Read-Only API" (needed to place orders)
- Click Apply/OK

**Step 3: Install ibapi into the project venv**

Run:
```bash
.venv/Scripts/pip install ibapi
```

**Step 4: Verify ibapi unlocks existing tests**

Run:
```bash
py -3 -m pytest tests/test_execution/test_ib_broker.py -v
```
Expected: 23 tests PASS (previously skipped)

**Step 5: Run the existing connection test**

Run:
```bash
py -3 -m pytest tests/test_execution/test_ib_broker.py -v --count
```
Expected: All 23 pass (they use MockIBConnection, no real gateway needed)

---

### Task 2: Validate Real IB Gateway Connection

**Prerequisite:** IB Gateway running and logged in (paper account)

**Files:**
- Test: `scripts/test_ib_connection.py` (existing script)

**Step 1: Run the connection diagnostic**

Run:
```bash
py -3 -m scripts.test_ib_connection
```
Expected output:
```
[1/5] Connecting to IB gateway at 127.0.0.1:7497 ...  PASS
[2/5] Querying account value ...                       PASS  (value > 0)
[3/5] Listing positions ...                            PASS  (may be empty)
[4/5] Skipping test order (use --test-order to enable)
[5/5] Disconnecting ...                                PASS
```

If connect fails: check IB Gateway is running, port 7497 is configured, API access is enabled.

**Step 2: Place a test order (1 share AAPL)**

Run:
```bash
py -3 -m scripts.test_ib_connection --test-order --ticker AAPL
```
Expected: Order placed, fill confirmed (paper account, market order, 1 share)

**Step 3: Verify the position appeared**

Run:
```bash
py -3 -m scripts.test_ib_connection
```
Expected: positions list now includes AAPL (1 share)

---

### Task 3: Fix get_account_summary() Placeholder

**Files:**
- Modify: `src/execution/broker.py:812-829` (get_account_summary)
- Modify: `src/execution/broker.py:974-976` (reqAccountSummary call in get_account_value)
- Test: `tests/test_execution/test_ib_broker.py`

**Step 1: Write the failing test**

Add to `tests/test_execution/test_ib_broker.py`:

```python
class TestIBBrokerAccountSummary:
    """get_account_summary returns real values, not placeholders."""

    @pytest.mark.skipif(not HAS_IBAPI, reason="ibapi not installed")
    def test_account_summary_has_real_cash_value(self) -> None:
        """Cash and invested should not be hardcoded zeros."""
        broker = IBBroker(config=_test_config())
        broker._connected = True
        broker._conn = MockIBConnection()
        # Set account values as IB would
        broker._conn._account_values["NetLiquidation"] = "150000"
        broker._conn._account_values["TotalCashValue"] = "50000"
        broker._conn._account_values["GrossPositionValue"] = "100000"
        broker._conn._account_ready.set()

        summary = broker.get_account_summary()
        assert summary["cash"] == 50000.0
        assert summary["invested"] == 100000.0
        assert summary["account_value"] == 150000.0
```

**Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_execution/test_ib_broker.py::TestIBBrokerAccountSummary -v`
Expected: FAIL (cash=0.0, invested=0.0 — placeholders)

**Step 3: Fix get_account_summary()**

In `src/execution/broker.py`, replace `get_account_summary()` (lines 812-829):

```python
def get_account_summary(self) -> dict[str, Any]:
    """Return account-level information matching PaperBroker interface.

    Requests NetLiquidation, TotalCashValue, and GrossPositionValue
    from IB to compute account overview.

    Returns:
        Dict with ``account_value``, ``cash``, ``invested``,
        ``positions_count``.
    """
    if not self._connected or self._conn is None:
        return {
            "account_value": 0.0,
            "cash": 0.0,
            "invested": 0.0,
            "positions_count": 0,
        }

    with self._conn._lock:
        self._conn._account_values.clear()
    self._conn._account_ready.clear()

    self._conn.reqAccountSummary(
        9002, "All",
        "NetLiquidation,TotalCashValue,GrossPositionValue",
    )

    if not self._conn._account_ready.wait(timeout=self.timeout):
        logger.warning(
            "Timeout waiting for account summary (%.0fs)",
            self.timeout,
        )

    try:
        self._conn.cancelAccountSummary(9002)
    except Exception:
        pass

    with self._conn._lock:
        vals = self._conn._account_values

    def _parse(key: str) -> float:
        try:
            return float(vals.get(key, "0"))
        except ValueError:
            return 0.0

    positions = self.get_positions()
    return {
        "account_value": _parse("NetLiquidation"),
        "cash": _parse("TotalCashValue"),
        "invested": _parse("GrossPositionValue"),
        "positions_count": len(positions),
    }
```

**Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_execution/test_ib_broker.py::TestIBBrokerAccountSummary -v`
Expected: PASS

**Step 5: Run full IB broker test suite**

Run: `py -3 -m pytest tests/test_execution/test_ib_broker.py -v`
Expected: All pass (24+ tests)

---

### Task 4: Add Contract Validation

**Files:**
- Modify: `src/execution/broker.py` (add validate_contract method)
- Modify: `scripts/test_ib_connection.py` (add --validate-contracts flag)
- Test: `tests/test_execution/test_ib_broker.py`

**Step 1: Write the failing test**

Add to `tests/test_execution/test_ib_broker.py`:

```python
class TestIBBrokerContractValidation:
    """Contract validation method exists and works."""

    @pytest.mark.skipif(not HAS_IBAPI, reason="ibapi not installed")
    def test_validate_contract_returns_dict(self) -> None:
        """validate_contract should return contract details dict."""
        broker = IBBroker(config=_test_config())
        broker._connected = True
        broker._conn = MockIBConnection()

        # Mock the contractDetails callback
        broker._conn._contract_details = {"conId": 12345, "longName": "Test Corp"}
        broker._conn._contract_ready = threading.Event()
        broker._conn._contract_ready.set()

        result = broker.validate_contract("AAPL")
        assert isinstance(result, dict)
        assert "conId" in result or "error" in result
```

**Step 2: Add _contract_ready Event and contractDetails callback to _IBConnection**

In `src/execution/broker.py`, add to `_IBConnection.__init__()` (around line 438):

```python
self._contract_ready = threading.Event()
self._contract_details: dict[str, Any] = {}
```

Add callback method to `_IBConnection` (after accountSummaryEnd):

```python
def contractDetails(  # noqa: N802
    self,
    reqId: int,  # noqa: N803
    contractDetails: Any,  # noqa: N803
) -> None:
    """Receive contract details from IB."""
    self._touch()
    with self._lock:
        self._contract_details = {
            "conId": contractDetails.contract.conId,
            "longName": contractDetails.longName,
            "secType": contractDetails.contract.secType,
            "exchange": contractDetails.contract.exchange,
            "currency": contractDetails.contract.currency,
        }

def contractDetailsEnd(  # noqa: N802
    self,
    reqId: int,  # noqa: N803
) -> None:
    """All contract details received."""
    self._touch()
    self._contract_ready.set()
```

**Step 3: Add validate_contract() to IBBroker**

In `src/execution/broker.py`, add after `_make_contract()` (after line 930):

```python
def validate_contract(self, ticker: str) -> dict[str, Any]:
    """Validate a ticker resolves to a real IB contract.

    Uses ``reqContractDetails`` to check whether IB recognises
    the contract built by ``_make_contract``.

    Args:
        ticker: System ticker symbol (e.g. ``"SHEL.L"``).

    Returns:
        Dict with contract details if valid, or ``{"error": "..."}``
        if not found or not connected.
    """
    if not self._connected or self._conn is None:
        return {"error": "Not connected to IB gateway"}

    contract = self._make_contract(ticker)

    with self._conn._lock:
        self._conn._contract_details.clear()
    self._conn._contract_ready.clear()

    self._conn.reqContractDetails(9003, contract)

    if not self._conn._contract_ready.wait(timeout=self.timeout):
        return {"error": f"Timeout resolving contract for {ticker}"}

    with self._conn._lock:
        details = dict(self._conn._contract_details)

    if not details:
        return {"error": f"No contract found for {ticker}"}

    return details
```

**Step 4: Add --validate-contracts flag to test_ib_connection.py**

In `scripts/test_ib_connection.py`, add a new check after the positions check that validates contracts for the user's watchlist tickers:

```python
# After existing position check
if args.validate_contracts:
    tickers = ["SHEL.L", "AAPL", "BTC.CFD"]  # Representative set
    for ticker in tickers:
        result = broker.validate_contract(ticker)
        if "error" in result:
            print(f"  {ticker}: FAIL - {result['error']}")
        else:
            print(f"  {ticker}: OK - {result.get('longName', 'resolved')}")
```

Add argparse argument:
```python
parser.add_argument(
    "--validate-contracts",
    action="store_true",
    help="Validate contract resolution for representative tickers",
)
```

**Step 5: Run tests**

Run: `py -3 -m pytest tests/test_execution/test_ib_broker.py -v`
Expected: All pass

---

### Task 5: Test Orders for All Asset Types (Manual Validation)

**Prerequisite:** IB Gateway running, Tasks 1-4 complete

**Step 1: Test UK stock order**

Run:
```bash
py -3 -m scripts.test_ib_connection --test-order --ticker SHEL.L
```
Expected: 1-share market order fills on LSE

**Step 2: Test US stock order**

Run:
```bash
py -3 -m scripts.test_ib_connection --test-order --ticker AAPL
```
Expected: 1-share market order fills on SMART

**Step 3: Test crypto CFD order**

Run:
```bash
py -3 -m scripts.test_ib_connection --test-order --ticker BTC.CFD
```
Expected: Order fills (or note if CFDs not available on paper account — some IB paper accounts don't support CFDs)

**Step 4: Validate all contracts**

Run:
```bash
py -3 -m scripts.test_ib_connection --validate-contracts
```
Expected: All 3 tickers resolve successfully

---

### Task 6: Wire Full Pipeline to IBBroker

**Files:**
- Modify: `config/settings.yaml` (switch broker.type)
- Test via: `scripts/run_rebalance.py`

**Step 1: Switch config to use IBBroker**

In `config/settings.yaml`, change:
```yaml
execution:
  mode: paper
  broker:
    type: ibkr    # was: paper
```

**Step 2: Run a signal-driven rebalance**

Run:
```bash
py -3 -m scripts.run_rebalance --alpha momentum
```

This triggers the full pipeline:
1. Loads strategy signals from `strategy_registry`
2. `SignalBridge.to_weights()` converts signals to target weights
3. `Optimiser` adjusts weights
4. `OMS.generate_orders()` compares targets to current positions
5. `IBBroker.place_order()` sends each order to IB Gateway
6. Orders fill on the paper account

Expected: Orders placed and filled (or partially filled during market hours). Report printed with order details.

**Step 3: Verify positions updated**

Run:
```bash
py -3 -m scripts.test_ib_connection
```
Expected: Positions list reflects the rebalance trades

**Step 4: Revert config to paper for safety**

In `config/settings.yaml`, change back:
```yaml
execution:
  broker:
    type: paper    # safe default
```

---

### Task 7: Write Integration Tests

**Files:**
- Create: `tests/test_execution/test_ib_integration.py`

**Step 1: Create integration test file**

Create `tests/test_execution/test_ib_integration.py`:

```python
"""Integration tests for IBBroker with real IB Gateway.

All tests require a running IB Gateway paper trading session.
They are marked with ``@pytest.mark.integration`` and skip
automatically if the gateway is not reachable.
"""

from __future__ import annotations

import threading
import time

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
    global _gateway_available
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
        assert value > 0

    def test_get_account_summary(self, ib_broker: IBBroker) -> None:
        summary = ib_broker.get_account_summary()
        assert "account_value" in summary
        assert "cash" in summary
        assert "invested" in summary
        assert summary["account_value"] > 0

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


class TestIBOrders:
    """Paper trading order placement."""

    def test_place_market_order(self, ib_broker: IBBroker) -> None:
        """Place a 1-share market order on AAPL (paper account)."""
        result = ib_broker.place_order(
            ticker="AAPL",
            quantity=1,
            side="BUY",
            order_type="market",
        )
        assert result["status"] in ("filled", "submitted", "cancelled")
        assert result["ticker"] == "AAPL"
```

**Step 2: Run integration tests (requires IB Gateway)**

Run:
```bash
py -3 -m pytest tests/test_execution/test_ib_integration.py -v -m integration
```
Expected: All pass if IB Gateway running; all skip if not

**Step 3: Verify integration tests don't break normal test suite**

Run:
```bash
py -3 -m pytest tests/ -v --ignore=tests/test_models/test_automl.py -m "not integration"
```
Expected: All existing tests pass (integration tests skipped)

---

### Task 8: Run Full Test Suite

**Step 1: Full test suite without integration**

Run:
```bash
py -3 -m pytest tests/ -v --ignore=tests/test_models/test_automl.py
```
Expected: 1017+ passed, 0 failures

**Step 2: Full test suite with integration (IB Gateway running)**

Run:
```bash
py -3 -m pytest tests/ -v --ignore=tests/test_models/test_automl.py -m "integration or not integration"
```
Expected: All pass including integration tests

**Step 3: Smoke test the execution pipeline**

Run:
```bash
py -3 -m scripts.smoke_test_execution
```
Expected: All 7 steps pass
