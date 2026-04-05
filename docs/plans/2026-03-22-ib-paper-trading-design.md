# IB Paper Trading Integration Design — 2026-03-22

## Problem

The quant-stack has a fully built IBBroker with threading, heartbeat, and auto-reconnect,
but it has never been tested against a real IB Gateway instance. The ibapi library is not
installed, TWS/IB Gateway is not configured, and several placeholder values exist in the
code. The goal is to wire the complete pipeline (signals -> rebalance -> paper fills) using
real IB Gateway paper trading.

## Scope

- Install IB Gateway + ibapi
- Validate connection and fix placeholder implementations
- Test contract resolution for UK stocks (LSE), US stocks (SMART), crypto CFDs
- Wire full pipeline: scheduler/rebalance -> SignalBridge -> OMS -> IBBroker -> paper fills
- Add integration tests (skipped when IB Gateway not available)

## Architecture

No new architecture needed — the existing pipeline and IBBroker are already designed for
this. The work is connecting the last mile: installing dependencies, fixing placeholders,
and testing with real IB Gateway.

```
Scheduler / run_rebalance.py
  -> FeaturePipeline.generate()        (already wired)
  -> strategy_registry.get_signals()   (already wired)
  -> SignalBridge.to_weights()          (already built)
  -> Optimiser.optimise()              (already built)
  -> OMS.generate_orders()             (already built)
  -> IBBroker.place_order()            (connect here)
```

## Phase 1: Installation & Setup

- Install IB Gateway (from user's Downloads folder)
- Configure paper account: API enabled, port 7497, trusted IP 127.0.0.1
- Install ibapi: `pip install ibapi` + add to pyproject.toml [execution] extras
- Verify: 23 existing IB tests should now run (not skip)

## Phase 2: Connection Validation & Placeholder Fixes

- Run `py -3 -m scripts.test_ib_connection` against real IB Gateway
- Fix `get_account_summary()` in broker.py:
  - Replace hardcoded TotalCashValue/GrossPositionValue
  - Request via `reqAccountSummary()` with proper callbacks
- Test: account value, positions, connection lifecycle

## Phase 3: Contract Validation

Test `_make_contract()` for all 3 asset types:

| Ticker | Exchange | Currency | SecType |
|--------|----------|----------|---------|
| SHEL.L | LSE | GBP | STK |
| AAPL | SMART | USD | STK |
| BTC.CFD | SMART | USD | CFD |

Add `validate_contract()` method using `reqContractDetails()`.
Add `--validate-contracts` flag to test_ib_connection.py.

## Phase 4: Pipeline Wiring

- Switch config `broker.type` from `paper` to `ibkr`
- Run `py -3 -m scripts.run_rebalance --alpha momentum` with IBBroker
- Verify orders flow through SignalBridge -> OMS -> IBBroker -> paper fills
- Verify scheduler auto-rebalance works with IBBroker (paper mode only)

## Phase 5: Integration Tests

New `tests/test_execution/test_ib_integration.py`:
- All marked `@pytest.mark.integration`
- Skip if IB Gateway not reachable
- Tests: connect, account query, position query, place order, cancel order
- Paper account only

## Safety

- Config defaults to `broker.type: paper`
- `ibkr` is explicit config change
- Live mode requires separate `mode: live` AND confirmation
- Scheduler never auto-executes in live mode
- All test orders use 1 share minimum

## Files to Modify

- `pyproject.toml` — add ibapi to [execution] extras
- `src/execution/broker.py` — fix get_account_summary(), add validate_contract()
- `scripts/test_ib_connection.py` — add --validate-contracts flag
- `config/settings.yaml` — switch broker.type to ibkr (manual step)
- `tests/test_execution/test_ib_integration.py` — new file

## Files Already Complete (no changes needed)

- `src/execution/broker.py` — IBBroker, _IBConnection, _make_contract, translate_ticker
- `src/execution/oms.py` — OrderManagementSystem
- `src/execution/signal_bridge.py` — SignalBridge
- `src/services/execution_service.py` — connect_ib_broker()
- `scripts/run_rebalance.py` — already uses create_broker()
- `src/scheduler/pipeline.py` — already calls broker
