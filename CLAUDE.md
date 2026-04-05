# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Quant Stack — Automated Trading Workflow

## Project Overview
A modular, open-source Python-based quantitative trading system following a pipeline architecture:
**Data → Features → Models → Portfolio Optimisation → Backtest → Execution**

Built for a solo quant / small team to prototype, test, and deploy trading strategies
without proprietary software. Designed to scale from personal capital to small fund operations.

UI: FastAPI + HTMX + Alpine.js production interface served by the existing
FastAPI backend with no JavaScript build chain.

## Architecture Principles
- **Pipeline pattern**: Each stage is independent and composable
- **Config-driven**: All parameters in YAML, never hardcoded
- **No lookahead bias**: Strictly enforced in all feature/model/backtest code
- **Dual-mode data**: Synthetic data for testing, live APIs for production
- **Fail-safe execution**: All trading code defaults to paper trading unless explicitly overridden
- **Production UI**: FastAPI + HTMX for the web interface. The service layer
  is UI-agnostic. UI technology never drives decisions in the quant stack.

## Commands
- **Install**: `pip install -e ".[all]"`
- **Test (all)**: `py -3 -m pytest tests/ -v --ignore=tests/test_models/test_automl.py`
- **Test (single file)**: `py -3 -m pytest tests/test_data/test_fetcher.py -v`
- **Test (integration)**: `py -3 -m pytest tests/ -v -m integration`
- **Lint**: `py -3 -m ruff check src/ tests/`
- **Format**: `py -3 -m ruff format src/ tests/`
- **Type check**: `py -3 -m mypy src/`
- **Synthetic data**: `py -3 -m scripts.fetch_data --source synthetic`
- **API server**: `py -3 -m scripts.run_api --port 8000 --reload`
- **Smoke test (pipeline)**: `py -3 -m scripts.smoke_test`
- **Smoke test (full)**: `.venv/Scripts/python scripts/smoke_test_full.py`
- **Smoke test (execution)**: `py -3 -m scripts.smoke_test_execution`
- **Pipeline scheduler**: `py -3 -m scripts.run_pipeline --schedule`
- **Backtest**: `py -3 -m scripts.run_backtest`
- **Rebalance**: `py -3 -m scripts.run_rebalance [--execute] [--alpha momentum]`
- **IB connection test**: `py -3 -m scripts.test_ib_connection [--test-order] [--validate-contracts]`
- **Test baseline**: 1169 passed, 34 skipped, 0 failures (main stack)
- **Smart-DCA tests**: `py -3 -m pytest smart-dca/tests/ -v`
- **Smart-DCA test baseline**: 102 passed, 0 failures

## Directory Structure
```
quant-stack/
├── CLAUDE.md              # This file — project context
├── README.md              # User-facing documentation
├── pyproject.toml         # Dependencies and project metadata
├── data/
│   ├── raw/               # Untouched downloaded data
│   ├── processed/         # Cleaned, normalised Parquet files
│   ├── synthetic/         # Generated test data
│   ├── dca/               # DCA purchase history JSON files
│   └── news/              # Cached news articles JSON
├── notebooks/             # Jupyter research notebooks (import from src/)
├── config/
│   ├── settings.yaml      # API keys, universe definitions, risk params
│   └── dashboard.yaml     # UI-specific configuration
├── src/
│   ├── __init__.py
│   ├── data/              # Data fetching & cleaning
│   │   ├── __init__.py
│   │   ├── fetcher.py     # Abstract fetcher + yfinance/OpenBB implementations
│   │   ├── cleaner.py     # Normalisation, missing data, corporate actions
│   │   └── synthetic.py   # Synthetic data generator for testing
│   ├── features/          # Feature engineering & alpha factors
│   │   ├── __init__.py
│   │   ├── technical.py   # Technical indicators (RSI, MACD, Bollinger, etc.)
│   │   ├── fundamental.py # Value, quality, growth factors
│   │   └── pipeline.py    # Feature pipeline orchestration
│   ├── models/            # ML models
│   │   ├── __init__.py
│   │   ├── base.py        # Abstract model interface
│   │   ├── classical.py   # Scikit-learn models
│   │   └── evaluation.py  # Cross-validation, metrics, diagnostics
│   ├── portfolio/         # Optimisation & risk
│   │   ├── __init__.py
│   │   ├── optimiser.py   # Riskfolio-Lib wrapper
│   │   ├── risk.py        # Risk metrics (VaR, CVaR, drawdown)
│   │   └── analysis.py    # Pyfolio/Alphalens integration
│   ├── backtest/          # Backtesting engines
│   │   ├── __init__.py
│   │   ├── engine.py      # VectorBT wrapper
│   │   └── strategy.py    # Strategy base class and templates (4 strategies + DCA)
│   ├── execution/         # Live/paper trading
│   │   ├── __init__.py
│   │   ├── broker.py      # PaperBroker, IBBroker, _IBConnection, translate_ticker, validate_contract
│   │   ├── oms.py         # Order management system
│   │   └── signal_bridge.py # Strategy signals -> portfolio target weights
│   ├── analysis/          # Statistical analysis modules
│   │   └── monte_carlo.py # Monte Carlo simulation engine
│   ├── services/          # Service layer between UI and quant modules
│   │   ├── market_data_service.py   # Intraday prices, fundamentals, FX
│   │   ├── chart_service.py         # Chart data preparation (OHLC, overlays)
│   │   ├── data_service.py          # Core data access (fetcher integration)
│   │   ├── dca_service.py           # DCA purchase history and analysis
│   │   ├── dca_storage.py           # SQLite backend for DCA purchases
│   │   ├── dca_analysis_service.py  # Claude API DCA chat integration
│   │   ├── execution_service.py     # Broker status, orders, positions
│   │   ├── model_service.py         # ML model training and predictions
│   │   ├── monte_carlo_service.py   # Monte Carlo service wrapper
│   │   ├── news_service.py          # RSS ingestion and Claude filtering
│   │   ├── portfolio_service.py     # Portfolio analytics (equity, risk, allocation)
│   │   ├── strategy_service.py      # Strategy registry and evaluation
│   │   └── alert_delivery.py        # Email/webhook alert dispatch
│   ├── scheduler/         # APScheduler-based pipeline automation
│   └── utils/             # Shared utilities
│       ├── __init__.py
│       ├── config.py      # YAML config loader
│       ├── logging.py     # Structured logging setup
│       └── validators.py  # Data validation helpers
├── tests/                 # pytest test suite
│   ├── conftest.py        # Shared fixtures (synthetic data, configs)
│   ├── test_data/
│   ├── test_features/
│   ├── test_portfolio/
│   ├── test_services/     # Service layer tests (12 files)
│   └── test_web/          # Route + auth tests
├── web/                   # Production UI — FastAPI + HTMX
│   ├── templates/         # Jinja2 HTML templates
│   ├── static/            # CSS and minimal JS (chart helpers only)
│   └── routes/            # FastAPI route handlers returning HTML fragments
│       ├── ui.py          # Main route handler (all /ui/ pages + partials)
│       ├── watchlist.py   # /ui/watchlist card partials
│       ├── chart.py       # /ui/chart + overlay fragments
│       ├── dca_chat.py    # /ui/portfolio/dca SSE chat
│       └── news.py        # /ui/news feed + filters
├── smart-dca/             # Smart DCA subsystem (reference — DCA logic ported to main stack)
│   ├── config/            # settings.yaml + assets.yaml
│   ├── src/
│   │   ├── data/          # price_feed, funding_feed, sentiment_feed
│   │   ├── signals/       # mean_reversion, funding, sentiment, seasonality, scorer
│   │   ├── execution/     # budget_tracker, order_manager, buy_engine
│   │   ├── notify/        # Telegram/console alerter
│   │   ├── monitor/       # performance_log, dashboard
│   │   └── utils/         # config_loader, logger, validators, exceptions
│   ├── tests/             # 102 tests across all modules
│   └── scripts/           # run_scheduler, backtest_signals
└── scripts/               # CLI entry points
    ├── fetch_data.py       # py -3 -m scripts.fetch_data
    ├── run_api.py          # py -3 -m scripts.run_api
    ├── run_backtest.py     # py -3 -m scripts.run_backtest
    ├── run_pipeline.py     # py -3 -m scripts.run_pipeline
    ├── run_real_backtest.py
    ├── run_rebalance.py    # py -3 -m scripts.run_rebalance [--alpha momentum]
    ├── generate_report.py
    ├── test_ib_connection.py  # py -3 -m scripts.test_ib_connection
    ├── smoke_test.py       # 7-step pipeline smoke test
    ├── smoke_test_full.py  # 18-check integration smoke test
    └── smoke_test_execution.py  # 7-step execution pipeline smoke test
```

## Coding Conventions
- **Python 3.13+** (developed on 3.13.7)
- **UK English** in all comments, docstrings, and documentation
- **Type hints** on all public functions and methods
- **Docstrings**: Google style
- **Pandas DataFrames** use DatetimeIndex for all time-series data
- **Config via YAML** files in `config/`, loaded through `src/utils/config.py`
- **All monetary values** in base currency (GBP unless specified in config)
- **Logging**: Use structured logging via `src/utils/logging.py`, not print()

## Critical Constraints
1. **NO lookahead bias** — Features must only use data available at the point in time they represent. All rolling calculations use `min_periods`. No future data leakage in train/test splits.
2. **Time-series aware splitting** — Never use random train/test splits. Always use temporal splits (walk-forward or expanding window).
3. **Transaction costs** — All backtests must account for commissions, slippage, and market impact.
4. **Data separation** — Fetching is always separate from transformation. Raw data is immutable once downloaded.
5. **Fail-safe trading** — Execution module defaults to paper trading. Live trading requires `execution.mode: live` and `broker.type: ibkr` in config. The `create_broker()` factory warns loudly when in live mode.

## Key Dependencies
### Core
- numpy, pandas, scipy — numerical computing
- matplotlib, seaborn — visualisation
- pyyaml — config loading
- pytest — testing

### ML
- scikit-learn — classical ML
- pycaret — AutoML prototyping (27 tests skipped due to dependency conflict — expected)

### Portfolio & Risk
- riskfolio-lib — portfolio optimisation
- alphalens-reloaded — factor analysis
- pyfolio-reloaded — performance reporting

### Backtesting
- vectorbt — fast vectorised backtesting

### Data Sources (requires network)
- yfinance — Yahoo Finance data
- openbb — investment research terminal

### Execution (requires broker)
- ibapi — Interactive Brokers API

### UI
- fastapi, uvicorn — API server
- jinja2, python-multipart — server-side HTML templating
- htmx and alpinejs loaded from CDN — no pip install required
- plotly — interactive charts

### Research
- anthropic>=0.20.0 — Claude API for research interpretation
- jupyter, nbformat — notebook infrastructure
- nbstripout — strip outputs before git commit

## Testing Strategy
- **Synthetic data** via `src/data/synthetic.py` for offline testing — no network required
- **Integration tests** marked with `@pytest.mark.integration` (need network)
- **Property-based testing** for numerical code where appropriate

## Workflow for Adding a New Strategy
1. Create a research notebook in `notebooks/` to explore the idea
2. Build features in `src/features/` with proper time-awareness
3. If ML-based, train model in `src/models/` with walk-forward validation
4. Evaluate with Alphalens in `src/portfolio/analysis.py`
5. Optimise portfolio weights with `src/portfolio/optimiser.py`
6. Backtest in `src/backtest/` with realistic transaction costs
7. Review tear sheet from Pyfolio
8. If validated, deploy to paper trading via `src/execution/`

## UI Layer

### Architecture

FastAPI + HTMX + Alpine.js. The FastAPI backend serves both JSON API
routes and server-rendered HTML via Jinja2 templates. HTMX handles
interactivity via HTML attributes — no npm, no build step, no JavaScript
framework to maintain. Alpine.js handles local UI state (dropdowns, toggles,
confirmation dialogs).

```
  Browser
    ↕  HTMX requests (returns HTML fragments, not JSON)
  FastAPI
    ├── /api/...   JSON routes
    └── /ui/...    HTML routes (in web/routes/)
          ↓
    Service Layer (src/services/)
```

### Information Hierarchy (Design North Star)

Every page answers questions in order, from macro to specific:

- **Level 1** — What is the system's state right now?
  Portfolio value, daily P&L, rolling Sharpe, active signal count
- **Level 2** — What does that mean in aggregate?
  Position heatmap: holdings sized by weight, coloured by return
- **Level 3** — What is the system saying?
  Active signals: which tickers, which direction, which features
- **Level 4** — Individual position detail
  Click a position -> equity curve, features, Claude interpretation

### Service Layer Pattern

The web UI NEVER imports from src/data, src/features, or src/models
directly. Route handlers call service functions that handle caching, error
handling, and data formatting. Services return plain dicts or DataFrames.

### Live Data Strategy (unchanged)

- End-of-day: yfinance (free, already integrated)
- Intraday: IB market data (when connected for execution)
- Fallback chain: IB -> yfinance -> cached data -> synthetic

### UI Conventions

- Templates extend web/templates/base.html via Jinja2 inheritance
- Route handlers in web/routes/ do all computation; templates only render
- HTMX hx-trigger="every 30s" for live-updating panels during market hours
- Plotly charts: data injected via JSON in the template, rendered by a thin
  charts.js helper — no business logic in JavaScript
- All monetary formatting and timezone conversion via Jinja2 filters

## Development Environment
- **Windows**: use `py -3` not `python`, `.venv/Scripts/` not `bin/`
- **Encoding**: cp1252 breaks Unicode in logging — use ASCII `->` not arrow glyphs in log strings
- **Dev servers**: launch via `.claude/launch.json` with `preview_start` — `fastapi-server` (port 8000), `pipeline-scheduler` (no port)
- **HTMX dev**: FastAPI serves HTML at http://localhost:8000/ui/ — same server entry as the JSON API, no second process needed
- **Template changes**: Jinja2 reloads on each request with `--reload` flag
- **Parquet quirk**: DatetimeIndex loses `freq` attribute on round-trip — use `check_freq=False` in `assert_frame_equal`
- **Auth**: HTTP Basic Auth — credentials `dex` / `changeme` (from `config/settings.yaml`, password via `${UI_PASSWORD}`)

## UI Gotchas
- **Jinja2 templates instance**: Each route file (`ui.py`, `chart.py`, `analyse.py`) creates its own `Jinja2Templates` object. Custom filters (e.g. `to_svg_points`) must be registered on the same instance the template is rendered with, or they'll be undefined at render time. The analyse route imports `templates` from `ui.py`.
- **HTMX swap boundaries**: If an interactive element (toggle button, toolbar) is outside the `hx-target` swap zone, its state goes stale after HTMX swaps. Fix: include the element inside the swap target so it re-renders with correct state.
- **Jinja2 `group.items` collision**: Dict key `items` collides with Python's `dict.items()` method in Jinja2. Use a different key name (e.g. `entries`).
- **Overlay sub-panels**: RSI/MACD/ML/Monte Carlo templates in `partials/chart_*.html` must be `{% include %}`d from `analyse_chart.html` — they won't render by themselves.
- **Visual verification**: Always test UI changes via the preview server. Template parse success does NOT mean render success — filters, context variables, and HTMX interactions can all fail silently.

## UI User Actions (Analyse Page)
The analyse page (`/ui/analyse`) is a buy/sell research workbench. Every element helps the user decide whether to act on a ticker.

| Action | Element | Effect |
|--------|---------|--------|
| Select ticker | Ticker bar dropdown (click company name) | Full page reload with new ticker |
| Change period | 1D/1W/1M/3M/6M/1Y buttons | HTMX swap of chart zone |
| Toggle Bollinger | Overlay button | Amber band overlay on price chart |
| Toggle RSI | Overlay button | RSI (14) sub-panel below chart |
| Toggle MACD | Overlay button | MACD (12,26,9) sub-panel below chart |
| Toggle ML Signal | Overlay button | ML confidence sub-panel below chart |
| Toggle Monte Carlo | Overlay button | Fan chart overlay (P5/P50/P95) on price SVG |
| Toggle Log scale | Log button | Logarithmic Y-axis on price chart |
| Get AI Deep Dive | Orange button | POST to `/ui/analyse/ai-dive`, returns plain-English analysis |
| Add to Watchlist | Action card link | Navigates to watchlist page |
| View in Portfolio | Action card link | Navigates to portfolio page |
| Set Price Alert | Action card link | Navigates to alerts |

## Project Status
**Last updated**: 2026-03-22

All build phases are complete: core pipeline, hardened foundation (CI, real data),
production UI (FastAPI + HTMX), UI intelligence (watchlist, charts, DCA, news, Monte Carlo),
and live execution pipeline (IB integration, signal-to-execution bridge, multi-market support).

Smart-DCA subsystem (`smart-dca/`) is kept as reference; DCA scoring logic has been ported
to the main strategy registry as `DCAStrategy` for execution via IB crypto CFDs.

### IB Gateway Status
- **ibapi 9.81.1** installed and active (38 unit tests + 7 integration tests)
- **IB Gateway connected** in live mode on port 7496 (two managed accounts: U24394034, U24394035)
- **Config**: `execution.mode: live`, `broker.type: ibkr`, `port: 7496`
- **Contract validation** confirmed against real IB: SHEL.L (SHELL PLC / LSE), AAPL (APPLE INC / SMART), BTC.CFD (resolves to Grayscale Bitcoin Mini ETF)
- **`get_account_summary()`** requests real NetLiquidation, TotalCashValue, GrossPositionValue from IB
- **`validate_contract()`** uses `reqContractDetails` to verify tickers resolve to real IB contracts
- **`test_ib_connection.py`** has 6 steps: connect, account value, positions, test order, validate contracts, disconnect

### Recent Changes (2026-03-22)
- **IB Gateway live connection**: ibapi installed, IB Gateway connected on live port 7496.
  38 IB unit tests pass (was 30 pass / 3 fail), 7 new integration tests pass against real gateway.
- **`get_account_summary()` fixed**: Requests real TotalCashValue and GrossPositionValue from IB
  instead of returning placeholder zeros.
- **`validate_contract()` added**: New IBBroker method uses `reqContractDetails` to verify ticker
  resolution. `_IBConnection` extended with `contractDetails`/`contractDetailsEnd` callbacks.
- **`--validate-contracts` flag**: `test_ib_connection.py` expanded to 6 steps with contract
  validation for SHEL.L, AAPL, BTC.CFD.
- **`disconnect()` fix**: Checks `is_alive()` before `join()` to handle unstarted threads.
- **MockIBConnection improved**: Test data populated via callbacks (not just at init), matching
  real `_IBConnection` behaviour where `get_positions()`/`get_account_value()` clear-then-request.

### Recent Changes (2026-03-14)
- **IBBroker live wiring**: `_IBConnection(EWrapper, EClient)` composition pattern with daemon
  thread, Event-based sync, heartbeat monitoring, auto-reconnection, ticker-to-contract mapping.
- **Alert delivery hardened**: Retry with exponential backoff, per-channel rate limiting,
  subject deduplication, HTML email support via `MIMEMultipart`. Default channel set to `log`.
- **DCA storage migrated to SQLite**: `DCAStorage` class with ACID transactions, WAL journal
  mode, thread-safe connection-per-call. Auto-migrates existing JSON files on first startup.
- **Multi-market contract mapping**: `_make_contract()` supports configurable `sec_type` per
  suffix rule. `.L` -> LSE/GBP/STK, `.CFD` -> SMART/USD/CFD, default -> SMART/USD/STK.
- **Signal-to-execution bridge**: `SignalBridge` converts strategy signals ({-1,0,1}) to
  portfolio target weights (equal-weight-filtered, signal-weighted, long-short).
- **DCA strategy ported**: `DCAStrategy` in `strategy.py` — mean-reversion + momentum +
  seasonality scoring. Registered as `strategy_registry.register("dca")`.
- **`run_rebalance.py` enhanced**: `--alpha <strategy>` flag uses strategy signals as
  expected-return input to the portfolio optimiser.

### Known Limitations
- Alert email/webhook delivery untested with real SMTP server or endpoint (mocked in tests)
- Pyfolio/Alphalens integration is read-only (analysis wrappers, no custom extensions)
- PyCaret AutoML: 27 tests skipped (expected — dependency conflict)
- HTTP Basic Auth is placeholder — upgrade to session-based auth before public deployment
- News feed depends on external RSS feeds — synthetic sample data available for offline use
- Monte Carlo uses parametric bootstrap (Gaussian); fat-tailed models not yet implemented
- Claude API news filtering requires ANTHROPIC_API_KEY; falls back to keyword matching without it
- BTC.CFD resolves to Grayscale Bitcoin Mini ETF, not a true crypto CFD — verify CFD availability on the live account
