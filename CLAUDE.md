# Quant Stack — Automated Trading Workflow

## Project Overview
A modular, open-source Python-based quantitative trading system following a pipeline architecture:
**Data → Features → Models → Portfolio Optimisation → Backtest → Execution**

Built for a solo quant / small team to prototype, test, and deploy trading strategies
without proprietary software. Designed to scale from personal capital to small fund operations.

## Architecture Principles
- **Pipeline pattern**: Each stage is independent and composable
- **Config-driven**: All parameters in YAML, never hardcoded
- **No lookahead bias**: Strictly enforced in all feature/model/backtest code
- **Dual-mode data**: Synthetic data for testing, live APIs for production
- **Fail-safe execution**: All trading code defaults to paper trading unless explicitly overridden

## Directory Structure
```
quant-stack/
├── CLAUDE.md              # This file — project context
├── README.md              # User-facing documentation
├── pyproject.toml         # Dependencies and project metadata
├── data/
│   ├── raw/               # Untouched downloaded data
│   ├── processed/         # Cleaned, normalised Parquet files
│   └── synthetic/         # Generated test data
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
│   │   └── strategy.py    # Strategy base class and templates
│   ├── execution/         # Live/paper trading
│   │   ├── __init__.py
│   │   ├── broker.py      # IBAPI connection
│   │   └── oms.py         # Order management system
│   ├── dashboard/         # Streamlit app and page modules
│   ├── services/          # Service layer between UI and quant modules
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
│   └── test_portfolio/
└── scripts/               # CLI entry points
    ├── fetch_data.py       # python -m scripts.fetch_data
    ├── run_backtest.py     # python -m scripts.run_backtest
    └── generate_report.py  # python -m scripts.generate_report
```

## Coding Conventions
- **Python 3.11+**
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
5. **Fail-safe trading** — Execution module defaults to paper trading. Live trading requires explicit `--live` flag AND confirmation.

## Key Dependencies
### Core (always needed)
- numpy, pandas, scipy — numerical computing
- matplotlib, seaborn — visualisation
- pyyaml — config loading
- pytest — testing

### Research & ML (Phase 2-3)
- scikit-learn — classical ML
- pycaret — AutoML prototyping

### Portfolio & Risk (Phase 4)
- riskfolio-lib — portfolio optimisation
- alphalens-reloaded — factor analysis
- pyfolio — performance reporting (use quantopian-pyfolio or pyfolio-reloaded)

### Backtesting (Phase 5)
- vectorbt — fast vectorised backtesting

### Data Sources (Phase 1, requires network)
- yfinance — Yahoo Finance data
- openbb — investment research terminal

### Execution (Phase 6, requires broker)
- ibapi — Interactive Brokers API

### Dashboard & Services (Sprint 1-4)
- streamlit — dashboard UI
- plotly — interactive charts
- apscheduler — pipeline scheduling
- requests — API fallback clients

## Testing Strategy
- **Unit tests** with pytest for all modules
- **Synthetic data** via `src/data/synthetic.py` for offline testing
- **Property-based testing** for numerical code where appropriate
- **Integration tests** marked with `@pytest.mark.integration` (need network)
- Run: `pytest tests/ -v` (unit only) or `pytest tests/ -v -m integration` (all)

## Workflow for Adding a New Strategy
1. Create a research notebook in `notebooks/` to explore the idea
2. Build features in `src/features/` with proper time-awareness
3. If ML-based, train model in `src/models/` with walk-forward validation
4. Evaluate with Alphalens in `src/portfolio/analysis.py`
5. Optimise portfolio weights with `src/portfolio/optimiser.py`
6. Backtest in `src/backtest/` with realistic transaction costs
7. Review tear sheet from Pyfolio
8. If validated, deploy to paper trading via `src/execution/`

## Dashboard & UI Layer

### Architecture
The UI is a Streamlit dashboard that reads from the existing data pipeline outputs.
No direct database — the file system (parquet files in data/processed/) IS the data store.
A scheduler service runs the pipeline daily; the dashboard reads the results.

Directory additions:
- src/dashboard/           — Streamlit app and page modules
- src/services/            — Service layer between UI and quant modules
- src/scheduler/           — APScheduler-based pipeline automation
- config/dashboard.yaml    — UI-specific configuration

### Service Layer Pattern
The dashboard NEVER imports from src/data, src/features, src/models directly.
It calls service functions that handle caching, error handling, and data formatting.
Services return plain dicts or DataFrames — no internal types leak to the UI.

### Live Data Strategy
- End-of-day: yfinance (free, already integrated)
- Intraday: IB market data (when connected for execution)
- Fallback chain: IB → yfinance → cached data → synthetic
- All data access goes through DataService, which manages the fallback

### Dashboard Pages
1. Overview — portfolio value, daily P&L, key risk metrics, signals
2. Strategy — backtest results, strategy comparison, signal charts
3. Portfolio — current weights, target weights, rebalance orders
4. Research — feature explorer, correlation heatmap, model diagnostics
5. Execution — paper trade controls, order history, reconciliation

### UI Conventions
- Streamlit with st.set_page_config(layout="wide")
- Colour scheme: dark theme compatible
- Charts: plotly for interactive, matplotlib for static exports
- All monetary values formatted with £ symbol and 2 decimal places
- Timestamps in Europe/London timezone
- Cache expensive computations with @st.cache_data (TTL from config)
