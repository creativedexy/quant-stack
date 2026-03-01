# Quant Stack

An open-source, modular Python quantitative trading system. From data ingestion
through feature engineering, ML modelling, portfolio optimisation, backtesting,
and execution — built for solo quants and small teams.

```
Data → Features → Models → Portfolio Optimisation → Backtest → Execution
```

Each stage is independent, config-driven, and testable offline with synthetic data.
All trading defaults to paper mode — live execution requires an explicit flag.

## Quick Start

```bash
# Clone and install
git clone <repo-url>
cd quant-stack
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e .

# Generate synthetic data (works offline, no API keys)
python -m scripts.fetch_data --source synthetic

# Run the test suite
pytest tests/ -v
```

### Install Optional Extras

```bash
pip install -e ".[dashboard]"    # Streamlit dashboard + Plotly charts
pip install -e ".[data-live]"    # yfinance + OpenBB for real market data
pip install -e ".[portfolio]"    # riskfolio-lib, alphalens, pyfolio
pip install -e ".[backtest]"     # vectorbt backtesting engine
pip install -e ".[ml-extended]"  # PyCaret, PyTorch
pip install -e ".[execution]"   # Interactive Brokers API
pip install -e ".[research]"    # Jupyter notebooks
pip install -e ".[api]"         # FastAPI server
pip install -e ".[dev]"         # pytest, ruff, mypy, pre-commit
pip install -e ".[all]"         # Everything above
```

## View the Dashboard

```bash
# 1. Install dashboard dependencies
pip install -e ".[dashboard]"

# 2. Generate sample data (works offline, no API keys needed)
python -m scripts.fetch_data --source synthetic

# 3. Launch the dashboard
streamlit run app.py
```

A browser tab will open at `http://localhost:8501` with two pages:

- **Overview** — data status, latest prices, risk metrics, allocation chart, pipeline health
- **Execution** — paper trading controls, positions, rebalance planner, order history

**Using Claude Code on the web?** The dashboard will appear in the preview panel automatically.

## Project Structure

```
quant-stack/
├── app.py                  # Streamlit dashboard entry point
├── config/
│   ├── settings.yaml       # Master config (universe, risk, execution, etc.)
│   └── dashboard.yaml      # Dashboard-specific settings
├── data/
│   ├── raw/                # Immutable downloaded data
│   ├── processed/          # Cleaned Parquet files (the dashboard reads from here)
│   └── synthetic/          # Generated test data
├── notebooks/
│   ├── 01_full_demo.ipynb
│   └── 01_research_template.ipynb
├── scripts/
│   ├── fetch_data.py       # python -m scripts.fetch_data
│   ├── run_backtest.py     # python -m scripts.run_backtest
│   ├── run_pipeline.py     # python -m scripts.run_pipeline
│   ├── run_rebalance.py    # python -m scripts.run_rebalance
│   ├── run_api.py          # python -m scripts.run_api
│   └── generate_report.py  # python -m scripts.generate_report
├── src/
│   ├── data/               # Fetching, cleaning, synthetic generation
│   ├── features/           # Technical indicators & feature pipeline
│   ├── models/             # ML models (scikit-learn, AutoML)
│   ├── portfolio/          # Optimisation, risk metrics, factor analysis
│   ├── backtest/           # Vectorised backtesting engine
│   ├── execution/          # Paper & live broker, order management
│   ├── dashboard/          # Streamlit pages & reusable components
│   ├── services/           # Service layer (data, portfolio, strategy, execution)
│   ├── scheduler/          # APScheduler pipeline automation & alerts
│   ├── api/                # FastAPI REST endpoints
│   └── utils/              # Config loader, structured logging, validators
└── tests/                  # pytest suite (~35 test files)
```

## CLI Scripts

| Command | Description |
|---------|-------------|
| `python -m scripts.fetch_data --source synthetic` | Generate synthetic OHLCV data |
| `python -m scripts.fetch_data --source yfinance` | Download real data (needs network) |
| `python -m scripts.run_backtest --strategy mean_reversion` | Run a backtest |
| `python -m scripts.run_pipeline` | Run the full data pipeline once |
| `python -m scripts.run_rebalance` | Generate a rebalance plan |
| `python -m scripts.run_rebalance --execute` | Execute rebalance (paper mode) |
| `python -m scripts.run_api` | Start the FastAPI server |
| `python -m scripts.generate_report` | Generate a portfolio risk report |

## Configuration

All settings live in `config/settings.yaml`. Key sections:

| Section | What it controls |
|---------|-----------------|
| `universe` | Tickers to trade (default: UK large-cap — SHEL.L, AZN.L, HSBA.L, ULVR.L, BP.L) |
| `data` | Source (synthetic/yfinance/alpha\_vantage), date range, fields |
| `features` | SMA windows, RSI period, return horizons |
| `models` | Model type, walk-forward parameters, retrain schedule |
| `portfolio` | Optimisation method (mean-variance, risk parity, equal weight), constraints |
| `risk` | Max drawdown, VaR confidence, position limits |
| `backtest` | Initial capital (£100,000), commission/slippage (10/5 bps) |
| `execution` | Broker, mode (paper/live), connection details |
| `scheduler` | Daily run time (17:30 London), retrain schedule |

Dashboard-specific settings are in `config/dashboard.yaml`.

## Architecture

### Pipeline Pattern

Each module reads from the previous stage's output and writes to `data/processed/`.
No module reaches back into another's internals.

### Service Layer

The dashboard and API never import from `src/data`, `src/features`, or `src/models` directly.
They call service functions that handle caching, error handling, and data formatting:

- **DataService** — prices, returns, features, data status
- **PortfolioService** — weights, risk metrics, equity curves, allocation
- **StrategyService** — strategy evaluation, backtest results
- **ExecutionService** — broker state, positions, rebalance plans, order history

### Safety

- Execution defaults to **paper trading** — live mode requires `--live` flag and confirmation
- No lookahead bias — all features use only data available at each point in time
- All backtests include transaction costs (commission + slippage)
- Raw data is immutable once downloaded

### Scheduler

An APScheduler-based runner executes the daily pipeline at 17:30 London time
(30 minutes after the LSE close). Model retraining runs weekly on Sundays.
Alerts are sent on failure via log, email, or webhook.

## Testing

```bash
# Unit tests (offline, fast)
pytest tests/ -v

# Include integration tests (need network)
pytest tests/ -v -m integration

# With coverage
pytest tests/ -v --cov=src
```

Tests use synthetic data generated by `src/data/synthetic.py` so they run
without API keys or network access.

## Dependencies

**Python 3.11+** required.

| Group | Key packages |
|-------|-------------|
| Core | numpy, pandas, scipy, matplotlib, seaborn, scikit-learn, pyarrow, apscheduler |
| Dashboard | streamlit, plotly |
| Data (live) | yfinance, openbb |
| Portfolio | riskfolio-lib, alphalens-reloaded, pyfolio-reloaded |
| Backtest | vectorbt |
| ML Extended | pycaret, torch |
| Execution | ibapi (Interactive Brokers) |
| API | fastapi, uvicorn, websockets |
| Dev | pytest, ruff, mypy, pre-commit |

## Using with Claude Code

This project includes a `CLAUDE.md` that gives Claude Code full context about
the architecture, conventions, and constraints. Open the project in Claude Code
and it will understand the system immediately.

**Example prompts:**
- "Build the technical indicators module for Phase 2"
- "Add RSI, MACD, and Bollinger Bands to src/features/technical.py"
- "Create a mean-reversion strategy and backtest it"
- "Run the pipeline and show me the dashboard"

## Licence

MIT
