[![CI](https://github.com/<your-username>/quant-stack/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/quant-stack/actions/workflows/ci.yml)
[![Lint](https://github.com/<your-username>/quant-stack/actions/workflows/lint.yml/badge.svg)](https://github.com/<your-username>/quant-stack/actions/workflows/lint.yml)

# Quant Stack 📈

An open-source, modular Python-based quantitative trading workflow for solo
quants and small teams. From raw data to live execution, in a single coherent
system.

## Architecture

Each stage is independent, config-driven, and testable offline using synthetic
data. The system scales from personal research to live trading with Interactive
Brokers.

## UI

**FastAPI + HTMX + Alpine.js** — a proper web application served by the
existing FastAPI backend, with no JavaScript build chain and no separate
frontend ecosystem to maintain.

## Quick Start

```bash
git clone <repo-url>
cd quant-stack
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -e ".[all]"

# Generate offline test data (no API keys needed)
py -3 -m scripts.fetch_data --source synthetic

# Run the test suite
py -3 -m pytest tests/ -v --ignore=tests/test_models/test_automl.py

# Launch the dashboard
uvicorn src.api.main:app --reload
# Open http://localhost:8000/ui/overview
```

## Research Notebooks

The `notebooks/` directory contains four Jupyter notebooks with Claude API
integration — each notebook sends structured metrics to Claude and receives
actionable interpretations alongside the charts.

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration` | Data quality, correlations, outlier detection |
| `02_feature_research` | Feature ICs, importance, autocorrelation |
| `03_strategy_development` | Parameter sweeps, sensitivity heatmaps |
| `04_backtest_tearsheet` | Full tearsheet, equity curve, next steps |

All notebooks run without an API key — charts and metrics render normally;
interpretation cells show "Interpretation unavailable". Set the key to enable
Claude analysis (~£0.02 per notebook run).

```bash
pip install -e ".[research]"
export ANTHROPIC_API_KEY=sk-ant-...   # optional — notebooks work without it
nbstripout --install                   # prevent committing notebook outputs
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Running the Production UI

The production interface is served by FastAPI with Jinja2 templates, HTMX
for interactivity, and Plotly for charts.

```bash
# Copy the example config and edit credentials
cp config/settings.yaml.example config/settings.yaml

# Generate synthetic data (if not already present)
py -3 -m scripts.fetch_data --source synthetic

# Start the server
uvicorn src.api.main:app --reload
```

Open http://localhost:8000/ui/overview and log in with the credentials from
`config/settings.yaml` (default: `admin` / `changeme`).

![Overview](docs/screenshot_overview.png)

## Project Structure

The full architecture, coding conventions, constraints, and roadmap are
documented in [CLAUDE.md](CLAUDE.md). Key directories:

| Directory | Purpose |
|-----------|---------|
| `src/` | Core pipeline: data, features, models, portfolio, backtest, execution |
| `src/services/` | Service layer consumed by both UIs |
| `web/` | Production UI: Jinja2 templates, CSS, route handlers |
| `config/` | YAML configuration (`settings.yaml.example` committed) |
| `notebooks/` | Jupyter research notebooks with Claude API integration |
| `scripts/` | CLI entry points and smoke tests |
| `tests/` | pytest suite (900+ tests) |

## Configuration

All settings live in `config/settings.yaml` (copy from `settings.yaml.example`).
Key sections: universe, data, features, models, portfolio, risk, backtest,
execution, notebooks, ui, alerts.

## Roadmap

| Phase | Sessions | Goal |
|-------|----------|------|
| 1 — Core pipeline | 1–6 | Full workflow validated |
| 2 — Foundation | 7–9 | CI, intelligent notebooks, real FTSE data |
| 3 — Production UI | 10–13 | FastAPI + HTMX interface |
| 4 — UI Intelligence | 14–20 | Watchlist, charts, DCA, news, Monte Carlo |

See [CLAUDE.md](CLAUDE.md) for full architecture documentation.

## CI

Every push and pull request runs two GitHub Actions workflows in parallel:

- **CI** (green badge) — installs the project, generates synthetic data, and
  runs the full pytest suite. A green badge means all tests pass.
- **Lint** (green badge) — runs ruff to catch syntax errors, undefined names,
  and unused imports.

A red badge means the most recent run on the default branch failed. Click
the badge to see the workflow logs and identify the failure.

See [.github/workflows/README.md](.github/workflows/README.md) for full
details on both workflows.

## Licence

MIT
