# Quant Stack 📈

An open-source, modular Python-based quantitative trading workflow. From data to execution, built for solo quants and small teams.

## Architecture

```
Data → Features → Models → Portfolio Optimisation → Backtest → Execution
```

Each stage is independent, config-driven, and testable offline using synthetic data.

## Quick Start

```bash
# Clone and set up
git clone <repo-url>
cd quant-stack
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install core dependencies
pip install -e .

# Install with all extras (when you need them)
pip install -e ".[all]"

# Fetch synthetic data (works offline)
python -m scripts.fetch_data --source synthetic

# Fetch real data (requires network)
python -m scripts.fetch_data --source yfinance

# Run tests
pytest tests/ -v
```

## View the Dashboard

Three steps to see the dashboard:

```bash
# 1. Install dashboard dependencies
pip install -e ".[dashboard]"

# 2. Generate sample data (works offline, no API keys needed)
python -m scripts.fetch_data --source synthetic

# 3. Launch the dashboard
streamlit run app.py
```

A browser tab will open automatically at `http://localhost:8501`.

**Using Claude Code on the web?** The dashboard will appear in the preview panel — no extra setup needed.

## Project Structure

See [CLAUDE.md](CLAUDE.md) for full architecture documentation.

## Configuration

All settings are in `config/settings.yaml`. Key sections:

- **universe**: Which assets to trade
- **data**: Data source and date range
- **features**: Technical indicator parameters
- **portfolio**: Optimisation method and constraints
- **risk**: Drawdown limits, position sizing
- **backtest**: Engine, capital, transaction costs
- **execution**: Broker connection (paper/live)

## Development Phases

| Phase | Module | Status |
|-------|--------|--------|
| 0 | Project scaffold | ✅ Complete |
| 1 | Data pipeline | ✅ Complete |
| 2 | Feature engineering | 🔲 Next |
| 3 | ML models | 🔲 Planned |
| 4 | Portfolio optimisation | 🔲 Planned |
| 5 | Backtesting | 🔲 Planned |
| 6 | Execution | 🔲 Planned |

## Using with Claude Code

This project includes a `CLAUDE.md` that gives Claude Code full context about the architecture, conventions, and constraints. Open the project folder in Claude Code and it will understand the system immediately.

**Example prompts for Claude Code:**
- "Build the technical indicators module for Phase 2"
- "Add RSI, MACD, and Bollinger Bands to src/features/technical.py"
- "Create a mean-reversion strategy and backtest it"

## Licence

MIT
