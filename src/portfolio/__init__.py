"""Portfolio optimisation and risk management.

Submodules:
- :mod:`~src.portfolio.risk` — Standalone risk metric functions.
- :mod:`~src.portfolio.optimiser` — Portfolio weight optimisation.
- :mod:`~src.portfolio.analysis` — Factor evaluation and performance reporting.
"""

from src.portfolio.risk import (
    conditional_var,
    correlation_report,
    max_drawdown,
    portfolio_returns,
    risk_summary,
    rolling_sharpe,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
)
from src.portfolio.optimiser import (
    PortfolioOptimiser,
    equal_weight,
    inverse_volatility,
)
from src.portfolio.analysis import (
    compare_strategies,
    evaluate_factor,
    generate_tearsheet,
)

__all__ = [
    # risk
    "portfolio_returns",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "value_at_risk",
    "conditional_var",
    "rolling_sharpe",
    "correlation_report",
    "risk_summary",
    # optimiser
    "PortfolioOptimiser",
    "equal_weight",
    "inverse_volatility",
    # analysis
    "evaluate_factor",
    "generate_tearsheet",
    "compare_strategies",
]
