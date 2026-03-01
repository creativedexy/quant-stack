"""Services module — dashboard data access and business logic.

Provides a clean service layer between the dashboard UI and the
underlying data, portfolio, and strategy modules.
"""

from src.services.data_service import DataService
from src.services.portfolio_service import PortfolioService
from src.services.strategy_service import StrategyService

__all__ = [
    "DataService",
    "PortfolioService",
    "StrategyService",
]
