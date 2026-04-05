"""Custom exceptions for smart-dca."""


class ConfigError(Exception):
    """Raised when configuration is invalid or missing required fields."""


class DataFeedError(Exception):
    """Raised when a data feed fails to return valid data."""


class OrderError(Exception):
    """Raised when order placement or management fails."""


class BudgetError(Exception):
    """Raised when a spend would exceed budget limits.

    Attributes:
        asset: The asset symbol.
        requested: The requested spend amount in GBP.
        remaining: The remaining budget in GBP.
    """

    def __init__(self, asset: str, requested: float, remaining: float) -> None:
        self.asset = asset
        self.requested = requested
        self.remaining = remaining
        super().__init__(
            f"Budget exceeded for {asset}: "
            f"requested {requested:.2f} GBP, remaining {remaining:.2f} GBP"
        )
