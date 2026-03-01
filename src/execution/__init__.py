"""Order execution and broker connectivity.

Public API::

    from src.execution import (
        Broker,
        PaperBroker,
        IBBroker,
        create_broker,
        OrderManagementSystem,
        Order,
        ExecutionReport,
    )
"""

from src.execution.broker import Broker, IBBroker, PaperBroker, create_broker
from src.execution.oms import ExecutionReport, Order, OrderManagementSystem

__all__ = [
    "Broker",
    "PaperBroker",
    "IBBroker",
    "create_broker",
    "OrderManagementSystem",
    "Order",
    "ExecutionReport",
]
