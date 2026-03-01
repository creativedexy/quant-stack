"""Execution module — broker connections and order management.

Provides paper and (future) live broker implementations together with
an order management system for executing rebalance plans.
"""

from src.execution.broker import Broker, PaperBroker
from src.execution.oms import Order, ExecutionReport, OrderManagementSystem
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
    "Order",
    "ExecutionReport",
    "OrderManagementSystem",
    "IBBroker",
    "create_broker",
    "OrderManagementSystem",
    "Order",
    "ExecutionReport",
]
