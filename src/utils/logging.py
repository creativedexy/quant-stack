"""Structured logging setup for the quant stack.

Usage:
    from src.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Fetched data", extra={"tickers": 10, "rows": 5000})
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from src.utils.config import load_config


_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    """Configure root logging for the project.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR). 
               Falls back to config setting if not provided.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    if level is None:
        try:
            config = load_config()
            level = config["general"].get("log_level", "INFO")
        except FileNotFoundError:
            level = "INFO"

    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt=date_format,
        stream=sys.stdout,
    )

    # Quieten noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured Logger instance.
    """
    setup_logging()
    return logging.getLogger(name)
