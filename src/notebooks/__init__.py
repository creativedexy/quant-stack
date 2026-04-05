"""Intelligent notebook utilities — Claude API integration for research."""

from src.notebooks.claude_interpreter import QuantInterpreter
from src.notebooks.formatters import (
    display_interpretation,
    format_backtest_metrics,
    format_feature_stats,
    format_risk_metrics,
)
from src.notebooks.research_log import ResearchLog

__all__ = [
    "QuantInterpreter",
    "ResearchLog",
    "display_interpretation",
    "format_backtest_metrics",
    "format_feature_stats",
    "format_risk_metrics",
]
