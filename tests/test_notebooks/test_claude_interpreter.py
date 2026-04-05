"""Tests for the intelligent notebook infrastructure.

All Claude API calls are mocked — no real API key required.
Run: py -3 -m pytest tests/test_notebooks/ -v
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.notebooks.claude_interpreter import QuantInterpreter
from src.notebooks.formatters import (
    display_interpretation,
    format_backtest_metrics,
    format_feature_stats,
    format_risk_metrics,
)
from src.notebooks.research_log import ResearchLog


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def nb_config() -> dict:
    """Minimal config for notebook tests."""
    return {
        "notebooks": {
            "claude_model": "claude-sonnet-4-6",
            "log_path": "data/processed/research_log.jsonl",
            "system_prompt": "You are a test assistant.",
        },
    }


@pytest.fixture
def mock_tearsheet() -> dict:
    """Realistic backtest metrics dict."""
    return {
        "sharpe": 1.42,
        "sortino": 1.85,
        "annualised_return": 0.1234,
        "max_drawdown": -0.1567,
        "calmar_ratio": 0.787,
        "win_rate": 0.58,
        "profit_factor": 1.73,
        "total_trades": 142,
        "avg_trade_duration_days": 12.3,
        "annualised_volatility": 0.0869,
        "var_95": -0.0234,
        "cvar_95": -0.0312,
    }


@pytest.fixture
def mock_features_df() -> pd.DataFrame:
    """Feature DataFrame with some NaN values and a target column."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "sma_20": np.random.randn(n),
        "rsi_14": np.random.uniform(20, 80, n),
        "macd_histogram": np.random.randn(n),
        "forward_return_5d": np.random.randn(n) * 0.01,
    })
    # Inject some NaN values.
    df.loc[5:7, "sma_20"] = np.nan
    df.loc[10:12, "rsi_14"] = np.nan
    return df


def _make_mock_response(text: str) -> MagicMock:
    """Build a mock Anthropic API response."""
    content_block = SimpleNamespace(text=text)
    usage = SimpleNamespace(input_tokens=500, output_tokens=200)
    return SimpleNamespace(content=[content_block], usage=usage)


# ---------------------------------------------------------------------------
# QuantInterpreter tests
# ---------------------------------------------------------------------------

def _make_mock_anthropic() -> MagicMock:
    """Build a mock anthropic module for injection into sys.modules."""
    mock_module = MagicMock()
    return mock_module


class TestQuantInterpreter:
    """Tests for QuantInterpreter."""

    def test_raises_without_api_key(self, nb_config: dict) -> None:
        """QuantInterpreter raises EnvironmentError when key is missing."""
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
                QuantInterpreter(nb_config)

    def test_interpret_success_json(self, nb_config: dict) -> None:
        """interpret() returns dict with all required keys on JSON response."""
        response_data = {
            "summary": "Strategy shows positive risk-adjusted returns.",
            "observations": ["Sharpe above 1.0", "Drawdown within limits"],
            "warnings": ["Small sample size"],
            "suggestions": ["Extend backtest period"],
            "confidence": "high",
        }
        mock_anthropic = _make_mock_anthropic()
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_mock_response(
            json.dumps(response_data)
        )
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test-key"}):
            with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
                interp = QuantInterpreter(nb_config)
                result = interp.interpret("backtest_tearsheet", {"sharpe": 1.42})

        assert result["summary"] == "Strategy shows positive risk-adjusted returns."
        assert "Sharpe above 1.0" in result["observations"]
        assert "Small sample size" in result["warnings"]
        assert "Extend backtest period" in result["suggestions"]
        assert result["confidence"] == "high"

    def test_interpret_success_plain_text(self, nb_config: dict) -> None:
        """interpret() handles plain text response (non-JSON)."""
        mock_anthropic = _make_mock_anthropic()
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_mock_response(
            "The strategy looks promising.\n"
            "Sharpe ratio is above benchmark.\n"
            "Warning: limited data range.\n"
            "Suggestion: add more features."
        )
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test-key"}):
            with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
                interp = QuantInterpreter(nb_config)
                result = interp.interpret("feature_analysis", {"feature_count": 10})

        assert "summary" in result
        assert "observations" in result
        assert "warnings" in result
        assert "suggestions" in result
        assert "confidence" in result

    def test_interpret_api_failure_returns_degraded(self, nb_config: dict) -> None:
        """interpret() returns degraded dict (never raises) on API failure."""
        mock_anthropic = _make_mock_anthropic()
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("Connection failed")
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test-key"}):
            with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
                interp = QuantInterpreter(nb_config)
                result = interp.interpret("risk_metrics", {"sharpe": 0.5})

        assert result["summary"] == "Interpretation unavailable"
        assert result["confidence"] == "low"
        assert len(result["warnings"]) > 0

    def test_interpret_unknown_task(self, nb_config: dict) -> None:
        """interpret() returns degraded dict for unknown task names."""
        mock_anthropic = _make_mock_anthropic()
        mock_anthropic.Anthropic.return_value = MagicMock()

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test-key"}):
            with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
                interp = QuantInterpreter(nb_config)
                result = interp.interpret("nonexistent_task", {})

        assert result["summary"] == "Interpretation unavailable"
        assert "Unknown task" in result["warnings"][0]

    def test_interpret_with_context(self, nb_config: dict) -> None:
        """interpret() passes context to the API call."""
        response_data = {
            "summary": "Building on prior analysis.",
            "observations": [],
            "warnings": [],
            "suggestions": [],
            "confidence": "medium",
        }
        mock_anthropic = _make_mock_anthropic()
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_mock_response(
            json.dumps(response_data)
        )
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test-key"}):
            with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
                interp = QuantInterpreter(nb_config)
                context = {"prior_sharpe": 0.8, "notes": "First iteration"}
                result = interp.interpret("next_steps", {"sharpe": 1.2}, context=context)

        # Verify context was included in the API call.
        call_args = mock_client.messages.create.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert "Prior context" in user_msg
        assert "First iteration" in user_msg
        assert result["summary"] == "Building on prior analysis."


# ---------------------------------------------------------------------------
# Formatter tests
# ---------------------------------------------------------------------------

class TestFormatBacktestMetrics:
    """Tests for format_backtest_metrics."""

    def test_correct_structure(self, mock_tearsheet: dict) -> None:
        """Produces correct keys and types."""
        result = format_backtest_metrics(mock_tearsheet)

        assert "sharpe_ratio" in result
        assert "sortino_ratio" in result
        assert "cagr" in result
        assert "max_drawdown" in result
        assert "calmar_ratio" in result
        assert "win_rate" in result
        assert "profit_factor" in result
        assert "total_trades" in result
        assert "avg_trade_duration_days" in result

        # Drawdown is a positive percentage string.
        assert result["max_drawdown"].endswith("%")
        assert not result["max_drawdown"].startswith("-")

        # total_trades is int.
        assert isinstance(result["total_trades"], int)

    def test_rounding(self, mock_tearsheet: dict) -> None:
        """All floats are rounded to 4dp."""
        result = format_backtest_metrics(mock_tearsheet)
        assert result["sharpe_ratio"] == 1.42
        assert result["sortino_ratio"] == 1.85

    def test_empty_tearsheet(self) -> None:
        """Handles empty dict gracefully."""
        result = format_backtest_metrics({})
        assert result["sharpe_ratio"] == 0.0
        assert result["total_trades"] == 0
        assert result["max_drawdown"] == "0.00%"


class TestFormatFeatureStats:
    """Tests for format_feature_stats."""

    def test_correct_structure(self, mock_features_df: pd.DataFrame) -> None:
        """Produces correct keys."""
        result = format_feature_stats(mock_features_df)

        assert result["feature_count"] == 4
        assert "sma_20" in result["missing_pct"]
        assert "sma_20" in result["autocorrelation"]

    def test_handles_nan(self, mock_features_df: pd.DataFrame) -> None:
        """Correctly reports missing percentages for features with NaN."""
        result = format_feature_stats(mock_features_df)
        assert result["missing_pct"]["sma_20"] > 0
        assert result["missing_pct"]["rsi_14"] > 0

    def test_target_correlations(self, mock_features_df: pd.DataFrame) -> None:
        """Detects target column and computes correlations."""
        result = format_feature_stats(mock_features_df)
        assert len(result["target_correlations"]) > 0
        for corr in result["target_correlations"]:
            assert "feature" in corr
            assert "correlation" in corr

    def test_no_target_column(self) -> None:
        """Works without a target column."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = format_feature_stats(df)
        assert result["target_correlations"] == []


class TestFormatRiskMetrics:
    """Tests for format_risk_metrics."""

    def test_var_as_positive_percentage(self, mock_tearsheet: dict) -> None:
        """VaR and CVaR are positive percentage strings."""
        result = format_risk_metrics(mock_tearsheet)
        assert result["var_95"].endswith("%")
        assert not result["var_95"].startswith("-")
        assert result["cvar_95"].endswith("%")

    def test_drawdown_periods_from_peak_trough(self) -> None:
        """Builds drawdown periods from peak_date/trough_date."""
        risk = {
            "max_drawdown": -0.15,
            "peak_date": "2024-01-15",
            "trough_date": "2024-03-10",
        }
        result = format_risk_metrics(risk)
        assert len(result["drawdown_periods"]) == 1
        assert result["drawdown_periods"][0]["start"] == "2024-01-15"


# ---------------------------------------------------------------------------
# display_interpretation tests
# ---------------------------------------------------------------------------

class TestDisplayInterpretation:
    """Tests for display_interpretation."""

    def test_plain_text_fallback(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Falls back to print when IPython is unavailable."""
        result = {
            "summary": "Test summary",
            "observations": ["obs1"],
            "warnings": ["warn1"],
            "suggestions": ["sug1"],
            "confidence": "high",
        }
        # Ensure IPython is not importable for this test.
        with patch.dict("sys.modules", {"IPython": None, "IPython.display": None}):
            display_interpretation(result)

        captured = capsys.readouterr()
        assert "Test summary" in captured.out
        assert "obs1" in captured.out
        assert "warn1" in captured.out
        assert "sug1" in captured.out


# ---------------------------------------------------------------------------
# ResearchLog tests
# ---------------------------------------------------------------------------

class TestResearchLog:
    """Tests for ResearchLog."""

    def test_log_entry_writes_valid_json(self, tmp_path: Path) -> None:
        """log_entry writes valid JSON to the log file."""
        config = {"notebooks": {"log_path": str(tmp_path / "test_log.jsonl")}}
        log = ResearchLog(config)

        log.log_entry(
            notebook="01_data_exploration",
            task="feature_analysis",
            data_summary={"feature_count": 10},
            interpretation={"summary": "Looks good"},
            notes="Test note",
        )

        log_file = tmp_path / "test_log.jsonl"
        assert log_file.exists()

        with open(log_file, "r", encoding="utf-8") as f:
            entry = json.loads(f.readline())

        assert entry["notebook"] == "01_data_exploration"
        assert entry["task"] == "feature_analysis"
        assert entry["data_summary"]["feature_count"] == 10
        assert entry["interpretation"]["summary"] == "Looks good"
        assert entry["notes"] == "Test note"
        assert "timestamp" in entry

    def test_load_recent_returns_at_most_n(self, tmp_path: Path) -> None:
        """load_recent(5) returns at most 5 entries."""
        config = {"notebooks": {"log_path": str(tmp_path / "test_log.jsonl")}}
        log = ResearchLog(config)

        for i in range(10):
            log.log_entry(
                notebook="test",
                task="task",
                data_summary={"i": i},
                interpretation={"summary": f"entry {i}"},
            )

        recent = log.load_recent(5)
        assert len(recent) == 5
        # Should be the last 5 entries.
        assert recent[0]["data_summary"]["i"] == 5
        assert recent[-1]["data_summary"]["i"] == 9

    def test_load_recent_empty_file(self, tmp_path: Path) -> None:
        """load_recent returns [] when log file does not exist."""
        config = {"notebooks": {"log_path": str(tmp_path / "nonexistent.jsonl")}}
        log = ResearchLog(config)
        assert log.load_recent(10) == []

    def test_summarise_session(self, tmp_path: Path) -> None:
        """summarise_session filters by notebook and today's date."""
        config = {"notebooks": {"log_path": str(tmp_path / "test_log.jsonl")}}
        log = ResearchLog(config)

        log.log_entry("nb_a", "task1", {}, {"summary": "a1"})
        log.log_entry("nb_b", "task2", {}, {"summary": "b1"})
        log.log_entry("nb_a", "task3", {}, {"summary": "a2"})

        result = log.summarise_session("nb_a")
        assert result["count"] == 2
        assert result["notebook"] == "nb_a"
        assert len(result["entries"]) == 2


# We need Path for tmp_path fixture type hint.
from pathlib import Path  # noqa: E402
