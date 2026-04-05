"""Claude API integration for quantitative research interpretation.

Sends structured metrics to Claude and returns actionable interpretations.
Never sends raw DataFrames or price data — only pre-formatted dicts.

Usage:
    from src.notebooks.claude_interpreter import QuantInterpreter
    from src.utils.config import load_config

    interpreter = QuantInterpreter(load_config())
    result = interpreter.interpret("backtest_tearsheet", metrics_dict)
"""

from __future__ import annotations

import json
import os
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

_VALID_TASKS = frozenset({
    "backtest_tearsheet",
    "feature_analysis",
    "risk_metrics",
    "strategy_comparison",
    "next_steps",
})

_DEFAULT_SYSTEM_PROMPT = (
    "You are a quantitative finance research assistant. You receive structured "
    "data from a Python-based trading system and return concise, actionable "
    "interpretations. Be specific about numbers. Flag risks prominently. "
    "Suggest concrete next steps."
)

_DEFAULT_MODEL = "claude-sonnet-4-6"

# Cost reference for logging (USD per million tokens).
_COST_PER_M_INPUT = 3.0
_COST_PER_M_OUTPUT = 15.0

_TASK_PROMPTS: dict[str, str] = {
    "backtest_tearsheet": (
        "Analyse the following backtest results. Highlight the risk-adjusted "
        "return profile, drawdown behaviour, and whether the strategy shows "
        "genuine edge or is likely curve-fitted. Data:\n{data}"
    ),
    "feature_analysis": (
        "Analyse the following feature statistics from a quantitative trading "
        "system. Identify which features look most predictive, flag any data "
        "quality issues, and suggest feature engineering improvements. Data:\n{data}"
    ),
    "risk_metrics": (
        "Analyse the following portfolio risk metrics. Flag any concerning "
        "risk concentrations, comment on tail risk, and suggest hedging or "
        "rebalancing actions. Data:\n{data}"
    ),
    "strategy_comparison": (
        "Compare the following strategy results side by side. Recommend which "
        "strategy (or blend) offers the best risk-adjusted return for a "
        "systematic portfolio. Data:\n{data}"
    ),
    "next_steps": (
        "Based on the following research results, suggest the three most "
        "impactful next steps for improving this trading strategy. Be "
        "concrete and specific. Data:\n{data}"
    ),
}


def _degraded_result(reason: str) -> dict[str, Any]:
    """Return a safe degraded interpretation when the API is unavailable."""
    return {
        "summary": "Interpretation unavailable",
        "observations": [],
        "warnings": [reason],
        "suggestions": [],
        "confidence": "low",
    }


class QuantInterpreter:
    """Sends structured quantitative data to Claude for interpretation.

    Args:
        config: Project configuration dict (from load_config()).

    Raises:
        EnvironmentError: If ANTHROPIC_API_KEY is not set.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable is not set.\n"
                "To use Claude interpretations in notebooks:\n"
                "  1. Get an API key from https://console.anthropic.com/\n"
                "  2. Set it: export ANTHROPIC_API_KEY=sk-ant-...\n"
                "  3. Restart your Jupyter kernel."
            )

        nb_config = config.get("notebooks", {})
        self._model = nb_config.get("claude_model", _DEFAULT_MODEL)
        self._system_prompt = nb_config.get("system_prompt", _DEFAULT_SYSTEM_PROMPT)

        # Late import so the module loads even without anthropic installed.
        import anthropic  # noqa: F811

        self._client = anthropic.Anthropic(api_key=api_key)
        logger.debug("QuantInterpreter initialised with model=%s", self._model)

    def interpret(
        self,
        task: str,
        data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send structured data to Claude and return an interpretation.

        Args:
            task: One of 'backtest_tearsheet', 'feature_analysis',
                  'risk_metrics', 'strategy_comparison', 'next_steps'.
            data: Quantitative results as a plain dict.
            context: Optional prior interpretations or user notes.

        Returns:
            Dict with keys: summary, observations, warnings, suggestions,
            confidence. On failure returns a degraded dict — never raises.
        """
        if task not in _VALID_TASKS:
            return _degraded_result(f"Unknown task: {task}")

        try:
            return self._call_api(task, data, context)
        except Exception as exc:
            logger.warning("Claude API call failed: %s", exc)
            return _degraded_result(f"API error: {exc}")

    def _call_api(
        self,
        task: str,
        data: dict[str, Any],
        context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build the prompt and call the Claude API."""
        data_str = json.dumps(data, indent=2, default=str)
        user_message = _TASK_PROMPTS[task].format(data=data_str)

        if context:
            user_message += (
                "\n\nPrior context:\n"
                + json.dumps(context, indent=2, default=str)
            )

        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=self._system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        # Log token usage and cost estimate.
        usage = response.usage
        input_cost = (usage.input_tokens / 1_000_000) * _COST_PER_M_INPUT
        output_cost = (usage.output_tokens / 1_000_000) * _COST_PER_M_OUTPUT
        total_cost = input_cost + output_cost
        logger.debug(
            "Claude API call: task=%s, input_tokens=%d, output_tokens=%d, "
            "estimated_cost=$%.4f",
            task,
            usage.input_tokens,
            usage.output_tokens,
            total_cost,
        )

        raw_text = response.content[0].text
        return self._parse_response(raw_text)

    @staticmethod
    def _parse_response(text: str) -> dict[str, Any]:
        """Parse Claude's response into the standard interpretation dict.

        Tries JSON first; falls back to structured text extraction.
        """
        # Try direct JSON parse.
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "summary" in parsed:
                return {
                    "summary": str(parsed.get("summary", "")),
                    "observations": list(parsed.get("observations", [])),
                    "warnings": list(parsed.get("warnings", [])),
                    "suggestions": list(parsed.get("suggestions", [])),
                    "confidence": str(parsed.get("confidence", "medium")),
                }
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: treat the entire response as the summary.
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        observations = []
        warnings = []
        suggestions = []

        for line in lines[1:]:
            lower = line.lower()
            if lower.startswith("warning") or lower.startswith("risk"):
                warnings.append(line)
            elif lower.startswith("suggest") or lower.startswith("next"):
                suggestions.append(line)
            else:
                observations.append(line)

        return {
            "summary": lines[0] if lines else "No summary available",
            "observations": observations[:10],
            "warnings": warnings[:5],
            "suggestions": suggestions[:5],
            "confidence": "medium",
        }
