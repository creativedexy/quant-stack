"""Pure formatting functions for notebook data -> Claude API payloads.

These functions contain no side effects and no imports from the src tree.
They convert raw quantitative objects into plain dicts suitable for the
QuantInterpreter.

Usage:
    from src.notebooks.formatters import format_backtest_metrics
    payload = format_backtest_metrics(tearsheet_dict)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def format_backtest_metrics(tearsheet: dict[str, Any]) -> dict[str, Any]:
    """Extract and normalise backtest metrics for Claude interpretation.

    Args:
        tearsheet: Raw metrics dict (e.g. from risk_summary or BacktestResult.metrics).

    Returns:
        Dict with standardised keys, all floats rounded to 4dp,
        drawdown as a positive percentage string.
    """

    def _round(val: Any, dp: int = 4) -> Any:
        if isinstance(val, (int, float, np.floating)):
            return round(float(val), dp)
        return val

    dd_raw = tearsheet.get("max_drawdown", 0.0)
    # Ensure drawdown is a positive percentage string.
    dd_val = abs(float(dd_raw)) if isinstance(dd_raw, (int, float, np.floating)) else 0.0
    dd_str = f"{dd_val * 100:.2f}%"

    return {
        "sharpe_ratio": _round(tearsheet.get("sharpe", tearsheet.get("sharpe_ratio", 0.0))),
        "sortino_ratio": _round(tearsheet.get("sortino", tearsheet.get("sortino_ratio", 0.0))),
        "cagr": _round(tearsheet.get("cagr", tearsheet.get("annualised_return", 0.0))),
        "max_drawdown": dd_str,
        "calmar_ratio": _round(tearsheet.get("calmar_ratio", 0.0)),
        "win_rate": _round(tearsheet.get("win_rate", 0.0)),
        "profit_factor": _round(tearsheet.get("profit_factor", 0.0)),
        "total_trades": int(tearsheet.get("total_trades", 0)),
        "avg_trade_duration_days": _round(
            tearsheet.get("avg_trade_duration_days", 0.0)
        ),
    }


def format_feature_stats(features_df: pd.DataFrame) -> dict[str, Any]:
    """Extract feature quality statistics for Claude interpretation.

    Args:
        features_df: DataFrame of features (rows = dates, columns = features).

    Returns:
        Dict with feature_count, missing_pct per feature, lag-1 autocorrelation,
        and top 5 correlations with target column if present.
    """
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_count = len(numeric_cols)

    # Missing percentage per feature.
    missing_pct = {}
    for col in numeric_cols:
        pct = float(features_df[col].isna().mean())
        missing_pct[col] = round(pct * 100, 2)

    # Lag-1 autocorrelation per feature.
    autocorrelation = {}
    for col in numeric_cols:
        try:
            ac = features_df[col].autocorr(lag=1)
            autocorrelation[col] = round(float(ac), 4) if pd.notna(ac) else None
        except Exception:
            autocorrelation[col] = None

    # Top 5 correlations with target column (if present).
    target_correlations: list[dict[str, Any]] = []
    target_candidates = [
        c for c in features_df.columns if "target" in c.lower() or "forward" in c.lower()
    ]
    if target_candidates:
        target_col = target_candidates[0]
        try:
            corrs = features_df[numeric_cols].corrwith(features_df[target_col]).dropna()
            # Exclude the target itself.
            corrs = corrs.drop(labels=[target_col], errors="ignore")
            top5 = corrs.abs().nlargest(5)
            for feat in top5.index:
                target_correlations.append({
                    "feature": feat,
                    "correlation": round(float(corrs[feat]), 4),
                    "abs_correlation": round(float(top5[feat]), 4),
                })
        except Exception:
            pass

    return {
        "feature_count": feature_count,
        "missing_pct": missing_pct,
        "autocorrelation": autocorrelation,
        "target_correlations": target_correlations,
    }


def format_risk_metrics(risk_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalise risk metrics for Claude interpretation.

    Args:
        risk_dict: Raw risk metrics dict (e.g. from risk_summary).

    Returns:
        Dict with all floats rounded, VaR as positive percentage,
        drawdown periods as structured list.
    """

    def _round(val: Any, dp: int = 4) -> Any:
        if isinstance(val, (int, float, np.floating)):
            return round(float(val), dp)
        return val

    # VaR as positive percentage.
    var_raw = risk_dict.get("var_95", risk_dict.get("value_at_risk", 0.0))
    var_val = abs(float(var_raw)) if isinstance(var_raw, (int, float, np.floating)) else 0.0
    var_str = f"{var_val * 100:.2f}%"

    cvar_raw = risk_dict.get("cvar_95", risk_dict.get("conditional_var", 0.0))
    cvar_val = abs(float(cvar_raw)) if isinstance(cvar_raw, (int, float, np.floating)) else 0.0
    cvar_str = f"{cvar_val * 100:.2f}%"

    # Drawdown periods as structured list.
    drawdown_periods: list[dict[str, Any]] = []
    dd_data = risk_dict.get("drawdown_periods", [])
    if isinstance(dd_data, list) and dd_data:
        for period in dd_data:
            if isinstance(period, dict):
                drawdown_periods.append({
                    "start": str(period.get("start", "")),
                    "end": str(period.get("end", "")),
                    "depth": _round(period.get("depth", 0.0)),
                })
    elif "peak_date" in risk_dict and "trough_date" in risk_dict:
        # Single drawdown from max_drawdown() output.
        drawdown_periods.append({
            "start": str(risk_dict.get("peak_date", "")),
            "end": str(risk_dict.get("trough_date", "")),
            "depth": _round(risk_dict.get("max_drawdown", 0.0)),
        })

    return {
        "sharpe_ratio": _round(risk_dict.get("sharpe", risk_dict.get("sharpe_ratio", 0.0))),
        "sortino_ratio": _round(risk_dict.get("sortino", risk_dict.get("sortino_ratio", 0.0))),
        "annualised_return": _round(risk_dict.get("annualised_return", 0.0)),
        "annualised_volatility": _round(risk_dict.get("annualised_volatility", 0.0)),
        "max_drawdown": _round(risk_dict.get("max_drawdown", 0.0)),
        "var_95": var_str,
        "cvar_95": cvar_str,
        "calmar_ratio": _round(risk_dict.get("calmar_ratio", 0.0)),
        "drawdown_periods": drawdown_periods,
    }


def display_interpretation(result: dict[str, Any]) -> None:
    """Render an interpretation dict as styled HTML in a Jupyter cell.

    Falls back to plain print() if IPython is not available.

    Args:
        result: Interpretation dict from QuantInterpreter.interpret().
    """
    confidence = result.get("confidence", "medium")
    confidence_colours = {
        "high": "#2e7d32",
        "medium": "#f57f17",
        "low": "#c62828",
    }
    badge_colour = confidence_colours.get(confidence, "#757575")

    summary = result.get("summary", "No summary available")
    observations = result.get("observations", [])
    warnings = result.get("warnings", [])
    suggestions = result.get("suggestions", [])

    # Build HTML.
    html_parts = [
        '<div style="font-family: sans-serif; padding: 12px; '
        'border: 1px solid #e0e0e0; border-radius: 8px; '
        'background: #fafafa; margin: 8px 0;">',
        f'<p style="font-size: 15px;"><strong>{summary}</strong></p>',
        f'<span style="display: inline-block; padding: 2px 10px; '
        f'border-radius: 4px; color: white; font-size: 12px; '
        f'background: {badge_colour};">Confidence: {confidence}</span>',
    ]

    if observations:
        html_parts.append("<h4>Observations</h4><ul>")
        for obs in observations:
            html_parts.append(f"<li>{obs}</li>")
        html_parts.append("</ul>")

    if warnings:
        html_parts.append('<h4 style="color: #e65100;">Warnings</h4><ul>')
        for warn in warnings:
            html_parts.append(
                f'<li style="color: #e65100;">\u26a0\ufe0f {warn}</li>'
            )
        html_parts.append("</ul>")

    if suggestions:
        html_parts.append("<h4>Suggestions</h4><ol>")
        for sug in suggestions:
            html_parts.append(f"<li>{sug}</li>")
        html_parts.append("</ol>")

    html_parts.append("</div>")
    html_str = "\n".join(html_parts)

    try:
        from IPython.display import HTML, display  # type: ignore[import-untyped]

        display(HTML(html_str))
    except ImportError:
        # Fallback: plain text.
        print(f"\n{'=' * 60}")
        print(f"  {summary}")
        print(f"  Confidence: {confidence}")
        if observations:
            print("\n  Observations:")
            for obs in observations:
                print(f"    - {obs}")
        if warnings:
            print("\n  Warnings:")
            for warn in warnings:
                print(f"    ! {warn}")
        if suggestions:
            print("\n  Suggestions:")
            for i, sug in enumerate(suggestions, 1):
                print(f"    {i}. {sug}")
        print(f"{'=' * 60}\n")
