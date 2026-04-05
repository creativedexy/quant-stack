# Manual test checklist:
# 1. Run with empty DB -> "no data yet" placeholders visible
# 2. Run with sample data -> all sections display correctly
# 3. Budget bars reflect current state
# 4. Fill history table shows last 20 entries
# 5. Performance table shows improvement percentages
# 6. Auto-refresh triggers after 5 minutes
"""Streamlit dashboard for smart-dca performance monitoring.

Read-only view of DCA performance vs naive baseline, budget utilisation,
signal scores, and fill history.  No trading actions are exposed.

Run with:
    streamlit run src/monitor/dashboard.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project root is importable when launched via `streamlit run`
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.execution.budget_tracker import AllocationState  # noqa: E402
from src.monitor.performance_log import FillRecord, PerformanceLog  # noqa: E402

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
_DB_PATH = _PROJECT_ROOT / "data" / "performance.db"
_BUDGET_STATE_PATH = _PROJECT_ROOT / "data" / "budget_state.json"
_CONFIG_PATH = _PROJECT_ROOT / "config" / "settings.yaml"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_performance_log() -> PerformanceLog | None:
    """Return PerformanceLog instance or None if the DB does not exist."""
    if not _DB_PATH.exists():
        return None
    return PerformanceLog(db_path=_DB_PATH)


def _load_budget_states() -> dict[str, AllocationState]:
    """Load budget state from JSON file.

    Returns:
        Dict of asset -> AllocationState, or empty dict if unavailable.
    """
    if not _BUDGET_STATE_PATH.exists():
        return {}
    try:
        raw = _BUDGET_STATE_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        return {asset: AllocationState(**state_dict) for asset, state_dict in data.items()}
    except (json.JSONDecodeError, TypeError, KeyError):
        return {}


def _load_config() -> dict:
    """Load settings.yaml for thresholds and signal weights.

    Returns:
        Settings dict or sensible defaults if the file is missing.
    """
    if not _CONFIG_PATH.exists():
        return {}
    try:
        import yaml

        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="smart-dca performance",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# Auto-refresh every 5 minutes (300 seconds)
st.markdown(
    '<meta http-equiv="refresh" content="300">',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Section 1 -- Header
# ---------------------------------------------------------------------------
st.title("smart-dca performance")
st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")

st.divider()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
perf_log = _load_performance_log()
budget_states = _load_budget_states()
config = _load_config()

all_fills: list[FillRecord] = []
perf_summary: list[dict] = []

if perf_log is not None:
    all_fills = perf_log.get_all_fills()
    perf_summary = perf_log.compute_performance_summary()


# ---------------------------------------------------------------------------
# Section 2 -- Current Week (budget utilisation)
# ---------------------------------------------------------------------------
st.header("Current Week")

if not budget_states:
    st.info("No budget state available yet. The scheduler has not run.")
else:
    cols = st.columns(len(budget_states))
    for idx, (asset, state) in enumerate(sorted(budget_states.items())):
        with cols[idx]:
            st.subheader(asset)
            allocation = state.weekly_allocation_gbp
            spent = state.spent_this_week_gbp
            remaining = max(allocation - spent, 0.0)

            # Count buys executed this week for this asset
            week_start_str = state.week_start_utc
            buys_this_week = 0
            for fill in all_fills:
                if fill.asset == asset and fill.timestamp >= week_start_str:
                    buys_this_week += 1

            st.metric("Weekly Allocation", f"\u00a3{allocation:.2f}")
            st.metric("Spent This Week", f"\u00a3{spent:.2f}")
            st.metric("Remaining", f"\u00a3{remaining:.2f}")
            st.metric("Buys Executed", buys_this_week)

            # Budget utilisation progress bar
            utilisation = spent / allocation if allocation > 0 else 0.0
            utilisation = min(utilisation, 1.0)
            st.progress(utilisation, text=f"Budget utilisation: {utilisation:.0%}")

st.divider()

# ---------------------------------------------------------------------------
# Section 3 -- Signal Score
# ---------------------------------------------------------------------------
st.header("Signal Score")

scoring_cfg = config.get("scoring", {})
buy_threshold = scoring_cfg.get("buy_threshold", 65)
signal_weights = scoring_cfg.get("signal_weights", {})

if not signal_weights:
    st.info("No signal configuration found. Ensure config/settings.yaml is present.")
else:
    # Display per-asset composite score from most recent fill
    # (The dashboard is read-only; we infer the last known score from fills.)
    assets_in_budget = sorted(budget_states.keys()) if budget_states else []
    assets_in_fills = sorted({f.asset for f in all_fills}) if all_fills else []
    all_assets = sorted(set(assets_in_budget) | set(assets_in_fills))

    if not all_assets:
        st.info("No assets configured or fills recorded yet.")
    else:
        score_cols = st.columns(len(all_assets))
        for idx, asset in enumerate(all_assets):
            with score_cols[idx]:
                # Find the most recent fill for this asset to get score
                asset_fills = [f for f in all_fills if f.asset == asset]
                if asset_fills:
                    latest = asset_fills[-1]  # fills are timestamp ASC
                    latest_score = latest.score_at_buy
                    should_buy = latest_score >= buy_threshold
                    st.metric(
                        f"{asset} Composite Score",
                        f"{latest_score:.1f} / 100",
                    )
                    if should_buy:
                        st.success("Would buy now: YES")
                    else:
                        st.warning("Would buy now: NO")
                else:
                    st.metric(f"{asset} Composite Score", "-- / 100")
                    st.info("No fills recorded")

        # Signal weight breakdown as horizontal bar chart
        st.subheader("Signal Weight Breakdown")
        weight_df = pd.DataFrame(
            {
                "Signal": list(signal_weights.keys()),
                "Weight": list(signal_weights.values()),
            }
        )
        st.bar_chart(
            weight_df.set_index("Signal"),
            horizontal=True,
        )

st.divider()

# ---------------------------------------------------------------------------
# Section 4 -- Performance vs Naive DCA
# ---------------------------------------------------------------------------
st.header("Performance vs Naive DCA")

if not perf_summary:
    st.info("No fill data yet. Performance comparison will appear after the first buy.")
else:
    # Performance table
    st.subheader("Per-Asset Summary")
    table_data = []
    total_saved = 0.0
    for entry in perf_summary:
        asset_name = entry["asset"]
        avg_fill = entry["avg_fill_price"]
        avg_naive = entry["avg_naive_price"]
        improvement = entry["improvement_pct"]
        total_spent = entry["total_spent_gbp"]
        fill_count = entry["fill_count"]

        # Estimate saved: improvement_pct of total_spent
        saved = (improvement / 100.0) * total_spent if improvement > 0 else 0.0
        total_saved += saved

        table_data.append(
            {
                "Asset": asset_name,
                "Avg Fill Price": f"{avg_fill:.6f}",
                "Avg Naive Price": f"{avg_naive:.6f}",
                "Improvement %": f"{improvement:+.2f}%",
                "Total Spent (GBP)": f"\u00a3{total_spent:.2f}",
                "Fills": fill_count,
            }
        )

    st.dataframe(
        pd.DataFrame(table_data),
        use_container_width=True,
        hide_index=True,
    )

    # Summary stat
    if total_saved > 0:
        st.success(
            f"Total estimated GBP saved vs naive DCA since inception: \u00a3{total_saved:.2f}"
        )
    elif total_saved < 0:
        st.warning(
            f"Smart DCA is currently underperforming naive DCA by: \u00a3{abs(total_saved):.2f}"
        )
    else:
        st.info("Smart DCA is currently on par with naive DCA.")

    # Cumulative spend and value chart
    st.subheader("Cumulative Spend and Value Over Time")

    if len(all_fills) > 1:
        cum_spent = 0.0
        cum_value_smart = 0.0
        cum_value_naive = 0.0
        chart_rows = []
        for fill in all_fills:
            cum_spent += fill.amount_gbp
            cum_value_smart += fill.quantity * fill.fill_price
            cum_value_naive += fill.quantity * fill.naive_price
            chart_rows.append(
                {
                    "Date": fill.timestamp[:10],
                    "Cumulative Spent (GBP)": round(cum_spent, 2),
                    "Cumulative Value (Smart)": round(cum_value_smart, 2),
                    "Cumulative Value (Naive)": round(cum_value_naive, 2),
                }
            )

        chart_df = pd.DataFrame(chart_rows)
        chart_df["Date"] = pd.to_datetime(chart_df["Date"])
        chart_df = chart_df.set_index("Date")
        st.line_chart(chart_df)
    else:
        st.info("At least two fills are needed to display the cumulative chart.")

st.divider()

# ---------------------------------------------------------------------------
# Section 5 -- Fill History
# ---------------------------------------------------------------------------
st.header("Fill History")

if not all_fills:
    st.info("No fills recorded yet. History will appear after the first buy.")
else:
    # Show last 20 fills, newest first
    recent_fills = list(reversed(all_fills))[:20]
    history_rows = []
    for fill in recent_fills:
        diff_pct = (
            ((fill.naive_price - fill.fill_price) / fill.naive_price * 100)
            if fill.naive_price > 0
            else 0.0
        )
        history_rows.append(
            {
                "Date": fill.timestamp[:19],
                "Asset": fill.asset,
                "Fill Price": f"{fill.fill_price:.6f}",
                "Naive Price": f"{fill.naive_price:.6f}",
                "Score": f"{fill.score_at_buy:.1f}",
                "Diff %": f"{diff_pct:+.2f}%",
            }
        )

    st.dataframe(
        pd.DataFrame(history_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(f"Showing {len(history_rows)} of {len(all_fills)} total fills.")
