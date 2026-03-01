"""Execution dashboard page — Streamlit UI for paper trading and rebalancing.

Provides five sections:
1. Broker status and connection
2. Current positions table
3. Rebalance planner with plan generation and execution
4. Execution history
5. Reconciliation of actual vs target weights

Usage:
    Run via ``streamlit run src/dashboard/pages/execution.py`` or import
    ``render_execution_page`` from another Streamlit app.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None  # type: ignore[assignment]

import matplotlib.pyplot as plt
import numpy as np

from src.services.execution_service import ExecutionService


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _colour_pnl(val: float) -> str:
    """Return CSS colour for P&L values."""
    if val > 0:
        return "color: green"
    elif val < 0:
        return "color: red"
    return ""


def _colour_drift(val: float) -> str:
    """Return CSS colour for drift values."""
    if abs(val) > 0.02:
        return "color: red"
    elif abs(val) > 0.01:
        return "color: orange"
    return "color: green"


def _format_currency(value: float, currency: str = "GBP") -> str:
    """Format a monetary value with currency symbol."""
    symbols = {"GBP": "\u00a3", "USD": "$", "EUR": "\u20ac"}
    sym = symbols.get(currency, currency + " ")
    return f"{sym}{value:,.2f}"


def _get_execution_service() -> ExecutionService:
    """Retrieve or create the ExecutionService in Streamlit session state."""
    if "execution_service" not in st.session_state:
        st.session_state["execution_service"] = ExecutionService()
    return st.session_state["execution_service"]


# ------------------------------------------------------------------
# Section 1: Broker Status
# ------------------------------------------------------------------

def _render_broker_status(svc: ExecutionService) -> None:
    """Render the broker connection and account summary section."""
    st.header("Broker Status")

    status = svc.get_broker_status()

    if status["connected"]:
        mode = status.get("mode", "paper")
        if mode == "paper":
            st.success("PAPER TRADING — Connected")
        else:
            st.error("LIVE TRADING — Connected")

        currency = status.get("base_currency", "GBP")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Account Value", _format_currency(status["account_value"], currency))
        col2.metric("Cash", _format_currency(status["cash"], currency))
        col3.metric("Invested", _format_currency(status["invested"], currency))
        col4.metric("Positions", str(status["positions_count"]))
    else:
        st.warning("Broker not connected")
        if st.button("Connect Paper Broker"):
            with st.spinner("Connecting..."):
                success = svc.connect_paper_broker()
            if success:
                st.success("Paper broker connected")
                st.rerun()
            else:
                st.error("Failed to connect")


# ------------------------------------------------------------------
# Section 2: Current Positions
# ------------------------------------------------------------------

def _render_positions(svc: ExecutionService) -> None:
    """Render the current positions table."""
    st.header("Current Positions")

    status = svc.get_broker_status()
    if not status["connected"]:
        st.info("Connect the broker to view positions.")
        return

    positions = svc.broker.get_positions()
    if not positions:
        st.info("No open positions.")
        return

    account_value = status["account_value"]
    rows: list[dict[str, Any]] = []
    for ticker, pos in sorted(positions.items()):
        weight = pos["market_value"] / account_value if account_value > 0 else 0
        rows.append({
            "Ticker": ticker,
            "Quantity": pos["quantity"],
            "Current Price": round(pos["current_price"], 2),
            "Market Value": round(pos["market_value"], 2),
            "Weight": f"{weight:.2%}",
            "P&L": round(pos["pnl"], 2),
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.map(_colour_pnl, subset=["P&L"]),
        use_container_width=True,
        hide_index=True,
    )


# ------------------------------------------------------------------
# Section 3: Rebalance Planner
# ------------------------------------------------------------------

def _render_rebalance_planner(svc: ExecutionService) -> None:
    """Render the rebalance plan generation and execution section."""
    st.header("Rebalance Planner")

    status = svc.get_broker_status()
    if not status["connected"]:
        st.info("Connect the broker to use the rebalance planner.")
        return

    # Target weights input
    with st.expander("Set target weights", expanded=False):
        weights_text = st.text_area(
            "Enter target weights (one per line: TICKER,WEIGHT)",
            placeholder="AAPL,0.25\nGOOG,0.25\nMSFT,0.25\nAMZN,0.25",
            key="target_weights_input",
        )
        if st.button("Save weights"):
            try:
                weights = {}
                for line in weights_text.strip().splitlines():
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        weights[parts[0].strip()] = float(parts[1].strip())
                svc.set_target_weights(weights)
                st.success(f"Saved weights for {len(weights)} tickers")
            except (ValueError, IndexError):
                st.error("Invalid format. Use: TICKER,WEIGHT (one per line)")

    # Generate plan
    if st.button("Generate Rebalance Plan"):
        try:
            plan = svc.generate_rebalance_plan()
            st.session_state["current_plan"] = plan
            st.success(f"Plan generated: {len(plan['orders'])} orders")
        except ValueError as e:
            st.error(str(e))

    # Display plan
    plan = st.session_state.get("current_plan")
    if plan:
        st.subheader("Proposed Orders")
        if plan["orders"]:
            plan_df = pd.DataFrame(plan["orders"])
            st.dataframe(plan_df, use_container_width=True, hide_index=True)
        else:
            st.info("Portfolio already at target weights — no orders needed.")

        col1, col2 = st.columns(2)
        col1.metric("Estimated Turnover", f"{plan['turnover']:.2%}")
        col2.metric("Estimated Cost", f"{plan['total_cost_estimate']:,.2f}")

        # Execute plan
        st.divider()
        confirmed = st.checkbox(
            "I confirm this is a paper-trade execution",
            key="confirm_execution",
        )
        execute_btn = st.button(
            "Execute Plan (Paper)",
            disabled=not confirmed or not plan["orders"],
        )
        if execute_btn:
            try:
                result = svc.execute_plan(plan["plan_id"])
                st.session_state["current_plan"] = None
                st.success(
                    f"Execution {result['status']}: "
                    f"{result['orders_filled']}/{result['orders_submitted']} filled"
                )
                st.rerun()
            except ValueError as e:
                st.error(str(e))


# ------------------------------------------------------------------
# Section 4: Execution History
# ------------------------------------------------------------------

def _render_execution_history(svc: ExecutionService) -> None:
    """Render the execution history table with expandable details."""
    st.header("Execution History")

    history = svc.get_execution_history(n=20)
    if not history:
        st.info("No execution history yet.")
        return

    for i, report in enumerate(history):
        fills_count = len(report.get("fills", []))
        status = report.get("status", "unknown")
        mode = report.get("mode", "paper")
        ts = report.get("timestamp", "unknown")

        label = f"{ts} | {fills_count} orders | {status} | {mode}"
        with st.expander(label):
            col1, col2, col3 = st.columns(3)
            col1.metric("Orders Filled", f"{report.get('orders_filled', 0)}/{report.get('orders_submitted', 0)}")
            col2.metric("Commission", f"{report.get('total_commission', 0):.2f}")
            col3.metric("Trade Value", f"{report.get('total_trade_value', 0):,.2f}")

            fills = report.get("fills", [])
            if fills:
                st.dataframe(
                    pd.DataFrame(fills),
                    use_container_width=True,
                    hide_index=True,
                )


# ------------------------------------------------------------------
# Section 5: Reconciliation
# ------------------------------------------------------------------

def _render_reconciliation(svc: ExecutionService) -> None:
    """Render target-vs-actual weight comparison with chart."""
    st.header("Reconciliation")

    status = svc.get_broker_status()
    if not status["connected"]:
        st.info("Connect the broker to view reconciliation.")
        return

    recon = svc.get_reconciliation()
    tickers_data = recon["tickers"]

    if not tickers_data:
        st.info("No target weights set — nothing to reconcile.")
        return

    # Summary
    total_drift = recon["total_drift"]
    aligned = recon["aligned"]
    if aligned:
        st.success(f"Portfolio aligned (total drift: {total_drift:.4f})")
    else:
        st.warning(f"Portfolio drifted (total drift: {total_drift:.4f})")

    # Table
    recon_df = pd.DataFrame(tickers_data)
    st.dataframe(
        recon_df.style.map(_colour_drift, subset=["drift"]),
        use_container_width=True,
        hide_index=True,
    )

    # Grouped bar chart: target vs actual
    if len(tickers_data) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        tickers = [r["ticker"] for r in tickers_data]
        target = [r["target_weight"] for r in tickers_data]
        actual = [r["actual_weight"] for r in tickers_data]

        x = np.arange(len(tickers))
        width = 0.35
        ax.bar(x - width / 2, target, width, label="Target", color="#4CAF50", alpha=0.8)
        ax.bar(x + width / 2, actual, width, label="Actual", color="#2196F3", alpha=0.8)
        ax.set_ylabel("Weight")
        ax.set_xticks(x)
        ax.set_xticklabels(tickers, rotation=45, ha="right")
        ax.legend()
        ax.set_title("Target vs Actual Weights")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ------------------------------------------------------------------
# Main page renderer
# ------------------------------------------------------------------

def render_execution_page() -> None:
    """Render the full execution dashboard page.

    Call this from a Streamlit app to display all execution sections.
    """
    st.title("Execution & Trading")

    svc = _get_execution_service()

    _render_broker_status(svc)
    st.divider()
    _render_positions(svc)
    st.divider()
    _render_rebalance_planner(svc)
    st.divider()
    _render_execution_history(svc)
    st.divider()
    _render_reconciliation(svc)

    # Footer
    st.divider()
    st.caption(
        "Live trading is only available via the CLI with the ``--live`` flag "
        "and requires explicit confirmation.  See the execution module "
        "documentation for details."
    )


# Allow running this page directly: streamlit run src/dashboard/pages/execution.py
if __name__ == "__main__":
    if st is not None:
        st.set_page_config(page_title="Execution — Quant Stack", layout="wide")
        render_execution_page()
