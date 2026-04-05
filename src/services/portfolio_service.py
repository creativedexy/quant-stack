"""Portfolio service — portfolio state and analytics for the dashboard.

Provides current weights, risk metrics, equity curves, and allocation
data without exposing underlying file I/O or computation details.

Usage:
    from src.services.data_service import DataService
    from src.services.portfolio_service import PortfolioService
    svc = PortfolioService(DataService())
    weights = svc.get_current_weights()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.services.data_service import DataService
from src.utils.logging import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class PortfolioService:
    """Portfolio state and analytics for the dashboard."""

    def __init__(
        self,
        data_service: DataService,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialise the portfolio service.

        Args:
            data_service: DataService instance for price data access.
            config: Optional configuration dict.
        """
        self.data = data_service
        self._config = config or {}

        data_dir_name = (
            self._config.get("general", {}).get("data_dir", "data")
        )
        self._data_dir = _PROJECT_ROOT / data_dir_name
        self._processed_dir = self._data_dir / "processed"

    def get_current_weights(self) -> pd.Series:
        """Load last saved target weights.

        Reads from data/processed/weights.parquet if available.
        Falls back to equal weights for the configured universe.

        Returns:
            Series indexed by ticker with portfolio weights summing to 1.
        """
        weights_path = self._processed_dir / "weights.parquet"

        if weights_path.exists():
            try:
                df = pd.read_parquet(weights_path)
                # Expect a single-column DataFrame or Series stored as parquet
                if isinstance(df, pd.DataFrame) and len(df.columns) >= 1:
                    weights = df.iloc[:, -1]  # Last column = most recent
                    weights.name = "weight"
                    return weights
            except Exception as exc:
                logger.error(f"Failed to load weights: {exc}")

        # Fall back to equal weights
        tickers = self.data._tickers
        if not tickers:
            return pd.Series(dtype=float, name="weight")

        n = len(tickers)
        return pd.Series(
            [1.0 / n] * n, index=tickers, name="weight"
        )

    def get_risk_metrics(self) -> dict[str, float]:
        """Compute current risk metrics for the portfolio.

        Returns:
            Dict with keys: annual_return, annual_volatility, sharpe_ratio,
            max_drawdown, var_95, cvar_95. Values are NaN if computation
            is not possible.
        """
        returns = self.data.get_returns()
        weights = self.get_current_weights()

        defaults: dict[str, float] = {
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
            "var_95": np.nan,
            "cvar_95": np.nan,
        }

        if returns.empty or weights.empty:
            return defaults

        # Align tickers
        common = returns.columns.intersection(weights.index)
        if len(common) == 0:
            return defaults

        returns = returns[common].dropna()
        w = weights[common]
        w = w / w.sum()  # Re-normalise

        portfolio_returns = returns @ w

        if len(portfolio_returns) < 2:
            return defaults

        annual_return = float(portfolio_returns.mean() * 252)
        annual_vol = float(portfolio_returns.std() * np.sqrt(252))
        sharpe = (
            annual_return / annual_vol if annual_vol > 0 else np.nan
        )

        # Max drawdown
        cum = (1 + portfolio_returns).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max
        max_dd = float(drawdown.min())

        # VaR and CVaR (historical, 95%)
        sorted_returns = portfolio_returns.sort_values()
        var_idx = int(len(sorted_returns) * 0.05)
        var_95 = float(sorted_returns.iloc[var_idx]) if var_idx > 0 else np.nan
        cvar_95 = (
            float(sorted_returns.iloc[:var_idx].mean())
            if var_idx > 0
            else np.nan
        )

        return {
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "var_95": var_95,
            "cvar_95": cvar_95,
        }

    def get_target_weights(self) -> pd.Series:
        """Load target weights from the optimiser output.

        Reads from data/processed/target_weights.parquet if available.
        Falls back to current weights (no rebalance needed).

        Returns:
            Series indexed by ticker with target weights summing to 1.
        """
        target_path = self._processed_dir / "target_weights.parquet"

        if target_path.exists():
            try:
                df = pd.read_parquet(target_path)
                if isinstance(df, pd.DataFrame) and len(df.columns) >= 1:
                    weights = df.iloc[:, -1]
                    weights.name = "target_weight"
                    return weights
            except Exception as exc:
                logger.error(f"Failed to load target weights: {exc}")

        # Fall back to current weights (portfolio is on target)
        current = self.get_current_weights()
        current.name = "target_weight"
        return current

    def get_rebalance_orders(
        self, portfolio_value: float = 100_000.0,
    ) -> pd.DataFrame:
        """Compute rebalance orders to move from current to target weights.

        Args:
            portfolio_value: Total portfolio value in base currency for
                estimating trade values.

        Returns:
            DataFrame with columns: ticker, current_weight, target_weight,
            delta, estimated_value_gbp.  Empty DataFrame if no data.
        """
        current = self.get_current_weights()
        target = self.get_target_weights()

        if current.empty or target.empty:
            return pd.DataFrame(
                columns=[
                    "ticker", "current_weight", "target_weight",
                    "delta", "estimated_value_gbp",
                ]
            )

        # Align on all tickers from both sets
        all_tickers = sorted(set(current.index) | set(target.index))
        rows = []
        for ticker in all_tickers:
            cw = float(current.get(ticker, 0.0))
            tw = float(target.get(ticker, 0.0))
            delta = tw - cw
            est_value = abs(delta) * portfolio_value
            rows.append({
                "ticker": ticker,
                "current_weight": round(cw, 6),
                "target_weight": round(tw, 6),
                "delta": round(delta, 6),
                "estimated_value_gbp": round(est_value, 2),
            })

        return pd.DataFrame(rows)

    def get_equity_curve(
        self, strategy_name: str | None = None
    ) -> pd.Series:
        """Load or compute an equity curve.

        If a strategy name is given, attempts to load cached backtest
        results. Otherwise computes from current weights and prices.

        Args:
            strategy_name: Optional strategy to load results for.

        Returns:
            Series representing cumulative portfolio value over time.
        """
        # Try loading cached backtest results
        if strategy_name is not None:
            backtest_path = (
                self._processed_dir
                / "backtests"
                / f"{strategy_name}_equity.parquet"
            )
            if backtest_path.exists():
                try:
                    df = pd.read_parquet(backtest_path)
                    if isinstance(df, pd.DataFrame) and len(df.columns) >= 1:
                        return df.iloc[:, 0]
                except Exception as exc:
                    logger.error(
                        f"Failed to load equity curve for "
                        f"{strategy_name}: {exc}"
                    )

        # Compute from current weights + prices
        returns = self.data.get_returns()
        weights = self.get_current_weights()

        if returns.empty or weights.empty:
            return pd.Series(dtype=float, name="equity")

        common = returns.columns.intersection(weights.index)
        if len(common) == 0:
            return pd.Series(dtype=float, name="equity")

        w = weights[common]
        w = w / w.sum()
        portfolio_returns = returns[common] @ w
        equity = (1 + portfolio_returns).cumprod()
        equity.name = "equity"
        return equity

    def get_allocation_chart_data(self) -> dict[str, Any]:
        """Return data formatted for a pie/bar chart of current allocation.

        Returns:
            Dict with 'labels' (list[str]) and 'values' (list[float])
            suitable for charting libraries.
        """
        weights = self.get_current_weights()

        if weights.empty:
            return {"labels": [], "values": []}

        # Filter out zero-weight positions
        weights = weights[weights > 0]

        return {
            "labels": weights.index.tolist(),
            "values": weights.values.tolist(),
        }

    # ------------------------------------------------------------------
    # Portfolio page helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_account_holdings(account_cfg: Any) -> list[dict[str, Any]]:
        """Extract holdings list from an account config section.

        Supports both the new format (dict with 'holdings' key) and the
        legacy format (plain list of ticker dicts).
        """
        if isinstance(account_cfg, dict):
            return account_cfg.get("holdings", []) or []
        if isinstance(account_cfg, list):
            return account_cfg
        return []

    @staticmethod
    def _get_account_label(section: str, account_cfg: Any) -> str:
        """Return the display label for an account section."""
        if isinstance(account_cfg, dict) and "label" in account_cfg:
            return account_cfg["label"]
        _DEFAULT_LABELS = {
            "chip_isa": "Chip ISA",
            "revolut_gia": "Revolut GIA",
            "crypto": "Crypto",
        }
        return _DEFAULT_LABELS.get(section, section)

    def _ticker_name_map(self) -> dict[str, str]:
        """Build ticker -> company name mapping from config."""
        name_map: dict[str, str] = {}
        portfolio_cfg = self._config.get("portfolio", {})
        for section in ("chip_isa", "revolut_gia", "crypto"):
            holdings = self._get_account_holdings(portfolio_cfg.get(section))
            for entry in holdings:
                if "ticker" in entry and "name" in entry:
                    name_map[entry["ticker"]] = entry["name"]
        return name_map

    def _ticker_account_map(self) -> dict[str, str]:
        """Build ticker -> account label mapping from config."""
        account_map: dict[str, str] = {}
        portfolio_cfg = self._config.get("portfolio", {})
        for section in ("chip_isa", "revolut_gia", "crypto"):
            account_cfg = portfolio_cfg.get(section)
            if account_cfg is None:
                continue
            label = self._get_account_label(section, account_cfg)
            holdings = self._get_account_holdings(account_cfg)
            for entry in holdings:
                if "ticker" in entry:
                    account_map[entry["ticker"]] = label
        return account_map

    def _ticker_value_map(self) -> dict[str, float]:
        """Build ticker -> config value_gbp mapping."""
        value_map: dict[str, float] = {}
        portfolio_cfg = self._config.get("portfolio", {})
        for section in ("chip_isa", "revolut_gia", "crypto"):
            holdings = self._get_account_holdings(portfolio_cfg.get(section))
            for entry in holdings:
                if "ticker" in entry and "value_gbp" in entry:
                    value_map[entry["ticker"]] = float(entry["value_gbp"])
        return value_map

    def get_positions(self) -> list[dict[str, Any]]:
        """Return enriched position list for the portfolio page.

        Each dict has: ticker, name, account, weight, daily_return,
        market_value, price, total_return.
        """
        weights = self.get_current_weights()
        prices = self.data.get_prices()
        name_map = self._ticker_name_map()
        account_map = self._ticker_account_map()
        value_map = self._ticker_value_map()

        if weights.empty or prices.empty:
            return []

        common = weights.index.intersection(prices.columns)
        if len(common) == 0:
            return []

        weights = weights[common]
        weights = weights / weights.sum()

        latest = prices[common].iloc[-1]
        if len(prices) >= 2:
            prev = prices[common].iloc[-2]
            daily_ret = (latest - prev) / prev
        else:
            daily_ret = pd.Series(0.0, index=common)

        first = prices[common].iloc[0]
        total_ret = (latest - first) / first.replace(0, 1)

        # Use config value_gbp when available; recompute weights from those
        has_values = any(t in value_map for t in common)
        if has_values:
            total_cfg_value = sum(value_map.get(t, 0) for t in common)

        positions: list[dict[str, Any]] = []
        for ticker in common:
            ret = float(daily_ret[ticker])
            mv = value_map.get(ticker, float(latest[ticker]) * 200)
            w = mv / total_cfg_value if has_values and total_cfg_value > 0 else float(weights[ticker])
            positions.append({
                "ticker": ticker,
                "name": name_map.get(ticker, ticker),
                "account": account_map.get(ticker, "Other"),
                "weight": w,
                "daily_return": ret,
                "total_return": float(total_ret[ticker]),
                "market_value": mv,
                "price": float(latest[ticker]),
            })

        positions.sort(key=lambda p: p["weight"], reverse=True)
        return positions

    def get_summary(self) -> dict[str, Any]:
        """Return portfolio summary for the header bar.

        Returns:
            Dict with total_value, daily_pnl, daily_pnl_pct,
            position_count, sector_count, best_today, worst_today.
        """
        positions = self.get_positions()
        defaults: dict[str, Any] = {
            "total_value": 0.0,
            "daily_pnl": 0.0,
            "daily_pnl_pct": 0.0,
            "position_count": 0,
            "sector_count": 0,
            "best_today": None,
            "worst_today": None,
        }

        if not positions:
            return defaults

        total_value = sum(p["market_value"] for p in positions)
        daily_pnl = sum(
            p["market_value"] * p["daily_return"] / (1 + p["daily_return"])
            for p in positions
        )
        daily_pnl_pct = daily_pnl / total_value if total_value > 0 else 0.0

        best = max(positions, key=lambda p: p["daily_return"])
        worst = min(positions, key=lambda p: p["daily_return"])

        # Count unique accounts from position data
        accounts = {p.get("account", "Other") for p in positions}
        sector_count = max(len(accounts), 1)

        return {
            "total_value": total_value,
            "daily_pnl": daily_pnl,
            "daily_pnl_pct": daily_pnl_pct,
            "position_count": len(positions),
            "sector_count": max(sector_count, 1),
            "best_today": {
                "ticker": best["ticker"],
                "return": best["daily_return"],
            },
            "worst_today": {
                "ticker": worst["ticker"],
                "return": worst["daily_return"],
            },
        }

    def get_heatmap_data(
        self,
        positions: list[dict[str, Any]],
        signals: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Enrich positions with heatmap display attributes.

        Returns positions with added: cell_size, colour_class, has_signal.
        """
        signals = signals or {}
        enriched: list[dict[str, Any]] = []

        for pos in positions:
            w = pos.get("weight", 0) * 100  # percentage
            ret = pos.get("daily_return", 0) * 100  # percentage

            # Cell size based on weight
            if w >= 15:
                cell_size = "big"
            elif w >= 10:
                cell_size = "medium"
            else:
                cell_size = "small"

            # Colour class based on return
            if ret > 5:
                colour_class = "pos-strong"
            elif ret > 1:
                colour_class = "pos-mild"
            elif ret > -1:
                colour_class = "neutral"
            elif ret > -5:
                colour_class = "neg-mild"
            else:
                colour_class = "neg-strong"

            enriched.append({
                **pos,
                "cell_size": cell_size,
                "colour_class": colour_class,
                "has_signal": pos["ticker"] in signals,
            })

        # Sort by account then weight descending so template can group
        _ACCOUNT_ORDER = {"Chip ISA": 0, "Revolut GIA": 1, "Crypto": 2}
        enriched.sort(
            key=lambda p: (
                _ACCOUNT_ORDER.get(p.get("account", "Other"), 99),
                -p.get("weight", 0),
            )
        )
        return enriched

    def get_risk_alert(
        self, risk_metrics: dict[str, float],
    ) -> dict[str, Any] | None:
        """Check if risk exceeds threshold and return alert data.

        Returns None if risk is within threshold, otherwise a dict with
        message, primary_driver, worst_case_loss, actions.
        """
        var_95 = risk_metrics.get("var_95", float("nan"))
        if np.isnan(var_95):
            return None

        # Threshold from config (default 2% daily)
        risk_cfg = self._config.get("risk", {})
        var_threshold = risk_cfg.get("var_threshold", -0.02)

        if var_95 >= var_threshold:
            return None

        # Find the primary driver (worst daily return position)
        positions = self.get_positions()
        if positions:
            worst = min(positions, key=lambda p: p["daily_return"])
            driver = worst["ticker"]
        else:
            driver = "unknown"

        # Estimate worst-case loss in GBP
        summary = self.get_summary()
        total_value = summary.get("total_value", 100_000)
        worst_case_loss = abs(var_95) * total_value

        return {
            "message": (
                f"Your portfolio risk is higher than usual. "
                f"Worst-case daily loss has increased to around "
                f"GBP{worst_case_loss:,.0f}, mainly driven by {driver}."
            ),
            "primary_driver": driver,
            "worst_case_loss": worst_case_loss,
            "actions": [
                {"label": f"Review {driver}", "href": f"/ui/analyse?ticker={driver}"},
                {"label": "Set Stop Loss", "href": "/ui/execution"},
            ],
        }
