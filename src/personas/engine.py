"""Persona engine -- lifecycle management and brief orchestration.

Central orchestrator for all trading personas.  Manages state
transitions (dormant -> pending -> active), runs event detection,
and coordinates research brief generation.

Usage::

    from src.personas.engine import PersonaEngine
    engine = PersonaEngine(config)
    pending = engine.check_events(market_data)
    engine.activate("crisis")
    brief = engine.generate_brief("crisis")
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.personas.analyst import PersonaAnalyst
from src.personas.capital import PersonaCapitalTracker
from src.personas.detectors import detector_registry
from src.personas.persona import (
    PersonaBrief,
    PersonaState,
    PersonaType,
    TradeRecommendation,
    TradingPersona,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class PersonaEngine:
    """Lifecycle manager and orchestrator for trading personas.

    Args:
        config: Project configuration dict containing ``personas`` section.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        personas_cfg = config.get("personas", {})
        self._enabled = personas_cfg.get("enabled", True)

        data_dir = config.get("general", {}).get("data_dir", "data")
        self._state_dir = _PROJECT_ROOT / data_dir / "processed" / "personas"
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self._state_dir / "state.json"

        # Build personas from config
        self._personas: dict[str, TradingPersona] = {}
        definitions = personas_cfg.get("definitions", {})
        for persona_id, defn in definitions.items():
            self._personas[persona_id] = TradingPersona.from_config(
                persona_id, defn
            )

        # Validate capital allocation
        total_pct = sum(p.capital_pct for p in self._personas.values())
        if total_pct > 1.0 + 1e-9:
            raise ValueError(
                f"Persona capital_pct values sum to {total_pct:.2%}, "
                "exceeding 100%"
            )

        # Restore persisted state (overrides initial state from config)
        self._load_state()

        # Sub-components
        self._capital = PersonaCapitalTracker(config)
        self._analyst = PersonaAnalyst(config)

        logger.info(
            "PersonaEngine initialised with %d personas "
            "(active: %d, dormant: %d)",
            len(self._personas),
            sum(1 for p in self._personas.values()
                if p.state == PersonaState.ACTIVE),
            sum(1 for p in self._personas.values()
                if p.state == PersonaState.DORMANT),
        )

    # ------------------------------------------------------------------
    # Public API -- lifecycle
    # ------------------------------------------------------------------

    def activate(self, persona_id: str) -> bool:
        """Activate a persona (user confirmation).

        Valid transitions: PENDING -> ACTIVE, DORMANT -> ACTIVE.

        Args:
            persona_id: Persona to activate.

        Returns:
            ``True`` if activation succeeded, ``False`` otherwise.
        """
        persona = self._personas.get(persona_id)
        if persona is None:
            logger.error("Unknown persona: %s", persona_id)
            return False

        if persona.state not in (PersonaState.PENDING, PersonaState.DORMANT):
            logger.warning(
                "Cannot activate %s from state %s",
                persona_id,
                persona.state.value,
            )
            return False

        persona.state = PersonaState.ACTIVE
        persona.activated_at = datetime.now(timezone.utc)
        self._persist_state()
        logger.info("Persona activated: %s", persona_id)
        return True

    def deactivate(self, persona_id: str) -> bool:
        """Deactivate a persona back to dormant.

        Args:
            persona_id: Persona to deactivate.

        Returns:
            ``True`` if deactivation succeeded.
        """
        persona = self._personas.get(persona_id)
        if persona is None:
            logger.error("Unknown persona: %s", persona_id)
            return False

        if persona.state == PersonaState.DORMANT:
            logger.info("Persona %s already dormant", persona_id)
            return True

        persona.state = PersonaState.DORMANT
        persona.deactivated_at = datetime.now(timezone.utc)
        self._persist_state()
        logger.info("Persona deactivated: %s", persona_id)
        return True

    def pause(self, persona_id: str) -> bool:
        """Pause an active persona (keep state, stop analysis).

        Args:
            persona_id: Persona to pause.

        Returns:
            ``True`` if pause succeeded.
        """
        persona = self._personas.get(persona_id)
        if persona is None:
            logger.error("Unknown persona: %s", persona_id)
            return False

        if persona.state != PersonaState.ACTIVE:
            logger.warning(
                "Cannot pause %s from state %s",
                persona_id,
                persona.state.value,
            )
            return False

        persona.state = PersonaState.PAUSED
        self._persist_state()
        logger.info("Persona paused: %s", persona_id)
        return True

    # ------------------------------------------------------------------
    # Public API -- event detection
    # ------------------------------------------------------------------

    def check_events(
        self, market_data: dict[str, pd.DataFrame]
    ) -> list[dict[str, Any]]:
        """Run detectors for all dormant event-triggered personas.

        Args:
            market_data: Dictionary of ticker -> OHLCV DataFrame.

        Returns:
            List of dicts for newly pending personas:
            ``[{persona_id, name, reason, metrics}, ...]``
        """
        newly_pending: list[dict[str, Any]] = []

        for persona_id, persona in self._personas.items():
            if persona.state != PersonaState.DORMANT:
                continue
            if persona.persona_type != PersonaType.EVENT_TRIGGERED:
                continue

            activation = persona.activation_config
            detector_name = activation.get("detector")
            if not detector_name:
                continue

            params = activation.get("params", {})
            try:
                detector = detector_registry.create(detector_name, params)
            except KeyError:
                logger.warning(
                    "Unknown detector '%s' for persona %s",
                    detector_name,
                    persona_id,
                )
                continue

            # Filter market data to persona's universe + detector-specific tickers
            persona_data = {
                t: df for t, df in market_data.items()
                if t in persona.universe or t in _detector_tickers(params)
            }

            result = detector.check(persona_data)
            if result.triggered:
                persona.state = PersonaState.PENDING
                self._persist_state()
                entry = {
                    "persona_id": persona_id,
                    "name": persona.name,
                    "reason": result.reason,
                    "metrics": result.metrics,
                }
                newly_pending.append(entry)
                logger.info(
                    "Persona %s -> PENDING: %s",
                    persona_id,
                    result.reason,
                )

        return newly_pending

    # ------------------------------------------------------------------
    # Public API -- brief generation
    # ------------------------------------------------------------------

    def generate_brief(
        self,
        persona_id: str,
        market_data: dict[str, pd.DataFrame] | None = None,
        total_capital: float = 100_000.0,
    ) -> PersonaBrief | None:
        """Generate a research brief for an active persona.

        Args:
            persona_id: Persona to generate a brief for.
            market_data: Ticker -> OHLCV DataFrames.  If ``None``, the
                brief will have empty market data.
            total_capital: Total portfolio capital for budget sizing.

        Returns:
            A :class:`PersonaBrief`, or ``None`` if the persona is not
            in a state that allows brief generation.
        """
        persona = self._personas.get(persona_id)
        if persona is None:
            logger.error("Unknown persona: %s", persona_id)
            return None

        if persona.state not in (PersonaState.ACTIVE, PersonaState.PENDING):
            logger.warning(
                "Cannot generate brief for %s in state %s",
                persona_id,
                persona.state.value,
            )
            return None

        market_data = market_data or {}

        # Allocate capital for sizing
        self._capital.allocate_capital(total_capital)

        # Build market snapshot for the brief
        market_snapshot = self._build_market_snapshot(
            persona.universe, market_data
        )

        # Generate strategy signals if we have data
        signals = self._run_strategy(persona, market_data)

        # Size trade recommendations
        recommendations = self._build_recommendations(
            persona, signals, market_data
        )

        # Generate the brief via analyst
        trigger_reason = ""
        if persona.persona_type == PersonaType.EVENT_TRIGGERED:
            trigger_reason = self._get_trigger_reason(persona_id)

        brief = self._analyst.generate_brief(
            persona,
            market_snapshot,
            signals,
            recommendations,
            trigger_reason,
        )

        # Update state
        persona.last_brief_at = datetime.now(timezone.utc)
        self._persist_state()
        self._persist_brief(persona_id, brief)

        return brief

    def generate_all_briefs(
        self,
        market_data: dict[str, pd.DataFrame] | None = None,
        total_capital: float = 100_000.0,
    ) -> dict[str, PersonaBrief]:
        """Generate briefs for all active personas.

        Returns:
            Dictionary of persona_id -> PersonaBrief for each active persona.
        """
        briefs: dict[str, PersonaBrief] = {}
        for persona_id, persona in self._personas.items():
            if persona.state == PersonaState.ACTIVE:
                brief = self.generate_brief(
                    persona_id, market_data, total_capital
                )
                if brief is not None:
                    briefs[persona_id] = brief
        return briefs

    # ------------------------------------------------------------------
    # Public API -- status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, dict[str, Any]]:
        """Return status of all personas for the UI."""
        return {
            pid: {
                "persona_id": p.persona_id,
                "name": p.name,
                "type": p.persona_type.value,
                "state": p.state.value,
                "strategy": p.strategy_name,
                "universe": p.universe,
                "capital_pct": p.capital_pct,
                "risk_params": p.risk_params,
                "activated_at": (
                    p.activated_at.isoformat() if p.activated_at else None
                ),
                "last_brief_at": (
                    p.last_brief_at.isoformat() if p.last_brief_at else None
                ),
            }
            for pid, p in self._personas.items()
        }

    def get_persona(self, persona_id: str) -> TradingPersona | None:
        """Return a specific persona, or ``None`` if not found."""
        return self._personas.get(persona_id)

    def list_personas(self) -> list[str]:
        """Return sorted list of persona IDs."""
        return sorted(self._personas.keys())

    def get_latest_brief(self, persona_id: str) -> dict[str, Any] | None:
        """Load the most recent brief for a persona from disk."""
        brief_path = self._state_dir / persona_id / "brief_latest.json"
        if not brief_path.exists():
            return None
        return json.loads(brief_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Internal -- strategy execution
    # ------------------------------------------------------------------

    def _run_strategy(
        self,
        persona: TradingPersona,
        market_data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame | None:
        """Run the persona's strategy to generate signals."""
        try:
            from src.backtest.strategy import strategy_registry
            from src.features.pipeline import FeaturePipeline

            # Collect OHLCV for persona's universe
            ticker_data: dict[str, pd.DataFrame] = {}
            for ticker in persona.universe:
                if ticker in market_data and not market_data[ticker].empty:
                    ticker_data[ticker] = market_data[ticker]

            if not ticker_data:
                logger.warning(
                    "No market data for %s universe", persona.persona_id
                )
                return None

            # Generate features
            pipeline = FeaturePipeline(self._config)
            features = pipeline.generate(ticker_data)

            if features.empty:
                return None

            # Run strategy
            strategy = strategy_registry.create(
                persona.strategy_name, persona.strategy_config
            )
            signals = strategy.generate_signals(features)
            return signals

        except Exception as exc:
            logger.error(
                "Strategy execution failed for %s: %s",
                persona.persona_id,
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Internal -- recommendations
    # ------------------------------------------------------------------

    def _build_recommendations(
        self,
        persona: TradingPersona,
        signals: pd.DataFrame | None,
        market_data: dict[str, pd.DataFrame],
    ) -> list[TradeRecommendation]:
        """Convert strategy signals into sized trade recommendations."""
        recommendations: list[TradeRecommendation] = []

        if signals is None or signals.empty:
            return recommendations

        # Get latest signal values
        latest = signals.iloc[-1]
        max_weight = persona.risk_params.get("max_weight", 0.25)

        for ticker in persona.universe:
            # Determine signal
            if "signal" in signals.columns:
                sig_val = int(latest.get("signal", 0))
            elif ticker in signals.columns:
                sig_val = int(latest.get(ticker, 0))
            else:
                continue

            if sig_val == 0:
                continue

            # Get current price
            price = self._get_latest_price(ticker, market_data)
            if price <= 0:
                continue

            direction = "BUY" if sig_val > 0 else "SELL"
            quantity = self._capital.size_recommendation(
                persona.persona_id, ticker, price, max_weight
            )

            if quantity <= 0:
                continue

            # Gather risk metrics
            risk_metrics = self._get_risk_metrics(ticker, market_data)

            reasoning = (
                f"{persona.strategy_name} signal={sig_val} "
                f"({direction.lower()} at {price:.2f})"
            )

            recommendations.append(
                TradeRecommendation(
                    ticker=ticker,
                    direction=direction,
                    suggested_size_gbp=round(quantity * price, 2),
                    suggested_quantity=quantity,
                    entry_price=price,
                    reasoning=reasoning,
                    risk_metrics=risk_metrics,
                )
            )

        return recommendations

    # ------------------------------------------------------------------
    # Internal -- market data helpers
    # ------------------------------------------------------------------

    def _build_market_snapshot(
        self,
        universe: list[str],
        market_data: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        """Build a summary snapshot of market data for the brief."""
        snapshot: dict[str, Any] = {}
        for ticker in universe:
            df = market_data.get(ticker)
            if df is None or df.empty:
                snapshot[ticker] = {"status": "no data"}
                continue

            close = df["Close"].dropna()
            if len(close) < 2:
                snapshot[ticker] = {"close": float(close.iloc[-1])}
                continue

            current = float(close.iloc[-1])
            prev = float(close.iloc[-2])
            change_pct = ((current - prev) / prev) * 100

            entry: dict[str, Any] = {
                "close": round(current, 4),
                "change_pct": f"{change_pct:+.2f}%",
                "high": round(float(df["High"].iloc[-1]), 4) if "High" in df.columns else None,
                "low": round(float(df["Low"].iloc[-1]), 4) if "Low" in df.columns else None,
            }

            # Add RSI if enough data
            if len(close) >= 14:
                delta = close.diff()
                gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
                loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
                rs = gain / loss.replace(0, float("inf"))
                rsi = 100 - (100 / (1 + rs))
                if not rsi.empty and pd.notna(rsi.iloc[-1]):
                    entry["rsi_14"] = round(float(rsi.iloc[-1]), 1)

            snapshot[ticker] = entry

        return snapshot

    def _get_latest_price(
        self, ticker: str, market_data: dict[str, pd.DataFrame]
    ) -> float:
        """Get the most recent closing price for a ticker."""
        df = market_data.get(ticker)
        if df is None or df.empty or "Close" not in df.columns:
            return 0.0
        close = df["Close"].dropna()
        return float(close.iloc[-1]) if len(close) > 0 else 0.0

    def _get_risk_metrics(
        self, ticker: str, market_data: dict[str, pd.DataFrame]
    ) -> dict[str, Any]:
        """Compute basic risk metrics for a ticker."""
        df = market_data.get(ticker)
        if df is None or df.empty or "Close" not in df.columns:
            return {}

        close = df["Close"].dropna()
        if len(close) < 10:
            return {}

        returns = close.pct_change().dropna()
        metrics: dict[str, Any] = {}

        if len(returns) >= 14:
            metrics["volatility_14d"] = round(
                float(returns.iloc[-14:].std() * (252 ** 0.5)), 4
            )

        # Drawdown from peak
        peak = close.cummax()
        dd = (close - peak) / peak
        metrics["current_drawdown"] = round(float(dd.iloc[-1]), 4)

        return metrics

    def _get_trigger_reason(self, persona_id: str) -> str:
        """Retrieve the trigger reason for a pending persona."""
        # Check persisted state for trigger info
        state_data = self._load_state_raw()
        persona_state = state_data.get(persona_id, {})
        return persona_state.get("trigger_reason", "Event conditions detected")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_state(self) -> None:
        """Save persona states to disk."""
        data = {}
        for pid, p in self._personas.items():
            data[pid] = {
                "state": p.state.value,
                "activated_at": (
                    p.activated_at.isoformat() if p.activated_at else None
                ),
                "deactivated_at": (
                    p.deactivated_at.isoformat() if p.deactivated_at else None
                ),
                "last_brief_at": (
                    p.last_brief_at.isoformat() if p.last_brief_at else None
                ),
            }
        self._state_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    def _load_state(self) -> None:
        """Restore persona states from disk."""
        data = self._load_state_raw()
        if not data:
            return

        for pid, state_info in data.items():
            persona = self._personas.get(pid)
            if persona is None:
                continue
            try:
                persona.state = PersonaState(state_info.get("state", "dormant"))
            except ValueError:
                pass

            if state_info.get("activated_at"):
                persona.activated_at = datetime.fromisoformat(
                    state_info["activated_at"]
                )
            if state_info.get("deactivated_at"):
                persona.deactivated_at = datetime.fromisoformat(
                    state_info["deactivated_at"]
                )
            if state_info.get("last_brief_at"):
                persona.last_brief_at = datetime.fromisoformat(
                    state_info["last_brief_at"]
                )

        logger.info("Restored persona states from %s", self._state_path)

    def _load_state_raw(self) -> dict[str, Any]:
        """Load raw state JSON from disk."""
        if not self._state_path.exists():
            return {}
        try:
            return json.loads(self._state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _persist_brief(self, persona_id: str, brief: PersonaBrief) -> None:
        """Save a research brief to disk."""
        brief_dir = self._state_dir / persona_id
        brief_dir.mkdir(parents=True, exist_ok=True)
        brief_path = brief_dir / "brief_latest.json"
        brief_path.write_text(
            json.dumps(brief.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        logger.debug("Brief persisted to %s", brief_path)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _detector_tickers(params: dict[str, Any]) -> set[str]:
    """Extract any extra tickers referenced by detector params."""
    tickers: set[str] = set()
    for key in ("index_ticker", "vix_ticker", "watch_ticker"):
        val = params.get(key)
        if val:
            tickers.add(val)
    return tickers
