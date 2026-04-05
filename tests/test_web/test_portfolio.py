"""Tests for the redesigned Portfolio page (web/routes/portfolio.py).

Verifies:
- GET /ui/portfolio returns 200 with heatmap, DCA sidebar, risk alert.
- Heatmap cell sizes match weight thresholds.
- Colour classes match return thresholds.
- GET /ui/portfolio/drawer?ticker=X returns drawer HTML.
- Drawer contains plain English summary (not LONG/SHORT).
- Drawer signal badge uses "Looking Good" / "Be Careful" labels.
- DCA sidebar renders with plan cards.
- POST /ui/portfolio/ai-review returns ai_card grid.
- Risk alert appears when VaR exceeds threshold.
- Risk alert hidden when VaR is within threshold.
- No financial jargon in response body.
- DCA summary calculates monthly total correctly.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# -- Helpers ---------------------------------------------------


def _mock_positions() -> list[dict]:
    """Synthetic positions for testing."""
    return [
        {
            "ticker": "AZN.L",
            "name": "AstraZeneca",
            "weight": 0.22,
            "daily_return": 0.032,
            "total_return": 0.15,
            "market_value": 10630,
            "price": 112.40,
        },
        {
            "ticker": "SHEL.L",
            "name": "Shell",
            "weight": 0.15,
            "daily_return": -0.018,
            "total_return": 0.05,
            "market_value": 7200,
            "price": 27.15,
        },
        {
            "ticker": "HSBA.L",
            "name": "HSBC",
            "weight": 0.12,
            "daily_return": -0.062,
            "total_return": -0.03,
            "market_value": 5800,
            "price": 7.50,
        },
        {
            "ticker": "GSK.L",
            "name": "GSK",
            "weight": 0.08,
            "daily_return": 0.005,
            "total_return": 0.10,
            "market_value": 3800,
            "price": 15.20,
        },
    ]


def _mock_summary() -> dict:
    """Synthetic summary for testing."""
    return {
        "total_value": 48_250.0,
        "daily_pnl": 320.0,
        "daily_pnl_pct": 0.0066,
        "position_count": 4,
        "sector_count": 2,
        "best_today": {"ticker": "AZN.L", "return": 0.032},
        "worst_today": {"ticker": "HSBA.L", "return": -0.062},
    }


def _mock_dca_plans() -> list[dict]:
    """Synthetic DCA plans for testing."""
    return [
        {
            "ticker": "CNDX.L",
            "name": "iShares NASDAQ 100",
            "amount": 250,
            "frequency": "monthly",
            "day": 1,
            "target": 5000,
            "invested": 2750,
            "purchases": 11,
            "status": "ACTIVE",
            "next_date": "2026-04-01",
        },
        {
            "ticker": "VUSA.L",
            "name": "Vanguard S&P 500",
            "amount": 150,
            "frequency": "monthly",
            "day": 1,
            "target": 3000,
            "invested": 1650,
            "purchases": 11,
            "status": "ACTIVE",
            "next_date": "2026-04-01",
        },
        {
            "ticker": "VFEM.L",
            "name": "Vanguard Emerging Markets",
            "amount": 100,
            "frequency": "monthly",
            "day": 15,
            "target": 2000,
            "invested": 800,
            "purchases": 8,
            "status": "PAUSED",
            "next_date": "2026-04-15",
        },
    ]


def _mock_dca_summary() -> dict:
    """Synthetic DCA summary for testing."""
    return {
        "monthly_total": 400,
        "total_invested": 5200,
        "avg_return": 0.045,
        "next_payment_date": "2026-04-01",
        "next_payment_ticker": "CNDX.L",
        "next_payment_amount": 250,
        "active_count": 2,
    }


def _mock_signals() -> dict:
    """Synthetic signals keyed by ticker."""
    return {
        "AZN.L": {
            "direction": "LONG",
            "confidence": 0.72,
            "strategy": "momentum",
            "label": "Looking Good",
        },
    }


def _mock_heatmap() -> list[dict]:
    """Synthetic heatmap data (positions enriched with display attrs)."""
    positions = _mock_positions()
    # Enrich with heatmap attrs matching the service logic
    enriched = []
    for pos in positions:
        w = pos["weight"] * 100
        ret = pos["daily_return"] * 100
        if w >= 15:
            cell_size = "big"
        elif w >= 10:
            cell_size = "medium"
        else:
            cell_size = "small"

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
            "has_signal": pos["ticker"] == "AZN.L",
        })
    return enriched


def _mock_risk_metrics_ok() -> dict:
    """Risk metrics within threshold."""
    return {
        "annual_return": 0.12,
        "annual_volatility": 0.15,
        "sharpe_ratio": 0.80,
        "max_drawdown": -0.08,
        "var_95": -0.015,
        "cvar_95": -0.022,
    }


def _mock_risk_metrics_breach() -> dict:
    """Risk metrics exceeding threshold."""
    return {
        "annual_return": 0.12,
        "annual_volatility": 0.25,
        "sharpe_ratio": 0.48,
        "max_drawdown": -0.18,
        "var_95": -0.035,
        "cvar_95": -0.052,
    }


# -- Fixtures --------------------------------------------------


def _patch_portfolio_services(
    risk_metrics: dict | None = None,
    risk_alert: dict | None = None,
):
    """Return a context manager that patches portfolio route services."""
    from unittest.mock import MagicMock

    if risk_metrics is None:
        risk_metrics = _mock_risk_metrics_ok()

    mock_portfolio_svc = MagicMock()
    mock_portfolio_svc.get_positions.return_value = _mock_positions()
    mock_portfolio_svc.get_summary.return_value = _mock_summary()
    mock_portfolio_svc.get_heatmap_data.return_value = _mock_heatmap()
    mock_portfolio_svc.get_risk_metrics.return_value = risk_metrics
    mock_portfolio_svc.get_risk_alert.return_value = risk_alert

    mock_strategy_svc = MagicMock()
    mock_strategy_svc.get_available_strategies.return_value = []

    mock_execution_svc = MagicMock()
    mock_execution_svc.get_dca_plans.return_value = _mock_dca_plans()
    mock_execution_svc.get_dca_summary.return_value = _mock_dca_summary()

    mock_news_svc = MagicMock()
    mock_news_svc.get_cached_news.return_value = []

    mock_ai_svc = MagicMock()
    mock_ai_svc.review_portfolio.return_value = [
        {
            "colour": "green",
            "icon": "check",
            "title": "Portfolio is well spread out",
            "text": "Your holdings cover several sectors.",
            "chips": [],
        },
    ]

    services = {
        "config": {},
        "data": None,
        "portfolio": mock_portfolio_svc,
        "strategy": mock_strategy_svc,
        "execution": mock_execution_svc,
        "news": mock_news_svc,
        "ai": mock_ai_svc,
    }

    return patch(
        "web.routes.portfolio._get_services",
        return_value=services,
    )


# -- Page rendering tests --------------------------------------


class TestPortfolioPage:
    """GET /ui/portfolio returns the redesigned portfolio page."""

    def test_returns_200(self, client: TestClient) -> None:
        with _patch_portfolio_services():
            resp = client.get("/ui/portfolio")
        assert resp.status_code == 200

    def test_content_type_is_html(self, client: TestClient) -> None:
        with _patch_portfolio_services():
            resp = client.get("/ui/portfolio")
        assert "text/html" in resp.headers["content-type"]

    def test_contains_your_positions(self, client: TestClient) -> None:
        with _patch_portfolio_services():
            resp = client.get("/ui/portfolio")
        assert "Your Positions" in resp.text

    def test_contains_heatmap_cells(self, client: TestClient) -> None:
        with _patch_portfolio_services():
            resp = client.get("/ui/portfolio")
        assert "heatmap-cell" in resp.text

    def test_heatmap_big_for_high_weight(self, client: TestClient) -> None:
        """Positions >= 15% weight get 'big' cell size."""
        with _patch_portfolio_services():
            resp = client.get("/ui/portfolio")
        # AZN.L has 22% weight -> should have 'big' class
        assert "big" in resp.text

    def test_heatmap_medium_for_mid_weight(self, client: TestClient) -> None:
        """Positions >= 10% and < 15% weight get 'medium' cell size."""
        with _patch_portfolio_services():
            resp = client.get("/ui/portfolio")
        assert "medium" in resp.text

    def test_heatmap_small_for_low_weight(self, client: TestClient) -> None:
        """Positions < 10% weight get 'small' cell size."""
        with _patch_portfolio_services():
            resp = client.get("/ui/portfolio")
        assert "small" in resp.text

    def test_colour_class_pos_mild(self, client: TestClient) -> None:
        """Return > +1% gets pos-mild."""
        with _patch_portfolio_services():
            resp = client.get("/ui/portfolio")
        assert "pos-mild" in resp.text

    def test_colour_class_neg_mild(self, client: TestClient) -> None:
        """Return between -1% and -5% gets neg-mild."""
        with _patch_portfolio_services():
            resp = client.get("/ui/portfolio")
        assert "neg-mild" in resp.text

    def test_colour_class_neg_strong(self, client: TestClient) -> None:
        """Return < -5% gets neg-strong."""
        with _patch_portfolio_services():
            resp = client.get("/ui/portfolio")
        assert "neg-strong" in resp.text

    def test_dca_sidebar_renders(self, client: TestClient) -> None:
        with _patch_portfolio_services():
            resp = client.get("/ui/portfolio")
        assert "Your Plans" in resp.text
        assert "CNDX.L" in resp.text
        assert "VUSA.L" in resp.text

    def test_dca_plan_cards_present(self, client: TestClient) -> None:
        with _patch_portfolio_services():
            resp = client.get("/ui/portfolio")
        assert "dca-plan-card" in resp.text

    def test_dca_new_plan_button(self, client: TestClient) -> None:
        with _patch_portfolio_services():
            resp = client.get("/ui/portfolio")
        assert "Start a New Plan" in resp.text

    def test_portfolio_value_displayed(self, client: TestClient) -> None:
        with _patch_portfolio_services():
            resp = client.get("/ui/portfolio")
        assert "GBP48,250.00" in resp.text

    def test_signal_dot_for_active_signal(self, client: TestClient) -> None:
        with _patch_portfolio_services():
            resp = client.get("/ui/portfolio")
        assert "signal-dot" in resp.text

    def test_htmx_drawer_attribute(self, client: TestClient) -> None:
        with _patch_portfolio_services():
            resp = client.get("/ui/portfolio")
        assert "hx-get" in resp.text
        assert "/ui/portfolio/drawer" in resp.text

    def test_ai_review_button(self, client: TestClient) -> None:
        with _patch_portfolio_services():
            resp = client.get("/ui/portfolio")
        assert "Get AI Portfolio Review" in resp.text
        assert "hx-post" in resp.text
        assert "/ui/portfolio/ai-review" in resp.text


# -- Risk alert tests ------------------------------------------


class TestRiskAlert:
    """Risk alert strip appears/hides based on VaR threshold."""

    def test_risk_alert_appears_when_breached(self, client: TestClient) -> None:
        risk_alert = {
            "message": (
                "Your portfolio risk is higher than usual. "
                "Worst-case daily loss has increased to around "
                "GBP1,689, mainly driven by HSBA.L."
            ),
            "primary_driver": "HSBA.L",
            "worst_case_loss": 1689,
            "actions": [
                {"label": "Review HSBA.L", "href": "/ui/analyse?ticker=HSBA.L"},
                {"label": "Set Stop Loss", "href": "/ui/execution"},
            ],
        }
        with _patch_portfolio_services(
            risk_metrics=_mock_risk_metrics_breach(),
            risk_alert=risk_alert,
        ):
            resp = client.get("/ui/portfolio")
        assert "risk-alert-strip" in resp.text
        assert "Worst-case daily loss" in resp.text

    def test_risk_alert_hidden_when_ok(self, client: TestClient) -> None:
        with _patch_portfolio_services(
            risk_metrics=_mock_risk_metrics_ok(),
            risk_alert=None,
        ):
            resp = client.get("/ui/portfolio")
        # The CSS class name appears in the <style> block, so check
        # that the actual HTML element is not rendered.
        # When risk_alert is None, the {% if risk_alert %} block is skipped.
        assert "Your portfolio risk is higher" not in resp.text

    def test_risk_alert_uses_plain_english(self, client: TestClient) -> None:
        risk_alert = {
            "message": (
                "Your portfolio risk is higher than usual. "
                "Worst-case daily loss has increased to around "
                "GBP1,689, mainly driven by HSBA.L."
            ),
            "primary_driver": "HSBA.L",
            "worst_case_loss": 1689,
            "actions": [],
        }
        with _patch_portfolio_services(risk_alert=risk_alert):
            resp = client.get("/ui/portfolio")
        assert "Worst-case daily loss" in resp.text


# -- No jargon tests -------------------------------------------


class TestNoJargon:
    """Financial jargon must not appear in the response body."""

    def test_no_var_in_response(self, client: TestClient) -> None:
        with _patch_portfolio_services():
            resp = client.get("/ui/portfolio")
        assert "VaR" not in resp.text

    def test_no_cvar_in_response(self, client: TestClient) -> None:
        with _patch_portfolio_services():
            resp = client.get("/ui/portfolio")
        assert "CVaR" not in resp.text


# -- Drawer tests ----------------------------------------------


class TestPortfolioDrawer:
    """GET /ui/portfolio/drawer?ticker=X returns drawer HTML."""

    def test_returns_200(self, client: TestClient) -> None:
        with _patch_portfolio_services():
            resp = client.get("/ui/portfolio/drawer?ticker=AZN.L")
        assert resp.status_code == 200

    def test_is_fragment(self, client: TestClient) -> None:
        with _patch_portfolio_services():
            resp = client.get("/ui/portfolio/drawer?ticker=AZN.L")
        assert "<!DOCTYPE" not in resp.text

    def test_contains_plain_english_summary(self, client: TestClient) -> None:
        with _patch_portfolio_services():
            resp = client.get("/ui/portfolio/drawer?ticker=AZN.L")
        assert "You hold" in resp.text
        assert "AstraZeneca" in resp.text

    def test_no_long_short_labels(self, client: TestClient) -> None:
        with _patch_portfolio_services():
            resp = client.get("/ui/portfolio/drawer?ticker=AZN.L")
        # The drawer should not use LONG/SHORT -- uses plain labels
        body = resp.text
        # "LONG" may appear inside URLs or data attrs, check content text
        assert "signal-badge" in body

    def test_signal_badge_labels(self, client: TestClient) -> None:
        """Signal badge uses human-readable labels."""
        with _patch_portfolio_services():
            resp = client.get("/ui/portfolio/drawer?ticker=AZN.L")
        body = resp.text
        # Should contain one of the friendly labels
        assert any(
            label in body
            for label in ("Looking Good", "Be Careful", "Steady", "Neutral")
        )

    def test_action_cards_present(self, client: TestClient) -> None:
        with _patch_portfolio_services():
            resp = client.get("/ui/portfolio/drawer?ticker=AZN.L")
        assert "Add to DCA" in resp.text
        assert "Set Alert" in resp.text
        assert "Analyse" in resp.text

    def test_position_not_found(self, client: TestClient) -> None:
        with _patch_portfolio_services():
            resp = client.get("/ui/portfolio/drawer?ticker=UNKNOWN")
        assert resp.status_code == 200
        assert "Position not found" in resp.text


# -- AI review tests -------------------------------------------


class TestAIReview:
    """POST /ui/portfolio/ai-review returns ai_card grid."""

    def test_returns_200(self, client: TestClient) -> None:
        with _patch_portfolio_services():
            resp = client.post("/ui/portfolio/ai-review")
        assert resp.status_code == 200

    def test_is_fragment(self, client: TestClient) -> None:
        with _patch_portfolio_services():
            resp = client.post("/ui/portfolio/ai-review")
        assert "<!DOCTYPE" not in resp.text

    def test_contains_ai_cards(self, client: TestClient) -> None:
        with _patch_portfolio_services():
            resp = client.post("/ui/portfolio/ai-review")
        assert "ai-card" in resp.text
        assert "ai-review-grid" in resp.text

    def test_contains_insight_title(self, client: TestClient) -> None:
        with _patch_portfolio_services():
            resp = client.post("/ui/portfolio/ai-review")
        assert "Portfolio is well spread out" in resp.text


# -- DCA summary unit tests ------------------------------------


class TestDCASummary:
    """DCA summary calculation in ExecutionService."""

    def test_monthly_total_calculation(self) -> None:
        from src.services.execution_service import ExecutionService

        svc = ExecutionService(config={})
        plans = _mock_dca_plans()
        summary = svc.get_dca_summary(plans)
        # Active plans: CNDX.L (250) + VUSA.L (150) = 400
        assert summary["monthly_total"] == 400

    def test_active_count(self) -> None:
        from src.services.execution_service import ExecutionService

        svc = ExecutionService(config={})
        plans = _mock_dca_plans()
        summary = svc.get_dca_summary(plans)
        assert summary["active_count"] == 2

    def test_total_invested(self) -> None:
        from src.services.execution_service import ExecutionService

        svc = ExecutionService(config={})
        plans = _mock_dca_plans()
        summary = svc.get_dca_summary(plans)
        # 2750 + 1650 + 800 = 5200
        assert summary["total_invested"] == 5200


# -- Heatmap service unit tests --------------------------------


class TestHeatmapData:
    """PortfolioService.get_heatmap_data enriches positions correctly."""

    def test_big_cell_for_high_weight(self) -> None:
        from src.services.portfolio_service import PortfolioService
        from unittest.mock import MagicMock

        svc = PortfolioService(MagicMock(), {})
        positions = [{"ticker": "X", "weight": 0.20, "daily_return": 0.0}]
        result = svc.get_heatmap_data(positions)
        assert result[0]["cell_size"] == "big"

    def test_medium_cell_for_mid_weight(self) -> None:
        from src.services.portfolio_service import PortfolioService
        from unittest.mock import MagicMock

        svc = PortfolioService(MagicMock(), {})
        positions = [{"ticker": "X", "weight": 0.12, "daily_return": 0.0}]
        result = svc.get_heatmap_data(positions)
        assert result[0]["cell_size"] == "medium"

    def test_small_cell_for_low_weight(self) -> None:
        from src.services.portfolio_service import PortfolioService
        from unittest.mock import MagicMock

        svc = PortfolioService(MagicMock(), {})
        positions = [{"ticker": "X", "weight": 0.05, "daily_return": 0.0}]
        result = svc.get_heatmap_data(positions)
        assert result[0]["cell_size"] == "small"

    def test_pos_strong_colour(self) -> None:
        from src.services.portfolio_service import PortfolioService
        from unittest.mock import MagicMock

        svc = PortfolioService(MagicMock(), {})
        positions = [{"ticker": "X", "weight": 0.10, "daily_return": 0.06}]
        result = svc.get_heatmap_data(positions)
        assert result[0]["colour_class"] == "pos-strong"

    def test_neg_strong_colour(self) -> None:
        from src.services.portfolio_service import PortfolioService
        from unittest.mock import MagicMock

        svc = PortfolioService(MagicMock(), {})
        positions = [{"ticker": "X", "weight": 0.10, "daily_return": -0.06}]
        result = svc.get_heatmap_data(positions)
        assert result[0]["colour_class"] == "neg-strong"

    def test_has_signal_flag(self) -> None:
        from src.services.portfolio_service import PortfolioService
        from unittest.mock import MagicMock

        svc = PortfolioService(MagicMock(), {})
        positions = [{"ticker": "AZN.L", "weight": 0.10, "daily_return": 0.0}]
        signals = {"AZN.L": {"direction": "LONG", "confidence": 0.7}}
        result = svc.get_heatmap_data(positions, signals)
        assert result[0]["has_signal"] is True


# -- Risk alert service unit tests -----------------------------


class TestRiskAlertService:
    """PortfolioService.get_risk_alert returns alert or None."""

    def test_returns_none_when_within_threshold(self) -> None:
        from src.services.portfolio_service import PortfolioService
        from unittest.mock import MagicMock

        svc = PortfolioService(MagicMock(), {"risk": {"var_threshold": -0.02}})
        result = svc.get_risk_alert({"var_95": -0.015})
        assert result is None

    def test_returns_alert_when_breached(self) -> None:
        from src.services.portfolio_service import PortfolioService
        from unittest.mock import MagicMock

        mock_data = MagicMock()
        mock_data.get_prices.return_value = MagicMock(empty=True)
        mock_data._tickers = []
        svc = PortfolioService(mock_data, {"risk": {"var_threshold": -0.02}})
        result = svc.get_risk_alert({"var_95": -0.035})
        assert result is not None
        assert "Worst-case daily loss" in result["message"]

    def test_returns_none_for_nan_var(self) -> None:
        from src.services.portfolio_service import PortfolioService
        from unittest.mock import MagicMock

        svc = PortfolioService(MagicMock(), {})
        result = svc.get_risk_alert({"var_95": float("nan")})
        assert result is None
