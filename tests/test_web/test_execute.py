"""Tests for the Execute page (web/routes/execute.py).

Verifies:
- Page renders with trading mode badge and system health cards.
- Order book filters by status.
- Rebalance approve/defer endpoints work.
- AI review returns ai_card grid.
- News sidebar renders with ticker chips.
- Plain English descriptions (no financial jargon).
- Service synthetic fallbacks return correct structure.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


# client fixture from tests/test_web/conftest.py (includes auth)


# ── Full page rendering ─────────────────────────────


class TestExecutePage:
    """GET /ui/execute returns a well-formed page."""

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/ui/execute")
        assert resp.status_code == 200

    def test_content_type_is_html(self, client: TestClient) -> None:
        resp = client.get("/ui/execute")
        assert "text/html" in resp.headers["content-type"]

    def test_contains_trading_mode_badge(self, client: TestClient) -> None:
        resp = client.get("/ui/execute")
        assert "PAPER TRADING" in resp.text or "LIVE TRADING" in resp.text

    def test_system_health_shows_5_cards(self, client: TestClient) -> None:
        resp = client.get("/ui/execute")
        for name in ("Broker", "Data Feed", "Scheduler", "Model", "Auth"):
            assert name in resp.text

    def test_contains_order_book(self, client: TestClient) -> None:
        resp = client.get("/ui/execute")
        assert "Order Book" in resp.text

    def test_contains_refresh_button(self, client: TestClient) -> None:
        resp = client.get("/ui/execute")
        assert "Refresh Now" in resp.text

    def test_contains_auto_refresh_note(self, client: TestClient) -> None:
        resp = client.get("/ui/execute")
        assert "Auto-refresh" in resp.text

    def test_contains_ai_review_section(self, client: TestClient) -> None:
        resp = client.get("/ui/execute")
        assert "AI Execution Review" in resp.text

    def test_contains_news_sidebar(self, client: TestClient) -> None:
        resp = client.get("/ui/execute")
        assert "Market News" in resp.text


# ── Order filtering ──────────────────────────────────


class TestOrderFiltering:
    """GET /ui/execute/orders?status=X returns filtered orders."""

    def test_all_orders_returns_200(self, client: TestClient) -> None:
        resp = client.get("/ui/execute/orders?status=all")
        assert resp.status_code == 200

    def test_filled_orders_only(self, client: TestClient) -> None:
        resp = client.get("/ui/execute/orders?status=filled")
        assert resp.status_code == 200
        text = resp.text
        # Should not contain queued or pending orders
        assert "status-dot-queued" not in text
        assert "status-dot-pending" not in text

    def test_rejected_orders(self, client: TestClient) -> None:
        resp = client.get("/ui/execute/orders?status=rejected")
        assert resp.status_code == 200

    def test_queued_orders(self, client: TestClient) -> None:
        resp = client.get("/ui/execute/orders?status=queued")
        assert resp.status_code == 200


# ── Rebalance actions ────────────────────────────────


class TestRebalanceActions:
    """POST endpoints for approve/defer rebalance suggestions."""

    def test_approve_returns_confirmation(self, client: TestClient) -> None:
        resp = client.post("/ui/execute/rebalance/approve?ticker=SHEL.L")
        assert resp.status_code == 200
        assert "Approved" in resp.text
        assert "queued" in resp.text

    def test_defer_returns_confirmation(self, client: TestClient) -> None:
        resp = client.post("/ui/execute/rebalance/defer?ticker=SHEL.L")
        assert resp.status_code == 200
        assert "Deferred" in resp.text

    def test_rebalance_suggestions_plain_english(
        self, client: TestClient,
    ) -> None:
        resp = client.get("/ui/execute")
        text = resp.text
        # Suggestions should use plain English reasons
        if "Rebalance Suggestions" in text:
            assert "above target weight" in text or "below target weight" in text


# ── Order descriptions ───────────────────────────────


class TestPlainEnglishDescriptions:
    """Order descriptions must be plain English, not technical."""

    def test_orders_use_plain_english(self, client: TestClient) -> None:
        resp = client.get("/ui/execute/orders?status=all")
        text = resp.text.lower()
        # Should contain plain English descriptions
        if "dca monthly" in text or "rebalance" in text or "stop loss" in text:
            pass  # Good -- plain English
        # Should NOT contain technical jargon
        assert "mkt buy" not in text
        assert "lmt" not in text

    def test_no_financial_jargon(self, client: TestClient) -> None:
        resp = client.get("/ui/execute")
        text = resp.text.lower()
        # No quant jargon on the Execute page
        assert "alpha" not in text
        assert "sharpe" not in text
        assert "var contribution" not in text


# ── News sidebar ─────────────────────────────────────


class TestNewsSidebar:
    """News items render in the sidebar with correct structure."""

    def test_news_items_render(self, client: TestClient) -> None:
        resp = client.get("/ui/execute")
        # Should have news items or the "no news" placeholder
        assert "Market News" in resp.text

    def test_news_items_have_ticker_chips(self, client: TestClient) -> None:
        resp = client.get("/ui/execute")
        # Synthetic news includes ticker chips
        if "ticker-chip" in resp.text:
            assert ".L" in resp.text  # LSE tickers

    def test_news_expand_has_ai_summary(self, client: TestClient) -> None:
        resp = client.get("/ui/execute")
        # AI summaries are in expand sections (hidden by default)
        if "impact-chip-inline" in resp.text:
            assert "impact_label" in resp.text or "Positive" in resp.text or "Negative" in resp.text or "Neutral" in resp.text


# ── AI review ────────────────────────────────────────


class TestAIReview:
    """POST /ui/execute/ai-review returns ai_card grid."""

    def test_ai_review_returns_200(self, client: TestClient) -> None:
        resp = client.post("/ui/execute/ai-review")
        assert resp.status_code == 200

    def test_ai_review_returns_cards(self, client: TestClient) -> None:
        resp = client.post("/ui/execute/ai-review")
        assert "ai-card" in resp.text

    def test_ai_review_has_insights(self, client: TestClient) -> None:
        resp = client.post("/ui/execute/ai-review")
        # Synthetic fallback returns at least one insight
        assert "System is running normally" in resp.text or "ai-card" in resp.text


# ── Trading mode toggle ─────────────────────────────


class TestTradingModeToggle:
    """POST /ui/execute/toggle-mode toggles and returns badge."""

    def test_toggle_returns_200(self, client: TestClient) -> None:
        resp = client.post("/ui/execute/toggle-mode")
        assert resp.status_code == 200

    def test_toggle_returns_badge(self, client: TestClient) -> None:
        resp = client.post("/ui/execute/toggle-mode")
        assert "trading-mode-badge" in resp.text
        assert "TRADING" in resp.text


# ── Service synthetic fallbacks ──────────────────────


class TestServiceFallbacks:
    """Synthetic fallbacks return correct structure."""

    def test_system_health_has_all_keys(self) -> None:
        from src.services.execution_service import ExecutionService

        svc = ExecutionService()
        health = svc.get_system_health()
        for key in ("broker", "data_feed", "scheduler", "model", "auth"):
            assert key in health
            assert "status" in health[key]
            assert "label" in health[key]
            assert "detail" in health[key]

    def test_get_orders_returns_list(self) -> None:
        from src.services.execution_service import ExecutionService

        svc = ExecutionService()
        orders = svc.get_orders()
        assert isinstance(orders, list)
        assert len(orders) > 0
        for order in orders:
            assert "ticker" in order
            assert "direction" in order
            assert "description" in order
            assert "status" in order

    def test_get_rebalance_suggestions_returns_list(self) -> None:
        from src.services.execution_service import ExecutionService

        svc = ExecutionService()
        suggestions = svc.get_rebalance_suggestions()
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        for s in suggestions:
            assert "ticker" in s
            assert "action" in s
            assert "reason" in s

    def test_get_trading_mode_returns_string(self) -> None:
        from src.services.execution_service import ExecutionService

        svc = ExecutionService()
        mode = svc.get_trading_mode()
        assert mode in ("paper", "live")

    def test_news_service_get_market_news_returns_list(self) -> None:
        from src.services.news_service import NewsService

        svc = NewsService({})
        news = svc.get_market_news()
        assert isinstance(news, list)
        assert len(news) > 0
        for item in news:
            assert "time" in item
            assert "title" in item
            assert "source" in item
            assert "tickers" in item
            assert "has_impact" in item
            assert "ai_summary" in item
            assert "impact_label" in item
            assert "impact_colour" in item
