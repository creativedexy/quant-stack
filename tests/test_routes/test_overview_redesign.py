"""Tests for the overview page redesign.

Covers: overview route, guide chat, watchlist CRUD, tab statuses,
and wl_row partial rendering.
"""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.utils.config import load_config
from web.routes.utils import get_tab_statuses


@pytest.fixture()
def client() -> TestClient:
    """FastAPI test client with valid Basic Auth credentials."""
    cfg = load_config()
    username = cfg.get("ui", {}).get("auth", {}).get("username", "admin")
    password = cfg.get("ui", {}).get("auth", {}).get("password", "") or "changeme"
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return TestClient(app, headers={"Authorization": f"Basic {token}"})


class TestOverviewPage:
    """GET /ui/overview tests."""

    def test_overview_returns_200(self, client: TestClient) -> None:
        resp = client.get("/ui/overview")
        assert resp.status_code == 200

    def test_overview_contains_chat_zone(self, client: TestClient) -> None:
        resp = client.get("/ui/overview")
        assert 'id="chatMessages"' in resp.text

    def test_overview_contains_nav_tiles(self, client: TestClient) -> None:
        resp = client.get("/ui/overview")
        assert "nav-tile" in resp.text

    def test_overview_contains_watchlist_rows(self, client: TestClient) -> None:
        resp = client.get("/ui/overview")
        assert "wl-row" in resp.text

    def test_overview_contains_news_items(self, client: TestClient) -> None:
        resp = client.get("/ui/overview")
        # Either news-snap-item or the "no news" fallback text
        assert "news-snap-item" in resp.text or "No news loaded yet" in resp.text


class TestWatchlistCRUD:
    """POST /ui/watchlist/remove, /refresh, /add tests."""

    def test_wl_remove_returns_empty(self, client: TestClient) -> None:
        resp = client.post("/ui/watchlist/remove/TEST.X")
        assert resp.status_code == 200
        assert resp.text == ""

    def test_wl_remove_updates_settings(self, client: TestClient) -> None:
        """After removing a ticker, it should not be in settings."""
        # Add a test ticker first, then remove it
        client.post("/ui/watchlist/add", data={"ticker": "TEST.REMOVE"})
        client.post("/ui/watchlist/remove/TEST.REMOVE")

        cfg = load_config()
        all_tickers: list[str] = []
        for g in cfg.get("ui", {}).get("watchlist", {}).get("groups", []):
            all_tickers.extend(g.get("tickers", []))
        assert "TEST.REMOVE" not in all_tickers

    def test_wl_add_appends_ticker(self, client: TestClient) -> None:
        resp = client.post("/ui/watchlist/add", data={"ticker": "LLOY.L"})
        assert resp.status_code == 200
        assert "LLOY" in resp.text
        # Clean up
        client.post("/ui/watchlist/remove/LLOY.L")

    def test_wl_add_duplicate_returns_409(self, client: TestClient) -> None:
        # Use a ticker we know is in the config
        cfg = load_config()
        groups = cfg.get("ui", {}).get("watchlist", {}).get("groups", [])
        if groups and groups[0].get("tickers"):
            existing = groups[0]["tickers"][0]
            resp = client.post("/ui/watchlist/add", data={"ticker": existing})
            assert resp.status_code == 409

    def test_wl_refresh_returns_row_fragment(self, client: TestClient) -> None:
        resp = client.post("/ui/watchlist/refresh/BARC.L")
        assert resp.status_code == 200
        assert "wl-row" in resp.text


class TestGuideChat:
    """POST /ui/guide/chat tests."""

    def test_guide_chat_returns_sse(self, client: TestClient) -> None:
        resp = client.post("/ui/guide/chat", data={"message": "hello"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_guide_chat_fallback_no_api_key(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        resp = client.post("/ui/guide/chat", data={"message": "hello"})
        assert resp.status_code == 200
        assert "data:" in resp.text


class TestTabStatuses:
    """get_tab_statuses helper tests."""

    def test_tab_statuses_helper(self) -> None:
        result = get_tab_statuses({"execution": {"mode": "paper"}}, signal_count=2)
        assert result["portfolio"] == "action"
        assert result["watchlist"] == "live"
        assert result["execution"] == "idle"

    def test_tab_statuses_live_execution(self) -> None:
        result = get_tab_statuses({"execution": {"mode": "live"}})
        assert result["execution"] == "live"

    def test_tab_statuses_news_with_unread(self) -> None:
        result = get_tab_statuses({}, unread_news=5)
        assert result["news"] == "live"


class TestWlRowPartial:
    """wl_row.html partial rendering tests."""

    def test_wl_row_partial_renders_data_id(self, client: TestClient) -> None:
        """The wl_row partial should contain a data-id attribute."""
        resp = client.post("/ui/watchlist/refresh/BARC.L")
        assert resp.status_code == 200
        assert "data-id" in resp.text
