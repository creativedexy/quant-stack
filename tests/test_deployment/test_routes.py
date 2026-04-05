"""Tests for web/routes/deployment.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_service():
    """Mock DeploymentService for route tests."""
    svc = MagicMock()
    svc.get_cycle_status.return_value = {"state": "IDLE"}
    svc.get_memory_summary.return_value = {
        "total_cycles": 3,
        "last_cycle_id": "cycle_001",
        "watchlist": ["NVDA"],
        "excluded_tickers": ["XOM"],
        "cost_summary": {"estimated_gbp_total": 1.50},
        "learnings_count": 2,
    }
    svc.get_ib_buying_power.return_value = None
    svc.get_scored_candidates.return_value = []
    svc.get_trade_queue.return_value = []
    svc.get_queue_summary.return_value = {
        "total_orders": 0,
        "status_counts": {},
        "total_approved_gbp": 0.0,
    }
    svc.approve_order.return_value = True
    svc.reject_order.return_value = True
    svc.approve_all_orders.return_value = 0
    svc.confirm_and_submit.return_value = []
    svc.emergency_stop.return_value = {
        "state": "HALTED",
        "orders_cancelled": 0,
    }
    return svc


@pytest.fixture
def client(mock_service):
    """FastAPI test client with mocked deployment service."""
    with patch("web.routes.deployment._get_service", return_value=mock_service):
        from src.api.main import app
        with TestClient(app) as c:
            yield c


# Basic auth credentials for all requests.
AUTH = ("dex", "changeme")


class TestDashboardRoute:
    """Tests for the deployment dashboard."""

    def test_dashboard_loads(self, client):
        """Dashboard returns 200."""
        resp = client.get("/ui/execute/deployment", auth=AUTH)
        assert resp.status_code == 200

    def test_dashboard_contains_title(self, client):
        """Dashboard HTML contains page title."""
        resp = client.get("/ui/execute/deployment", auth=AUTH)
        assert "Capital Deployment" in resp.text

    def test_dashboard_shows_memory_stats(self, client):
        """Dashboard shows memory summary data."""
        resp = client.get("/ui/execute/deployment", auth=AUTH)
        # Total cycles from mock.
        assert "3" in resp.text


class TestCheckpointRoute:
    """Tests for the checkpoint review page."""

    def test_checkpoint_loads(self, client):
        """Checkpoint page returns 200."""
        resp = client.get("/ui/execute/deployment/checkpoint", auth=AUTH)
        assert resp.status_code == 200

    def test_checkpoint_contains_title(self, client):
        """Checkpoint page contains title."""
        resp = client.get("/ui/execute/deployment/checkpoint", auth=AUTH)
        assert "Checkpoint Review" in resp.text

    def test_approve_candidate(self, client):
        """Approving a candidate returns HTML fragment."""
        resp = client.post(
            "/ui/execute/deployment/checkpoint/approve",
            data={"ticker": "NVDA"},
            auth=AUTH,
        )
        assert resp.status_code == 200
        assert "APPROVED" in resp.text

    def test_reject_candidate(self, client):
        """Rejecting a candidate returns HTML fragment."""
        resp = client.post(
            "/ui/execute/deployment/checkpoint/reject",
            data={"ticker": "NVDA"},
            auth=AUTH,
        )
        assert resp.status_code == 200
        assert "REJECTED" in resp.text


class TestQueueRoute:
    """Tests for the trade queue page."""

    def test_queue_loads(self, client):
        """Queue page returns 200."""
        resp = client.get("/ui/execute/deployment/queue", auth=AUTH)
        assert resp.status_code == 200

    def test_queue_contains_title(self, client):
        """Queue page contains title."""
        resp = client.get("/ui/execute/deployment/queue", auth=AUTH)
        assert "Trade Queue" in resp.text


class TestHaltRoute:
    """Tests for emergency halt."""

    def test_halt_returns_200(self, client):
        """Emergency halt returns 200."""
        resp = client.post("/ui/execute/deployment/halt", auth=AUTH)
        assert resp.status_code == 200
        assert "EMERGENCY HALT" in resp.text


class TestMemoryRoute:
    """Tests for the memory page."""

    def test_memory_loads(self, client, mock_service):
        """Memory page returns 200."""
        # Mock the _ensure_memory for sector trends.
        mock_memory = MagicMock()
        mock_memory.get_learnings.return_value = {"sector_trends": {}}
        mock_service._ensure_memory.return_value = mock_memory
        mock_service._ensure_config.return_value = {}

        resp = client.get("/ui/execute/deployment/memory", auth=AUTH)
        assert resp.status_code == 200

    def test_memory_contains_title(self, client, mock_service):
        """Memory page contains title."""
        mock_memory = MagicMock()
        mock_memory.get_learnings.return_value = {"sector_trends": {}}
        mock_service._ensure_memory.return_value = mock_memory
        mock_service._ensure_config.return_value = {}

        resp = client.get("/ui/execute/deployment/memory", auth=AUTH)
        assert "Deployment Memory" in resp.text


class TestStatusRoute:
    """Tests for the HTMX status polling endpoint."""

    def test_status_returns_fragment(self, client):
        """Status endpoint returns HTML fragment."""
        resp = client.get("/ui/execute/deployment/status", auth=AUTH)
        assert resp.status_code == 200
        assert "Pipeline Status" in resp.text
