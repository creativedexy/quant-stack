"""Tests for all Quant Stack API endpoints.

Uses FastAPI's TestClient to verify each endpoint returns the correct
HTTP status and response envelope structure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def _assert_envelope(response, expected_status: int = 200) -> dict:
    """Assert the standard API response envelope and return parsed JSON."""
    assert response.status_code == expected_status, (
        f"Expected {expected_status}, got {response.status_code}: {response.text}"
    )
    body = response.json()
    assert "data" in body
    assert "timestamp" in body
    assert body["status"] == "ok"
    assert isinstance(body["timestamp"], str)
    return body


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────


class TestHealth:
    def test_health_returns_200(self) -> None:
        resp = client.get("/api/health")
        body = _assert_envelope(resp)
        assert body["data"]["service"] == "quant-stack"
        assert "version" in body["data"]
        assert "uptime_seconds" in body["data"]
        assert isinstance(body["data"]["modules"], dict)

    def test_health_modules_listed(self) -> None:
        resp = client.get("/api/health")
        body = resp.json()
        modules = body["data"]["modules"]
        assert "data" in modules
        assert modules["data"] == "available"


# ─────────────────────────────────────────────
# Portfolio
# ─────────────────────────────────────────────


class TestPortfolio:
    def test_overview(self) -> None:
        resp = client.get("/api/portfolio/overview")
        body = _assert_envelope(resp)
        data = body["data"]
        assert "total_value" in data
        assert "sharpe_ratio" in data
        assert "max_drawdown_pct" in data
        assert "num_positions" in data
        assert data["currency"] == "GBP"

    def test_positions(self) -> None:
        resp = client.get("/api/portfolio/positions")
        body = _assert_envelope(resp)
        assert isinstance(body["data"], list)
        assert len(body["data"]) > 0
        pos = body["data"][0]
        assert "ticker" in pos
        assert "quantity" in pos
        assert "market_value" in pos
        assert "weight_pct" in pos

    def test_equity_curve(self) -> None:
        resp = client.get("/api/portfolio/equity-curve")
        body = _assert_envelope(resp)
        assert isinstance(body["data"], list)
        assert len(body["data"]) > 0
        point = body["data"][0]
        assert "date" in point
        assert "value" in point
        assert "drawdown_pct" in point

    def test_equity_curve_period_param(self) -> None:
        resp = client.get("/api/portfolio/equity-curve?period=3m")
        body = _assert_envelope(resp)
        assert len(body["data"]) <= 63 + 5  # ~3 months of trading days, with some slack

    def test_risk_metrics(self) -> None:
        resp = client.get("/api/portfolio/risk")
        body = _assert_envelope(resp)
        data = body["data"]
        assert "var_95" in data
        assert "cvar_95" in data
        assert "max_drawdown_pct" in data
        assert "volatility_annual" in data


# ─────────────────────────────────────────────
# Strategies
# ─────────────────────────────────────────────


class TestStrategies:
    def test_list_strategies(self) -> None:
        resp = client.get("/api/strategies")
        body = _assert_envelope(resp)
        assert isinstance(body["data"], list)
        assert len(body["data"]) > 0
        strat = body["data"][0]
        assert "name" in strat
        assert "description" in strat
        assert "status" in strat

    def test_strategy_results_found(self) -> None:
        resp = client.get("/api/strategies/momentum_12_1/results")
        body = _assert_envelope(resp)
        data = body["data"]
        assert data["strategy"] == "momentum_12_1"
        assert "total_return_pct" in data
        assert "equity_curve" in data
        assert isinstance(data["equity_curve"], list)

    def test_strategy_results_not_found(self) -> None:
        resp = client.get("/api/strategies/nonexistent/results")
        assert resp.status_code == 404

    def test_strategy_no_results(self) -> None:
        resp = client.get("/api/strategies/quality_value/results")
        assert resp.status_code == 404

    def test_compare_strategies(self) -> None:
        resp = client.get("/api/strategies/compare")
        body = _assert_envelope(resp)
        assert "strategies" in body["data"]
        assert isinstance(body["data"]["strategies"], list)
        # Only strategies with results should appear
        for s in body["data"]["strategies"]:
            assert s["total_return_pct"] is not None


# ─────────────────────────────────────────────
# Prices
# ─────────────────────────────────────────────


class TestPrices:
    def test_latest_prices(self) -> None:
        resp = client.get("/api/prices/latest")
        body = _assert_envelope(resp)
        assert isinstance(body["data"], list)
        assert len(body["data"]) > 0
        price = body["data"][0]
        assert "ticker" in price
        assert "price" in price
        assert "change_pct" in price

    def test_price_history(self) -> None:
        resp = client.get("/api/prices/TEST/history?start=2023-01-01")
        body = _assert_envelope(resp)
        data = body["data"]
        assert data["ticker"] == "TEST"
        assert isinstance(data["data"], list)
        assert len(data["data"]) > 0
        point = data["data"][0]
        assert "date" in point
        assert "open" in point
        assert "close" in point

    def test_price_features(self) -> None:
        resp = client.get("/api/prices/TEST/features?windows=1,5")
        body = _assert_envelope(resp)
        data = body["data"]
        assert data["ticker"] == "TEST"
        assert "ret_1d" in data["features"]
        assert "ret_5d" in data["features"]
        assert isinstance(data["data"], list)


# ─────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────


class TestExecution:
    def test_status(self) -> None:
        resp = client.get("/api/execution/status")
        body = _assert_envelope(resp)
        data = body["data"]
        assert data["mode"] == "paper"
        assert "connected" in data

    def test_plan(self) -> None:
        resp = client.post(
            "/api/execution/plan",
            json={"target_weights": {"SHEL.L": 0.5, "HSBA.L": 0.5}},
        )
        body = _assert_envelope(resp)
        data = body["data"]
        assert "orders" in data
        assert "estimated_commission" in data

    def test_plan_invalid_weights(self) -> None:
        resp = client.post(
            "/api/execution/plan",
            json={"target_weights": {"SHEL.L": 0.1, "HSBA.L": 0.1}},
        )
        assert resp.status_code == 422

    def test_execute_paper(self) -> None:
        resp = client.post(
            "/api/execution/execute",
            json={
                "orders": [
                    {
                        "ticker": "SHEL.L",
                        "side": "buy",
                        "quantity": 100,
                        "estimated_cost": 5000,
                        "current_weight_pct": 20,
                        "target_weight_pct": 30,
                    }
                ],
                "mode": "paper",
            },
        )
        body = _assert_envelope(resp)
        data = body["data"]
        assert data["submitted"] == 1
        assert data["mode"] == "paper"
        assert len(data["order_ids"]) == 1

    def test_execute_live_rejected(self) -> None:
        resp = client.post(
            "/api/execution/execute",
            json={
                "orders": [
                    {
                        "ticker": "SHEL.L",
                        "side": "buy",
                        "quantity": 100,
                        "estimated_cost": 5000,
                        "current_weight_pct": 20,
                        "target_weight_pct": 30,
                    }
                ],
                "mode": "live",
            },
        )
        assert resp.status_code == 403

    def test_history(self) -> None:
        resp = client.get("/api/execution/history")
        body = _assert_envelope(resp)
        assert isinstance(body["data"], list)
        if body["data"]:
            record = body["data"][0]
            assert "order_id" in record
            assert "ticker" in record
            assert "status" in record


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────


class TestPipeline:
    def test_status(self) -> None:
        resp = client.get("/api/pipeline/status")
        body = _assert_envelope(resp)
        data = body["data"]
        assert "running" in data
        assert "stages" in data
        assert isinstance(data["stages"], list)

    def test_run(self) -> None:
        resp = client.post("/api/pipeline/run", json={"source": "synthetic"})
        body = _assert_envelope(resp)
        data = body["data"]
        assert "job_id" in data
        assert "stages_queued" in data

    def test_run_specific_stages(self) -> None:
        resp = client.post(
            "/api/pipeline/run",
            json={"stages": ["data", "features"], "source": "synthetic"},
        )
        body = _assert_envelope(resp)
        assert body["data"]["stages_queued"] == ["data", "features"]

    def test_run_invalid_stage(self) -> None:
        resp = client.post(
            "/api/pipeline/run",
            json={"stages": ["nonexistent"], "source": "synthetic"},
        )
        assert resp.status_code == 422


# ─────────────────────────────────────────────
# WebSocket
# ─────────────────────────────────────────────


class TestWebSocket:
    def test_ws_prices_connect_and_receive(self) -> None:
        with client.websocket_connect("/ws/prices") as ws:
            msg = ws.receive_json()
            assert "data" in msg
            assert msg["status"] == "ok"
            assert isinstance(msg["data"], list)
            assert len(msg["data"]) > 0
            tick = msg["data"][0]
            assert "ticker" in tick
            assert "price" in tick
