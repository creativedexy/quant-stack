"""Integration tests for the Quant Stack API.

These tests exercise every API endpoint with the full service stack
initialised against synthetic data.  They verify HTTP status codes,
response envelope structure, and that no endpoint raises an unhandled
exception.

Mark: all tests carry ``@pytest.mark.integration``.
Skip condition: set ``RUN_INTEGRATION=1`` in the environment to run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("RUN_INTEGRATION"),
        reason="Integration only -- set RUN_INTEGRATION=1",
    ),
]

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def _ok(response, status: int = 200) -> dict:
    """Assert envelope and return parsed JSON."""
    assert response.status_code == status, (
        f"Expected {status}, got {response.status_code}: {response.text}"
    )
    body = response.json()
    assert "data" in body, "Missing 'data' key in response"
    assert "timestamp" in body, "Missing 'timestamp' key in response"
    assert body["status"] == "ok"
    return body


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────


class TestHealthIntegration:
    def test_health_status_200(self) -> None:
        resp = client.get("/api/health")
        body = _ok(resp)
        data = body["data"]
        assert data["service"] == "quant-stack"
        assert "version" in data
        assert "uptime_seconds" in data
        assert isinstance(data["modules"], dict)

    def test_health_modules_present(self) -> None:
        resp = client.get("/api/health")
        modules = resp.json()["data"]["modules"]
        for mod in ("data", "features", "models"):
            assert mod in modules


# ─────────────────────────────────────────────
# Portfolio
# ─────────────────────────────────────────────


class TestPortfolioIntegration:
    def test_overview_keys(self) -> None:
        resp = client.get("/api/portfolio/overview")
        body = _ok(resp)
        for key in ("total_value", "sharpe_ratio", "max_drawdown_pct",
                     "num_positions", "currency"):
            assert key in body["data"], f"Missing key: {key}"

    def test_positions_list(self) -> None:
        resp = client.get("/api/portfolio/positions")
        body = _ok(resp)
        assert isinstance(body["data"], list)
        assert len(body["data"]) > 0
        pos = body["data"][0]
        for key in ("ticker", "quantity", "market_value", "weight_pct"):
            assert key in pos

    def test_equity_curve_default(self) -> None:
        resp = client.get("/api/portfolio/equity-curve")
        body = _ok(resp)
        assert isinstance(body["data"], list)
        assert len(body["data"]) > 0
        pt = body["data"][0]
        assert "date" in pt
        assert "value" in pt
        assert "drawdown_pct" in pt

    def test_equity_curve_3m_period(self) -> None:
        resp = client.get("/api/portfolio/equity-curve?period=3m")
        body = _ok(resp)
        # ~63 trading days, with slack
        assert len(body["data"]) <= 70

    def test_risk_metrics(self) -> None:
        resp = client.get("/api/portfolio/risk")
        body = _ok(resp)
        for key in ("var_95", "cvar_95", "max_drawdown_pct",
                     "volatility_annual"):
            assert key in body["data"]


# ─────────────────────────────────────────────
# Strategies
# ─────────────────────────────────────────────


class TestStrategiesIntegration:
    def test_list_strategies(self) -> None:
        resp = client.get("/api/strategies")
        body = _ok(resp)
        assert isinstance(body["data"], list)
        assert len(body["data"]) > 0
        strat = body["data"][0]
        for key in ("name", "description", "status"):
            assert key in strat

    def test_strategy_results_found(self) -> None:
        resp = client.get("/api/strategies/momentum_12_1/results")
        body = _ok(resp)
        data = body["data"]
        assert data["strategy"] == "momentum_12_1"
        assert "total_return_pct" in data
        assert "equity_curve" in data
        assert isinstance(data["equity_curve"], list)

    def test_strategy_not_found_404(self) -> None:
        resp = client.get("/api/strategies/does_not_exist/results")
        assert resp.status_code == 404

    def test_strategy_no_backtest_results_404(self) -> None:
        resp = client.get("/api/strategies/quality_value/results")
        assert resp.status_code == 404

    def test_compare_strategies(self) -> None:
        resp = client.get("/api/strategies/compare")
        body = _ok(resp)
        assert "strategies" in body["data"]
        strats = body["data"]["strategies"]
        assert isinstance(strats, list)
        for s in strats:
            assert s["total_return_pct"] is not None


# ─────────────────────────────────────────────
# Prices
# ─────────────────────────────────────────────


class TestPricesIntegration:
    def test_latest_prices(self) -> None:
        resp = client.get("/api/prices/latest")
        body = _ok(resp)
        assert isinstance(body["data"], list)
        assert len(body["data"]) > 0
        price = body["data"][0]
        for key in ("ticker", "price", "change_pct"):
            assert key in price

    def test_price_history(self) -> None:
        resp = client.get("/api/prices/TEST/history?start=2023-01-01")
        body = _ok(resp)
        data = body["data"]
        assert data["ticker"] == "TEST"
        assert isinstance(data["data"], list)
        assert len(data["data"]) > 0
        pt = data["data"][0]
        for key in ("date", "open", "close"):
            assert key in pt

    def test_price_features(self) -> None:
        resp = client.get("/api/prices/TEST/features?windows=1,5")
        body = _ok(resp)
        data = body["data"]
        assert data["ticker"] == "TEST"
        assert "ret_1d" in data["features"]
        assert "ret_5d" in data["features"]
        assert isinstance(data["data"], list)


# ─────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────


class TestExecutionIntegration:
    def test_status(self) -> None:
        resp = client.get("/api/execution/status")
        body = _ok(resp)
        data = body["data"]
        assert data["mode"] == "paper"
        assert "connected" in data

    def test_plan_valid_weights(self) -> None:
        resp = client.post(
            "/api/execution/plan",
            json={"target_weights": {"SHEL.L": 0.5, "HSBA.L": 0.5}},
        )
        body = _ok(resp)
        assert "orders" in body["data"]
        assert "estimated_commission" in body["data"]

    def test_plan_invalid_weights_422(self) -> None:
        resp = client.post(
            "/api/execution/plan",
            json={"target_weights": {"SHEL.L": 0.1, "HSBA.L": 0.1}},
        )
        assert resp.status_code == 422

    def test_execute_paper(self) -> None:
        order = {
            "ticker": "SHEL.L",
            "side": "buy",
            "quantity": 100,
            "estimated_cost": 5000,
            "current_weight_pct": 20,
            "target_weight_pct": 30,
        }
        resp = client.post(
            "/api/execution/execute",
            json={"orders": [order], "mode": "paper"},
        )
        body = _ok(resp)
        assert body["data"]["submitted"] == 1
        assert body["data"]["mode"] == "paper"
        assert len(body["data"]["order_ids"]) == 1

    def test_execute_live_rejected_403(self) -> None:
        order = {
            "ticker": "SHEL.L",
            "side": "buy",
            "quantity": 100,
            "estimated_cost": 5000,
            "current_weight_pct": 20,
            "target_weight_pct": 30,
        }
        resp = client.post(
            "/api/execution/execute",
            json={"orders": [order], "mode": "live"},
        )
        assert resp.status_code == 403

    def test_execution_history(self) -> None:
        resp = client.get("/api/execution/history")
        body = _ok(resp)
        assert isinstance(body["data"], list)
        if body["data"]:
            rec = body["data"][0]
            for key in ("order_id", "ticker", "status"):
                assert key in rec


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────


class TestPipelineIntegration:
    def test_status(self) -> None:
        resp = client.get("/api/pipeline/status")
        body = _ok(resp)
        data = body["data"]
        assert "running" in data
        assert "stages" in data
        assert isinstance(data["stages"], list)

    def test_run_all_stages(self) -> None:
        resp = client.post(
            "/api/pipeline/run", json={"source": "synthetic"}
        )
        body = _ok(resp)
        assert "job_id" in body["data"]
        assert "stages_queued" in body["data"]

    def test_run_specific_stages(self) -> None:
        resp = client.post(
            "/api/pipeline/run",
            json={"stages": ["data", "features"], "source": "synthetic"},
        )
        body = _ok(resp)
        assert body["data"]["stages_queued"] == ["data", "features"]

    def test_run_invalid_stage_422(self) -> None:
        resp = client.post(
            "/api/pipeline/run",
            json={"stages": ["nonexistent"], "source": "synthetic"},
        )
        assert resp.status_code == 422


# ─────────────────────────────────────────────
# WebSocket
# ─────────────────────────────────────────────


class TestWebSocketIntegration:
    def test_ws_prices_connect(self) -> None:
        with client.websocket_connect("/ws/prices") as ws:
            msg = ws.receive_json()
            assert "data" in msg
            assert msg["status"] == "ok"
            assert isinstance(msg["data"], list)
            assert len(msg["data"]) > 0
            tick = msg["data"][0]
            assert "ticker" in tick
            assert "price" in tick


# ─────────────────────────────────────────────
# Cross-cutting: 404 for unknown ticker
# ─────────────────────────────────────────────


class TestNotFound:
    def test_unknown_strategy_404(self) -> None:
        resp = client.get("/api/strategies/UNKNOWN_STRAT/results")
        assert resp.status_code == 404
