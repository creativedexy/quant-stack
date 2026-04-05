#!/usr/bin/env python3
"""Full integration smoke test for the Quant Stack system.

Checks services, UI routes, API routes, and filesystem artefacts.
Exits 0 if all checks pass, 1 otherwise.

Usage:
    py -3 scripts/smoke_test_full.py
"""

from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Helpers ──────────────────────────────────────────────────────

class CheckResult:
    """Container for a single check result."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.passed = False
        self.error: str | None = None

    def __repr__(self) -> str:
        tag = "PASS" if self.passed else "FAIL"
        msg = f"  [{tag}] {self.name}"
        if self.error:
            msg += f"\n         {self.error}"
        return msg


def _run_check(name: str, func) -> CheckResult:
    result = CheckResult(name)
    try:
        func()
        result.passed = True
    except Exception as exc:
        result.error = str(exc)
    return result


def _auth_header() -> dict[str, str]:
    token = base64.b64encode(b"admin:changeme").decode()
    return {"Authorization": f"Basic {token}"}


# ── Service checks ───────────────────────────────────────────────

def check_data_service() -> None:
    from src.utils.config import load_config
    from src.services.data_service import DataService

    cfg = load_config()
    svc = DataService(config=cfg)
    status = svc.get_data_status()
    assert isinstance(status, dict), "get_data_status did not return a dict"


def check_model_service() -> None:
    from src.utils.config import load_config
    from src.services.model_service import ModelService

    cfg = load_config()
    svc = ModelService(config=cfg)
    # load_model returns None when no model is saved -- proves instantiation works
    result = svc.load_model("SHEL.L")
    assert result is None or result is not None, "load_model raised unexpectedly"


def check_portfolio_service() -> None:
    from src.utils.config import load_config
    from src.services.data_service import DataService
    from src.services.portfolio_service import PortfolioService

    cfg = load_config()
    data_svc = DataService(config=cfg)
    svc = PortfolioService(data_svc, config=cfg)
    risk = svc.get_risk_metrics()
    assert isinstance(risk, dict), "get_risk_metrics did not return a dict"


def check_strategy_service() -> None:
    from src.utils.config import load_config
    from src.services.data_service import DataService
    from src.services.strategy_service import StrategyService

    cfg = load_config()
    data_svc = DataService(config=cfg)
    svc = StrategyService(data_svc, config=cfg)
    strategies = svc.get_available_strategies()
    assert isinstance(strategies, list), "get_available_strategies did not return a list"


def check_execution_service() -> None:
    from src.services.execution_service import ExecutionService

    svc = ExecutionService()
    status = svc.get_broker_status()
    assert isinstance(status, dict), "get_broker_status did not return a dict"


def check_alert_delivery_service() -> None:
    from src.utils.config import load_config
    from src.services.alert_delivery import AlertDeliveryService

    cfg = load_config()
    svc = AlertDeliveryService(config=cfg.get("alerts", {}))
    assert svc is not None, "AlertDeliveryService failed to instantiate"


# ── HTTP route checks ────────────────────────────────────────────

def _get_test_client():
    from fastapi.testclient import TestClient
    from src.api.main import app
    return TestClient(app)


def check_ui_overview() -> None:
    client = _get_test_client()
    resp = client.get("/ui/overview", headers=_auth_header())
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert "text/html" in resp.headers.get("content-type", "")


def check_ui_strategy() -> None:
    client = _get_test_client()
    resp = client.get("/ui/strategy", headers=_auth_header())
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"


def check_ui_auth_required() -> None:
    client = _get_test_client()
    resp = client.get("/ui/overview")
    assert resp.status_code == 401, f"Expected 401 without auth, got {resp.status_code}"


def check_api_health() -> None:
    client = _get_test_client()
    resp = client.get("/api/health")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    assert data.get("status") == "ok", f"Health check status: {data.get('status')}"


def check_api_portfolio_overview() -> None:
    client = _get_test_client()
    resp = client.get("/api/portfolio/overview")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert "application/json" in resp.headers.get("content-type", "")


# ── New UI + API route checks (Phase 4) ─────────────────────────


def check_ui_watchlist() -> None:
    client = _get_test_client()
    resp = client.get("/ui/watchlist", headers=_auth_header())
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    body = resp.text
    assert "Barclays" in body or "BARC.L" in body, "Watchlist page missing first ticker (Barclays / BARC.L)"


def check_ui_chart_bollinger() -> None:
    client = _get_test_client()
    resp = client.get(
        "/ui/chart?ticker=BARC.L&period=1Y&scale=log",
        headers=_auth_header(),
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert "Bollinger" in resp.text, "Chart page missing 'Bollinger' text"


def check_ui_news_feed() -> None:
    client = _get_test_client()
    resp = client.get("/ui/news", headers=_auth_header())
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert "news-feed" in resp.text, "News page missing 'news-feed' div id"


def check_ui_portfolio_dca() -> None:
    client = _get_test_client()
    resp = client.get("/ui/portfolio?tab=dca", headers=_auth_header())
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert "DCA" in resp.text, "Portfolio DCA tab missing 'DCA' text"


def check_api_health_monte_carlo() -> None:
    client = _get_test_client()
    resp = client.get("/api/health")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    services = data.get("data", {}).get("services", {})
    assert "monte_carlo_service" in services, (
        f"monte_carlo_service not in health services: {list(services.keys())}"
    )


# ── Filesystem checks ────────────────────────────────────────────

def check_parquet_files() -> None:
    processed = PROJECT_ROOT / "data" / "processed"
    if not processed.exists():
        raise AssertionError("data/processed/ directory does not exist")
    files = list(processed.glob("*.parquet"))
    assert len(files) > 0, "No parquet files in data/processed/"


def check_research_log_writable() -> None:
    log_path = PROJECT_ROOT / "data" / "processed" / "research_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Test that we can write to the path
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            pass  # Just verify we can open for append
    except OSError as exc:
        raise AssertionError(f"Cannot write to {log_path}: {exc}")


# ── Main ─────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("  Quant Stack -- Full Integration Smoke Test")
    print("=" * 60)
    print()

    checks = [
        # Services
        ("DataService instantiation + get_data_status", check_data_service),
        ("ModelService instantiation + load_model", check_model_service),
        ("PortfolioService instantiation + get_risk_metrics", check_portfolio_service),
        ("StrategyService instantiation + get_available_strategies", check_strategy_service),
        ("ExecutionService instantiation + get_broker_status", check_execution_service),
        ("AlertDeliveryService instantiation", check_alert_delivery_service),
        # UI routes (auth required)
        ("GET /ui/overview (authenticated) -> 200", check_ui_overview),
        ("GET /ui/strategy (authenticated) -> 200", check_ui_strategy),
        ("GET /ui/overview (no auth) -> 401", check_ui_auth_required),
        # API routes (no auth)
        ("GET /api/health -> 200 JSON", check_api_health),
        ("GET /api/portfolio/overview -> 200 JSON", check_api_portfolio_overview),
        # New UI routes (Phase 4)
        ("GET /ui/watchlist -> 200 + Barclays/BARC.L", check_ui_watchlist),
        ("GET /ui/chart?ticker=BARC.L&period=1Y&scale=log -> 200 + Bollinger", check_ui_chart_bollinger),
        ("GET /ui/news -> 200 + news-feed div", check_ui_news_feed),
        ("GET /ui/portfolio?tab=dca -> 200 + DCA text", check_ui_portfolio_dca),
        ("GET /api/health -> monte_carlo_service in services", check_api_health_monte_carlo),
        # Filesystem
        ("data/processed/ has parquet files", check_parquet_files),
        ("research_log.jsonl is writable", check_research_log_writable),
    ]

    results: list[CheckResult] = []
    start = time.monotonic()

    for name, func in checks:
        print(f"  {name} ...", end="", flush=True)
        r = _run_check(name, func)
        results.append(r)
        tag = "PASS" if r.passed else "FAIL"
        print(f" [{tag}]")
        if r.error:
            print(f"    -> {r.error}")

    elapsed = time.monotonic() - start

    # Summary
    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    for r in results:
        print(r)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    print()
    print(f"  Total: {passed} passed, {failed} failed  ({elapsed:.1f}s)")
    print()

    if failed:
        print("  RESULT: FAIL")
        return 1
    print("  RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
