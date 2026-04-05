"""Tests for HTTP Basic Auth on /ui/ routes.

Verifies:
- Unauthenticated requests to /ui/ return 401.
- Correct credentials grant access (200).
- Wrong credentials are rejected (401).
- JSON API routes (/api/) remain unprotected.
"""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.utils.config import load_config


def _auth_header(username: str, password: str) -> dict[str, str]:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def _configured_credentials() -> tuple[str, str]:
    """Read auth credentials from config/settings.yaml."""
    cfg = load_config()
    username = cfg.get("ui", {}).get("auth", {}).get("username", "admin")
    password = cfg.get("ui", {}).get("auth", {}).get("password", "") or "changeme"
    return username, password


@pytest.fixture()
def raw_client() -> TestClient:
    """Client with no default auth headers."""
    return TestClient(app)


class TestUIAuth:
    """HTTP Basic Auth protects all /ui/ routes."""

    def test_no_credentials_returns_401(self, raw_client: TestClient) -> None:
        resp = raw_client.get("/ui/overview")
        assert resp.status_code == 401
        assert resp.headers.get("www-authenticate") == 'Basic realm="Quant Stack"'

    def test_correct_credentials_returns_200(self, raw_client: TestClient) -> None:
        username, password = _configured_credentials()
        resp = raw_client.get("/ui/overview", headers=_auth_header(username, password))
        assert resp.status_code == 200

    def test_wrong_credentials_returns_401(self, raw_client: TestClient) -> None:
        username, _ = _configured_credentials()
        resp = raw_client.get("/ui/overview", headers=_auth_header(username, "wrong"))
        assert resp.status_code == 401

    def test_api_routes_unprotected(self, raw_client: TestClient) -> None:
        resp = raw_client.get("/api/health")
        assert resp.status_code == 200
