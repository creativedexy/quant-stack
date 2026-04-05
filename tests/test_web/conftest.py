"""Shared fixtures for web/UI tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.utils.config import load_config


@pytest.fixture()
def client() -> TestClient:
    """FastAPI test client with valid Basic Auth credentials."""
    cfg = load_config()
    username = cfg.get("ui", {}).get("auth", {}).get("username", "admin")
    password = cfg.get("ui", {}).get("auth", {}).get("password", "") or "changeme"
    return TestClient(app, headers=_basic_auth(username, password))


@pytest.fixture()
def anon_client() -> TestClient:
    """FastAPI test client with no credentials."""
    return TestClient(app)


def _basic_auth(username: str, password: str) -> dict[str, str]:
    """Return an Authorization header dict for HTTP Basic Auth."""
    import base64

    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}
