"""Tests for the /info endpoint."""

import pytest
from fastapi.testclient import TestClient

from app.main import app, state


@pytest.fixture(autouse=True)
def _clean_state():
    state.model = None
    yield
    state.model = None


@pytest.fixture
def client():
    return TestClient(app)


def test_info_returns_version(client):
    resp = client.get("/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == "2.0.0"
    assert "settings" in body


def test_info_settings_has_port(client):
    body = client.get("/info").json()
    assert body["settings"]["port"] == 8000


def test_info_settings_has_threshold(client):
    body = client.get("/info").json()
    assert body["settings"]["uncertainty_threshold"] == 0.10
