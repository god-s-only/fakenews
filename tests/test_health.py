"""Tests for the liveness/readiness endpoints (audit B6 separation)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app


@pytest.fixture(autouse=True)
def _no_model():
    main_module.state.model = None
    yield
    main_module.state.model = None


class TestLive:
    def test_liveness_is_ok_without_model(self):
        client = TestClient(app)
        resp = client.get("/health/live")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_liveness_is_ok_even_with_no_model_loaded(self):
        assert main_module.state.model is None
        client = TestClient(app)
        assert client.get("/health/live").status_code == 200


class TestReady:
    def test_not_ready_without_model(self):
        client = TestClient(app)
        resp = client.get("/health/ready")
        assert resp.status_code == 503
        assert "not ready" in resp.json()["detail"]

    def test_not_ready_when_loaded_but_undescribed(self):
        class _LoadedOnly:
            model_is_loaded = True
            vectorizer_is_loaded = True
            is_loaded = True
            _backend = "sklearn"
            model_ready = False

        main_module.state.model = _LoadedOnly()
        client = TestClient(app)
        resp = client.get("/health/ready")
        assert resp.status_code == 503

    def test_ready_when_model_ready(self):
        class _Ready:
            model_is_loaded = True
            vectorizer_is_loaded = True
            is_loaded = True
            _backend = "sklearn"
            model_ready = True
            model_file = type("F", (), {"name": "my_model_lr.pkl"})()
            vectorizer_file = type("F", (), {"name": "my_tfidf_vectorizer.pkl"})()
            model_sha256 = "aa"
            vectorizer_sha256 = "bb"
            vocab_size = 36_862

        main_module.state.model = _Ready()
        client = TestClient(app)
        resp = client.get("/health/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"
        assert body["model_ready"] is True
        assert body["model_backend"] == "sklearn"
        assert body["vocab_size"] == 36_862
        assert body["model_sha256"] == "aa"

    def test_ready_uses_real_model_facts(self):
        # With no lifespan the module state has no loaded real model.
        client = TestClient(app)
        assert client.get("/health/ready").status_code == 503