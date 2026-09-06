"""Basic application-level tests (routes, docs, static serving)."""

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


class TestIndexRoute:
    def test_index_serves_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    def test_static_css_served(self, client):
        resp = client.get("/static/style.css")
        assert resp.status_code == 200
        assert "text/css" in resp.headers.get("content-type", "")

    def test_static_js_served(self, client):
        resp = client.get("/static/script.js")
        assert resp.status_code == 200
        assert "javascript" in resp.headers.get("content-type", "")


class TestDocs:
    def test_openapi_docs_available(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_redoc_available(self, client):
        resp = client.get("/redoc")
        assert resp.status_code == 200

    def test_openapi_schema_has_predict(self, client):
        schema = client.get("/openapi.json").json()
        assert "/predict" in schema["paths"]
        assert "/predict-url" in schema["paths"]
        assert "/health" in schema["paths"]

    def test_predict_request_body_is_documented(self, client):
        schema = client.get("/openapi.json").json()
        predict = schema["paths"]["/predict"]["post"]
        assert "requestBody" in predict


class TestPredictLimits:
    def test_over_length_text_rejected(self, client):
        payload = {"news": "a" * 25_000}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422


class TestCors:
    def test_health_response_has_cors_headers(self, client):
        resp = client.get("/health", headers={"Origin": "http://localhost:3000"})
        # Default config uses "*" so the ACAO header should be returned.
        assert "access-control-allow-origin" in resp.headers


class TestExceptionHandling:
    def test_health_always_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
