"""Tests for the GET /history endpoint."""

from unittest import mock

import pytest
from fastapi.testclient import TestClient

from app.main import app, state
from app.model import ModelService
from app.prediction_log import prediction_log, PredictionEntry


@pytest.fixture(autouse=True)
def _reset():
    state.model = None
    prediction_log.clear()
    yield
    state.model = None
    prediction_log.clear()


class _FakeModel:
    def predict(self, vector, verbose=0):
        import numpy as np
        return np.array([[0.9]])


class _FakeVectorizer:
    vocabulary_ = {"hello": 0, "world": 1}

    def transform(self, texts):
        import numpy as np
        result = mock.Mock()
        result.toarray.return_value = np.array([[1, 1]])
        return result

    def get_feature_names_out(self):
        import numpy as np
        return np.array(["hello", "world"])


@pytest.fixture
def client():
    svc = ModelService(model_file=__import__("pathlib").Path("/tmp/m.h5"),
                       vectorizer_file=__import__("pathlib").Path("/tmp/v.pkl"))
    svc._model = _FakeModel()
    svc._vectorizer = _FakeVectorizer()
    state.model = svc
    return TestClient(app)


def test_history_empty_by_default(client):
    resp = client.get("/history")
    assert resp.status_code == 200
    assert resp.json() == []


def test_history_after_prediction(client):
    client.post("/predict", json={"news": "hello world this is test text"})
    resp = client.get("/history")
    assert resp.status_code == 200
    entries = resp.json()
    assert len(entries) == 1
    assert entries[0]["source_type"] == "text"


def test_history_limit(client):
    for _ in range(5):
        client.post("/predict", json={"news": "hello world this is test text"})
    resp = client.get("/history?limit=3")
    assert len(resp.json()) == 3


def test_history_contains_fields(client):
    client.post("/predict", json={"news": "hello world this is test text"})
    entry = client.get("/history").json()[0]
    assert "label" in entry
    assert "confidence" in entry
    assert "probability_real" in entry
    assert "probability_fake" in entry
    assert "source_type" in entry
    assert "input_preview" in entry
