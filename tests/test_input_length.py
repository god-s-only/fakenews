"""Tests for the backend input-length enforcement on /predict."""

from unittest import mock

import pytest
from fastapi.testclient import TestClient

from app.main import app, state
from app.model import ModelService
from app.prediction_log import prediction_log


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
    vocabulary_ = {"hello": 0}
    def transform(self, texts):
        import numpy as np
        result = mock.Mock()
        result.toarray.return_value = np.array([[1]])
        return result
    def get_feature_names_out(self):
        return ["hello"]


@pytest.fixture
def client():
    svc = ModelService(
        model_file=__import__("pathlib").Path("/tmp/m.h5"),
        vectorizer_file=__import__("pathlib").Path("/tmp/v.pkl"),
    )
    svc._model = _FakeModel()
    svc._vectorizer = _FakeVectorizer()
    state.model = svc
    return TestClient(app)


def test_short_input_accepted(client):
    resp = client.post("/predict", json={"news": "hello world test text here"})
    assert resp.status_code == 200


def test_empty_after_strip_rejected(client):
    resp = client.post("/predict", json={"news": "   "})
    assert resp.status_code == 422


def test_oversized_input_rejected(client):
    text = "a" * 25000
    resp = client.post("/predict", json={"news": text})
    assert resp.status_code == 422
    body = resp.json()
    detail_str = str(body["detail"]).lower()
    assert "at most 20000" in detail_str or "too long" in detail_str
