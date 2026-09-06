"""Tests for prediction endpoints, uncertainty handling and validation.

These tests use a lightweight in-memory model/vectorizer so they do not
depend on the real trained network or external data.
"""

from unittest import mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app, state
from app import preprocessing
from app.scraper import ExtractResult, ScrapeError


class _FakeVectorizer:
    """A tiny vectorizer with a fixed feature vocabulary."""

    def __init__(self):
        self.vocabulary_ = {
            "real": 0,
            "news": 1,
            "great": 2,
            "storm": 3,
            "terribl": 4,
            "disast": 5,
        }

    def transform(self, texts):
        rows = []
        for text in texts:
            words = text.split()
            row = [0] * len(self.vocabulary_)
            for word in words:
                if word in self.vocabulary_:
                    row[self.vocabulary_[word]] += 1
            rows.append(row)
        arr = np.array(rows)
        result = mock.Mock()
        result.toarray.return_value = arr
        return result

    def get_feature_names_out(self):
        sorted_vocab = sorted(self.vocabulary_.items(), key=lambda x: x[1])
        return np.array([k for k, _ in sorted_vocab])


class _FakeModel:
    """A model whose sigmoid output biases strongly toward one class."""

    def __init__(self, prob_real):
        self._prob_real = prob_real

    def predict(self, vector, verbose=0):
        return np.array([[self._prob_real]])


def _install_fake_service(model_prob=0.9):
    """Install a mock ModelService on app.state for testing."""
    from app.model import ModelService, Prediction

    fake_vectorizer = _FakeVectorizer()
    fake_model = _FakeModel(model_prob)

    service = mock.Mock(spec=ModelService)
    service.is_loaded = True
    service.model_is_loaded = True
    service.vectorizer_is_loaded = True
    service._model = fake_model
    service._vectorizer = fake_vectorizer

    # Replicate the real predict logic using our fake model/vectorizer
    def _real_predict(raw_text):
        cleaned = preprocessing.clean_single_text(raw_text)
        vector = fake_vectorizer.transform([cleaned]).toarray()
        prob_real = float(fake_model.predict(vector)[0][0])
        prob_real = max(0.0, min(1.0, prob_real))
        prob_fake = 1.0 - prob_real

        from app.config import settings
        winner = max(prob_real, prob_fake)
        if winner - 0.5 < settings.UNCERTAINTY_THRESHOLD:
            label = "uncertain"
            confidence = round(winner * 100.0, 2)
        else:
            label = "real" if prob_real >= prob_fake else "fake"
            confidence = round(winner * 100.0, 2)

        return Prediction(
            probability_real=prob_real,
            probability_fake=prob_fake,
            label=label,
            confidence=confidence,
            explanation=[],
        )

    service.predict.side_effect = _real_predict
    state.model = service


@pytest.fixture(autouse=True)
def _setup_and_teardown():
    """Ensure a clean state for every test.

    We avoid running the application's lifespan (which would load the real
    TensorFlow model); tests install a mock service on ``state.model`` instead.
    """
    state.model = None
    yield
    state.model = None


@pytest.fixture
def client():
    # Do not enter the context manager so the lifespan (real model load) is
    # not triggered; the tests install a mock ModelService on the shared state.
    return TestClient(app)


class TestHealth:
    def test_returns_ok(self, client):
        _install_fake_service()
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "model_loaded" in body
        assert "vectorizer_loaded" in body

    def test_shows_loaded_state(self, client):
        _install_fake_service()
        resp = client.get("/health")
        body = resp.json()
        assert body["model_loaded"] is True

    def test_shows_not_loaded_when_missing(self, client):
        state.model = None
        resp = client.get("/health")
        body = resp.json()
        assert body["model_loaded"] is False

    def test_reports_separate_loaded_flags(self, client):
        from app.model import ModelService
        fake_vectorizer = _FakeVectorizer()
        fake_model = _FakeModel(0.9)
        service = mock.Mock(spec=ModelService)
        service.is_loaded = False
        service.model_is_loaded = True
        service.vectorizer_is_loaded = False
        service._model = fake_model
        service._vectorizer = fake_vectorizer
        state.model = service
        body = client.get("/health").json()
        assert body["model_loaded"] is True
        assert body["vectorizer_loaded"] is False


class TestPredictValidation:
    def test_valid_article_returns_200(self, client):
        _install_fake_service()
        resp = client.post("/predict", json={"news": "This is a valid news article with enough words."})
        assert resp.status_code == 200

    def test_missing_news_returns_422(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_empty_string_returns_422(self, client):
        resp = client.post("/predict", json={"news": ""})
        assert resp.status_code == 422

    def test_whitespace_only_returns_422(self, client):
        resp = client.post("/predict", json={"news": "     "})
        assert resp.status_code == 422

    def test_near_empty_returns_422(self, client):
        resp = client.post("/predict", json={"news": "short"})
        assert resp.status_code == 422

    def test_null_news_returns_422(self, client):
        resp = client.post("/predict", json={"news": None})
        assert resp.status_code == 422

    def test_validation_detail_message(self, client):
        resp = client.post("/predict", json={"news": ""})
        assert resp.status_code == 422
        detail = resp.json()["detail"][0]
        assert "News text must contain enough content to analyze." in detail["msg"]

    def test_no_model_returns_503(self, client):
        resp = client.post("/predict", json={"news": "This has enough content for validation."})
        assert resp.status_code == 503


class TestPredictResponseSchema:
    def test_schema_is_correct(self, client):
        _install_fake_service(model_prob=0.9)
        resp = client.post("/predict", json={"news": "An ordinary article about politics and economy."})
        body = resp.json()
        assert "label" in body
        assert "confidence" in body
        assert "probability_real" in body
        assert "probability_fake" in body
        assert "explanation" in body
        assert "source_type" in body

    def test_numeric_probabilities_not_strings(self, client):
        _install_fake_service(model_prob=0.9)
        resp = client.post("/predict", json={"news": "Some article text for numeric checking today."})
        body = resp.json()
        assert isinstance(body["confidence"], (int, float))
        assert isinstance(body["probability_real"], (int, float))
        assert isinstance(body["probability_fake"], (int, float))
        assert not isinstance(body["confidence"], bool)

    def test_valid_verdict_label(self, client):
        _install_fake_service(model_prob=0.9)
        resp = client.post("/predict", json={"news": "This article discusses a positive development story."})
        body = resp.json()
        assert body["label"] in ("real", "fake", "uncertain")

    def test_probabilities_sum_to_100(self, client):
        _install_fake_service(model_prob=0.87)
        resp = client.post("/predict", json={"news": "The economy grew strongly this quarter across regions."})
        body = resp.json()
        total = body["probability_real"] + body["probability_fake"]
        assert abs(total - 100.0) < 0.02

    def test_source_type_is_text_for_predict(self, client):
        _install_fake_service()
        resp = client.post("/predict", json={"news": "This is an ordinary news article about something real."})
        body = resp.json()
        assert body["source_type"] == "text"

    def test_explanation_is_present(self, client):
        from app.model import Prediction, ExplanationItem
        _install_fake_service()
        state.model.predict = mock.MagicMock(
            return_value=Prediction(
                probability_real=0.9,
                probability_fake=0.1,
                label="real",
                confidence=90.0,
                explanation=[
                    ExplanationItem(word="govern", impact=1.2, direction="real"),
                    ExplanationItem(word="claim", impact=0.8, direction="fake"),
                ],
            )
        )
        resp = client.post("/predict", json={"news": "Some article about politics and government news today."})
        body = resp.json()
        words = body["explanation"]["top_influential_words"]
        assert len(words) == 2
        first = words[0]
        assert set(first) == {"word", "impact", "direction"}
        assert first["direction"] in ("real", "fake")


class TestUncertainty:
    def test_uncertain_when_probabilities_close(self, client):
        _install_fake_service(model_prob=0.53)
        resp = client.post("/predict", json={"news": "A somewhat balanced neutral article topic today."})
        body = resp.json()
        assert body["label"] == "uncertain"

    def test_confident_when_probabilities_far_from_50(self, client):
        _install_fake_service(model_prob=0.95)
        resp = client.post("/predict", json={"news": "This article is clearly biased and slanted story."})
        body = resp.json()
        assert body["label"] in ("real", "fake")

    def test_uncertain_at_boundary(self, client):
        """Test that prob 0.55 is uncertain with default threshold of 0.10."""
        _install_fake_service(model_prob=0.55)
        resp = client.post("/predict", json={"news": "Some moderate news content about economy today."})
        body = resp.json()
        # max(0.55, 0.45) = 0.55; 0.55 - 0.5 = 0.05 < 0.10
        assert body["label"] == "uncertain"


class TestPredictUrlEndpoint:
    def test_predict_url_uses_shared_pipeline(self, client):
        """Mocked scraping feeds text through the same prediction pipeline."""
        from app.model import Prediction
        _install_fake_service()
        state.model.predict = mock.MagicMock(
            return_value=Prediction(
                probability_real=0.9,
                probability_fake=0.1,
                label="real",
                confidence=90.0,
                explanation=[],
            )
        )
        with mock.patch(
            "app.main.fetch_article",
            return_value=ExtractResult(
                text="Article text from the url",
                title="Test headline",
                final_url="http://example.com/article",
            ),
        ) as fetcher:
            resp = client.post(
                "/predict-url", json={"url": "http://example.com/article"}
            )
        assert resp.status_code == 200
        fetcher.assert_called_once_with("http://example.com/article")
        body = resp.json()
        assert body["source_type"] == "url"
        assert body["source"] == "http://example.com/article"
        assert body["label"] == "real"
        assert body["page_title"] == "Test headline"

    def test_predict_url_returns_422_on_scrape_failure(self, client):
        _install_fake_service()
        with mock.patch(
            "app.main.fetch_article",
            side_effect=ScrapeError("The page content is too large to analyse."),
        ):
            resp = client.post(
                "/predict-url", json={"url": "http://example.com/big" }
            )
        assert resp.status_code == 422
        assert "too large" in resp.json()["detail"]

    def test_predict_url_requires_loaded_model(self, client):
        state.model = None
        resp = client.post("/predict-url", json={"url": "http://example.com/a"})
        assert resp.status_code == 503