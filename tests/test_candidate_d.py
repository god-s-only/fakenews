"""Regression tests for the promoted Candidate D detector (TF-IDF + LR).

Locks the frozen model's behaviour on stable, prominent inputs:
  * Reuters-style REAL article
  * known FAKE-style article
  * ambiguous (near-50%) article
  * non-Reuters REAL writing
  * empty / near-empty input
  * URL extraction, 404 URL, homepage/non-article URL
  * SSRF / private-network protection
  * API probability/label parity with the offline evaluator
  * explanation output (influential features, not proof)
  * backend selection (sklearn) and deterministic loading

Skips automatically when the promoted artifacts are not present.
"""

import csv
import pickle
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from app import preprocessing
from app.config import settings
from app.main import app
from app.model import ModelService
from app.scraper import ExtractResult, ScrapeError


def _promoted_available() -> bool:
    return settings.model_file.exists() and settings.vectorizer_file.exists()


pytestmark = pytest.mark.skipif(
    not _promoted_available(),
    reason="Promoted model/vectorizer artifacts are not available",
)

REUTERS_REAL = (
    "WASHINGTON (Reuters) - The head of a conservative Republican faction in "
    "the U.S. Congress said on Thursday that party leaders were working to "
    "resolve differences over the fiscal measure."
)
FAKE_STYLE = (
    "Donald Trump just couldn't wish all Americans a Happy New Year and leave "
    "it at that. Instead he turned the holiday into another one of his "
    "signature rants."
)
UNCERTAIN_TEXT = "Officials announced the hospital wing will open next month."
NEUTRAL_REAL = (
    "Forecasters say the storm system will move inland by the weekend and "
    "bring heavy rain to the coastal cities before weakening."
)


def _offline_probability(text: str) -> float:
    with open(settings.model_file, "rb") as fh:
        model = pickle.load(fh)
    with open(settings.vectorizer_file, "rb") as fh:
        vec = pickle.load(fh)
    # Mirror the production inference path exactly: the Phase 9 source-marker
    # normaliser runs before cleaning (it strips datelines/bylines/meta).
    normalized = preprocessing.normalize_news_markers(text) or text
    return float(model.predict_proba(
        vec.transform([preprocessing.clean_single_text(normalized)])
    )[0][1])


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestPromotedModel:
    def test_loads_as_sklearn_backend(self):
        service = ModelService(settings.model_file, settings.vectorizer_file).load()
        assert service._backend == "sklearn"
        assert service.is_loaded
        assert service.model_is_loaded
        assert service.vectorizer_is_loaded

    def test_loading_is_deterministic(self):
        a = ModelService(settings.model_file, settings.vectorizer_file).load()
        b = ModelService(settings.model_file, settings.vectorizer_file).load()
        text = REUTERS_REAL
        assert a.predict(text).probability_real == b.predict(text).probability_real


class TestStableInputs:
    def test_reuters_style_real_is_real(self):
        svc = ModelService(settings.model_file, settings.vectorizer_file).load()
        pred = svc.predict(REUTERS_REAL)
        assert pred.label == "real"
        # Re-locked with the adopted source-marker normaliser active:
        # "WASHINGTON (Reuters) - " is stripped at inference, so the stored
        # precision dropped from 0.999993 to 0.999827 while remaining REAL.
        assert pred.probability_real == pytest.approx(0.999827, abs=1e-4)

    def test_known_fake_is_fake(self):
        svc = ModelService(settings.model_file, settings.vectorizer_file).load()
        pred = svc.predict(FAKE_STYLE)
        assert pred.label == "fake"
        assert pred.probability_real == pytest.approx(0.050833, abs=1e-4)

    def test_ambiguous_is_uncertain(self):
        svc = ModelService(settings.model_file, settings.vectorizer_file).load()
        pred = svc.predict(UNCERTAIN_TEXT)
        assert pred.label == "uncertain"
        assert pred.probability_real == pytest.approx(0.494867, abs=1e-4)

    def test_non_reuters_real_writing_kept_real(self):
        svc = ModelService(settings.model_file, settings.vectorizer_file).load()
        pred = svc.predict(NEUTRAL_REAL)
        assert pred.label == "real"
        assert pred.probability_real > 0.5

    def test_empty_cleaned_text_is_uncertain(self):
        svc = ModelService(settings.model_file, settings.vectorizer_file).load()
        with mock.patch.object(preprocessing, "clean_single_text", return_value=""):
            pred = svc.predict("the and of and")
        assert pred.label == "uncertain"
        assert pred.probability_real == 0.5


class TestNonReutersRegression:
    def test_guardian_article_stays_real(self):
        """A real non-Reuters article from the OOD corpus must be REAL."""
        import pathlib
        from app import main as app_main

        svc = ModelService(settings.model_file, settings.vectorizer_file).load()
        root = pathlib.Path(app_main.__file__).resolve().parent.parent
        rows = []
        try:
            with open(root / "data" / "splits" / "generalization.csv",
                      newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
        except OSError:
            pytest.skip("generalization corpus unavailable")
        guardian = next((r for r in rows
                 if r["source"] == "guardian"
                 and r["text"].startswith("Pedro Sánchez")), None)
        if guardian is None:
            pytest.skip("no matched guardian article in corpus")
        pred = svc.predict(guardian["text"])
        assert pred.probability_real == pytest.approx(0.993426, abs=1e-4)


class TestApiParity:
    def test_predict_matches_offline_evaluator(self, client):
        for text in (REUTERS_REAL, FAKE_STYLE, UNCERTAIN_TEXT):
            resp = client.post("/predict", json={"news": text})
            assert resp.status_code == 200
            body = resp.json()
            expected = _offline_probability(text)
            assert body["probability_real"] == pytest.approx(expected * 100.0, abs=0.02)
            assert body["probability_fake"] == pytest.approx((1.0 - expected) * 100.0, abs=0.02)
            winner = max(expected, 1 - expected)
            if winner - 0.5 < settings.UNCERTAINTY_THRESHOLD:
                assert body["label"] == "uncertain"
            else:
                assert body["label"] == ("real" if expected >= 0.5 else "fake")

    def test_history_records_promoted_predictions(self, client):
        client.post("/predict", json={"news": REUTERS_REAL})
        history = client.get("/history").json()
        assert any(h["label"] in ("real", "fake", "uncertain") for h in history)


class TestExplanation:
    def test_explanation_is_influential_features(self, client):
        resp = client.post("/predict", json={"news": REUTERS_REAL})
        assert resp.status_code == 200
        expl = resp.json()["explanation"]
        assert expl, "linear-model explanation should be non-empty"
        words = [w["word"] for w in expl["top_influential_words"]]
        assert len(words) <= settings.TOP_FEATURES
        svc = ModelService(settings.model_file, settings.vectorizer_file).load()
        vocab = set(svc._vectorizer.vocabulary_.keys())
        for w in expl["top_influential_words"]:
            assert w["word"] in vocab
            assert w["direction"] in ("real", "fake")
            assert 0.0 <= w["impact"] <= 100.0


class TestUrlAndProtections:
    def test_predict_url_uses_same_pipeline(self, client):
        with mock.patch.object(
            __import__("app.main", fromlist=["fetch_article"]),
            "fetch_article",
            return_value=ExtractResult(
                text=NEUTRAL_REAL, title="Storm", final_url="http://example.com/a",
            ),
        ):
            resp = client.post("/predict-url", json={"url": "http://example.com/a"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["source_type"] == "url"
        assert body["label"] == "real"

    def test_404_url_is_rejected_cleanly(self, client):
        with mock.patch.object(
            __import__("app.main", fromlist=["fetch_article"]),
            "fetch_article",
            side_effect=ScrapeError("The URL could not be fetched: HTTP 404"),
        ):
            resp = client.post("/predict-url", json={"url": "http://example.com/missing"})
        assert resp.status_code == 422
        assert "404" in resp.json()["detail"]

    def test_homepage_non_article_is_rejected(self, client):
        with mock.patch.object(
            __import__("app.main", fromlist=["fetch_article"]),
            "fetch_article",
            return_value=ExtractResult(
                text="", title=None, final_url="http://example.com/",
            ),
        ):
            resp = client.post("/predict-url", json={"url": "http://example.com/"})
        assert resp.status_code == 422
        assert "couldn't identify the article" in resp.json()["detail"]

    def test_ssrf_private_network_url_is_blocked(self, client):
        resp = client.post("/predict-url", json={"url": "http://127.0.0.1:9/x"})
        assert resp.status_code == 422
        assert resp.json()["category"] == "blocked_network"

    def test_near_empty_input_rejected(self, client):
        resp = client.post("/predict", json={"news": "hi"})
        assert resp.status_code == 422