"""Production deployment regression tests.

Guard rails so a running server can never silently serve the wrong detector:

* The default configuration MUST point at the promoted Candidate D artifacts
  (``my_model_lr.pkl`` + ``my_tfidf_vectorizer.pkl``), never at the legacy
  Keras baseline under ``artifacts/baseline/``.
* The loaded backend MUST be scikit-learn ``LogisticRegression`` (TF-IDF
  backend), and the artifact hashes MUST match ``reports/release_manifest.json``.
* ``GET /health`` MUST expose the running backend + artifact fingerprints so
  a stale process is detectable at runtime.
* API predictions MUST agree with the frozen offline candidate to the API's
  display precision.

If any of these fail the server is serving the wrong model.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from app.config import settings
from app.main import app
from app.model import ModelService

ROOT = Path(__file__).resolve().parent.parent

CBN_REAL = (
    "The Central Bank of Nigeria said commercial banks will continue to operate "
    "under the existing cash withdrawal guidelines while customers are encouraged "
    "to use electronic payment channels. The bank said the policy is intended to "
    "improve the efficiency of the country's payment system and reduce reliance "
    "on physical cash."
)


def _release_hashes() -> dict[str, str]:
    manifest = json.loads((ROOT / "reports/release_manifest.json").read_text())
    new = manifest["new_artifacts"]
    return {
        "my_model_lr.pkl": new["my_model_lr.pkl"]["sha256"],
        "my_tfidf_vectorizer.pkl": new["my_tfidf_vectorizer.pkl"]["sha256"],
    }


def test_default_settings_point_at_promoted_artifacts():
    assert settings.model_file.name == "my_model_lr.pkl"
    assert settings.vectorizer_file.name == "my_tfidf_vectorizer.pkl"
    assert settings.model_file.exists()
    assert settings.vectorizer_file.exists()


def test_promoted_backend_is_sklearn_lr_and_not_baseline():
    service = ModelService(settings.model_file, settings.vectorizer_file).load()
    try:
        assert service._backend == "sklearn"
        assert isinstance(service._model, LogisticRegression)
        assert service.model_file.name == "my_model_lr.pkl"
        assert service.vectorizer_file.name == "my_tfidf_vectorizer.pkl"
        assert "artifacts/baseline" not in str(service.model_file)
    finally:
        del service


def test_production_artifacts_match_release_manifest_hashes():
    import hashlib

    expected = _release_hashes()

    def sha(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    assert sha(ROOT / "my_model_lr.pkl") == expected["my_model_lr.pkl"]
    assert sha(ROOT / "my_tfidf_vectorizer.pkl") == expected["my_tfidf_vectorizer.pkl"]


def test_offline_production_is_bit_exact_with_frozen_candidate():
    from app import preprocessing

    service = ModelService(settings.model_file, settings.vectorizer_file).load()
    with (ROOT / "artifacts/candidates/expD_lr.pkl").open("rb") as handle:
        blob = pickle.load(handle)
    vec, model = blob["vec"], blob["model"]
    try:
        cleaned = preprocessing.clean_single_text(CBN_REAL)
        expected = float(model.predict_proba(vec.transform([cleaned]))[0][1])
        actual = service.predict(CBN_REAL).probability_real
        assert actual == expected
        assert actual == pytest.approx(0.97756865, abs=1e-8)
    finally:
        del service


def test_health_reports_backend_and_artifact_fingerprints():
    expected = _release_hashes()
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["vectorizer_loaded"] is True
    assert body["model_backend"] == "sklearn"
    assert body["model_file"] == "my_model_lr.pkl"
    assert body["vectorizer_file"] == "my_tfidf_vectorizer.pkl"
    assert body["model_sha256"] == expected["my_model_lr.pkl"]
    assert body["vectorizer_sha256"] == expected["my_tfidf_vectorizer.pkl"]
    assert body["vocab_size"] == 36862


def test_api_prediction_matches_offline_candidate_via_http():
    with TestClient(app) as client:
        resp = client.post("/predict", json={"news": CBN_REAL})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "real"
    assert body["probability_real"] == pytest.approx(97.76, abs=0.01)
    assert body["probability_fake"] == pytest.approx(2.24, abs=0.01)