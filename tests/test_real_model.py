"""Real-model integration tests.

These tests exercise the actual trained model and vectorizer pointed to by the
application defaults (the promoted robust TF-IDF + LogisticRegression detector).
The legacy Keras backend is still covered in TestLegacyKerasBackend, which loads
the preserved baseline artifacts explicitly.

They are skipped automatically when the assets are unavailable, so the rest of
the suite still runs in lightweight environments.
"""

import numpy as np
import pytest

from app.config import settings
from app.model import ModelService, _has_input_gradients, _rebuild_functional_model


def _assets_available() -> bool:
    return settings.model_file.exists() and settings.vectorizer_file.exists()


pytestmark = pytest.mark.skipif(
    not _assets_available(),
    reason="Model/vectorizer assets are not available",
)


def _load_service() -> ModelService:
    return ModelService(settings.model_file, settings.vectorizer_file).load()


class TestRealLoading:
    def test_model_and_vectorizer_load(self):
        service = _load_service()
        assert service.is_loaded
        assert service.model_is_loaded
        assert service.vectorizer_is_loaded

    def test_backend_is_sklearn_for_promoted_detector(self):
        service = _load_service()
        assert service._backend == "sklearn"

    def test_output_is_probability_in_unit_range(self):
        service = _load_service()
        vector = service._vectorizer.transform(["authorit"]).toarray()
        prob_real = service._probability_real("authorit news")
        assert 0.0 <= prob_real <= 1.0


class TestRealPrediction:
    def test_probabilities_sum_to_one(self):
        svc = _load_service()
        pred = svc.predict(
            "Government announces new economic policy with broad support"
        )
        assert 0.0 <= pred.probability_real <= 1.0
        assert 0.0 <= pred.probability_fake <= 1.0
        assert abs((pred.probability_real + pred.probability_fake) - 1.0) < 1e-6
        assert isinstance(pred.confidence, float)

    def test_label_is_verdict(self):
        svc = _load_service()
        for text in [
            "Breaking news on the latest election results in the capital",
            "A completely normal article about weather and sports today",
            "Experts announce a major new discovery in medical research",
        ]:
            pred = svc.predict(text)
            assert pred.label in ("real", "fake", "uncertain")

    def test_explainability_returns_structured_words(self):
        svc = _load_service()
        pred = svc.predict(
            "Scientists announce the new mission control center opened downtown"
        )
        assert pred.explanation, "linear-model explanation should be non-empty"
        for item in pred.explanation:
            assert item.word
            assert isinstance(item.impact, float)
            assert item.direction in ("real", "fake")
            assert 0.0 <= item.impact <= 100.0

    def test_explanations_are_influential_features_not_proof(self):
        """Attribution words come from the vectorizer vocabulary."""
        from app import preprocessing

        svc = _load_service()
        text = "Officials announce a new discovery at the research center downtown"
        pred = svc.predict(text)
        cleaned = preprocessing.clean_single_text(text)
        vocab = set(svc._vectorizer.vocabulary_.keys())
        for item in pred.explanation:
            assert item.word in vocab

    def test_manual_pipeline_matches_service(self):
        """The service predict() path equals a manual vectorize + model path."""
        from app import preprocessing

        svc = _load_service()
        text = (
            "Officials announced the new hospital wing will open next month after "
            "delays in construction across the northern region were resolved."
        )
        pred = svc.predict(text)
        cleaned = preprocessing.clean_single_text(text)
        vector = svc._vectorizer.transform([cleaned]).toarray()
        if svc._backend == "sklearn":
            raw = float(svc._model.predict_proba(vector)[0][1])
        else:
            raw = float(svc._model.predict(vector, verbose=0)[0][0])
        raw = float(np.clip(raw, 0.0, 1.0))
        assert pred.probability_real == pytest.approx(raw, abs=1e-5)
        assert pred.probability_fake == pytest.approx(1.0 - raw, abs=1e-5)

    def test_label_derived_from_actual_probability(self):
        """Label must follow the documented threshold rule from P(real)."""
        svc = _load_service()
        for text in [
            "Breaking news on the latest election results in the capital",
            "A completely normal article about weather and sports today",
            "Experts announce a major new discovery in medical research",
        ]:
            pred = svc.predict(text)
            label, confidence = svc._verdict(pred.probability_real, pred.probability_fake)
            assert pred.label == label
            assert pred.confidence == confidence
            winner = max(pred.probability_real, pred.probability_fake)
            assert pred.label in ("real", "fake", "uncertain")
            assert pred.confidence == round(winner * 100.0, 2)

    def test_prediction_pipeline_is_deterministic(self):
        """The same input always produces the same output (stability/parity)."""
        svc = _load_service()
        text = (
            "Government announces new economic policy with broad support across "
            "several regions and industries this quarter."
        )
        first = svc.predict(text)
        second = svc.predict(text)
        assert first.probability_real == second.probability_real
        assert first.probability_fake == second.probability_fake
        assert first.label == second.label


class TestLegacyKerasBackend:
    """The preserved Keras baseline backend must keep working (rollback path)."""

    @classmethod
    def _load_baseline(cls) -> ModelService:
        from app.config import BASE_DIR
        return ModelService(
            BASE_DIR / "artifacts" / "baseline" / "my_model.h5",
            BASE_DIR / "artifacts" / "baseline" / "countvectorizer.pkl",
        ).load()

    def test_baseline_backend_is_keras(self):
        service = self._load_baseline()
        assert service._backend == "keras"
        assert _has_input_gradients(service._model)

    def test_baseline_rebuild_preserves_weights(self):
        service = self._load_baseline()
        rebuilt = _rebuild_functional_model(service._model)
        for original, rebuilt_weight in zip(
            service._model.get_weights(), rebuilt.get_weights()
        ):
            assert np.allclose(original, rebuilt_weight)

    def test_baseline_input_dimension_matches_vocab(self):
        service = self._load_baseline()
        input_dim = service._model.input_shape[-1]
        vocab_size = len(service._vectorizer.vocabulary_)
        assert input_dim == vocab_size

    def test_baseline_predicts_sigmoid_style(self):
        service = self._load_baseline()
        vector = service._vectorizer.transform(["authorit"]).toarray()
        prob_real = float(service._model.predict(vector, verbose=0)[0][0])
        assert 0.0 <= prob_real <= 1.0