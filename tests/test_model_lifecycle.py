"""Model lifecycle tests: loading, readiness, fingerprints, concurrency.

These exercise the real promoted TF-IDF LogisticRegression detector and are
skipped when the frozen artifacts are unavailable (same policy as
``test_real_model.py``).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from app.config import BASE_DIR, settings
from app.model import ModelLoadError, ModelService

# Pinned in reports/release_manifest.json — the model must never drift.
MODEL_SHA256 = "7f555ef4793ca4464f49cfe4195dc81223ac97e435ed30cc2f63b272c791e4ec"
VECTORIZER_SHA256 = "6d50472067072913932d461114bd67644732f04078d58b1f8a3720d2c1a4f05a"


def _assets_available() -> bool:
    return settings.model_file.exists() and settings.vectorizer_file.exists()


pytestmark = pytest.mark.skipif(
    not _assets_available(),
    reason="Model/vectorizer assets are not available",
)


@pytest.fixture(scope="module")
def service() -> ModelService:
    return ModelService(settings.model_file, settings.vectorizer_file).load()


class TestLoadLifecycle:
    def test_not_loaded_before_load(self):
        svc = ModelService(settings.model_file, settings.vectorizer_file)
        assert not svc.is_loaded
        assert not svc.model_ready
        assert svc.loaded_at is None

    def test_load_populates_fingerprints(self, service):
        """Load() itself captures the artifact digests and model metadata."""
        assert service.model_sha256 == MODEL_SHA256
        assert service.vectorizer_sha256 == VECTORIZER_SHA256

    def test_load_populates_vocab_and_timestamp(self, service):
        assert service.vocab_size == 36_862
        assert service.loaded_at is not None
        assert service.model_ready

    def test_load_is_idempotent(self, service):
        """Re-entering load() reloads cleanly and keeps identical facts."""
        service.load()
        assert service.model_sha256 == MODEL_SHA256
        assert service.vectorizer_sha256 == VECTORIZER_SHA256
        assert service.model_ready

    def test_model_ready_requires_fingerprints(self, service):
        """A service with no fingerprints is loaded but not ready."""
        service.model_sha256 = None
        try:
            assert service.is_loaded
            assert not service.model_ready
        finally:
            service.model_sha256 = MODEL_SHA256

    def test_ready_distinguishes_from_loaded(self):
        svc = ModelService(settings.model_file, settings.vectorizer_file)
        assert svc.is_loaded is False
        assert svc.model_ready is False


class TestLoadFailure:
    def test_missing_model_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "MODEL_PATH", str(tmp_path / "nope.pkl"))
        svc = ModelService(settings.model_file, settings.vectorizer_file)
        with pytest.raises(ModelLoadError, match="Model file not found"):
            svc.load()

    def test_model_load_error_is_predictable_type(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "MODEL_PATH", str(tmp_path / "nope.pkl"))
        svc = ModelService(settings.model_file, settings.vectorizer_file)
        with pytest.raises(ModelLoadError):
            svc.load()
        # After a failed load the service is not ready/loaded.
        assert not svc.is_loaded
        assert not svc.model_ready
        assert svc.model_sha256 is None


class TestConcurrentPrediction:
    TEXT = (
        "Government announces new economic policy with broad support across "
        "several regions and industries this quarter while officials noted "
        "continued progress on infrastructure spending and job growth."
    )

    def test_concurrent_predictions_are_deterministic(self, service):
        """Many threads share one ModelService without races or exceptions."""
        reference = service.predict(self.TEXT)

        def _predict(_):
            return service.predict(self.TEXT)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(_predict, range(40)))
        for result in results:
            assert result.probability_real == pytest.approx(
                reference.probability_real
            )
            assert result.label == reference.label
            assert result.explanation == reference.explanation

    def test_concurrent_explanation_is_bounded(self, service):
        def _predict(_):
            return service.predict(
                "Scientists announce the new mission control center opened "
                "downtown with officials from several agencies attending."
            )

        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(_predict, range(18)))
        for result in results:
            assert len(result.explanation) <= max(1, min(settings.TOP_FEATURES, 50))


class TestBaselineArtifactsStillPinned:
    def test_baseline_rollback_files_unchanged(self):
        baseline_model = BASE_DIR / "artifacts" / "baseline" / "my_model.h5"
        baseline_vec = BASE_DIR / "artifacts" / "baseline" / "countvectorizer.pkl"
        assert baseline_model.exists()
        assert baseline_vec.exists()