"""Unit tests for the verdict/uncertainty decision helper.

The verdict logic does not require TensorFlow, so it is tested directly on a
ModelService instance (which only needs file paths for construction).
"""

from unittest import mock

from app import preprocessing
from app.model import ModelService


def _service(tmp_path):
    model_file = tmp_path / "model.h5"
    vectorizer_file = tmp_path / "vectorizer.pkl"
    model_file.touch()
    vectorizer_file.touch()
    return ModelService(model_file, vectorizer_file)


class TestVerdict:
    def test_confident_real(self, tmp_path):
        service = _service(tmp_path)
        label, confidence = service._verdict(0.91, 0.09)
        assert label == "real"
        assert confidence == 91.0

    def test_confident_fake(self, tmp_path):
        service = _service(tmp_path)
        label, confidence = service._verdict(0.05, 0.95)
        assert label == "fake"
        assert confidence == 95.0

    def test_uncertain_at_53_percent(self, tmp_path):
        # winner 0.53 - 0.5 = 0.03 < 0.10 threshold -> uncertain
        service = _service(tmp_path)
        label, confidence = service._verdict(0.53, 0.47)
        assert label == "uncertain"
        assert confidence == 53.0

    def test_uncertain_at_55_percent(self, tmp_path):
        # winner 0.55 - 0.5 = 0.05 < 0.10 threshold -> uncertain
        service = _service(tmp_path)
        label, confidence = service._verdict(0.55, 0.45)
        assert label == "uncertain"

    def test_confident_at_61_percent(self, tmp_path):
        # winner 0.61 - 0.5 = 0.11 >= 0.10 threshold -> confident real
        service = _service(tmp_path)
        label, _ = service._verdict(0.61, 0.39)
        assert label == "real"

    def test_confidence_uses_winner(self, tmp_path):
        service = _service(tmp_path)
        _, confidence = service._verdict(0.42, 0.58)
        assert confidence == 58.0

    def test_exact_50_50_is_uncertain(self, tmp_path):
        service = _service(tmp_path)
        label, _ = service._verdict(0.50, 0.50)
        assert label == "uncertain"


class TestPredictRobustness:
    def test_empty_cleaned_text_returns_uncertain(self, tmp_path):
        service = _service(tmp_path)
        with mock.patch.object(preprocessing, "clean_single_text", return_value=""):
            prediction = service.predict("the and of")
        assert prediction.label == "uncertain"
        assert prediction.probability_real == 0.5
        assert prediction.probability_fake == 0.5
        assert prediction.explanation == []