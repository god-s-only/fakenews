"""Tests for the empty-vector probability guard in model."""

import numpy as np
from unittest import mock
from app.model import ModelService, Prediction


class _FakeVectorizer:
    def __init__(self, vocab=None):
        self.vocabulary_ = vocab or {"hello": 0, "world": 1}

    def transform(self, texts):
        rows = []
        for text in texts:
            words = text.split()
            row = [0] * len(self.vocabulary_)
            for w in words:
                if w in self.vocabulary_:
                    row[self.vocabulary_[w]] += 1
            rows.append(row)
        result = mock.Mock()
        result.toarray.return_value = np.array(rows)
        return result

    def get_feature_names_out(self):
        return sorted(self.vocabulary_, key=self.vocabulary_.get)


class _FakeModel:
    def __init__(self, prob=0.8):
        self._prob = prob

    def predict(self, vector, verbose=0):
        return np.array([[self._prob]])


def _make_service(tmp_path):
    model_file = tmp_path / "m.h5"
    vec_file = tmp_path / "v.pkl"
    model_file.touch()
    vec_file.touch()
    svc = ModelService(model_file, vec_file)
    svc._model = _FakeModel()
    svc._vectorizer = _FakeVectorizer()
    return svc


def test_empty_vector_returns_uncertain(tmp_path):
    svc = _make_service(tmp_path)
    pred = svc.predict("the and a of")
    assert pred.label == "uncertain"
    assert pred.probability_real == 0.5


def test_all_stopwords_returns_uncertain(tmp_path):
    svc = _make_service(tmp_path)
    pred = svc.predict("the a and of an or is are was were")
    assert pred.label == "uncertain"
    assert pred.probability_fake == 0.5


def test_valid_text_returns_confidence(tmp_path):
    svc = _make_service(tmp_path)
    pred = svc.predict("hello world news today here something")
    assert pred.label in ("real", "fake")
    assert pred.confidence > 50.0
