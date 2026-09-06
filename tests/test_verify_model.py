"""Tests for the model verify_model module (structure only, no TF needed)."""

import pytest
from unittest import mock


def test_run_checks_passes_for_well_formed_prediction():
    from app.verify_model import _run_checks
    service = mock.Mock()
    pred = mock.Mock()
    pred.probability_real = 0.8
    pred.probability_fake = 0.2
    pred.label = "real"
    pred.explanation = [mock.Mock(), mock.Mock()]
    service.predict.return_value = pred
    ok, returned = _run_checks(service)
    assert ok is True
    assert returned is pred


def test_run_checks_fails_on_degenerate_probability():
    from app.verify_model import _run_checks
    service = mock.Mock()
    pred = mock.Mock()
    pred.probability_real = -1.0
    pred.probability_fake = 2.0
    pred.label = "real"
    pred.explanation = []
    service.predict.return_value = pred
    ok, _ = _run_checks(service)
    assert ok is False


def test_main_returns_1_when_model_missing(tmp_path):
    from app.verify_model import main
    with mock.patch("app.verify_model.settings") as s:
        s.model_file = tmp_path / "nonexistent.pkl"
        s.vectorizer_file = tmp_path / "nonexistent.pkl"
        assert main() == 1


def test_main_returns_0_for_valid_loader(tmp_path):
    from app.verify_model import main
    from app import verify_model as vm
    service = mock.Mock()
    service._backend = "sklearn"
    service._model = mock.Mock()
    service._vectorizer = mock.Mock()
    service._vectorizer.vocabulary_ = {"a": 0}
    pred = mock.Mock()
    pred.probability_real = 0.9
    pred.probability_fake = 0.1
    pred.label = "real"
    pred.explanation = []
    service.predict.return_value = pred
    model_file = tmp_path / "m.pkl"
    vec_file = tmp_path / "v.pkl"
    model_file.touch()
    vec_file.touch()
    with mock.patch("app.verify_model.settings") as s, \
         mock.patch.object(vm, "_load_service", return_value=service) as load:
        s.model_file = model_file
        s.vectorizer_file = vec_file
        assert main() == 0
        load.assert_called_once_with(model_file, vec_file)