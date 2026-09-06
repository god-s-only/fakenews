"""Tests for Settings.validate()."""

from unittest import mock

import pytest

from app.config import Settings


def test_default_config_no_warnings():
    s = Settings()
    assert s.validate() == []


def test_threshold_zero_warns():
    s = Settings()
    s.UNCERTAINTY_THRESHOLD = 0.0
    warnings = s.validate()
    assert any("UNCERTAINTY_THRESHOLD" in w for w in warnings)


def test_threshold_negative_warns():
    s = Settings()
    s.UNCERTAINTY_THRESHOLD = -0.5
    warnings = s.validate()
    assert any("UNCERTAINTY_THRESHOLD" in w for w in warnings)


def test_threshold_one_warns():
    s = Settings()
    s.UNCERTAINTY_THRESHOLD = 1.0
    warnings = s.validate()
    assert any("UNCERTAINTY_THRESHOLD" in w for w in warnings)


def test_max_input_too_short_warns():
    s = Settings()
    s.MAX_INPUT_LENGTH = 50
    warnings = s.validate()
    assert any("MAX_INPUT_LENGTH" in w for w in warnings)


def test_port_out_of_range_warns():
    s = Settings()
    s.PORT = 99999
    warnings = s.validate()
    assert any("PORT" in w for w in warnings)


def test_top_features_zero_warns():
    s = Settings()
    s.TOP_FEATURES = 0
    warnings = s.validate()
    assert any("TOP_FEATURES" in w for w in warnings)


def test_all_bad_values_warns():
    s = Settings()
    s.UNCERTAINTY_THRESHOLD = -1.0
    s.MAX_INPUT_LENGTH = 10
    s.PORT = 0
    s.TOP_FEATURES = -5
    warnings = s.validate()
    assert len(warnings) == 4
