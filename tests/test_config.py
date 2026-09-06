"""Tests for the configuration module."""

from app.config import Settings, BASE_DIR


def test_settings_summary_returns_dict():
    s = Settings()
    summary = s.summary()
    assert isinstance(summary, dict)
    assert summary["port"] == 8000
    assert summary["uncertainty_threshold"] == 0.10


def test_settings_model_file_resolves_to_absolute():
    s = Settings()
    assert s.model_file.is_absolute()


def test_settings_vectorizer_file_resolves_to_absolute():
    s = Settings()
    assert s.vectorizer_file.is_absolute()


def test_allowed_origins_wildcard():
    s = Settings()
    assert s.allowed_origins == ["*"]


def test_allowed_origins_list():
    s = Settings()
    s.CORS_ORIGINS = "http://a.com, http://b.com"
    assert s.allowed_origins == ["http://a.com", "http://b.com"]


def test_base_dir_is_project_root():
    assert BASE_DIR.name == "fakenews"
