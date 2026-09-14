"""Application lifecycle tests (lifespan, shutdown, failure, re-creation).

Unlike most of the suite, these tests DO enter the FastAPI lifespan (via the
``with TestClient(app)`` context manager), so they must be careful: the real
promoted model is small and fast to load, and the failure test points the
lifespan at a missing artifact to prove predictable startup failure.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import create_app
from app.model import ModelLoadError

import app.main as main_module


@pytest.fixture(autouse=True)
def _reset_default_state():
    main_module.state.model = None
    yield
    main_module.state.model = None


def test_default_app_state_is_bound_to_module_state():
    """The default application must keep using the module-level state object,
    which existing tests mutate directly."""
    assert main_module.app.state.app_state is main_module.state


def test_successful_startup_loads_model_and_shuts_down():
    with TestClient(main_module.app) as client:
        assert main_module.state.model is not None
        assert main_module.state.model.model_ready
        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["model_loaded"] is True
        assert health["model_sha256"]
    # After the lifespan exits the model slot is released.
    assert main_module.state.model is None


def test_shutdown_releases_model_on_multiple_cycles():
    with TestClient(main_module.app):
        assert main_module.state.model is not None
        first = main_module.state.model
    with TestClient(main_module.app):
        assert main_module.state.model is not None
        assert main_module.state.model is not first


def test_startup_fails_predictably_when_model_missing(monkeypatch):
    """A missing artifact must raise during startup (fail-fast), not serve 503s."""
    monkeypatch.setattr(settings, "MODEL_PATH", "does-not-exist.pkl")
    with pytest.raises(RuntimeError) as excinfo:
        with TestClient(main_module.app):
            raise AssertionError("lifespan should not have started")
    assert "Model file not found" in str(excinfo.value)
    assert main_module.state.model is None


def test_startup_failure_is_model_load_error(monkeypatch):
    monkeypatch.setattr(settings, "MODEL_PATH", "does-not-exist.pkl")
    with pytest.raises(RuntimeError) as excinfo:
        with TestClient(main_module.app):
            raise AssertionError("lifespan should not have started")
    assert isinstance(excinfo.value.__cause__, ModelLoadError)


def test_repeated_application_creation_is_isolated():
    """Every create_app() call owns its own AppState (no cross-app sharing)."""
    app_a = create_app()
    app_b = create_app()
    assert app_a.state.app_state is not app_b.state.app_state

    app_a.state.app_state = object()  # any sentinel state
    assert app_b.state.app_state is not app_a.state.app_state

    # And neither is the module default app's state.
    assert app_b.state.app_state is not main_module.state