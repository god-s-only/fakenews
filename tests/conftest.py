"""Shared pytest fixtures.

The URL-extraction cache lives on the module-level app state and outlives
individual tests; without an explicit clear, a URL cached by one test could
short-circuit the fetch mock of a later test using the same URL. This autouse
fixture clears the bounded cache before and after every test so tests never see
each other's cached extractions.
"""

from __future__ import annotations

import pytest

import app.main as main_module


@pytest.fixture(autouse=True)
def _isolate_shared_state(monkeypatch):
    # The URL-extraction cache and rate limiter live on the module-level app
    # state and outlive individual tests; without an explicit reset, one test
    # could short-circuit the fetch mock (cached URL) or exhaust the request
    # quota of a later test. Reset both before and after every test.
    #
    # Rate limiting is a production concern; tests that exercise it re-enable
    # RATE_LIMIT_ENABLED themselves.
    monkeypatch.setattr("app.config.settings.RATE_LIMIT_ENABLED", False)
    cache = getattr(main_module.state, "url_cache", None)
    if cache is not None:
        cache.clear()
    limiter = getattr(main_module.state, "rate_limiter", None)
    if limiter is not None:
        limiter.reset()
    yield
    monkeypatch.setattr("app.config.settings.RATE_LIMIT_ENABLED", False)
    if cache is not None:
        cache.clear()
    if limiter is not None:
        limiter.reset()