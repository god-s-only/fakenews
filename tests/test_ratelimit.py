"""Tests for the sliding-window rate limiter (unit + middleware + end-to-end)."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.config import settings
from app.ratelimit import RateLimitMiddleware, SlidingWindowRateLimiter


async def _noop(app_scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


class _SendSpy:
    def __init__(self):
        self.messages: list[dict] = []

    async def __call__(self, message):
        self.messages.append(message)


class TestSlidingWindowRateLimiter:
    def test_allows_up_to_limit_then_denies(self):
        limiter = SlidingWindowRateLimiter(limit=3, window_seconds=60)
        assert all(limiter.allow("ip")[0] for _ in range(3))
        allowed, retry_after = limiter.allow("ip")
        assert not allowed
        assert retry_after > 0

    def test_window_slides_after_elapse(self):
        limiter = SlidingWindowRateLimiter(limit=2, window_seconds=0.05)
        assert limiter.allow("ip")[0]
        time.sleep(0.03)
        assert limiter.allow("ip")[0]
        time.sleep(0.03)
        assert limiter.allow("ip")[0]  # earliest hit now expired

    def test_keys_are_independent(self):
        limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60)
        assert limiter.allow("a")[0]
        assert limiter.allow("b")[0]
        assert not limiter.allow("a")[0]
        assert not limiter.allow("b")[0]

    def test_key_table_is_bounded(self):
        limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60, max_keys=4)
        for i in range(6):
            limiter.allow(f"ip{i}")
        assert len(limiter) <= 4

    def test_reset_clears(self):
        limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60)
        limiter.allow("ip")
        assert not limiter.allow("ip")[0]
        limiter.reset()
        assert limiter.allow("ip")[0]
        assert limiter.remaining("ip") == 0

    def test_remaining_counts_down(self):
        limiter = SlidingWindowRateLimiter(limit=2, window_seconds=60)
        assert limiter.remaining("ip") == 2
        limiter.allow("ip")
        assert limiter.remaining("ip") == 1

    def test_invalid_arguments_rejected(self):
        with pytest.raises(ValueError):
            SlidingWindowRateLimiter(limit=0, window_seconds=60)
        with pytest.raises(ValueError):
            SlidingWindowRateLimiter(limit=1, window_seconds=0)
        with pytest.raises(ValueError):
            SlidingWindowRateLimiter(limit=1, window_seconds=60, max_keys=0)

    def test_thread_safety(self):
        import threading

        limiter = SlidingWindowRateLimiter(limit=1000, window_seconds=60, max_keys=10)
        errors: list[BaseException] = []
        barrier = threading.Barrier(8)

        def _worker(i: int):
            barrier.wait()
            try:
                for _ in range(200):
                    limiter.allow(f"ip{i % 4}")
            except BaseException as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


class TestRateLimitMiddleware:
    def _call(self, scope):
        spy = _SendSpy()
        calls = []

        async def downstream(app_scope, receive, send):
            calls.append(None)
            await _noop(app_scope, receive, send)

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        middleware = RateLimitMiddleware(downstream)
        asyncio.run(middleware(scope, receive, spy))
        return spy, calls

    def _scope(self, limiter):
        return {
            "type": "http",
            "method": "GET",
            "path": "/health",
            "client": ("10.0.0.1", 1234),
            "app": SimpleNamespace(
                state=SimpleNamespace(app_state=SimpleNamespace(rate_limiter=limiter))
            ),
        }

    @pytest.fixture(autouse=True)
    def _enable_rate_limit(self, monkeypatch):
        monkeypatch.setattr("app.config.settings.RATE_LIMIT_ENABLED", True)

    def test_denies_and_sets_retry_after(self):
        limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60)
        scope = self._scope(limiter)
        spy, calls = self._call(scope)
        assert calls == [None]
        spy, calls = self._call(scope)
        assert calls == []
        start = spy.messages[0]
        assert start["status"] == 429
        assert (b"retry-after", b"60") in start["headers"]

    def test_fail_open_on_limiter_error(self):
        limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60)

        def broken(*_):
            raise RuntimeError("boom")

        limiter.allow = broken  # type: ignore[assignment]
        scope = self._scope(limiter)
        spy, calls = self._call(scope)
        assert calls == [None]  # request proceeded despite limiter failure
        assert spy.messages[0]["status"] == 200

    def test_no_limiter_passes_through(self):
        scope = self._scope(None)
        spy, calls = self._call(scope)
        assert calls == [None]


class TestEndToEnd:
    def test_real_app_429s_after_quota(self, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
        monkeypatch.setattr(settings, "RATE_LIMIT_REQUESTS", 3)
        real = create_app_for_ratelimit()
        client = TestClient(real)

        for _ in range(3):
            resp = client.get("/health")
            assert resp.status_code == 200
        resp = client.get("/health")
        assert resp.status_code == 429
        assert int(resp.headers["retry-after"]) >= 1

    def test_disabled_rate_limit_never_429s(self, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)
        monkeypatch.setattr(settings, "RATE_LIMIT_REQUESTS", 1)
        real = create_app_for_ratelimit()
        client = TestClient(real)
        for _ in range(5):
            assert client.get("/health").status_code == 200


def create_app_for_ratelimit():
    from app.main import create_app
    return create_app()