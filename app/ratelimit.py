"""In-process sliding-window rate limiting (audit B5).

A bounded sliding-window limiter keyed by client IP, enforced by a pure-ASGI
middleware. Limits are per-process: under multiple workers each process enforces
its own counters (documented limitation — cross-worker/cross-instance limiting
would need a shared store, out of scope). The limiter is strictly fail-open —
any internal error logs a warning and lets the request through, because rate
limiting must never take the application down.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger("fakenews.ratelimit")

BodyReceiver = Callable[[], Awaitable[dict[str, Any]]]
ASGIApp = Callable[..., Awaitable[None]]
Sender = Callable[..., Awaitable[None]]


class SlidingWindowRateLimiter:
    """Rate-limits keys (typically client IPs) with a sliding window.

    Thread-safe and bounded: at most ``max_keys`` tracked keys, and each key's
    hit list is trimmed to the window. Uses 1-NTP-relative monotonic timestamps.
    """

    def __init__(
        self,
        limit: int,
        window_seconds: float,
        max_keys: int = 10_000,
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if max_keys < 1:
            raise ValueError("max_keys must be >= 1")
        self.limit = int(limit)
        self.window_seconds = float(window_seconds)
        self.max_keys = int(max_keys)
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.RLock()

    def allow(self, key: str) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds)."""
        with self._lock:
            now = time.monotonic()
            timestamps = self._hits.get(key)
            if timestamps is None:
                timestamps = deque()
                self._hits[key] = timestamps
                # Bounded key table: drop the least-recently-active key.
                while len(self._hits) > self.max_keys:
                    self._hits.popitem(last=False)
            while timestamps and now - timestamps[0] >= self.window_seconds:
                timestamps.popleft()
            if len(timestamps) >= self.limit:
                retry_after = max(self.window_seconds - (now - timestamps[0]), 0.0)
                return False, retry_after
            timestamps.append(now)
            self._hits.move_to_end(key)
            return True, 0.0

    def remaining(self, key: str) -> int:
        with self._lock:
            timestamps = self._hits.get(key)
            if timestamps is None:
                return self.limit
            now = time.monotonic()
            while timestamps and now - timestamps[0] >= self.window_seconds:
                timestamps.popleft()
            return max(self.limit - len(timestamps), 0)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._hits)


def _client_ip(scope: dict[str, Any]) -> str:
    client = scope.get("client")
    if client and client[0]:
        return str(client[0])
    return "unknown"


class RateLimitMiddleware:
    """Enforces the app-state rate limiter keyed by client IP.

    The limiter instance is resolved from ``scope["app"].state.rate_limiter`` so
    every application (and test) can own its own counters. Fail-open on any
    unexpected error.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: BodyReceiver, send: Sender) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        try:
            settings_proxy = getattr(scope.get("app"), "state", None)
            from app.config import settings

            if not settings.RATE_LIMIT_ENABLED:
                await self.app(scope, receive, send)
                return
            # The limiter lives on AppState (application.state.app_state),
            # never directly on application.state.
            app_state = getattr(settings_proxy, "app_state", None)
            limiter = getattr(app_state, "rate_limiter", None)
            if limiter is None:
                await self.app(scope, receive, send)
                return
            allowed, retry_after = limiter.allow(_client_ip(scope))
        except Exception:  # noqa: BLE001 - fail-open
            logger.warning("Rate limiter misbehaved; allowing request", exc_info=True)
            await self.app(scope, receive, send)
            return

        if allowed:
            await self.app(scope, receive, send)
            return

        await _send_429(send, retry_after)


async def _send_429(send: Sender, retry_after: float) -> None:
    """Respond 429 with a Retry-After header (seconds, integer, >= 1)."""
    body = json.dumps(
        {"detail": "Too many requests. Please slow down and try again later."}
    ).encode("utf-8")
    retry_after = max(1, math.ceil(retry_after))
    await send(
        {
            "type": "http.response.start",
            "status": 429,
            "headers": [
                (b"content-type", b"application/json; charset=utf-8"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"retry-after", str(retry_after).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})