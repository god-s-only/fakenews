"""Request correlation & structured access logging (audit B7).

Every HTTP request gets a ``request_id`` (either echoed from a well-formed
``X-Request-ID`` header or generated), stored in a context variable so any log
line emitted while handling the request carries it, propagated on the response
as ``X-Request-ID``, and logged as a single structured access record with
timing. A logging filter injects ``request_id`` into every record.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any

logger = logging.getLogger("fakenews.access")

BodyReceiver = Callable[[], Awaitable[dict[str, Any]]]
ASGIApp = Callable[..., Awaitable[None]]
Sender = Callable[..., Awaitable[None]]

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_INCOMING_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


class RequestIdFilter(logging.Filter):
    """Attach the active request id to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def attach_request_id_filter() -> None:
    """Attach the filter to every handler that can format ``fakenews`` records.

    ``fakenews`` records propagate to the root logger (whose handler is what
    ``configure_logging()`` configures via ``basicConfig`` and what uvicorn may
    replace or extend at startup), so the filter must be present on the whole
    ``fakenews`` tree *and* on the root logger's handlers — otherwise a handler
    using a ``request_id``-aware format raises ``ValueError``.
    """
    loggers = [
        logging.getLogger("fakenews"),
        logging.root,
    ]
    loggers += [
        logging.getLogger(name)
        for name in sorted(logging.root.manager.loggerDict)
        if name == "fakenews" or name.startswith("fakenews.")
    ]
    seen: set[int] = set()
    for target in loggers:
        for handler in target.handlers:
            if id(handler) in seen:
                continue
            seen.add(id(handler))
            if not any(
                isinstance(hf, RequestIdFilter)
                for hf in getattr(handler, "filters", [])
            ):
                handler.addFilter(RequestIdFilter())


def _resolve_request_id(scope: dict[str, Any]) -> str:
    for name, value in scope.get("headers", []):
        if name.lower() == b"x-request-id":
            candidate = value.decode("latin1").strip()
            if _INCOMING_ID_RE.match(candidate):
                return candidate
    return new_request_id()


class RequestIDMiddleware:
    """Assigns/echoes a request id and logs a structured access record."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: BodyReceiver, send: Sender) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _resolve_request_id(scope)
        scope["request_id"] = request_id
        token = request_id_var.set(request_id)
        start = time.monotonic()
        status = {"code": 0}

        async def send_wrapper(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message.get("status", 0)
                headers: list[tuple[bytes, bytes]] = message.setdefault("headers", [])
                headers.append((b"x-request-id", request_id.encode("ascii")))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = (time.monotonic() - start) * 1000.0
            logger.info(
                "method=%s path=%s status=%s duration_ms=%.1f request_id=%s",
                scope.get("method", ""),
                scope.get("path", ""),
                status["code"] or "-",
                duration_ms,
                request_id,
            )
            request_id_var.reset(token)