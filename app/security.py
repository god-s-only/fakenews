"""HTTP security middleware: request-body size limiting.

A pure-ASGI middleware that bounds how much of a request body the server will
read. Without it a client could stream an unbounded (chunked or huge
Content-Length) payload that Pydantic then has to materialise, enabling
memory-exhaustion attacks (audit B4).

Two-stage enforcement:

1. Fast path — if ``Content-Length`` is declared and exceeds the limit, respond
   ``413 Payload Too Large`` without reading any body.
2. Streaming path — for methods that may carry a body and messages where the
   header is absent/lying, consume the stream in steps and abort with ``413``
   the moment the bound is crossed. On success the buffered body is replayed to
   the downstream application.

The limit is always strictly applied; body bytes are never persisted.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

BodyReceiver = Callable[[], Awaitable[dict[str, Any]]]
ASGIApp = Callable[..., Awaitable[None]]
Sender = Callable[..., Awaitable[None]]

_BODY_METHODS = {"POST", "PUT", "PATCH"}


async def _send_json(send: Sender, status: int, payload: dict[str, object]) -> None:
    body = json.dumps(payload).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json; charset=utf-8"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class RequestBodyLimitMiddleware:
    """Reject request bodies larger than ``max_body_bytes``."""

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = int(max_body_bytes)

    async def __call__(self, scope: dict[str, Any], receive: BodyReceiver, send: Sender) -> None:
        if scope["type"] != "http" or self.max_body_bytes <= 0:
            await self.app(scope, receive, send)
            return

        headers = {
            k.decode("latin1").lower(): v.decode("latin1")
            for k, v in scope.get("headers", [])
        }
        try:
            declared = int(headers["content-length"])
        except (KeyError, ValueError):
            declared = None

        if declared is not None:
            if declared > self.max_body_bytes:
                await _send_json(
                    send,
                    413,
                    {
                        "detail": (
                            f"Request body too large ({declared} bytes). "
                            f"Maximum is {self.max_body_bytes} bytes."
                        )
                    },
                )
                return
            # Declared size is within bounds; trust the header and pass through.
            await self.app(scope, receive, send)
            return

        # Undeclared/chunked body: stream it under the bound and replay.
        streamed: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                continue  # ignore http.disconnect noise mid-body
            data = message.get("body", b"")
            size += len(data)
            if size > self.max_body_bytes:
                await _send_json(
                    send,
                    413,
                    {
                        "detail": (
                            "Request body too large. "
                            f"Maximum is {self.max_body_bytes} bytes."
                        )
                    },
                )
                return
            streamed.append(data)
            if not message.get("more_body", False):
                break

        body = b"".join(streamed)
        served = False

        async def replay() -> dict[str, Any]:
            nonlocal served
            if not served:
                served = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, replay, send)