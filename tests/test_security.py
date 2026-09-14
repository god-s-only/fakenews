"""Tests for request-body size limiting and centralized URL length limits."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app, create_app
from app.schemas import UrlRequest
from app.security import RequestBodyLimitMiddleware


def _scope(method="POST", headers=()):
    return {
        "type": "http",
        "method": method,
        "headers": [(k.encode(), v.encode()) for k, v in headers],
        "path": "/predict",
    }


async def _noop_downstream(app_scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


def _counting_downstream(calls):
    async def downstream(app_scope, receive, send):
        calls.append(None)
        await _noop_downstream(app_scope, receive, send)

    return downstream


class _SendSpy:
    def __init__(self):
        self.messages: list[dict[str, Any]] = []

    async def __call__(self, message: dict[str, Any]) -> None:
        self.messages.append(message)


def _response_body(send_spy: _SendSpy) -> bytes:
    for message in send_spy.messages:
        if message["type"] == "http.response.body":
            return message.get("body", b"")
    return b""


async def _call(middleware, scope, chunks, downstream_app=None):
    send_spy = _SendSpy()
    downstream_calls = {"count": 0, "body": b""}

    async def spy(app_scope, receive, send):
        assert app_scope is scope
        downstream_calls["count"] += 1
        msg = await receive()
        downstream_calls["body"] = msg.get("body", b"")
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    def make_receive():
        i = 0
        async def receive():
            nonlocal i
            if i < len(chunks):
                body = chunks[i]
                last = i == len(chunks) - 1
                i += 1
                return {"type": "http.request", "body": body, "more_body": not last}
            return {"type": "http.request", "body": b"", "more_body": False}
        return receive

    app = downstream_app if downstream_app is not None else spy
    await middleware(scope, make_receive(), send_spy)
    return send_spy, downstream_calls


class TestRequestBodyLimitMiddleware:
    def test_content_length_over_limit_413(self):
        middleware = RequestBodyLimitMiddleware(lambda *_: None, max_body_bytes=1_000)
        scope = _scope(headers=[("content-length", "5000")])
        send_spy, downstream = asyncio.run(_call(middleware, scope, [b"x" * 5000]))
        assert send_spy.messages[0]["type"] == "http.response.start"
        assert send_spy.messages[0]["status"] == 413
        assert downstream["count"] == 0

    def test_content_length_within_limit_passes_through(self):
        calls: list[None] = []
        middleware = RequestBodyLimitMiddleware(
            _counting_downstream(calls), max_body_bytes=10_000
        )
        scope = _scope(headers=[("content-length", "5")])
        send_spy, _ = asyncio.run(_call(middleware, scope, [b"hello"]))
        assert calls == [None]  # downstream invoked
        assert send_spy.messages[0]["status"] == 200

    def test_streamed_body_over_limit_413(self):
        middleware = RequestBodyLimitMiddleware(lambda *_: None, max_body_bytes=1_000)
        scope = _scope()  # no content-length -> chunked path
        chunks = [b"0" * 800, b"0" * 400]  # 1200 > 1000
        send_spy, downstream = asyncio.run(_call(middleware, scope, chunks))
        assert send_spy.messages[0]["status"] == 413
        assert downstream["count"] == 0
        # The 413 detail names the limit.
        assert b"Maximum is 1000" in _response_body(send_spy)

    def test_streamed_body_within_limit_replays_to_app(self):
        captured: dict[str, Any] = {}

        async def downstream(app_scope, receive, send):
            captured["body"] = (await receive()).get("body", b"")
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        middleware = RequestBodyLimitMiddleware(downstream, max_body_bytes=100)
        scope = _scope()
        send_spy, _ = asyncio.run(_call(middleware, scope, [b"aa", b"bb", b"cc"]))
        assert captured["body"] == b"aabbcc"
        assert send_spy.messages[0]["status"] == 200

    def test_high_max_rejects_nothing(self):
        calls: list[None] = []
        middleware = RequestBodyLimitMiddleware(
            _counting_downstream(calls), max_body_bytes=0
        )
        scope = _scope(headers=[("content-length", "500000")])
        send_spy, _ = asyncio.run(_call(middleware, scope, [b"x" * 500000]))
        assert calls == [None]  # disabled bounds -> pass-through
        assert send_spy.messages[0]["status"] == 200


class TestEndToEnd:
    def test_real_app_rejects_oversized_body(self, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "MAX_REQUEST_BODY_BYTES", 10_000)
        real = create_app()  # fresh app; body limit read at middleware creation
        client = TestClient(real)
        # Well under the bound: passes middleware, fails on model/schema, never 413.
        ok = client.post("/predict", json={"news": "x" * 9_000})
        assert ok.status_code != 413
        big = client.post(
            "/predict", content=b'{"news":"' + b"x" * 60_000 + b'"}'
        )
        assert big.status_code == 413

    def test_default_url_max_length_is_centralized(self):
        field = UrlRequest.model_json_schema()["properties"]["url"]
        assert field["maxLength"] == 2048

    def test_url_longer_than_max_length_rejected(self):
        client = TestClient(app)
        resp = client.post("/predict-url", json={"url": "http://a.example/" + "b" * 2100})
        assert resp.status_code == 422

    def test_normal_predict_still_works_with_middleware(self):
        import app.main as main_module

        class _Fake:
            is_loaded = True

            def predict(self, text):
                return type(
                    "P",
                    (),
                    {
                        "label": "real",
                        "confidence": 0.9,
                        "probability_real": 0.9,
                        "probability_fake": 0.1,
                        "explanation": [],
                    },
                )()

        original = main_module.state.model
        main_module.state.model = _Fake()
        try:
            client = TestClient(app)
            resp = client.post("/predict", json={"news": "A normal amount of text here."})
            assert resp.status_code == 200
        finally:
            main_module.state.model = original