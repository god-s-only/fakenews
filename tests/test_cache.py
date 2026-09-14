"""Tests for the bounded TTL cache used to memoise URL extractions."""

from __future__ import annotations

import threading
import time

import pytest

from app.cache import TTLCache, cache_key


class TestTTLCache:
    def test_get_miss_returns_none(self):
        cache = TTLCache(max_items=10)
        assert cache.get("nope") is None

    def test_put_then_get(self):
        cache = TTLCache(max_items=10)
        cache.put("k", {"text": "x"})
        assert cache.get("k") == {"text": "x"}

    def test_ttl_expiry_removes_entry(self):
        cache = TTLCache(ttl_seconds=0.05, max_items=10)
        cache.put("k", "v")
        assert cache.get("k") == "v"
        time.sleep(0.08)
        assert cache.get("k") is None

    def test_max_items_lru_eviction(self):
        cache = TTLCache(ttl_seconds=60, max_items=3)
        for i in range(3):
            cache.put(f"k{i}", i)
        cache.get("k0")  # refresh k0 -> it becomes most recently used
        cache.put("k3", 3)
        # The least recently used item (k1) was evicted, not k0.
        assert cache.get("k0") == 0
        assert cache.get("k1") is None
        assert cache.get("k2") == 2
        assert cache.get("k3") == 3
        assert len(cache) == 3

    def test_zero_ttl_never_expires(self):
        cache = TTLCache(ttl_seconds=0, max_items=5)
        cache.put("k", "v")
        assert cache.get("k") == "v"

    def test_invalid_arguments_rejected(self):
        with pytest.raises(ValueError):
            TTLCache(ttl_seconds=-1)
        with pytest.raises(ValueError):
            TTLCache(max_items=0)

    def test_clear_drops_everything(self):
        cache = TTLCache(max_items=10)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        assert len(cache) == 0
        assert cache.get("a") is None

    def test_prune_removes_only_expired(self):
        cache = TTLCache(ttl_seconds=0.05, max_items=10)
        cache.put("fresh", "f")
        time.sleep(0.15)  # definitively expired
        cache.put("old", "o")  # inserted immediately before prune
        removed = cache.prune()
        assert removed == 1
        assert cache.get("old") == "o"

    def test_concurrent_get_put_is_safe(self):
        cache = TTLCache(ttl_seconds=60, max_items=100)
        errors: list[BaseException] = []
        barrier = threading.Barrier(8)

        def _worker(i: int):
            barrier.wait()
            try:
                for j in range(200):
                    cache.put(f"k{i}-{j % 50}", j)
                    cache.get(f"k{(i + 1) % 8}-{j % 50}")
            except BaseException as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(cache) <= 100


class TestCacheKey:
    def test_key_is_a_sha256_digest(self):
        key = cache_key("https://Example.com/Article")
        assert len(key) == 64
        int(key, 16)  # hex

    def test_key_normalises_trailing_slash_and_fragment(self):
        assert cache_key("https://example.com/path") == cache_key(
            "https://example.com/path/"
        )
        assert cache_key("https://example.com/path") == cache_key(
            "https://example.com/path#section"
        )

    def test_key_normalises_host_case(self):
        assert cache_key("https://Example.COM/article") == cache_key(
            "https://example.com/article"
        )

    def test_key_distinguishes_paths(self):
        assert cache_key("https://example.com/a") != cache_key(
            "https://example.com/b"
        )


class TestUrlExtractionCaching_EndToEnd:
    def test_second_analysis_is_served_from_cache(self):
        """The route must not re-fetch a URL analysed moments ago."""
        from unittest import mock

        from fastapi.testclient import TestClient

        from app.main import app
        from app.scraper import ExtractResult
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

        main_module.state.model = _Fake()
        main_module.state.url_cache.clear()

        article = (
            "Officials confirmed the new bridge funding plan yesterday with "
            "bi-partisan backing and a construction timeline of three years."
        )

        with mock.patch(
            "app.main.fetch_article",
            return_value=ExtractResult(
                text=article, title="Bridge Plan", final_url="http://example.com/news/bridge"
            ),
        ) as fake_fetch:
            client = TestClient(app)
            first = client.post(
                "/predict-url", json={"url": "http://example.com/news/bridge"}
            )
            second = client.post(
                "/predict-url", json={"url": "http://example.com/news/bridge"}
            )
        assert first.status_code == 200 and second.status_code == 200
        assert fake_fetch.call_count == 1  # cached on the second call

    def test_cache_can_be_disabled(self, monkeypatch):
        from unittest import mock

        from fastapi.testclient import TestClient

        from app.config import settings
        from app.main import app
        from app.scraper import ExtractResult
        import app.main as main_module

        monkeypatch.setattr(settings, "CACHE_URL_ENABLED", False)

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

        main_module.state.model = _Fake()
        main_module.state.url_cache.clear()

        article = (
            "The mayor announced infrastructure spending for the eastern "
            "districts spanning road repairs, rail lines and water systems."
        )

        with mock.patch(
            "app.main.fetch_article",
            return_value=ExtractResult(
                text=article, title="Spending", final_url="http://example.com/a/b"
            ),
        ) as fake_fetch:
            client = TestClient(app)
            client.post("/predict-url", json={"url": "http://example.com/a/b"})
            client.post("/predict-url", json={"url": "http://example.com/a/b"})
        assert fake_fetch.call_count == 2  # disabled -> every call fetches