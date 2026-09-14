"""Concurrency tests for the URL fetcher's thread-local, pooled sessions.

Each worker thread owns exactly one ``requests.Session`` (sessions are not
thread-safe), while all ``UrlFetcher`` instances within a thread share it so
TCP/TLS connections are reused. These tests prove session locality and that
concurrent fetch calls never bleed responses across threads.
"""

from __future__ import annotations

import socket
import threading
from concurrent.futures import ThreadPoolExecutor

import requests

from app.scraper import UrlFetcher

DNS_IP = "93.184.216.34"


def _dnslookup(*_):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (DNS_IP, 80))]


def _article_response(text: str) -> requests.Response:
    response = requests.Response()
    response.status_code = 200
    response.url = "https://example.com/article"
    response._content = f"<html><body><article>{text}</article></body></html>".encode()
    response.headers["Content-Type"] = "text/html; charset=utf-8"
    response.headers["Content-Length"] = str(len(response.content))
    return response


def _fetch_article(url: str) -> str:
    return UrlFetcher().fetch_article(url).text


def test_url_fetchers_on_same_thread_share_one_session():
    first = UrlFetcher()
    second = UrlFetcher()
    assert first.session is second.session


def test_distinct_threads_own_distinct_sessions():
    observed: dict[int, object] = {}
    lock = threading.Lock()
    barrier = threading.Barrier(4)

    def _worker():
        barrier.wait()
        with lock:
            observed[threading.get_ident()] = UrlFetcher().session

    threads = [threading.Thread(target=_worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not observed[threads[0].ident] is None
    sessions = set(map(id, observed.values()))
    assert len(sessions) == 4  # one distinct session per thread


def test_concurrent_fetches_are_isolated(monkeypatch):
    import socket as socket_mod
    import app.scraper as scraper_mod

    text_by_path = {
        f"/story/{i}": f"Distinct body number {i} with enough words to "
        f"clear the minimum article length threshold for extraction "
        f"verification purposes across the concurrent worker threads."
        for i in range(6)
    }
    paths_served: list[str] = []
    served_lock = threading.Lock()

    def fake_get(self, url, **kwargs):
        path = url.partition("://")[2].partition("/")[2]
        with served_lock:
            paths_served.append(f"/{path}")
        return _article_response(text_by_path[f"/{path}"])

    monkeypatch.setattr(scraper_mod.socket, "getaddrinfo", _dnslookup)
    monkeypatch.setattr(scraper_mod.requests.Session, "get", fake_get)

    barrier = threading.Barrier(6)

    def _worker(i: int) -> str:
        barrier.wait()
        return _fetch_article(f"http://example.com/story/{i}")

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(_worker, range(6)))

    assert len(results) == 6
    for i, text in enumerate(results):
        assert f"Distinct body number {i}" in text
    assert sorted(paths_served) == [f"/story/{i}" for i in range(6)]


def test_adapter_connection_pool_is_bounded():
    session = UrlFetcher().session
    adapter = session.get_adapter("https://example.com")
    assert adapter._pool_connections > 0
    assert adapter._pool_maxsize >= 1
    assert adapter._pool_maxsize <= 64  # bounded, never unbounded growth


def test_concurrent_bare_origin_still_fetches(monkeypatch):
    import socket as socket_mod
    import app.scraper as scraper_mod

    monkeypatch.setattr(scraper_mod.socket, "getaddrinfo", _dnslookup)

    def fake_get(self, url, **kwargs):
        return _article_response(
            "A clear single-article page about a fact-checked government "
            "statement covering policy details, agency responses and public "
            "reaction across several districts and sources."
        )

    monkeypatch.setattr(scraper_mod.requests.Session, "get", fake_get)

    def _worker(_):
        return _fetch_article("http://example.com/article/concurrent")

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_worker, range(8)))
    assert len(results) == 8
    assert all("government" in t for t in results)