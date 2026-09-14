"""Performance regression guards (NOT benchmarks).

Thresholds are deliberately generous (well above Phase 10 baseline medians:
prediction 45-50 ms, model load 36 ms, mocked URL pipeline 2.5 ms) so CI stays
deterministic while still catching order-of-magnitude regressions (e.g. a
per-request feature-name rebuild, an unbounded fetch loop, an accidental
re-tokenisation per call). For precise numbers run scripts/benchmark.py.
"""

from __future__ import annotations

import time
from unittest import mock

import pytest

from app.config import settings
from app.model import ModelService
from app.scraper import UrlFetcher

ARTICLE_TEXT = (
    "Government officials announced a new economic package on Tuesday that "
    "will fund infrastructure projects across several regions, including road "
    "repairs, rail upgrades and water systems, while legislators from both "
    "parties said the plan enjoys broad bipartisan support and could be "
    "approved before the end of the quarter, with officials noting continued "
    "progress on job growth and regional development initiatives."
)

LARGE_TEXT = (
    "The city council voted to approve the revised transportation plan for the "
    "coming fiscal year, a decision that follows months of public hearings and "
    "consultations with neighborhood associations. " * 200
)


def _assets_available() -> bool:
    return settings.model_file.exists() and settings.vectorizer_file.exists()


skip_no_assets = pytest.mark.skipif(
    not _assets_available(),
    reason="Model/vectorizer assets are not available",
)


@pytest.fixture(scope="module")
def service() -> ModelService:
    return ModelService(settings.model_file, settings.vectorizer_file).load()


@skip_no_assets
class TestPredictionLatency:
    def test_single_prediction_under_threshold(self, service):
        start = time.monotonic()
        service.predict(ARTICLE_TEXT)
        assert (time.monotonic() - start) < 1.0  # baseline ~0.05 s

    def test_explanation_included_in_latency(self, service):
        start = time.monotonic()
        prediction = service.predict(ARTICLE_TEXT)
        assert (time.monotonic() - start) < 1.0
        assert prediction.explanation  # explainability path exercised

    def test_large_input_under_threshold(self, service):
        start = time.monotonic()
        service.predict(LARGE_TEXT)
        assert (time.monotonic() - start) < 5.0  # 20k chars, generous bound

    def test_consecutive_predictions_stay_fast(self, service):
        start = time.monotonic()
        for _ in range(20):
            service.predict(ARTICLE_TEXT)
        elapsed = time.monotonic() - start
        assert elapsed < 5.0  # would blow up if features rebuilt per request


@skip_no_assets
class TestModelLoadLatency:
    def test_model_load_under_threshold(self):
        start = time.monotonic()
        svc = ModelService(settings.model_file, settings.vectorizer_file).load()
        assert (time.monotonic() - start) < 5.0  # baseline ~0.04 s + describe
        assert svc.model_ready


def _dnslookup(*_):
    import socket

    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))]


def _html_response(body_text: str):
    import requests

    response = requests.Response()
    response.status_code = 200
    response.url = "https://example.com/article"
    response._content = (
        f"<html><head><title>T</title></head><body><article>{body_text}</article>"
        "</body></html>"
    ).encode()
    response.headers["Content-Type"] = "text/html; charset=utf-8"
    response.headers["Content-Length"] = str(len(response.content))
    return response


class TestUrlPipelineLatency:
    def test_mocked_url_fetch_plus_extract_under_threshold(self):
        import app.scraper as scraper_mod

        with mock.patch("app.scraper.socket.getaddrinfo", side_effect=_dnslookup):
            with mock.patch.object(
                scraper_mod.requests.Session,
                "get",
                return_value=_html_response(ARTICLE_TEXT),
            ):
                start = time.monotonic()
                result = UrlFetcher().fetch_article("https://example.com/article")
        assert (time.monotonic() - start) < 1.0  # baseline ~0.003 s
        assert "Government officials" in result.text

    def test_div_heavy_page_parse_under_threshold(self):
        import app.scraper as scraper_mod

        # Simulate a page with many short divs (worst case for extraction).
        noise = "".join(
            f'<div class="side">{i}</div><nav itemid="{i}">link {i}</nav>'
            for i in range(400)
        )
        html = (
            "<html><body>"
            + noise
            + f"<div id='content'>{ARTICLE_TEXT}</div>"
            + "</body></html>"
        )

        response_factory = mock.Mock(return_value=_html_response(html))
        with mock.patch("app.scraper.socket.getaddrinfo", side_effect=_dnslookup):
            with mock.patch.object(
                scraper_mod.requests.Session, "get", response_factory
            ):
                start = time.monotonic()
                result = UrlFetcher().fetch_article("https://example.com/busy")
        assert (time.monotonic() - start) < 2.0
        assert "Government officials" in result.text