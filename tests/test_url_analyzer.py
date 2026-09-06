"""End-to-end tests for the URL analysis pipeline.

Covers fetching, redirects (incl. HTTP->HTTPS), error categories and
user-facing messages, JSON-LD and semantic extraction, non-article rejection,
the /predict-url response contract, and manual-vs-URL pipeline parity.
"""

from unittest import mock

import pytest
from fastapi.testclient import TestClient

from app.main import app, state
from app.model import Prediction
from app.scraper import ExtractResult, ScrapeError, UrlFetcher

ARTICLE_URL = "https://example.com/news/article"

ARTICLE_HTML = b"""<!DOCTYPE html>
<html><head><title>Climate Study Released</title>
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"NewsArticle",
 "headline":"Climate Study Released",
 "articleBody":"Scientists released a comprehensive study of temperatures. The report found that the average temperature increased steadily over the past decade across multiple regions and seasons around the world."}
</script></head>
<body><article><h1>Climate Study Released</h1>
<p>Scientists released a comprehensive study of temperatures.</p>
<p>The report found the average temperature increased steadily over the past decade.</p>
</article></body></html>
"""

SEMANTIC_HTML = b"""<!DOCTYPE html>
<html><head><title>Local Election Coverage</title><style>.x{}</style></head>
<body><nav>Home News Sports</nav>
<div class="content"><h1>Local Election Coverage</h1>
<p>The polling stations opened at eight o'clock and will remain open until the evening hours for all registered voters.</p>
<p>Voters are deciding on the new city budget, which officials say will fund public transport improvements across the whole region for the coming five years.</p>
</div><footer>Copyright</footer></body></html>
"""

HOMEPAGE_HTML = b"""<html><head><title>The Daily Paper</title></head>
<body><h1>The Daily Paper</h1><p>Welcome</p>
<a href="/a">Story one</a><a href="/b">Story two</a><a href="/c">Story three</a>
</body></html>
"""

SHORT_HTML = b"""<html><body><article>
<h1>Tiny piece</h1><p>Just a short note.</p></article></body></html>
"""

EMPTY_HTML = b"<html><body></body></html>"


@pytest.fixture(autouse=True)
def _clean_state():
    state.model = None
    yield
    state.model = None


def _article_response(body: bytes = ARTICLE_HTML, status: int = 200):
    response = mock.Mock()
    response.status_code = status
    response.is_redirect = False
    response.is_permanent_redirect = False
    type(response).headers = mock.PropertyMock(
        return_value={"content-type": "text/html; charset=utf-8"}
    )
    response.content = body
    response.close = mock.Mock()
    return response


def _redirect_response(url: str, status: int = 301):
    response = mock.Mock()
    response.status_code = status
    response.is_redirect = True
    response.is_permanent_redirect = True
    type(response).headers = mock.PropertyMock(return_value={"location": url})
    response.close = mock.Mock()
    return response


def _dnslookup(host: str, *args):
    return [(None, None, None, None, ("93.184.216.34", 0))]


class _FakeService:
    def __init__(self, prob_real: float = 0.7) -> None:
        self._prob_real = prob_real
        self.calls: list[str] = []

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def model_is_loaded(self) -> bool:
        return True

    @property
    def vectorizer_is_loaded(self) -> bool:
        return True

    def predict(self, raw_text: str) -> Prediction:
        self.calls.append(raw_text)
        prob_real = round(self._prob_real, 6)
        return Prediction(
            probability_real=prob_real,
            probability_fake=round(1.0 - prob_real, 6),
            label="real" if prob_real >= 0.5 else "fake",
            confidence=round(max(prob_real, 1.0 - prob_real) * 100.0, 2),
            explanation=[],
        )


class TestValidArticleExtraction:
    def test_json_ld_extraction(self):
        with mock.patch("app.scraper.socket.getaddrinfo", side_effect=_dnslookup):
            with mock.patch(
                "app.scraper.requests.Session.get", return_value=_article_response()
            ):
                result = UrlFetcher().fetch_article(ARTICLE_URL)
        assert isinstance(result, ExtractResult)
        assert result.extraction_method == "json-ld"
        assert result.title == "Climate Study Released"
        assert "comprehensive study" in result.text
        assert len(result.text) >= 100

    def test_semantic_extraction_fallback(self):
        with mock.patch("app.scraper.socket.getaddrinfo", side_effect=_dnslookup):
            with mock.patch(
                "app.scraper.requests.Session.get",
                return_value=_article_response(SEMANTIC_HTML),
            ):
                result = UrlFetcher().fetch_article(ARTICLE_URL)
        assert result.extraction_method == "semantic"
        assert result.title == "Local Election Coverage"
        assert "polling stations" in result.text
        assert "city budget" in result.text


class TestHttpErrors:
    def test_404_maps_to_http_error(self):
        import requests

        response = _article_response(status=404)

        def _raise_for_status():
            raise requests.exceptions.HTTPError("404 Client Error")

        response.raise_for_status = _raise_for_status
        with mock.patch("app.scraper.socket.getaddrinfo", side_effect=_dnslookup):
            with mock.patch("app.scraper.requests.Session.get", return_value=response):
                with pytest.raises(ScrapeError) as excinfo:
                    UrlFetcher().fetch_article(ARTICLE_URL)
        error = excinfo.value
        assert error.category == "http_error"
        assert "couldn't find an article" in str(error)
        assert error.final_url == ARTICLE_URL

    def test_dns_failure_maps_to_dns_failure(self):
        import socket

        with mock.patch(
            "app.scraper.socket.getaddrinfo",
            side_effect=socket.gaierror("no such host"),
        ):
            with pytest.raises(ScrapeError) as excinfo:
                UrlFetcher().fetch_article("https://no-such-host-xyz.invalid/a")
        assert excinfo.value.category == "dns_failure"
        assert "couldn't find an article" in str(excinfo.value)


class TestRedirects:
    def test_http_to_https_redirect_is_followed(self):
        fetcher = UrlFetcher()
        start = "http://example.com/story"

        def fake_get(url, **kwargs):
            if url == start:
                return _redirect_response("https://example.com/news/story-final")
            assert url == "https://example.com/news/story-final"
            return _article_response()

        with mock.patch("app.scraper.socket.getaddrinfo", side_effect=_dnslookup):
            with mock.patch("app.scraper.requests.Session.get", side_effect=fake_get):
                result = fetcher.fetch_article(start)
        assert result.final_url == "https://example.com/news/story-final"
        assert result.redirects == (start, "https://example.com/news/story-final")
        assert "comprehensive study" in result.text

    def test_redirect_without_location_rejected(self):
        response = mock.Mock()
        response.status_code = 302
        response.is_redirect = True
        response.is_permanent_redirect = False
        type(response).headers = mock.PropertyMock(return_value={})
        response.close = mock.Mock()
        with mock.patch("app.scraper.socket.getaddrinfo", side_effect=_dnslookup):
            with mock.patch("app.scraper.requests.Session.get", return_value=response):
                with pytest.raises(ScrapeError) as excinfo:
                    UrlFetcher().fetch_article(ARTICLE_URL)
        assert excinfo.value.category == "redirect_invalid"
        assert "could not be retrieved" in str(excinfo.value)

    def test_redirect_loop_hits_limit(self):
        fetcher = UrlFetcher()
        with mock.patch("app.scraper.socket.getaddrinfo", side_effect=_dnslookup):
            with mock.patch(
                "app.scraper.requests.Session.get",
                return_value=_redirect_response("https://example.com/loop"),
            ) as mock_get:
                with pytest.raises(ScrapeError) as excinfo:
                    fetcher.fetch_article(ARTICLE_URL)
        assert excinfo.value.category == "too_many_redirects"
        assert "could not be retrieved" in str(excinfo.value)
        assert mock_get.call_count >= fetcher.session.max_redirects + 1


class TestNonArticleRejection:
    def test_category_url_rejected(self):
        with mock.patch("app.scraper.socket.getaddrinfo", side_effect=_dnslookup):
            with mock.patch(
                "app.scraper.requests.Session.get", return_value=_article_response()
            ):
                with pytest.raises(ScrapeError) as excinfo:
                    UrlFetcher().fetch_article("https://example.com/category/science")
        assert excinfo.value.category == "not_article"
        assert "doesn't appear to contain a single news article" in str(excinfo.value)

    def test_site_root_rejected(self):
        with mock.patch("app.scraper.socket.getaddrinfo", side_effect=_dnslookup):
            with mock.patch(
                "app.scraper.requests.Session.get", return_value=_article_response()
            ):
                with pytest.raises(ScrapeError) as excinfo:
                    UrlFetcher().fetch_article("https://example.com")
        assert excinfo.value.category == "not_article"
        assert "doesn't appear to contain a single news article" in str(excinfo.value)

    def test_homepage_content_rejected(self):
        with mock.patch("app.scraper.socket.getaddrinfo", side_effect=_dnslookup):
            with mock.patch(
                "app.scraper.requests.Session.get",
                return_value=_article_response(HOMEPAGE_HTML),
            ):
                with pytest.raises(ScrapeError) as excinfo:
                    UrlFetcher().fetch_article("https://example.com/page")
        assert excinfo.value.category == "not_article"

    def test_short_content_rejected(self):
        with mock.patch("app.scraper.socket.getaddrinfo", side_effect=_dnslookup):
            with mock.patch(
                "app.scraper.requests.Session.get",
                return_value=_article_response(SHORT_HTML),
            ):
                with pytest.raises(ScrapeError) as excinfo:
                    UrlFetcher().fetch_article(ARTICLE_URL)
        assert excinfo.value.category == "not_article"

    def test_empty_content_rejected_as_extraction_failure(self):
        with mock.patch("app.scraper.socket.getaddrinfo", side_effect=_dnslookup):
            with mock.patch(
                "app.scraper.requests.Session.get",
                return_value=_article_response(EMPTY_HTML),
            ):
                with pytest.raises(ScrapeError) as excinfo:
                    UrlFetcher().fetch_article(ARTICLE_URL)
        assert excinfo.value.category == "extraction_failed"
        assert "couldn't identify the article content" in str(excinfo.value)


class TestPredictUrlContract:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def _install_fake_service(self):
        service = _FakeService()
        state.model = service
        return service

    def test_success_response_contract(self, client):
        service = self._install_fake_service()
        with mock.patch(
            "app.main.fetch_article",
            return_value=ExtractResult(
                text="Government announces new economic policy with broad support.",
                title="Policy Announcement",
                final_url=ARTICLE_URL,
            ),
        ):
            resp = client.post("/predict-url", json={"url": ARTICLE_URL})
        assert resp.status_code == 200
        body = resp.json()
        assert body["label"] in ("real", "fake", "uncertain")
        assert isinstance(body["confidence"], float)
        assert body["probability_real"] + body["probability_fake"] == pytest.approx(100.0)
        assert body["source_type"] == "url"
        assert body["source"] == ARTICLE_URL
        assert body["page_title"] == "Policy Announcement"
        assert "explanation" in body
        assert service.calls == ["Government announces new economic policy with broad support."]

    def test_not_article_error_has_category(self, client):
        self._install_fake_service()
        with mock.patch(
            "app.main.fetch_article",
            side_effect=ScrapeError(
                "This page doesn't appear to contain a single news article.",
                category="not_article",
            ),
        ):
            resp = client.post("/predict-url", json={"url": "https://example.com/category"})
        assert resp.status_code == 422
        body = resp.json()
        assert body["category"] == "not_article"
        assert "doesn't appear" in body["detail"]

    def test_blocked_network_error_has_category(self, client):
        self._install_fake_service()
        with mock.patch(
            "app.main.fetch_article",
            side_effect=ScrapeError(
                "This URL points to a private network and is blocked.",
                category="blocked_network",
            ),
        ):
            resp = client.post("/predict-url", json={"url": "http://10.0.0.5/"})
        assert resp.status_code == 422
        assert resp.json()["category"] == "blocked_network"

    def test_404_error_has_category(self, client):
        self._install_fake_service()
        with mock.patch(
            "app.main.fetch_article",
            side_effect=ScrapeError(
                "We couldn't find an article at this URL. Check the link and try again.",
                category="http_error",
            ),
        ):
            resp = client.post(
                "/predict-url", json={"url": "https://example.com/not-found"}
            )
        assert resp.status_code == 422
        body = resp.json()
        assert body["category"] == "http_error"
        assert "couldn't find an article" in body["detail"]


class TestManualVsUrlParity:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_shared_pipeline_receives_identical_text(self, client):
        service = _FakeService()
        state.model = service
        text = "Officials announced the new hospital wing will open next month."
        with mock.patch(
            "app.main.fetch_article",
            return_value=ExtractResult(text=text, title="News", final_url=ARTICLE_URL),
        ):
            url_resp = client.post("/predict-url", json={"url": ARTICLE_URL})
            text_resp = client.post("/predict", json={"news": text})
        assert url_resp.status_code == 200
        assert text_resp.status_code == 200
        assert service.calls == [text, text]
        url_body, text_body = url_resp.json(), text_resp.json()
        assert url_body["probability_real"] == text_body["probability_real"]
        assert url_body["probability_fake"] == text_body["probability_fake"]
        assert url_body["label"] == text_body["label"]
        assert url_body["source_type"] == "url"
        assert text_body["source_type"] == "text"


class TestPrivateNetworkGuard:
    def test_internal_ip_literal_rejected(self):
        def _dnslookup_private(host: str, *args):
            return [(None, None, None, None, ("10.0.0.5", 0))]

        with mock.patch("app.scraper.socket.getaddrinfo", side_effect=_dnslookup_private):
            with pytest.raises(ScrapeError) as excinfo:
                UrlFetcher().fetch_article("http://10.0.0.5/admin")
        assert excinfo.value.category == "blocked_network"
        assert "private network" in str(excinfo.value)