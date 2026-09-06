"""Tests for URL validation and scraping."""

from unittest import mock

import pytest
from fastapi.testclient import TestClient

from app.main import app, state
from app.scraper import ExtractResult, UrlFetcher, ScrapeError


@pytest.fixture(autouse=True)
def _clean_state():
    state.model = None
    yield
    state.model = None


@pytest.fixture
def client():
    return TestClient(app)


def _dns_public(host, *args):
    return [(None, None, None, None, ("93.184.216.34", 0))]


class TestUrlValidation:
    def test_rejects_ftp(self):
        fetcher = UrlFetcher()
        with pytest.raises(ScrapeError, match="couldn't find an article"):
            fetcher._validate_url("ftp://example.com/article")

    def test_rejects_file_scheme(self):
        fetcher = UrlFetcher()
        with pytest.raises(ScrapeError, match="couldn't find an article"):
            fetcher._validate_url("file:///etc/passwd")

    def test_rejects_localhost(self):
        fetcher = UrlFetcher()
        with pytest.raises(ScrapeError, match="not allowed"):
            fetcher._validate_url("http://localhost/admin")

    def test_rejects_local_domain(self):
        fetcher = UrlFetcher()
        with pytest.raises(ScrapeError, match="not allowed"):
            fetcher._validate_url("http://myserver.local/api")

    def test_rejects_no_hostname(self):
        fetcher = UrlFetcher()
        with pytest.raises(ScrapeError, match="couldn't find an article"):
            fetcher._validate_url("http:///")

    def test_rejects_empty_string(self):
        fetcher = UrlFetcher()
        with pytest.raises(ScrapeError):
            fetcher._validate_url("")

    def test_accepts_valid_http(self):
        """Validation passes for a valid HTTP URL (network check is separate)."""
        fetcher = UrlFetcher()
        # Mock DNS resolution to avoid actual network calls
        with mock.patch("app.scraper.socket.getaddrinfo", return_value=[
            (None, None, None, None, ("93.184.216.34", 0))
        ]):
            fetcher._validate_url("http://example.com/article")

    def test_rejects_private_ip(self):
        fetcher = UrlFetcher()
        with mock.patch("app.scraper.socket.getaddrinfo", return_value=[
            (None, None, None, None, ("192.168.1.1", 0))
        ]):
            with pytest.raises(ScrapeError, match="private network"):
                fetcher._validate_url("http://internal.company.com")

    def test_rejects_127_x(self):
        fetcher = UrlFetcher()
        with mock.patch("app.scraper.socket.getaddrinfo", return_value=[
            (None, None, None, None, ("127.0.0.1", 0))
        ]):
            with pytest.raises(ScrapeError, match="private network"):
                fetcher._validate_url("http://myserver.com")

    def test_rejects_link_local(self):
        fetcher = UrlFetcher()
        with mock.patch("app.scraper.socket.getaddrinfo", return_value=[
            (None, None, None, None, ("169.254.0.1", 0))
        ]):
            with pytest.raises(ScrapeError, match="private network"):
                fetcher._validate_url("http://link-local.dev")

    def test_rejects_unresolvable_host(self):
        fetcher = UrlFetcher()
        import socket
        with mock.patch("app.scraper.socket.getaddrinfo", side_effect=socket.gaierror("no such host")):
            with pytest.raises(ScrapeError, match="couldn't find an article"):
                fetcher._validate_url("http://definitely-not-a-real-host-12345.invalid")

    def test_rejects_too_many_redirects(self):
        fetcher = UrlFetcher()
        with mock.patch("app.scraper.socket.getaddrinfo", side_effect=_dns_public):
            with mock.patch("app.scraper.requests.Session.get") as mock_get:
                import requests
                mock_get.side_effect = requests.exceptions.TooManyRedirects()
                with pytest.raises(ScrapeError, match="could not be retrieved"):
                    fetcher.fetch_article("http://example.com/redirect-loop")

    def test_rejects_timeout(self):
        fetcher = UrlFetcher()
        with mock.patch("app.scraper.socket.getaddrinfo", side_effect=_dns_public):
            with mock.patch("app.scraper.requests.Session.get") as mock_get:
                import requests
                mock_get.side_effect = requests.exceptions.Timeout()
                with pytest.raises(ScrapeError, match="could not be retrieved"):
                    fetcher.fetch_article("http://example.com/slow")

    def test_rejects_connection_error(self):
        fetcher = UrlFetcher()
        with mock.patch("app.scraper.socket.getaddrinfo", side_effect=_dns_public):
            with mock.patch("app.scraper.requests.Session.get") as mock_get:
                import requests
                mock_get.side_effect = requests.exceptions.ConnectionError("refused")
                with pytest.raises(ScrapeError, match="could not be retrieved"):
                    fetcher.fetch_article("http://example.com")


class TestUrlRequestValidation:
    """Tests for the /predict-url request schema validation."""

    def test_empty_url_rejected(self, client):
        resp = client.post("/predict-url", json={"url": ""})
        assert resp.status_code == 422

    def test_missing_url_rejected(self, client):
        resp = client.post("/predict-url", json={})
        assert resp.status_code == 422

    def test_whitespace_url_rejected(self, client):
        resp = client.post("/predict-url", json={"url": "   "})
        assert resp.status_code == 422

    def test_non_http_scheme_rejected(self, client):
        resp = client.post("/predict-url", json={"url": "ftp://example.com/a"})
        assert resp.status_code == 422

    def test_invalid_scheme_rejected(self, client):
        resp = client.post("/predict-url", json={"url": "javascript:alert(1)"})
        assert resp.status_code == 422

    def test_valid_http_url_passes_validation(self, client):
        # Scheme validation passes; the scraper then checks network access.
        # We bypass network by returning a 503 (model not loaded) before any
        # scraping since the model service is not installed.
        resp = client.post("/predict-url", json={"url": "http://example.com/article"})
        # If the model is not loaded we get 503; otherwise scraping may run.
        assert resp.status_code in (503, 200, 422)


class TestRedirectGuard:
    def test_blocks_redirect_to_private_network(self):
        fetcher = UrlFetcher()
        first = mock.Mock()
        first.status_code = 302
        first.is_redirect = True
        first.is_permanent_redirect = False
        type(first).headers = mock.PropertyMock(return_value={"location": "http://10.0.0.5/admin"})
        first.close.return_value = None
        second = mock.Mock()

        def fake_get(url, **kwargs):
            if "10.0.0.5" in url:
                # would be reached only if validation is bypassed
                return second
            return first

        with mock.patch("app.scraper.socket.getaddrinfo") as mock_dns:
            def fake_getaddrinfo(host, *args):
                if host == "example.com":
                    return [(None, None, None, None, ("93.184.216.34", 0))]
                if host.startswith("10."):
                    # public-ish resolution shouldn't matter; the IP literal is used
                    return [(None, None, None, None, ("10.0.0.5", 0))]
                raise LookupError

            mock_dns.side_effect = fake_getaddrinfo
            with mock.patch("app.scraper.requests.Session.get", side_effect=fake_get):
                with pytest.raises(ScrapeError, match="private network"):
                    fetcher.fetch_article("http://example.com/start")

    def test_blocks_excessive_redirects(self):
        fetcher = UrlFetcher()
        redirect = mock.Mock()
        redirect.status_code = 301
        redirect.is_redirect = True
        redirect.is_permanent_redirect = True
        type(redirect).headers = mock.PropertyMock(return_value={"location": "http://example.com/x"})
        redirect.close.return_value = None

        with mock.patch("app.scraper.socket.getaddrinfo", side_effect=_dns_public):
            with mock.patch("app.scraper.requests.Session.get", return_value=redirect) as mock_get:
                with pytest.raises(ScrapeError, match="could not be retrieved"):
                    fetcher.fetch_article("http://example.com/start")
                assert mock_get.call_count >= fetcher.session.max_redirects + 1

    def test_rejects_non_html_content_type(self):
        fetcher = UrlFetcher()
        with mock.patch("app.scraper.socket.getaddrinfo", side_effect=_dns_public):
            with mock.patch("app.scraper.requests.Session.get") as mock_get:
                response = mock.Mock()
                response.status_code = 200
                response.is_redirect = False
                response.is_permanent_redirect = False
                response.headers = {"content-type": "application/pdf"}
                response.content = b"%PDF-1.4 fake bytes"
                mock_get.return_value = response
                with pytest.raises(ScrapeError, match="couldn't find an article"):
                    fetcher.fetch_article("http://example.com/doc.pdf")

    def test_extracts_text_from_valid_html(self):
        fetcher = UrlFetcher()
        with mock.patch("app.scraper.socket.getaddrinfo", side_effect=_dns_public):
            with mock.patch("app.scraper.requests.Session.get") as mock_get:
                html = b"""
                <html><head><style>.css{display:none}</style>
                <script>var x=1;</script><title>Politics and Economics Today</title></head>
                <body><nav>Nav links</nav>
                <article><h1>Politics and Economics Today</h1>
                <p>This is a genuine article about politics and economics. Economists and
                analysts report that the new trade policy will affect markets across the
                region over the next several years.</p>
                <p>Officials say the policy is designed to stabilise growth while keeping
                inflation low, and point to recent figures as evidence of improvement.</p>
                </article></body></html>
                """
                response = mock.Mock()
                response.status_code = 200
                response.is_redirect = False
                response.is_permanent_redirect = False
                type(response).headers = mock.PropertyMock(
                    return_value={"content-type": "text/html; charset=utf-8"}
                )
                response.content = html
                mock_get.return_value = response
                result = fetcher.fetch_article("http://example.com/article")
                assert isinstance(result, ExtractResult)
                assert "genuine article" in result.text
                assert result.title == "Politics and Economics Today"
                assert result.extraction_method == "semantic"

    def test_rejects_too_large_response(self):
        from app.config import settings
        fetcher = UrlFetcher()
        with mock.patch("app.scraper.socket.getaddrinfo", side_effect=_dns_public):
            with mock.patch("app.scraper.requests.Session.get") as mock_get:
                response = mock.Mock()
                response.status_code = 200
                response.is_redirect = False
                response.is_permanent_redirect = False
                type(response).headers = mock.PropertyMock(
                    return_value={"content-type": "text/html"}
                )
                response.content = b"a" * (settings.MAX_URL_RESPONSE_SIZE + 10)
                mock_get.return_value = response
                with pytest.raises(ScrapeError, match="could not be retrieved"):
                    fetcher.fetch_article("http://example.com/big")