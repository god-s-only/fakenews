"""Safe article URL fetching and text extraction.

Treats URLs as fully untrusted input. Enforces HTTPS/HTTP only, blocks
private/reserved network ranges (SSRF guard), caps response size, applies
connect/read timeouts and redirect limits, extracts the dominant article via
JSON-LD (``NewsArticle``/``Article``) or a semantic HTML fallback, and rejects
pages that do not contain a single news article.

Every failure mode maps to a distinct :attr:`ScrapeError.category` with a
user-friendly message, so the API can surface actionable errors to the frontend
instead of leaking raw exceptions.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import socket
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from app.config import settings

logger = logging.getLogger("fakenews.scraper")

# --------------------------------------------------------------------------- #
# Error categories / messages
# --------------------------------------------------------------------------- #

USER_FACING_MESSAGES: dict[str, str] = {
    "invalid_url": "We couldn't find an article at this URL. Check the link and try again.",
    "http_error": "We couldn't find an article at this URL. Check the link and try again.",
    "dns_failure": "We couldn't find an article at this URL. Check the link and try again.",
    "not_html": "We couldn't find an article at this URL. Check the link and try again.",
    "blocked_domain": "This URL type is not allowed for analysis.",
    "blocked_network": "This URL points to a private network and is blocked.",
    "timeout": "The article could not be retrieved right now. Please try again.",
    "connection_error": "The article could not be retrieved right now. Please try again.",
    "too_many_redirects": "The article could not be retrieved right now. Please try again.",
    "redirect_invalid": "The article could not be retrieved right now. Please try again.",
    "response_read_error": "The article could not be retrieved right now. Please try again.",
    "oversize": "The article could not be retrieved right now. Please try again.",
    "not_article": "This page doesn't appear to contain a single news article.",
    "extraction_failed": "We retrieved the page, but couldn't identify the article content.",
    "generic": "The article could not be retrieved right now. Please try again.",
}


class ScrapeError(Exception):
    """Raised when a URL cannot be fetched or its article text extracted.

    ``message`` is the user-facing text. ``category`` is a stable machine
    identifier for the failure, and ``detail`` holds the technical reason for
    debug logging.
    """

    def __init__(
        self,
        message: str | None = None,
        *,
        category: str = "generic",
        detail: str | None = None,
        final_url: str | None = None,
        redirects: tuple[str, ...] = (),
    ) -> None:
        if message is None:
            message = USER_FACING_MESSAGES.get(category, USER_FACING_MESSAGES["generic"])
        super().__init__(message)
        self.category = category
        self.message = message
        self.detail = detail
        self.final_url = final_url
        self.redirects = redirects


def _error(category: str, detail: str | None = None, **kwargs: Any) -> ScrapeError:
    """Build a :class:`ScrapeError` with the standard user-facing message."""
    return ScrapeError(
        USER_FACING_MESSAGES.get(category, USER_FACING_MESSAGES["generic"]),
        category=category,
        detail=detail,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Extraction constants
# --------------------------------------------------------------------------- #

_UNSUPPORTED_PATTERNS = ("localhost", ".local")

# Minimum character count to consider extracted content a news article.
_MIN_ARTICLE_CHARS = 100
# Minimum JSON-LD article body length before trusting JSON-LD over HTML fallback.
_MIN_JSONLD_BODY_CHARS = 80

_ARTICLE_TYPES = (
    "NewsArticle",
    "Article",
    "Report",
    "BlogPosting",
    "AnalysisNewsArticle",
    "OpinionNewsArticle",
)

# Path prefixes that indicate listing/index pages rather than an article.
_LISTING_PATH_MARKERS = (
    "/category/",
    "/categories/",
    "/tag/",
    "/tags/",
    "/author/",
    "/about",
    "/contact",
    "/careers",
    "/privacy",
    "/terms",
    "/login",
    "/register",
    "/signup",
    "/sign-in",
    "/cart",
    "/search",
    "/feed",
    "/sitemap",
    "/archive",
)


@dataclass
class ExtractResult:
    """Extracted article content ready for downstream analysis."""

    text: str
    title: str = ""
    extraction_method: str = ""
    final_url: str = ""
    redirects: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# JSON-LD helpers
# --------------------------------------------------------------------------- #

def _iter_json_ld_nodes(data: Any) -> Any:
    """Yield every dict node in a JSON-LD document (incl. ``@graph`` lists)."""
    if isinstance(data, dict):
        yield data
        for value in data.values():
            yield from _iter_json_ld_nodes(value)
    elif isinstance(data, list):
        for item in data:
            yield from _iter_json_ld_nodes(item)


def _json_ld_node_is_article(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    node_types = node.get("@type")
    if isinstance(node_types, str):
        return node_types in _ARTICLE_TYPES
    if isinstance(node_types, list):
        return any(isinstance(t, str) and t in _ARTICLE_TYPES for t in node_types)
    return False


class UrlFetcher:
    """Fetches and extracts article text from a validated URL."""

    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.max_redirects = settings.MAX_REDIRECTS

    @staticmethod
    def _headers() -> dict[str, str]:
        return {
            "User-Agent": UrlFetcher.USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

    @staticmethod
    def _timeout() -> tuple[float, float]:
        return (settings.CONNECT_TIMEOUT, settings.REQUEST_TIMEOUT)

    # ------------------------------------------------------------------ #
    # URL validation / SSRF guard
    # ------------------------------------------------------------------ #
    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise _error("invalid_url", detail=f"unsupported scheme {parsed.scheme!r}")
        if not parsed.hostname:
            raise _error("invalid_url", detail="missing hostname")
        hostname = parsed.hostname.lower()
        if any(pattern in hostname for pattern in _UNSUPPORTED_PATTERNS):
            raise _error(
                "blocked_domain", detail=f"hostname {hostname!r} is not allowed"
            )
        self._check_resolved_addresses(hostname)

    def _check_resolved_addresses(self, hostname: str) -> None:
        try:
            infos = socket.getaddrinfo(hostname, None)
        except socket.gaierror as exc:
            raise _error("dns_failure", detail=f"getaddrinfo failed for {hostname}") from exc

        for info in infos:
            try:
                address = info[4][0]
                ip = ipaddress.ip_address(address)
            except ValueError:
                continue
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                raise _error(
                    "blocked_network", detail=f"{hostname} resolves to {address}"
                )

    # ------------------------------------------------------------------ #
    # Fetching
    # ------------------------------------------------------------------ #
    def _fetch_with_redirect_guard(
        self, url: str
    ) -> tuple[requests.Response, tuple[str, ...]]:
        """Fetch a URL, following redirects manually and validating each hop.

        Each hop is re-validated (scheme/host + SSRF) so a public URL cannot
        smuggle the client onto a private network via a redirect. Logs every hop
        and the final status so redirect behaviour is fully traceable at DEBUG.
        """
        current = url
        visited: list[str] = []
        timeout = self._timeout()
        headers = self._headers()

        for _ in range(settings.MAX_REDIRECTS + 1):
            self._validate_url(current)
            visited.append(current)
            logger.debug("Fetching URL: %s", current)

            try:
                response = self.session.get(
                    current,
                    timeout=timeout,
                    headers=headers,
                    allow_redirects=False,
                    stream=True,
                )
            except requests.exceptions.Timeout as exc:
                raise _error(
                    "timeout", detail=f"read/connect timeout fetching {current}"
                ) from exc
            except requests.exceptions.TooManyRedirects as exc:
                raise _error(
                    "too_many_redirects", detail=f"redirect loop at {current}"
                ) from exc
            except requests.exceptions.ConnectionError as exc:
                raise _error(
                    "connection_error", detail=f"connection error for {current}: {exc}"
                ) from exc
            except requests.exceptions.RequestException as exc:
                raise _error(
                    "connection_error", detail=f"request failed for {current}: {exc}"
                ) from exc

            logger.debug(
                "Hop %d: %s -> HTTP %s (content-type=%s)",
                len(visited),
                current,
                response.status_code,
                response.headers.get("content-type"),
            )

            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("location")
                if not location:
                    response.close()
                    raise _error(
                        "redirect_invalid", detail=f"redirect from {current} without Location"
                    )
                current = requests.compat.urljoin(current, location)
                response.close()
                continue

            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as exc:
                status = getattr(exc.response, "status_code", None)
                response.close()
                raise _error(
                    "http_error",
                    detail=f"HTTP {status} for {current}",
                    final_url=current,
                    redirects=tuple(visited),
                ) from exc
            return response, tuple(visited)

        raise _error("too_many_redirects", detail="redirect limit exceeded")

    @staticmethod
    def _check_html_content_type(response: requests.Response) -> None:
        content_type = (response.headers.get("content-type") or "").lower()
        if not content_type:
            return
        if not any(
            marker in content_type for marker in ("html", "xml", "xhtml")
        ):
            raise _error("not_html", detail=f"content-type {content_type!r}")

    def _read_response(self, response: requests.Response) -> bytes:
        """Read the response body, enforcing the configured size cap."""
        content_length = response.headers.get("content-length")
        if (
            content_length
            and content_length.isdigit()
            and int(content_length) > settings.MAX_URL_RESPONSE_SIZE
        ):
            response.close()
            raise _error("oversize", detail=f"content-length {content_length}")

        if isinstance(response, requests.Response):
            chunks: list[bytes] = []
            total = 0
            try:
                for chunk in response.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > settings.MAX_URL_RESPONSE_SIZE:
                        raise _error(
                            "oversize",
                            detail=f"body exceeded {settings.MAX_URL_RESPONSE_SIZE} bytes",
                        )
                    chunks.append(chunk)
            except ScrapeError:
                raise
            except requests.exceptions.ChunkedEncodingError as exc:
                raise _error("response_read_error", detail=str(exc)) from exc
            finally:
                response.close()
            return b"".join(chunks)

        # Test double path (e.g. a mock.Response in the test-suite).
        data = response.content
        response.close()
        if len(data) > settings.MAX_URL_RESPONSE_SIZE:
            raise _error("oversize", detail=f"body is {len(data)} bytes")
        return data

    # ------------------------------------------------------------------ #
    # Extraction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_json_ld(soup: BeautifulSoup) -> tuple[str, str]:
        """Return the longest JSON-LD article body and its headline, if any."""
        best_body = ""
        best_title = ""
        for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
            raw = script.string or script.get_text()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except (ValueError, TypeError):
                continue
            for node in _iter_json_ld_nodes(data):
                if not _json_ld_node_is_article(node):
                    continue
                body = node.get("articleBody") or node.get("text") or ""
                if isinstance(body, str) and len(body.strip()) > len(best_body):
                    best_body = body.strip()
                headline = node.get("headline") or node.get("name") or ""
                if isinstance(headline, str) and headline.strip():
                    best_title = headline.strip()
        return best_body, best_title

    @staticmethod
    def _text_score(element: Any) -> int:
        return len(" ".join(element.get_text(" ", strip=True).split()))

    @staticmethod
    def _container_candidates(soup: BeautifulSoup) -> list[Any]:
        """Return the strongest article containers, most relevant first."""
        preferred: list[Any] = []
        for selector in ("article", "main", '[role="main"]'):
            for element in soup.select(selector):
                preferred.append(element)
        if preferred:
            return preferred
        scored = sorted(
            soup.find_all(["div", "section"]),
            key=UrlFetcher._text_score,
            reverse=True,
        )
        return [el for el in scored if UrlFetcher._text_score(el) >= _MIN_ARTICLE_CHARS]

    @staticmethod
    def _extract_text_from_container(container: Any) -> str:
        paragraphs = container.find_all("p")
        if paragraphs:
            parts = [p.get_text(" ", strip=True) for p in paragraphs]
            return " ".join(" ".join(parts).split())
        return " ".join(container.get_text(" ", strip=True).split())

    @staticmethod
    def _extract_title(soup: BeautifulSoup, container: Any = None) -> str:
        candidates = []
        if container is not None:
            candidates.append(container.find("h1"))
        candidates.append(soup.find("h1"))
        for heading in candidates:
            if heading is not None and heading.get_text(strip=True):
                return " ".join(heading.get_text(strip=True).split())
        if soup.title is not None and soup.title.get_text(strip=True):
            return " ".join(soup.title.get_text(strip=True).split())
        return ""

    def _extract_article(self, html: str, url: str) -> tuple[str, str, str]:
        """Return ``(text, title, method)`` for the dominant article in ``html``."""
        soup = BeautifulSoup(html, "html.parser")
        json_ld_body, json_ld_title = self._extract_json_ld(soup)
        for element in soup(
            ["script", "style", "noscript", "header", "footer", "nav", "aside", "form", "iframe"]
        ):
            element.decompose()

        if len(json_ld_body) >= _MIN_JSONLD_BODY_CHARS:
            return (
                json_ld_body,
                json_ld_title or self._extract_title(soup),
                "json-ld",
            )

        container = None
        for candidate in self._container_candidates(soup):
            if len(self._extract_text_from_container(candidate)) >= _MIN_ARTICLE_CHARS:
                container = candidate
                break

        if container is not None:
            return (
                self._extract_text_from_container(container),
                self._extract_title(soup, container),
                "semantic",
            )

        body = soup.body.get_text(" ", strip=True) if soup.body is not None else ""
        logger.debug("Extraction found only %d chars of body text for %s", len(body), url)
        return " ".join(body.split()), self._extract_title(soup), "fallback"

    @staticmethod
    def _reject_listing_url(url: str) -> None:
        parsed = urlparse(url)
        path = parsed.path.lower()
        for marker in _LISTING_PATH_MARKERS:
            if marker in path:
                raise _error("not_article", detail=f"listing path marker {marker!r} in {url}")
        if path.lower() in ("", "/") and not parsed.query:
            raise _error("not_article", detail=f"site root without query: {url}")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fetch_article(self, url: str) -> ExtractResult:
        """Validate, fetch, extract and classify the article at ``url``."""
        self._validate_url(url)
        response, redirects = self._fetch_with_redirect_guard(url)
        final_url = redirects[-1]

        try:
            self._check_html_content_type(response)
        except ScrapeError as exc:
            response.close()
            exc.final_url = final_url
            exc.redirects = redirects
            raise

        try:
            html_bytes = self._read_response(response)
        except ScrapeError as exc:
            exc.final_url = exc.final_url or final_url
            exc.redirects = exc.redirects or redirects
            raise

        encoding = getattr(response, "encoding", None)
        if not isinstance(encoding, str) or not encoding:
            encoding = "utf-8"
        html_text = html_bytes.decode(encoding, errors="replace")

        self._reject_listing_url(final_url)
        text, title, method = self._extract_article(html_text, final_url)

        if not text.strip():
            raise _error(
                "extraction_failed",
                detail=f"no article text found at {final_url}",
                final_url=final_url,
                redirects=redirects,
            )
        if len(text) < _MIN_ARTICLE_CHARS:
            raise _error(
                "not_article",
                detail=f"only {len(text)} chars extracted from {final_url}",
                final_url=final_url,
                redirects=redirects,
            )

        logger.debug(
            "Extracted %d chars via %s from %s (title=%r; redirects=%s)",
            len(text),
            method,
            final_url,
            title,
            redirects,
        )
        return ExtractResult(
            text=text,
            title=title,
            extraction_method=method,
            final_url=final_url,
            redirects=redirects,
        )


def fetch_article(url: str) -> ExtractResult:
    """Convenience wrapper returning the extracted article for ``url``."""
    return UrlFetcher().fetch_article(url)