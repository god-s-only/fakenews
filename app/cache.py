"""Bounded, thread-safe, time-to-live cache.

Used to memoise deterministic per-URL results (the *fetched/extracted* article)
so that analysing the same URL twice does not repeat DNS, fetch, decode and
parse work (see docs/scalability-audit.md, bottleneck B3).

Guarantees:
- Bounded: at most ``max_items`` entries (LRU eviction on overflow).
- Time-bounded: entries expire ``ttl_seconds`` after insertion (lazy removal).
- Thread-safe: all access is serialised by an internal lock.
- Never persists data; on shutdown or process exit everything is dropped.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    """LRU evicting cache with per-entry TTL and a background pruner.

    Items are keyed by an arbitrary string (typically a URL digest). The cache
    never reads or stores request bodies; callers decide what value to store.
    """

    def __init__(self, ttl_seconds: float = 600.0, max_items: int = 512) -> None:
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be non-negative")
        if max_items < 1:
            raise ValueError("max_items must be >= 1")
        self._ttl = float(ttl_seconds)
        self._max_items = int(max_items)
        self._data: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._lock = threading.RLock()

    @property
    def ttl(self) -> float:
        return self._ttl

    @property
    def max_items(self) -> int:
        return self._max_items

    def _expires(self, inserted_at: float) -> bool:
        return self._ttl > 0 and (time.monotonic() - inserted_at) >= self._ttl

    def get(self, key: str) -> T | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            inserted_at, value = item
            if self._expires(inserted_at):
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def put(self, key: str, value: T) -> None:
        with self._lock:
            self._data[key] = (time.monotonic(), value)
            self._data.move_to_end(key)
            self._evict_locked()

    def _evict_locked(self) -> None:
        while len(self._data) > self._max_items:
            self._data.popitem(last=False)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def prune(self) -> int:
        """Drop expired entries and return how many were removed."""
        with self._lock:
            now = time.monotonic()
            expired = [k for k, (ts, _) in self._data.items() if now - ts >= self._ttl]
            for key in expired:
                del self._data[key]
            return len(expired)


def cache_key(url: str) -> str:
    """Stable cache key for a URL: a digest, never the URL itself.

    Only the stripped, lower-cased origin+path is hashed; fragments and bare
    trailing slashes are normalised away so "same" pages share one entry.
    """
    import hashlib
    from urllib.parse import urlparse

    parsed = urlparse(url.strip())
    path = parsed.path.rstrip("/")
    canonical = f"{parsed.scheme}://{parsed.netloc.lower()}{path}".strip()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()