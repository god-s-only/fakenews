"""In-memory prediction log for API export.

Stores the last N predictions made via the API.  Unlike the frontend
history (which lives in localStorage), this is server-side and can be
exported via a future admin endpoint.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

Verdict = Literal["real", "fake", "uncertain"]

_MAX_ENTRIES = 100


@dataclass
class PredictionEntry:
    label: Verdict
    confidence: float
    probability_real: float
    probability_fake: float
    source_type: str
    input_preview: str = ""
    timestamp: str = ""


class PredictionLog:
    """Thread-safe ring-buffer of recent predictions."""

    def __init__(self, max_entries: int = _MAX_ENTRIES) -> None:
        self._buffer: deque[PredictionEntry] = deque(maxlen=max_entries)
        self._lock = threading.Lock()

    def append(self, entry: PredictionEntry) -> None:
        with self._lock:
            self._buffer.append(entry)

    def recent(self, limit: int = 20) -> list[PredictionEntry]:
        with self._lock:
            return list(self._buffer)[-limit:]

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


prediction_log = PredictionLog()
