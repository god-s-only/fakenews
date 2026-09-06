"""Tests for the server-side prediction log."""

from app.prediction_log import PredictionEntry, PredictionLog, prediction_log


def test_append_and_recent():
    log = PredictionLog(max_entries=5)
    for i in range(3):
        log.append(PredictionEntry(
            label="real",
            confidence=95.0,
            probability_real=0.95,
            probability_fake=0.05,
            source_type="text",
        ))
    assert len(log) == 3
    recent = log.recent(limit=10)
    assert len(recent) == 3


def test_ring_buffer_eviction():
    log = PredictionLog(max_entries=2)
    log.append(PredictionEntry(label="real", confidence=80.0,
                               probability_real=0.8, probability_fake=0.2,
                               source_type="text"))
    log.append(PredictionEntry(label="fake", confidence=90.0,
                               probability_real=0.1, probability_fake=0.9,
                               source_type="text"))
    log.append(PredictionEntry(label="uncertain", confidence=50.0,
                               probability_real=0.5, probability_fake=0.5,
                               source_type="text"))
    assert len(log) == 2
    recent = log.recent()
    assert recent[0].label == "fake"
    assert recent[1].label == "uncertain"


def test_clear():
    log = PredictionLog()
    log.append(PredictionEntry(label="real", confidence=80.0,
                               probability_real=0.8, probability_fake=0.2,
                               source_type="text"))
    log.clear()
    assert len(log) == 0
    assert log.recent() == []


def test_recent_limit():
    log = PredictionLog(max_entries=50)
    for _ in range(10):
        log.append(PredictionEntry(label="real", confidence=80.0,
                                   probability_real=0.8, probability_fake=0.2,
                                   source_type="text"))
    assert len(log.recent(limit=5)) == 5
    assert len(log.recent(limit=20)) == 10


def test_global_log_exists():
    assert prediction_log is not None
    assert isinstance(prediction_log, PredictionLog)
