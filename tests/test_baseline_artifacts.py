"""Phase 1 — baseline artifacts remain immutable copies of the originals."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_baseline_model_is_byte_identical_to_production():
    original = ROOT / "my_model.h5"
    baseline = ROOT / "artifacts" / "baseline" / "my_model.h5"
    assert original.exists(), "production model missing"
    assert baseline.exists(), "baseline copy missing"
    assert _sha256(original) == _sha256(baseline)


def test_baseline_vectorizer_is_byte_identical_to_production():
    original = ROOT / "countvectorizer.pkl"
    baseline = ROOT / "artifacts" / "baseline" / "countvectorizer.pkl"
    assert original.exists(), "production vectorizer missing"
    assert baseline.exists(), "baseline copy missing"
    assert _sha256(original) == _sha256(baseline)


def test_baseline_report_and_manifest_exist():
    assert (ROOT / "reports" / "baseline_report.md").exists()
    manifest = (ROOT / "reports" / "baseline_sha256.txt").read_text()
    assert "my_model.h5" in manifest and "countvectorizer.pkl" in manifest