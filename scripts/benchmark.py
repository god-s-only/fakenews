"""Deterministic baseline measurement for the Fake News Detector.

Captures machine-readable engineering baselines that the Phase 10
scalability work is measured against:

* ``tests``   — pytest collection count.
* ``startup`` — wall-clock time to create the FastAPI application.
* ``model``   — model load time and prediction latency on the promoted
  TF-IDF LogisticRegression detector (real frozen artifacts).
* ``url``     — URL analysis pipeline latency using an in-process fixture
  (no network; DNS and HTTP are mocked) so the measurement is
  deterministic and offline.
* ``memory``  — resident memory of the app before and after model load
  (measured in a fresh subprocess for isolation).

Every subcommand writes a JSON file into ``reports/baseline/`` and prints a
short human-readable summary. Timing uses ``time.perf_counter`` for monotonic
precision; latency numbers report mean/median over a repeated run to smooth
out scheduler noise.

Usage::

    python scripts/benchmark.py all
    python scripts/benchmark.py predict --runs 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports" / "baseline"

# Make the project root importable when invoked as ``python scripts/benchmark.py``.
sys.path.insert(0, str(BASE_DIR))

# Representative prose for prediction latency (real, fake, uncertain, and a
# Reuters-format stress case). These are run through the real frozen model.
SAMPLE_TEXT = {
    "real": (
        "The central bank left interest rates unchanged on Thursday, saying "
        "the economy continued to expand at a moderate pace while inflation "
        "remained contained. Policymakers noted that household spending had "
        "picked up and that the labor market stayed resilient despite global "
        "trade uncertainty."
    ),
    "fake": (
        "Scientists have confirmed that drinking a single glass of water "
        "every morning completely eliminates all known diseases, according "
        "to a study that has supposedly been peer reviewed by leading "
        "medical institutions around the world. The miracle cure was "
        "reportedly hidden from the public for decades by pharmaceutical "
        "companies and is now spreading rapidly across social media."
    ),
    "uncertain": (
        "A report about recent economic developments has been published. "
        "It notes some improvements in certain sectors while also "
        "mentioning several ongoing challenges. Analysts have given mixed "
        "opinions about what the future may hold for the region."
    ),
    "reuters_stress": (
        "WASHINGTON (Reuters) - A fabricated claim about an emergency "
        "government program is being spread by anonymous accounts online. "
        "The story references officials, ministries and unnamed sources in "
        "an attempt to look like a genuine wire report."
    ),
}

_ARTICLE_HTML = """
<!doctype html>
<html>
<head><title>Washington Tackles New Infrastructure Plan</title></head>
<body>
<article>
<h1>Washington Tackles New Infrastructure Plan</h1>
<p>The city council approved a sweeping infrastructure plan on Monday that
will upgrade roads, bridges and public transit across the metropolitan area.
Officials said the first contracts could be awarded before the end of the
year and that the financing package has already been secured.</p>
<p>Supporters argued the investment would create thousands of jobs and reduce
congestion, while critics questioned the projected costs and asked for more
independent oversight during the construction phase.</p>
<p>The mayor is expected to sign the measure into law next week, after which
the transportation authority will publish a timeline for the first projects
and a public comment period will be opened for affected neighborhoods.</p>
</article>
</body>
</html>
"""


class _FakeResponse:
    """A minimal ``requests.Response`` double used by the URL benchmark."""

    def __init__(self, html: str, url: str) -> None:
        self._html_bytes = html.encode("utf-8")
        self.headers = {"content-type": "text/html; charset=utf-8"}
        self.status_code = 200
        self.encoding = "utf-8"
        self.url = url
        self.content = self._html_bytes

    @property
    def is_redirect(self) -> bool:
        return False

    @property
    def is_permanent_redirect(self) -> bool:
        return False

    def raise_for_status(self) -> None:  # pragma: no cover - trivially safe
        return None

    def close(self) -> None:  # pragma: no cover - trivially safe
        return None


def _measure(func, runs: int) -> dict[str, float]:
    """Run ``func`` ``runs`` times and return summary statistics."""
    timings: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        func()
        timings.append(time.perf_counter() - start)
    return {
        "runs": runs,
        "mean_ms": round(statistics.mean(timings) * 1000.0, 3),
        "median_ms": round(statistics.median(timings) * 1000.0, 3),
        "min_ms": round(min(timings) * 1000.0, 3),
        "max_ms": round(max(timings) * 1000.0, 3),
    }


def _write_reports(name: str, payload: dict[str, Any]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    target = REPORTS_DIR / f"{name}.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[benchmark] wrote {target.relative_to(BASE_DIR)}")


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_tests() -> dict[str, Any]:
    # ``-o addopts=`` neutralises the ``-q`` addopts in pytest.ini which would
    # otherwise suppress the "N tests collected" summary line we parse.
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-o", "addopts="],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        timeout=600,
    )
    summary = "N/A"
    for line in (result.stdout or "").splitlines():
        if "tests collected" in line:
            summary = line.strip()
            break
    return {"pytest": sys.executable, "collected": summary}


def cmd_startup() -> dict[str, Any]:
    start = time.perf_counter()
    from app.main import app  # noqa: F401 - this is the measurement target

    elapsed = time.perf_counter() - start
    return {
        "import_and_create_app_ms": round(elapsed * 1000.0, 3),
        "note": "module import + create_app() only; model load is measured in 'model'.",
    }


def cmd_model(runs: int) -> dict[str, Any]:
    from app.config import settings
    from app.model import ModelService

    service = ModelService(settings.model_file, settings.vectorizer_file)
    load_start = time.perf_counter()
    service.load()
    load_ms = round((time.perf_counter() - load_start) * 1000.0, 3)

    vocab = getattr(service._vectorizer, "vocabulary_", None)
    return {
        "model_file": str(settings.model_file),
        "vectorizer_file": str(settings.vectorizer_file),
        "backend": service._backend,
        "vocab_size": len(vocab) if vocab else None,
        "load_ms": load_ms,
        "prediction": {
            name: _measure(lambda t=text: service.predict(t), runs)
            for name, text in SAMPLE_TEXT.items()
        },
    }


def cmd_url(runs: int) -> dict[str, Any]:
    from unittest import mock

    from app.scraper import UrlFetcher

    url = "https://example.com/article/world/2024/infrastructure"
    dns_result = [(
        2, 1, 6, "", ("93.184.216.34", 0),
    )]

    def _one_fetch() -> None:
        fetcher = UrlFetcher()
        with mock.patch("app.scraper.socket.getaddrinfo", return_value=dns_result):
            with mock.patch(
                "app.scraper.requests.Session.get",
                return_value=_FakeResponse(_ARTICLE_HTML, url),
            ):
                fetcher.fetch_article(url)

    return _measure(_one_fetch, runs)


def cmd_memory() -> dict[str, Any]:
    """Measure RSS in a fresh subprocess (isolation from this process)."""
    script = r"""
import json, resource, sys
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
scale = 1024.0 if sys.platform.startswith("linux") else 1.0  # linux=KB, darwin=bytes
before_mb = round(rss() / scale / 1024.0 / 1024.0, 2)
from app.config import settings
from app.model import ModelService
service = ModelService(settings.model_file, settings.vectorizer_file)
service.load()
after_mb = round(rss() / scale / 1024.0 / 1024.0, 2)
print(json.dumps({
    "peak_rss_before_load_mb": before_mb,
    "peak_rss_after_load_mb": after_mb,
    "delta_mb": round(after_mb - before_mb, 2),
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        return {
            "error": result.stderr.strip().splitlines()[-1:]
            if result.stderr.strip()
            else "subprocess failed",
        }
    return json.loads(result.stdout)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        choices=["all", "tests", "startup", "model", "url", "memory"],
    )
    parser.add_argument("--runs", type=int, default=20, help="repetitions per sample")
    args = parser.parse_args()

    targets = (
        ["tests", "startup", "model", "url", "memory"]
        if args.target == "all"
        else [args.target]
    )
    for name in targets:
        payload = {
            "tests": cmd_tests,
            "startup": cmd_startup,
            "model": lambda: cmd_model(args.runs),
            "url": lambda: cmd_url(args.runs),
            "memory": cmd_memory,
        }[name]()
        payload = {"target": name, **payload}
        _write_reports(name, payload)
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()