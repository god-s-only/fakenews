"""Phase 3D: build an out-of-distribution GENERALIZATION eval corpus.

Purpose: measure whether non-Reuters REAL news writing is classified as FAKE
by the production model, and whether the Reuters dateline is disproportionately
influential. This corpus is EVAL-ONLY:

* it is NOT used for training,
* it is NOT used for model selection or uncertainty-threshold tuning
  (it is designated a final-test-style set and inspected once).

Articles are discovered LIVE from reputable news outlets via their public RSS
feeds, then the article body is extracted with the application's own scraper
(``app.scraper.fetch_article``). Every entry records its source outlet, title,
final URL, extraction method and retrieval timestamp for provenance. The label
is 1 (REAL) by construction because the sources are established news
organisations — this is an audit/evaluation corpus, not a training dataset.

Outputs:
    data/splits/generalization.csv
    reports/generalization_corpus.json
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.scraper import ScrapeError, fetch_article  # noqa: E402
OUT_CSV = ROOT / "data/splits/generalization.csv"
REPORT = ROOT / "reports/generalization_corpus.json"

FEEDS = {
    "bbc": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "guardian": "https://www.theguardian.com/world/rss",
    "npr": "https://feeds.npr.org/1001/rss.xml",
    "cbs": "https://www.cbsnews.com/latest/rss/main",
    "dw": "https://rss.dw.com/rdf/rss-en-all",
    "ap": "https://apnews.com/rss/topnews",
    "aljazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "timesofindia": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    "cbc": "https://www.cbc.ca/rss/cbc-topstories.xml",
}

MIN_ARTICLE_CHARS = 300
TARGET_ARTICLES = 20
FETCH_TIMEOUT = (10, 30)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")


def fetch_feed(url: str) -> str:
    return requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": UA}).text


def _ns(tag: str) -> str:
    return tag


def parse_items(raw: str, outlet: str) -> list[dict]:
    """Extract title/link/date from RSS or Atom items (namespace-agnostic)."""
    items: list[dict] = []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return items
    for node in root.iter():
        if node.tag.split("}")[-1] not in ("item", "entry"):
            continue
        fields = {
            el.tag.split("}")[-1]: (el.text or "").strip()
            for el in node if el.tag.split("}")[-1] in ("title", "description", "pubDate", "updated", "link")
        }
        if not fields.get("title"):
            continue
        link = fields.get("link", "")
        if not link.startswith("http"):
            for el in node.iter():
                if el.tag.split("}")[-1] == "link":
                    href = el.get("href", "").strip()
                    if href.startswith("http"):
                        link = href
                        break
        if not link.startswith("http"):
            continue
        items.append({
            "outlet": outlet,
            "title": fields["title"],
            "link": link,
            "pubdate": (fields.get("pubDate") or fields.get("updated") or ""),
            "description": re.sub(r"\s+", " ", fields.get("description", "")).strip(),
        })
    return items


def main() -> None:
    picked: list[dict] = []
    feed_errors: list[str] = []
    for outlet, url in FEEDS.items():
        try:
            raw = fetch_feed(url)
            items = [it for it in parse_items(raw, outlet) if it["link"]]
        except Exception as exc:  # noqa: BLE001 - report & continue
            feed_errors.append(f"{outlet}: {url} -> {type(exc).__name__}: {exc}")
            continue
        seen = set()
        for it in items:
            if it["link"] in seen:
                continue
            seen.add(it["link"])
            picked.append(it)

    by_outlet: dict[str, list[dict]] = {}
    for it in picked:
        by_outlet.setdefault(it["outlet"], []).append(it)

    if by_outlet:
        per = max(2, TARGET_ARTICLES // max(1, len(by_outlet)))
    else:
        per = 0

    chosen: list[dict] = []
    for outlet, its in by_outlet.items():
        chosen.extend(its[:per])
    if len(chosen) > TARGET_ARTICLES:
        chosen = chosen[:TARGET_ARTICLES]

    records: list[dict] = []
    errors: list[dict] = []
    for it in chosen:
        try:
            # Increase scraper timeouts slightly for slower sites by providing
            # a fresh fetcher via settings override (module default kept).
            res = fetch_article(it["link"])
        except ScrapeError as exc:
            errors.append({
                "url": it["link"], "outlet": it["outlet"], "title": it["title"],
                "category": exc.category, "detail": exc.detail,
                "final_url": exc.final_url,
            })
            continue
        if len(res.text) < MIN_ARTICLE_CHARS:
            errors.append({
                "url": it["link"], "outlet": it["outlet"], "title": it["title"],
                "category": "too_short", "detail": f"{len(res.text)} chars",
                "final_url": res.final_url,
            })
            continue
        records.append({
            "text": res.text,
            "label": 1,
            "source": it["outlet"],
            "dataset": "generalization",
            "url": it["link"],
            "final_url": res.final_url,
            "title": res.title or it["title"],
            "feed_title": it["title"],
            "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "extraction_method": res.extraction_method,
            "char_count": len(res.text),
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(records[0].keys()) if records else FIELDS
        )
        if records:
            writer.writeheader()
            writer.writerows(records)

    summary = {
        "purpose": "eval-only OOD generalization corpus; not for training or tuning",
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "feeds": FEEDS,
        "feed_errors": feed_errors,
        "candidates_discovered": len(picked),
        "chosen_to_fetch": len(chosen),
        "records": len(records),
        "fetch_errors": errors,
        "outlet_counts": dict(Counter(r["source"] for r in records)),
        "char_min_max": {"min": min(r["char_count"] for r in records),
                         "max": max(r["char_count"] for r in records)} if records else None,
        "label_note": "label=1(REAL) by source credibility; audit corpus only",
    }
    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    print("feed errors:", len(feed_errors))
    for e in feed_errors:
        print("   ", e)
    print("records:", len(records))
    for r in records:
        print(f"   {r['source']:11s} {r['char_count']:6d}  {r['title'][:60]}")
    if errors:
        print("fetch errors:", len(errors))
        for e in errors:
            print(f"   {e['outlet']:11s} {e['category']:18s} {e['url'][:70]}")
    print("wrote", OUT_CSV, "and", REPORT)


FIELDS = ["text", "label", "source", "dataset", "url", "final_url", "title",
          "feed_title", "retrieved_at", "extraction_method", "char_count"]


if __name__ == "__main__":
    main()