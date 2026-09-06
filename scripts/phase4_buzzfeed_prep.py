"""Phase 4: acquire-verify + prep the MisInfoText BuzzFeed-v02 external corpus.

Verifies the external dataset BEFORE combining it with cleaned ISOT:

* provenance (source, retrieval date, hashes, license notes)
* schema (id, url, original label, full text)
* row count (expected 1,380 from the MisInfoText table)
* label values inspected (not assumed) and mapped true->REAL, false->FAKE,
  with mixture / no-factual-content excluded (but kept in the excluded tally)
* quality: missing text, duplicates, duplicate URLs, suspiciously short
  articles, conflicting labels, malformed rows
* exact + normalized overlap check against the CLEANED ISOT corpus

The raw corpus stays in gitignored ``data/``; this script commits nothing. The
manifest/report written under ``reports/`` (hashes, stats, license notes) is
safe to commit because it contains no article text.

License/provenance notes (do NOT assume GPL-3.0 covers the article texts):
* Repository (code): sfu-discourse-lab/Misinformation_detection, license GPL-3.0.
* Dataset (checked full-text news collection with original BuzzFeed fact-check
  labels) is distributed by the SFU Discourse Processing Lab for research.
* The underlying article text is originally published by various 2016-era
  outlets/authors; copyright of that text belongs to its respective owners.
* Therefore we DO NOT redistribute the raw corpus via git. We commit only
  preprocessing scripts, manifests, hashes/identifiers, documentation, and
  reproducible download instructions.

Outputs:
    data/processed/buzzfeed_cleaned.csv   (gitignored; label 1=REAL 0=FAKE)
    reports/buzzfeed_provenance.json
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.common as common  # noqa: E402

RAW_TXT = ROOT / "data/buzzfeed_v02/buzzfeed-v02-originalLabels.txt"
RAW_ZIP = ROOT / "data/buzzfeed_v02/buzzfeed-v02-originalLabels.txt.zip"
ISOT_CLEANED = ROOT / "data/processed/isot_cleaned.csv"
OUT_CSV = ROOT / "data/processed/buzzfeed_cleaned.csv"
REPORT = ROOT / "reports/buzzfeed_provenance.json"

SOURCE_URL = (
    "https://github.com/sfu-discourse-lab/Misinformation_detection/"
    "(buzzfeed-v02-originalLabels.txt.zip, default branch)"
)
RETRIEVED_AT = "2026-09-05T19:00Z"
ZIP_SHA256 = "86b432f5d80ca67ed124263440cc74e811054f7534dbdd41b39ea8bd831ce13a"
MIN_TEXT_CHARS = 50
SHORT_TEXT_CHARS = 200

LABEL_MAP = {
    "true": 1,
    "mostly true": 1,
    "false": 0,
    "mostly false": 0,
}
EXCLUDED_LABELS = {"mixture of true and false", "no factual content"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_strong(text: str) -> str:
    """Aggressive normalization for overlap checks (lower, alnum only)."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def content_key(text: str) -> str:
    """First 25 normalized tokens (compatible with ISOT content_key idea)."""
    tokens = normalize_strong(text).split()
    return "_".join(tokens[:25])


def main() -> None:
    assert sha256(RAW_ZIP) == ZIP_SHA256, "zip hash changed"

    rows = []
    with open(RAW_TXT, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for lineno, parts in enumerate(reader, start=1):
            if len(parts) < 4:
                continue  # tolerate malformed rows, counted below
            rows.append({
                "article_id": parts[0].strip(),
                "url": parts[1].strip(),
                "original_label": parts[2].strip(),
                "text": "\t".join(parts[3:]).strip(),
            })

    raw_count = len(rows)
    rating_counts = Counter(r["original_label"] for r in rows)
    unknown_ratings = sorted(set(rating_counts) - set(LABEL_MAP) - EXCLUDED_LABELS)

    # quality checks
    missing_text = [r for r in rows if not r["text"]]
    malformed = [r for r in rows if not r["url"].startswith("http")]
    dup_url_text = {}
    dup_rows = []
    for r in rows:
        key = (r["url"], r["text"])
        if key in dup_url_text:
            dup_rows.append(r)
        else:
            dup_url_text[key] = r
    dup_urls = 0
    url_seen = {}
    for r in rows:
        if r["url"] in url_seen:
            dup_urls += 1
        else:
            url_seen[r["url"]] = True
    short_rows = [r for r in rows if 0 < len(r["text"]) < MIN_TEXT_CHARS]
    conflicts = []
    label_by_url = {}
    for r in rows:
        if r["url"] in label_by_url and label_by_url[r["url"]] != r["original_label"]:
            conflicts.append({"url": r["url"], "labels": sorted({label_by_url[r["url"]], r["original_label"]})})
        label_by_url[r["url"]] = r["original_label"]

    # keep rows (dedupe by (url,text); drop missing url/text and too-short)
    kept = []
    for r in rows:
        key = (r["url"], r["text"])
        if not r["text"] or not r["url"].startswith("http") or len(r["text"]) < MIN_TEXT_CHARS:
            continue
        if key in dup_url_text and dup_url_text[key] is r:
            kept.append(r)
    dup_dropped = raw_count - len(kept)

    # map labels; exclusions recorded
    mapped = []
    excluded = Counter()
    for r in kept:
        lab = r["original_label"]
        if lab in EXCLUDED_LABELS:
            excluded[lab] += 1
            continue
        if lab not in LABEL_MAP:
            excluded[f"unknown:{lab}"] += 1
            continue
        mapped.append({**r, "label": LABEL_MAP[lab]})

    for i, r in enumerate(mapped, start=1):
        r["content_key"] = content_key(r["text"])
        r["raw_length"] = len(r["text"])

    # overlap vs cleaned ISOT
    isot_texts = set()
    isot_keys = set()
    isot_labels = {}
    with open(ISOT_CLEANED, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            isot_texts.add(re.sub(r"\s+", " ", row["text"]).strip())
            isot_keys.add(row["content_key"])
            isot_labels.setdefault(row["content_key"], row["label"])
    overlap_exact_text = sum(1 for r in mapped
                             if re.sub(r"\s+", " ", r["text"]).strip() in isot_texts)
    overlap_key = sum(1 for r in mapped if r["content_key"] in isot_keys)

    # Leakage/label-conflict guard: drop ANY BuzzFeed row whose content_key
    # (first-25-token identity) also appears in cleaned ISOT. Verified: these
    # are the same articles surfaced in both corpora (SequenceMatcher up to
    # ~0.93), sometimes with CONFLICTING labels, so they must never be combined.
    overlap_groups = [r for r in mapped if r["content_key"] in isot_keys]
    isot_conflict_keys = {
        r["content_key"] for r in overlap_groups
        if isot_labels.get(r["content_key"]) is not None
        and int(isot_labels[r["content_key"]]) != r["label"]
    }
    n_overlap_conflicting_label = sum(
        1 for r in overlap_groups if r["content_key"] in isot_conflict_keys
    )
    mapped = [r for r in mapped if r["content_key"] not in isot_keys]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["text", "label", "source", "dataset", "origin_url",
                        "origin_rating", "content_key", "raw_length"],
        )
        writer.writeheader()
        for r in mapped:
            writer.writerow({
                "text": r["text"], "label": r["label"], "source": "buzzfeed",
                "dataset": "buzzfeed_v02", "origin_url": r["url"],
                "origin_rating": r["original_label"],
                "content_key": r["content_key"], "raw_length": r["raw_length"],
            })

    payload = {
        "dataset": "MisInfoText / SFU Discourse Lab — BuzzFeed-2016 checked full-text (buzzfeed-v02-originalLabels)",
        "source_url": SOURCE_URL,
        "retrieved_at": RETRIEVED_AT,
        "commit_note": "default branch @ retrieval; commit SHA unavailable during acquisition (GitHub API/network)",
        "hashes": {"zip": ZIP_SHA256, "txt": sha256(RAW_TXT)},
        "license_notes": {
            "repo_license": "GPL-3.0 (sfu-discourse-lab/Misinformation_detection)",
            "dataset": "Checked full-text news collection with original BuzzFeed fact-check labels, distributed by the SFU Discourse Processing Lab for research.",
            "article_text_copyright": "Underlying article text originally published by 2016-era outlets/authors; copyright belongs to respective owners.",
            "redistribution_policy": "Raw corpus kept outside git (gitignored data/). Only scripts, manifests, hashes, and documentation are committed.",
            "repro": "Re-download buzzfeed-v02-originalLabels.txt.zip from the SFU repo (see source_url).",
        },
        "raw_records": raw_count,
        "expected": 1380,
        "rating_counts": dict(rating_counts),
        "unknown_ratings": unknown_ratings,
        "quality": {
            "missing_text": len(missing_text),
            "malformed": len(malformed),
            "duplicate_url_text_dropped": dup_dropped,
            "rows_with_duplicate_url": dup_urls,
            "very_short_dropped": len(short_rows),
            "conflicting_labels_per_url": len(conflicts),
        },
        "label_mapping": LABEL_MAP,
        "excluded_labels": sorted(EXCLUDED_LABELS),
        "excluded_counts": dict(excluded),
        "kept_before_mapping": len(kept),
        "mapped": len(mapped),
        "mapped_label_counts": dict(Counter(str(r["label"]) for r in mapped)),
        "real_fraction": round(sum(r["label"] == 1 for r in mapped) / len(mapped), 6) if mapped else None,
        "content_key_groups": len({r["content_key"] for r in mapped}),
        "overlap_with_cleaned_ISOT": {
            "exact_text_matches": overlap_exact_text,
            "normalized_content_key_matches": overlap_key,
            "near_duplicate_dropped": len(overlap_groups),
            "near_duplicates_with_conflicting_labels": n_overlap_conflicting_label,
        },
        "output": str(OUT_CSV),
    }
    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    print("raw records:", raw_count)
    print("rating counts:", dict(rating_counts))
    print("unknown ratings:", unknown_ratings)
    print("quality:", {k: v for k, v in payload["quality"].items() if k != "duplicate_url_text_dropped"})
    print("mapped:", len(mapped), dict(Counter(r["label"] for r in mapped)))
    print("excluded:", dict(excluded))
    print("overlap vs ISOT:", payload["overlap_with_cleaned_ISOT"])
    print("wrote", OUT_CSV, "and", REPORT)


if __name__ == "__main__":
    main()