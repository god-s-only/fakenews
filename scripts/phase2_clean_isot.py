"""Phase 2 — Reproducible cleaning pipeline for the ISOT dataset.

Reads the raw (untouched) ``data/True.csv`` and ``data/Fake.csv`` and writes a
cleaned, normalized artifact plus a quality-report JSON:

    data/processed/isot_cleaned.csv
    reports/isot_quality_report.json

Cleaning applied (in order) and logged:
1. Drop blank / empty text rows.
2. Drop extremely short, unusable rows (raw text < MIN_TEXT_CHARS).
3. Drop exact duplicate (title, text) rows.
4. Investigate and remove any full-text duplicate that appears in BOTH the
   real and fake files with different labels (ambiguous → kept out).
5. Neutralise the ``CITY (Reuters) -`` dateline artifact (leading prefix removed).
6. Assign a ``content_key`` (first 25 normalised tokens) so that near-duplicate
   articles can be kept together and kept out of different splits in Phase 4.

The raw CSV files are never modified.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATA_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    read_csv_rows,
    save_json,
    strip_reuters_dateline,
    write_csv,
)

MIN_TEXT_CHARS = 50
CONTENT_KEY_TOKENS = 25


def content_key(text: str) -> str:
    """Stable near-duplicate key: first CONTENT_KEY_TOKENS normalised words."""
    norm = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    return "_".join(norm.split()[:CONTENT_KEY_TOKENS])


def clean_rows(rows: Iterable[dict[str, str]], label: int, source: str) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for r in rows:
        raw_text = (r.get("text") or "").strip()
        if not raw_text:
            continue
        if len(raw_text) < MIN_TEXT_CHARS:
            continue
        text = strip_reuters_dateline(raw_text)
        cleaned.append(
            {
                "text": text,
                "label": str(label),
                "source": source,
                "dataset": "ISOT",
                "origin_title": (r.get("title") or "").strip(),
                "origin_subject": (r.get("subject") or "").strip(),
                "origin_date": (r.get("date") or "").strip(),
                "content_key": content_key(text),
                "raw_length": str(len(raw_text)),
            }
        )
    return cleaned


def main() -> int:
    true_rows = read_csv_rows(DATA_DIR / "True.csv")
    fake_rows = read_csv_rows(DATA_DIR / "Fake.csv")

    stats: dict = {
        "raw_rows": {"real": len(true_rows), "fake": len(fake_rows)},
        "min_text_chars": MIN_TEXT_CHARS,
        "blank_text_dropped": 0,
        "short_text_dropped": 0,
        "exact_duplicate_rows_dropped": 0,
        "conflicting_real_fake_texts": [],
        "post_clean_rows": None,
        "reuters_stripped_count": 0,
        "reuters_token_leftover_real": 0,
    }

    # 1 + 2 blank / short
    true_keep: list[dict[str, str]] = []
    fake_keep: list[dict[str, str]] = []
    for rows, keep in ((true_rows, true_keep), (fake_rows, fake_keep)):
        for r in rows:
            text = (r.get("text") or "").strip()
            if not text:
                stats["blank_text_dropped"] += 1
            elif len(text) < MIN_TEXT_CHARS:
                stats["short_text_dropped"] += 1
            else:
                keep.append(r)

    # 3 exact duplicate (title, text) within each class
    def dedupe_exact(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        seen: set[tuple[str, str]] = set()
        out: list[dict[str, str]] = []
        for r in rows:
            key = ((r.get("title") or "").strip(), (r.get("text") or "").strip())
            if key in seen:
                stats["exact_duplicate_rows_dropped"] += 1
                continue
            seen.add(key)
            out.append(r)
        return out

    true_keep = dedupe_exact(true_keep)
    fake_keep = dedupe_exact(fake_keep)

    # 4 conflicting full-text duplicate across REAL/FAKE (ambiguous label → drop both).
    # Compute on the raw rows FIRST so the investigation is recorded even when the
    # offending row is later dropped by the blank/short rules (as happens here: the
    # single cross-set duplicate has an empty body).
    true_text_by_text = {(r.get("text") or "").strip(): r for r in true_rows}
    conflict_texts = {r["text"].strip() for r in fake_rows if r["text"].strip() in true_text_by_text}
    stats["conflicting_real_fake_texts"] = [
        {
            "text_length": len(t),
            "text_prefix": t[:120],
            "real_title": true_text_by_text[t]["title"],
            "fake_title": next(r["title"] for r in fake_rows if r["text"].strip() == t),
            "real_subject": true_text_by_text[t]["subject"],
            "fake_subject": next(r["subject"] for r in fake_rows if r["text"].strip() == t),
        }
        for t in conflict_texts
    ]
    # Regardless of when it was reported, ensure no kept row carries the
    # ambiguous conflicting text (drop from both classes).
    true_texts = {r["text"].strip() for r in true_keep}
    conflict_kept = {r["text"].strip() for r in fake_keep if r["text"].strip() in true_texts}
    true_keep = [r for r in true_keep if r["text"].strip() not in conflict_kept]
    fake_keep = [r for r in fake_keep if r["text"].strip() not in conflict_kept]

    # 5 Reuters dateline neutralisation + normalised rows
    true_clean = clean_rows(true_keep, 1, "reuters")
    fake_clean = clean_rows(fake_keep, 0, "politifact-flagged")
    stats["reuters_stripped_count"] = sum(
        1 for r in [*true_clean, *fake_clean] if len(r["text"]) < int(r["raw_length"])
    )

    stats["reuters_token_leftover_real"] = sum(
        1 for r in true_clean if re.search(r"\breuter\b", r["text"], re.I)
    )
    stats["reuters_token_pct_leftover_real"] = (
        round(stats["reuters_token_leftover_real"] / len(true_clean) * 100, 2)
        if true_clean
        else 0.0
    )

    all_clean = true_clean + fake_clean
    stats["post_clean_rows"] = {
        "real": len(true_clean),
        "fake": len(fake_clean),
        "total": len(all_clean),
    }
    stats["content_key_groups"] = len({r["content_key"] for r in all_clean})
    stats["reuters_dateline_pattern"] = r"^\s*CITY (Reuters) -"

    write_csv(PROCESSED_DIR / "isot_cleaned.csv", all_clean)
    save_json(REPORTS_DIR / "isot_quality_report.json", stats)

    print("raw:", stats["raw_rows"])
    print("blank dropped:", stats["blank_text_dropped"],
          "| short dropped:", stats["short_text_dropped"],
          "| exact dup dropped:", stats["exact_duplicate_rows_dropped"])
    print("conflicting real/fake texts:", stats["conflicting_real_fake_texts"])
    print("post-clean real/fake/total:", stats["post_clean_rows"])
    print("content-key groups:", stats["content_key_groups"])
    print("REAL with 'reuter' token left after strip:",
          stats["reuters_token_leftover_real"],
          f"({stats['reuters_token_pct_leftover_real']}%)")
    print("wrote", PROCESSED_DIR / "isot_cleaned.csv",
          "and", REPORTS_DIR / "isot_quality_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())