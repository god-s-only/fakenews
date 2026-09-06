"""Phase 3E: controlled Reuters-dateline shortcut experiments (EVAL ONLY).

Demonstrates whether the production model depends on Reuters formatting, using
the SAME production predict path as the API (``app.model.ModelService``).

Families probed (each item reports P(real) and the resulting label):

Reuters family (REAL articles from the untouched ISOT test split):
    original   - raw article text, including the "CITY (Reuters) -" dateline
    stripped   - identical text with the Reuters dateline removed
    normalized - dateline removed AND formatting normalized (lowercase,
                 punctuation removed, whitespace collapsed)

Non-Reuters family (REAL articles from the Phase 3D generalization corpus,
from BBC/Guardian/NPR/Wire-based outlets etc.):
    plain          - article as published (non-Reuters REAL writing)
    reutersstyled  - the SAME factual content, wrapped in a Reuters-style
                     "CITY (Reuters) - " dateline

Deltas quantify the influence of the Reuters formatting. This is an evaluation
experiment only: NO Reuters-specific handling is added to the classifier.

Output: reports/reuters_experiment.json
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.common as common  # noqa: E402

from app.model import ModelService  # noqa: E402

BASELINE_MODEL = ROOT / "artifacts/baseline/my_model.h5"
BASELINE_VECTORIZER = ROOT / "artifacts/baseline/countvectorizer.pkl"

ISOT_TEST = ROOT / "data/splits/isot_test.csv"
RAW_TRUE = ROOT / "data/True.csv"
GEN_CSV = ROOT / "data/splits/generalization.csv"
REPORT = ROOT / "reports/reuters_experiment.json"

N_REUTERS = 10
N_NONREUTERS = 20

PUNCT_RE = re.compile(r"[^a-z0-9\s]+", re.IGNORECASE)


def normalize_formatting(text: str) -> str:
    """Strip dateline, lowercase, drop punctuation, collapse whitespace."""
    t = re.sub(r"\s+", " ", PUNCT_RE.sub(" ", common.strip_reuters_dateline(text).lower())).strip()
    return t


def main() -> None:
    service = ModelService(BASELINE_MODEL, BASELINE_VECTORIZER).load()

    # ---- Reuters family: RAW text from True.csv for rows in the test split ----
    raw_by_title: dict[str, list[str]] = defaultdict(list)
    with open(RAW_TRUE, newline="", encoding="utf-8-sig", errors="replace") as fh:
        for row in csv.DictReader(fh):
            raw_by_title[row["title"]].append(row["text"])

    test_real = []
    with open(ISOT_TEST, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["label"] == "1":
                test_real.append(row)

    reuters_items: list[dict] = []
    seen_titles = set()
    for row in test_real:
        title = row["origin_title"]
        if title in seen_titles or title not in raw_by_title:
            continue
        for raw in raw_by_title[title]:
            if common.is_reuters_dateline(raw):
                seen_titles.add(title)
                reuters_items.append({"title": title, "raw": raw})
                break
        if len(reuters_items) >= N_REUTERS:
            break

    # ---- Non-Reuters family: generalization corpus ----
    gen_items = []
    with open(GEN_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not common.is_reuters_dateline(row["text"]):
                gen_items.append(row)
    gen_items = gen_items[:N_NONREUTERS]
    for it in gen_items:
        assert common.is_reuters_dateline(it["text"]) is False

    results: list[dict] = []
    deltas: dict[str, list[float]] = defaultdict(list)

    for i, item in enumerate(reuters_items):
        original = item["raw"]
        stripped = common.strip_reuters_dateline(original)
        normalized = normalize_formatting(original)
        preds = {}
        for name, text in (
            ("original", original),
            ("stripped", stripped),
            ("normalized", normalized),
        ):
            p = service.predict(text)
            preds[name] = {"p_real": round(p.probability_real, 6), "label": p.label}
        results.append({
            "family": "reuters",
            "title": item["title"],
            "preview_original": original[:160],
            "preview_stripped": stripped[:80],
            "preview_normalized": normalized[:80],
            "preds": preds,
        })
        deltas["stripped_delta"].append(preds["stripped"]["p_real"] - preds["original"]["p_real"])
        deltas["normalized_delta"].append(preds["normalized"]["p_real"] - preds["original"]["p_real"])

    for item in gen_items:
        plain = item["text"]
        styled = f"WASHINGTON (Reuters) - {plain}"
        preds = {}
        for name, text in (("plain", plain), ("reutersstyled", styled)):
            p = service.predict(text)
            preds[name] = {"p_real": round(p.probability_real, 6), "label": p.label}
        results.append({
            "family": "nonreuters",
            "title": item.get("title", item.get("feed_title", "")),
            "url": item.get("url", ""),
            "preview_plain": plain[:120],
            "preds": preds,
        })
        deltas["styled_delta"].append(preds["reutersstyled"]["p_real"] - preds["plain"]["p_real"])
        deltas["styled_label_flip"].append(
            1 if preds["plain"]["label"] != preds["reutersstyled"]["label"] else 0
        )

    def summarize(family: str, key: str) -> dict:
        pairs = [(r["preds"][key]["p_real"], r["preds"][key]["label"])
                 for r in results if r["family"] == family]
        return {
            "n": len(pairs),
            "mean_p_real": round(sum(vals) / len(vals), 6) if (vals := [p for p, _ in pairs]) else None,
            "min_p_real": round(min(vals), 6) if vals else None,
            "max_p_real": round(max(vals), 6) if vals else None,
            "label_counts": dict(Counter(l for _, l in pairs)),
        }

    aggregate = {
        "reuters_original": summarize("reuters", "original"),
        "reuters_stripped": summarize("reuters", "stripped"),
        "reuters_normalized": summarize("reuters", "normalized"),
        "nonreuters_plain": summarize("nonreuters", "plain"),
        "nonreuters_reutersstyled": summarize("nonreuters", "reutersstyled"),
        "mean_delta_strip_dateline": round(sum(deltas["stripped_delta"]) / len(deltas["stripped_delta"]), 6) if deltas["stripped_delta"] else None,
        "mean_delta_normalize": round(sum(deltas["normalized_delta"]) / len(deltas["normalized_delta"]), 6) if deltas["normalized_delta"] else None,
        "mean_delta_add_dateline": round(sum(deltas["styled_delta"]) / len(deltas["styled_delta"]), 6) if deltas["styled_delta"] else None,
        "add_dateline_label_flips": int(sum(deltas["styled_label_flip"])),
        "uncertainty_threshold": 0.10,
        "note": "label from production verdict() with UNCERTAINTY_THRESHOLD=0.10",
        "model": str(BASELINE_MODEL),
    }

    payload = {"aggregate": aggregate, "items": results}
    common.save_json(REPORT, payload)

    print("== Reuters family (REAL, dateline present) ==")
    for r in results:
        if r["family"] != "reuters":
            continue
        o, s, n = r["preds"]["original"], r["preds"]["stripped"], r["preds"]["normalized"]
        print(f"  orig={o['label']:<8} p={o['p_real']:.4f} | "
              f"stripped={s['label']:<8} p={s['p_real']:.4f} | "
              f"norm={n['label']:<8} p={n['p_real']:.4f} | {r['title'][:50]}")
    print("== Non-Reuters family (REAL, no dateline) ==")
    for r in results:
        if r["family"] != "nonreuters":
            continue
        p, s = r["preds"]["plain"], r["preds"]["reutersstyled"]
        print(f"  plain={p['label']:<8} p={p['p_real']:.4f} | "
              f"styled={s['label']:<8} p={s['p_real']:.4f} | {r['title'][:50]}")
    print()
    print("aggregate:", json.dumps(aggregate, indent=2, sort_keys=True))
    print("wrote", REPORT)


if __name__ == "__main__":
    main()