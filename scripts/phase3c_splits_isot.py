"""Phase 3C: leakage-free, group-stratified ISOT train/val/test split.

Splits are made at the level of ``content_key`` (an approximation of an
article identity), so no two rows that share near-identical content can land in
different splits, and exact ``(title, text)`` duplication cannot cross splits
(already deduplicated in Phase 2). Mixed-label groups (near-duplicates that
carry contradictory labels) are dropped before splitting because they are
ambiguous. Splits are stratified by class with a deterministic greedy
assignment, so each split preserves the global REAL/FAKE ratio as closely as
possible.

Outputs:
    data/splits/isot_train.csv
    data/splits/isot_val.csv
    data/splits/isot_test.csv
    reports/splits_isot.json

The test split is designated untouched: it is not used for model selection or
uncertainty-threshold tuning. The vectorizer is fitted only on training rows
(Phase 5).
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLEANED = ROOT / "data/processed/isot_cleaned.csv"
SPLITS = ROOT / "data/splits"
REPORT = ROOT / "reports/splits_isot.json"

SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
SEED = 42

FIELDS = ["text", "label", "source", "dataset", "origin_title",
          "origin_subject", "origin_date", "content_key", "raw_length"]


def main() -> None:
    SPLITS.mkdir(parents=True, exist_ok=True)

    with open(CLEANED, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    # Group rows by content_key; drop mixed-label groups (ambiguous).
    by_key: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_key[row["content_key"]].append(row)

    mixed = {k for k, v in by_key.items()
             if len({int(r["label"]) for r in v}) > 1}
    dropped_mixed = sum(len(by_key[k]) for k in mixed)
    for k in mixed:
        del by_key[k]

    group_labels: dict[str, int] = {k: int(v[0]["label"]) for k, v in by_key.items()}
    g_keys = list(by_key)

    counts = Counter(group_labels.values())
    total = sum(len(v) for v in by_key.values())
    targets: dict[str, dict[int, float]] = {
        split: {
            label: SPLIT_RATIOS[split] * counts[label] for label in counts
        }
        for split in SPLIT_RATIOS
    }

    rng = random.Random(SEED)
    rng.shuffle(g_keys)

    assigned: dict[str, list[str]] = {s: [] for s in SPLIT_RATIOS}
    load = {s: {label: 0 for label in counts} for s in SPLIT_RATIOS}

    def deficit(split: str, label: int) -> float:
        return targets[split][label] - load[split][label]

    for key in g_keys:
        label = group_labels[key]
        best = max(SPLIT_RATIOS, key=lambda s: deficit(s, label))
        assigned[best].append(key)
        load[best][label] += 1

    paths = {s: SPLITS / f"isot_{s}.csv" for s in SPLIT_RATIOS}
    per_split: dict[str, dict] = {}
    for split, keys in assigned.items():
        rows_out = [r for k in keys for r in by_key[k]]
        with open(paths[split], "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows_out)
        per_split[split] = {
            "rows": len(rows_out),
            "groups": len(keys),
            "label_counts": dict(Counter(int(r["label"]) for r in rows_out)),
            "real_frac": round(sum(1 for r in rows_out if r["label"] == "1") / len(rows_out), 6),
        }

    # Verification: no content_key straddles splits; no exact (title,text) overlap.
    split_keys = {s: set(assigned[s]) for s in SPLIT_RATIOS}
    overlap_groups = 0
    for i, a in enumerate(split_keys.values()):
        for j, b in enumerate(split_keys.values()):
            if j > i:
                overlap_groups += len(a & b)

    def row_tuples(fname: Path) -> set[tuple]:
        with open(fname, newline="", encoding="utf-8") as fh:
            return {(r["origin_title"], r["text"]) for r in csv.DictReader(fh)}

    tt = {s: row_tuples(paths[s]) for s in SPLIT_RATIOS}
    overlap_titletext = sum(len(tt[a] & tt[b])
                            for i, a in enumerate(SPLIT_RATIOS)
                            for j, b in enumerate(SPLIT_RATIOS) if j > i)

    summary = {
        "input_rows": len(rows),
        "mixed_label_groups_dropped": len(mixed),
        "rows_dropped_from_mixed_groups": dropped_mixed,
        "groups_used": len(g_keys),
        "rows_used": total,
        "ratios": SPLIT_RATIOS,
        "seed": SEED,
        "splits": per_split,
        "verification": {
            "content_key_overlap_between_splits": overlap_groups,
            "titletext_overlap_between_splits": overlap_titletext,
        },
        "class_totals": dict(counts),
    }

    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    for s in SPLIT_RATIOS:
        p = per_split[s]
        print(f"{s:6s} rows={p['rows']:6d} groups={p['groups']:6d} "
              f"labels={p['label_counts']} real_frac={p['real_frac']}")
    print(f"mixed-label groups dropped: {len(mixed)} ({dropped_mixed} rows)")
    print(f"content_key overlap between splits: {overlap_groups} "
          f"| title,text overlap: {overlap_titletext}")
    print("wrote", *paths.values(), "and", REPORT)


if __name__ == "__main__":
    main()