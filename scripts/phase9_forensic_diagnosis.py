"""Phase 9: forensic diagnosis of the promoted Candidate D production model.

READ-ONLY. Loads the frozen production model (my_model_lr.pkl) and its
vectorizer, decomposes individual predictions into feature-level
contributions (coefficient x TF-IDF), traces the driving features back to the
training corpora (ISOT train split + BuzzFeed-v02 train split as actually
fitted), checks whether generic legitimate-news terms carry learned FAKE
associations, runs controlled synthetic probes, and re-runs the Reuters
dateline stress test.

No retraining, no threshold changes, no label changes, no artifact writes.
Outputs reports/forensic_diagnosis.json (new file only) and prints tables.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import preprocessing  # noqa: E402
from app.model import ModelService  # noqa: E402
from app.config import settings  # noqa: E402
from scripts.common import read_csv_rows, save_json, sha256  # noqa: E402
from scripts.phase5_experiments import buzzfeed_splits  # noqa: E402

UNCERTAINTY = settings.UNCERTAINTY_THRESHOLD

FAIL_TEXT_1 = (
    "The Central Bank of Nigeria said commercial banks will continue to operate "
    "under the existing cash withdrawal guidelines while customers are encouraged "
    "to use electronic payment channels. The bank said the policy is intended to "
    "improve the efficiency of the country's payment system and reduce reliance "
    "on physical cash."
)
FAIL_TEXT_2 = (
    "Researchers studying coral reefs have found that periods of unusually warm "
    "ocean temperatures can cause widespread coral bleaching. During bleaching "
    "events, corals expel the algae that live within their tissues, leaving them "
    "vulnerable to disease and other environmental stresses. Scientists say "
    "reducing greenhouse gas emissions remains important for limiting the "
    "frequency and severity of these events."
)

GENERIC_TERMS = [
    "researchers", "scientists", "study", "government", "bank", "policy", "said",
    "according", "environmental", "researcher", "scientist", "studies",
    "official", "ministry", "commission", "reported", "report", "minister",
    "president", "election", "climate", "research", "published", "found",
    "science", "data", "system", "economy", "economic",
]

SYNTHETIC_CASES = {
    "neutral_science": {
        "kind": "neutral",
        "text": (
            "Researchers studying plant photosynthesis reported that leaves "
            "convert sunlight into chemical energy more efficiently under "
            "moderate temperatures. The study measured growth rates over a four "
            "month period and compared results across several controlled "
            "conditions. Scientists said the findings may help improve crop "
            "yields in regions that experience variable weather."
        ),
    },
    "fabricated_science": {
        "kind": "fabricated",
        "text": (
            "SHOCKING! Scientists just made a miracle discovery: plants can be "
            "taught to SPEAK! One simple trick doctors do not want you to know is "
            "destroying the fake news mainstream narrative forever. Share before "
            "this is deleted!"
        ),
    },
    "neutral_economics": {
        "kind": "neutral",
        "text": (
            "The Ministry of Finance said inflation slowed to four percent in the "
            "third quarter, according to the latest official statistics. The "
            "central bank expects interest rates to remain stable while the "
            "government reviews its monetary policy framework."
        ),
    },
    "fabricated_economics": {
        "kind": "fabricated",
        "text": (
            "THE ECONOMY IS COLLAPSING! The government is hiding the truth about "
            "your money! Banks are stealing deposits and printing fake cash "
            "around the clock. This is the end of the world as we know it, "
            "official sources confirm."
        ),
    },
    "neutral_nigeria": {"kind": "neutral", "text": FAIL_TEXT_1},
    "fabricated_nigeria": {
        "kind": "fabricated",
        "text": (
            "BREAKING: President announces the Central Bank will seize all "
            "private savings and replace the currency with digital coins! Banks "
            "closing tomorrow forever. Stand up and refuse, this is total theft "
            "approved by corrupt bankers."
        ),
    },
    "neutral_technology": {
        "kind": "neutral",
        "text": (
            "Engineers at a research laboratory developed a prototype battery "
            "that stores more energy and charges faster than current designs. "
            "The system uses silicon anodes and the team expects to begin "
            "commercial production within two years. Industry analysts said the "
            "approach remains promising but unproven at scale."
        ),
    },
    "fabricated_technology": {
        "kind": "fabricated",
        "text": (
            "TECH GIANTS ARE SECRETLY IMPLANTING CHIPS IN YOUR PHONE! Engineers "
            "confirmed the government can control your bank account through "
            "billboard screens. Scientists are losing their minds over this "
            "leaked footage."
        ),
    },
}


def verdict(p: float) -> str:
    winner = max(p, 1.0 - p)
    if winner - 0.5 < UNCERTAINTY:
        return "uncertain"
    return "real" if p >= 0.5 else "fake"


def analyze_text(service: ModelService, text: str) -> dict:
    vec = service._vectorizer
    coef = np.asarray(service._model.coef_).ravel()
    intercept = float(np.asarray(service._model.intercept_).ravel()[0])
    feature_names = list(vec.get_feature_names_out())
    cleaned = preprocessing.clean_single_text(text)
    tokens = cleaned.split()
    matrix = vec.transform([cleaned])
    idx = matrix.indices
    vals = matrix.data
    contribs = {int(i): float(coef[i] * v) for i, v in zip(idx, vals)}
    logit = intercept + sum(contribs.values())
    from scipy.special import expit
    p_real = float(expit(logit))

    vocab = set(feature_names)
    dropped_stop = [t for t in preprocessing.tokenize(text) if t in _stopwords()]
    in_vocab = sum(1 for t in tokens if t in vocab)
    out_of_vocab = len(tokens) - in_vocab

    tfidf_rows = sorted(
        ((feature_names[int(i)], float(v), float(coef[int(i)]),
          float(coef[int(i)] * v)) for i, v in zip(idx, vals)),
        key=lambda r: r[1], reverse=True)[:20]

    contrib_rows = sorted(
        ((feature_names[int(i)], float(coef[int(i)] * v),
          "real" if coef[int(i)] * v > 0 else "fake")
         for i, v in zip(idx, vals)),
        key=lambda r: abs(r[1]), reverse=True)[:20]

    neg = [(row[0], row[1]) for row in contrib_rows if row[2] == "fake"]
    neg_abs = sorted(neg, key=lambda r: r[1])
    total_neg = sum((c for c in contribs.values() if c < 0))
    cum = 0.0
    leaders: list[dict] = []
    for term, c in neg_abs:
        cum += c
        leaders.append({"term": term, "contribution": c,
                        "cum_share_neg": cum / total_neg if total_neg else None})
    return {
        "text": text,
        "cleaned": cleaned,
        "n_tokens": len(tokens),
        "n_in_vocab": in_vocab,
        "n_out_of_vocab": out_of_vocab,
        "n_surviving_from_stopwords": len(dropped_stop),
        "logit": logit,
        "intercept": intercept,
        "p_real": p_real,
        "p_fake": 1.0 - p_real,
        "verdict": verdict(p_real),
        "top20_by_tfidf": [dict(zip(("token", "tfidf", "coef", "contribution"), r))
                          for r in tfidf_rows],
        "top20_by_abs_contribution": [{"token": r[0], "contribution": r[1], "direction": r[2]}
                                      for r in contrib_rows],
        "negative_leaders_cumulative": leaders[:15],
        "sum_positive_contrib": float(sum((c for c in contribs.values() if c > 0))),
        "sum_negative_contrib": float(sum((c for c in contribs.values() if c < 0))),
        "n_contributing": len(contribs),
    }


def _stopwords():
    return preprocessing._get_stopwords()


def build_provenance(service: ModelService) -> dict:
    """Per-corpus x per-label document frequencies of the driving terms."""
    feature_names = list(service._vectorizer.get_feature_names_out())
    wanted = set(GENERIC_TERMS)
    WANT_EXTRA_STEM = {
        "central", "bank", "commerci", "custom", "payment", "channel",
        "cash", "guidelin", "effici", "reli", "physic", "coral", "reef",
        "bleach", "alga", "tissu", "diseas", "greenhous", "emiss", "ocean",
        "temperature", "warm", "weather", "studied", "research", "found",
        "report", "researcher", "scientist", "govern", "polici", "said",
        "accord", "environ",
    }
    wanted |= WANT_EXTRA_STEM

    isot_rows = read_csv_rows(ROOT / "data/splits/isot_train.csv")
    bf = read_csv_rows(ROOT / "data/processed/buzzfeed_cleaned.csv")
    bf_splits = buzzfeed_splits()
    bf_train = bf_splits[0]["train"]

    corpora = {
        "isot_train": (isot_rows, "1"),
        "bf_train": (bf_train, "1"),
        "bf_full": (bf, "1"),
    }

    df: dict[str, dict[str, Counter]] = {}  # corpus -> label -> Counter
    total: dict[str, Counter] = {}
    vocab_seen: dict[str, set] = {c: set() for c in corpora}
    vocab = set(feature_names)

    for cname, (rows, _label_col) in corpora.items():
        df[cname] = {"REAL": Counter(), "FAKE": Counter()}
        total[cname] = Counter()
        for r in rows:
            lab = "REAL" if r["label"] == "1" else "FAKE"
            total[cname][lab] += 1
            cleaned = preprocessing.clean_single_text(r["text"])
            toks = set(cleaned.split())
            for t in toks:
                if t in wanted:
                    df[cname][lab][t] += 1
                if t in vocab:
                    vocab_seen[cname].add(t)

    out: dict[str, dict] = {}
    for cname in corpora:
        n_real = total[cname]["REAL"]
        n_fake = total[cname]["FAKE"]
        c = out[cname] = {"n_rows": int(n_real + n_fake), "n_real": int(n_real),
                          "n_fake": int(n_fake)}
        terms = sorted(set(wanted))
        c["vocab_overlap"] = {
            "vocab_terms": len(vocab),
            "terms_present_from_this_corpus": len(vocab_seen[cname]),
        }
        c["features"] = {}
        for t in terms:
            dr = df[cname]["REAL"][t]
            dfq = df[cname]["FAKE"][t]
            c["features"][t] = {
                "df_real": int(dr),
                "df_fake": int(dfq),
                "freq_real": float(dr / n_real) if n_real else None,
                "freq_fake": float(dfq / n_fake) if n_fake else None,
                "in_vocab": t in vocab,
                "coef": float(np.asarray(service._model.coef_).ravel()[
                    feature_names.index(t)]) if t in vocab else None,
            }
    return out


def reuters_rerun(service: ModelService) -> dict:
    """Full-text re-run replicating phase8_final_eval probe construction
    exactly (isot_test REAL rows -> raw True.csv full text; GEN full texts)."""
    from collections import defaultdict
    from scripts.common import is_reuters_dateline, strip_reuters_dateline

    isot = read_csv_rows(ROOT / "data/splits/isot_test.csv")
    raw_by_title: dict[str, list[str]] = defaultdict(list)
    with open(ROOT / "data/True.csv", newline="", encoding="utf-8-sig",
              errors="replace") as fh:
        import csv as _csv
        for row in _csv.DictReader(fh):
            raw_by_title[row["title"]].append(row["text"])

    reuters, plain = [], []
    seen = set()
    for row in isot:
        if row["label"] != "1" or row["origin_title"] in seen:
            continue
        found = next((t for t in raw_by_title.get(row["origin_title"], [])
                      if is_reuters_dateline(t)), None)
        if found:
            seen.add(row["origin_title"])
            reuters.append(found)
        if len(reuters) >= 10:
            break
    gen = read_csv_rows(ROOT / "data/splits/generalization.csv")
    plain = [r["text"] for r in gen if not is_reuters_dateline(r["text"])][:20]

    p_orig = [service.predict(t).probability_real for t in reuters]
    p_strip = [service.predict(strip_reuters_dateline(t)).probability_real
               for t in reuters]
    p_plain = [service.predict(t).probability_real for t in plain]
    p_styled = [service.predict(f"WASHINGTON (Reuters) - {t}").probability_real
                for t in plain]

    def flips(a, b, mode="binary"):
        n = len(a)
        if mode == "binary":
            return sum(1 for x, y in zip(a, b) if (x > 0.5) != (y > 0.5))
        return sum(1 for x, y in zip(a, b) if verdict(x) != verdict(y))

    return {
        "reuters_family_n": len(reuters),
        "probe_source": "full texts from data/True.csv + isot_test REAL rows "
                        "(identical construction to phase8_final_eval)",
        "original": {"n": len(p_orig), "mean_p_real": float(np.mean(p_orig)),
                     "min_p_real": float(np.min(p_orig)),
                     "max_p_real": float(np.max(p_orig)),
                     "labels": dict(Counter(verdict(p) for p in p_orig))},
        "stripped": {"n": len(p_strip), "mean_p_real": float(np.mean(p_strip)),
                     "min_p_real": float(np.min(p_strip)),
                     "max_p_real": float(np.max(p_strip)),
                     "labels": dict(Counter(verdict(p) for p in p_strip))},
        "strip_delta_mean_p_real": float(np.mean(np.array(p_strip) - np.array(p_orig))),
        "strip_verdict_flips": flips(p_orig, p_strip, "verdict"),
        "strip_binary_flips": flips(p_orig, p_strip, "binary"),
        "nonreuters_family_n": len(plain),
        "plain": {"means_p_real": float(np.mean(p_plain)),
                  "labels": dict(Counter(verdict(p) for p in p_plain))},
        "styled": {"mean_p_real": float(np.mean(p_styled)),
                   "labels": dict(Counter(verdict(p) for p in p_styled))},
        "add_delta_mean_p_real": float(np.mean(np.array(p_styled) - np.array(p_plain))),
        "add_verdict_flips": flips(p_plain, p_styled, "verdict"),
        "add_binary_flips": flips(p_plain, p_styled, "binary"),
        "phase8_reference": {
            "strip_delta": -0.0181, "strip_verdict_flips": 0, "strip_binary_flips": 0,
            "add_delta": 0.0687, "add_verdict_flips": 3, "add_binary_flips": 0,
        },
    }


def main() -> None:
    svc = ModelService(ROOT / "my_model_lr.pkl", ROOT / "my_tfidf_vectorizer.pkl").load()
    assert svc._backend == "sklearn"
    coef = np.asarray(svc._model.coef_).ravel()
    intercept = float(np.asarray(svc._model.intercept_).ravel()[0])
    from scipy.special import expit

    # Parity against the frozen candidate artifact (must be bit-exact).
    with (ROOT / "artifacts/candidates/expD_lr.pkl").open("rb") as fh:
        blob = pickle.load(fh)
    cand_vec = blob["vec"]
    cand_model = blob["model"]
    parity = {}
    for name, t in (("fail1", FAIL_TEXT_1), ("fail2", FAIL_TEXT_2)):
        c = preprocessing.clean_single_text(t)
        p_cand = float(cand_model.predict_proba(cand_vec.transform([c]))[0][1])
        p_prod = svc.predict(t).probability_real
        parity[name] = {"production": p_prod, "candidate": p_cand,
                        "identical": p_cand == p_prod}

    failed = {
        "nigeria_cbn_real": analyze_text(svc, FAIL_TEXT_1),
        "coral_reef_real": analyze_text(svc, FAIL_TEXT_2),
    }
    synthetic = {name: analyze_text(svc, c["text"]) | {"kind": c["kind"]}
                 for name, c in SYNTHETIC_CASES.items()}

    generic = {}
    fnames = list(svc._vectorizer.get_feature_names_out())
    for term in GENERIC_TERMS:
        stem = preprocessing.clean_single_text(term).strip()
        generic[term] = {
            "raw_term": term,
            "resulting_token": stem or None,
            "dropped_as_stopword": (stem == ""),
            "in_vocab": stem in set(fnames),
            "coef": float(coef[fnames.index(stem)]) if stem in fnames else None,
        }

    provenance = build_provenance(svc)
    reuters = reuters_rerun(svc)

    report = {
        "meta": {
            "model_file": "my_model_lr.pkl",
            "model_sha256": sha256(ROOT / "my_model_lr.pkl"),
            "vectorizer_file": "my_tfidf_vectorizer.pkl",
            "vectorizer_sha256": sha256(ROOT / "my_tfidf_vectorizer.pkl"),
            "vocab_size": len(fnames),
            "backend": svc._backend,
            "decision_rule": {"threshold": 0.5, "uncertainty_band": UNCERTAINTY},
        },
        "base_rate": {
            "intercept": intercept,
            "p_real_across_fully_out_of_vocab_text": float(expit(intercept)),
            "note": "P(real) the model emits for an input whose cleaned tokens are "
                    "all out-of-vocabulary (raw logit = intercept). The API guards "
                    "zero-vector inputs to 0.5, but nearly-empty in-vocab inputs "
                    "are not guarded.",
        },
        "parity_check": parity,
        "failed_real_texts": failed,
        "generic_terms": generic,
        "provenance_by_corpus": provenance,
        "synthetic": synthetic,
        "reuters_stress": reuters,
    }
    save_json(ROOT / "reports/forensic_diagnosis.json", report)

    # ---------- console tables ----------
    print("=" * 88)
    print("BASE RATE  intercept=%.4f  -> all-OOV P(real)=%.4f"
          % (intercept, expit(intercept)))
    print("=" * 88)
    for name, r in failed.items():
        print(f"\n### {name}  P(real)={r['p_real']:.4f}  verdict={r['verdict']}  "
              f"logit={r['logit']:.3f}")
        print(f"tokens={r['n_tokens']} in-vocab={r['n_in_vocab']} "
              f"OOV={r['n_out_of_vocab']} stopword-tokens={r['n_surviving_from_stopwords']} "
              f"sum(+){r['sum_positive_contrib']:.3f} sum(-){r['sum_negative_contrib']:.3f}")
        print(f"{'token':<18}{'tfidf':>9}{'coef':>9}{'contrib':>10}  dir")
        for row in r["top20_by_tfidf"][:12]:
            d = "real" if row["contribution"] > 0 else "fake"
            print(f"{row['token']:<18}{row['tfidf']:>9.4f}{row['coef']:>9.3f}"
                  f"{row['contribution']:>10.4f}  {d}")
        neg = [l for l in r["negative_leaders_cumulative"] if l["cum_share_neg"] <= 0.95]
        print(" negative leaders cover 95%% of negative logit: %d items"
              % len(neg))

    print("\n" + "=" * 88)
    print("GENERIC TERMS -> learned association")
    print("=" * 88)
    print(f"{'term':<15}{'token':<12}{'stopped':<8}{'in_vocab':<9}{'coef':>9}  dir")
    for term, g in generic.items():
        d = "" if g["coef"] is None else ("real" if g["coef"] > 0 else "fake")
        print(f"{term:<15}{str(g['resulting_token']):<12}{str(g['dropped_as_stopword']):<8}"
              f"{str(g['in_vocab']):<9}{str(g['coef']):>9}  {d}")

    print("\n" + "=" * 88)
    print("PROVENANCE (document frequency per corpus x label, train splits as fitted)")
    print("=" * 88)
    prov_terms = [t for t in GENERIC_TERMS
                  + ["central", "bank", "coral", "reef", "bleach", "greenhous",
                     "emiss", "ocean", "payment", "cash", "polici", "govern",
                     "research", "environ"]] if True else []
    prov_terms = sorted(set(prov_terms))
    print(f"{'term':<14}{'coef':>9}  | 'isot_train REAL/FAKE'   'bf_train REAL/FAKE'   'bf_full REAL/FAKE'")
    for t in prov_terms:
        row = provenance["isot_train"]["features"].get(t)
        if not row:
            continue
        cells = []
        for cname in ("isot_train", "bf_train", "bf_full"):
            f = provenance[cname]["features"][t]
            fr = f["freq_real"] or 0
            ff = f["freq_fake"] or 0
            cells.append(f"{f['df_real']:>4}/{fr:>5.3f} {f['df_fake']:>4}/{ff:>5.3f}")
        print(f"{t:<14}{str(row['coef']):>9}  |  {'  '.join(cells)}")

    for cname in ("isot_train", "bf_train"):
        o = provenance[cname]["vocab_overlap"]
        print(f"[vocab] terms present in {cname}: {o['terms_present_from_this_corpus']} "
              f"of {o['vocab_terms']} vocab ({o['terms_present_from_this_corpus']/o['vocab_terms']*100:.1f}%)")

    print("\n" + "=" * 88)
    print("SYNTHETIC CONTROLLED PROBES")
    print("=" * 88)
    print(f"{'case':<20}{'kind':<11}{'P(real)':>9}  {'verdict':<10}{'logit':>8}"
          f"{'sumpos':>8}{'sumneg':>9}")
    for name, r in synthetic.items():
        print(f"{name:<20}{r['kind']:<11}{r['p_real']:>9.4f}  {r['verdict']:<10}"
              f"{r['logit']:>8.3f}{r['sum_positive_contrib']:>8.3f}"
              f"{r['sum_negative_contrib']:>9.3f}")

    print("\n" + "=" * 88)
    print("REUTERS DATELINE STRESS (rerun, production model)")
    print("=" * 88)
    rt = reuters["reuters_stress"] if "reuters_stress" in reuters else reuters
    print(f"reuters orig mean P(real)={rt['original']['mean_p_real']:.4f} "
          f"min={rt['original']['min_p_real']:.4f} labels={rt['original']['labels']}")
    print(f"reuters stripped mean P(real)={rt['stripped']['mean_p_real']:.4f} "
          f"min={rt['stripped']['min_p_real']:.4f} labels={rt['stripped']['labels']}")
    print(f"strip Δ={rt['strip_delta_mean_p_real']:+.4f} "
          f"binary flips={rt['strip_binary_flips']}/{rt['reuters_family_n']} "
          f"verdict flips={rt['strip_verdict_flips']}/{rt['reuters_family_n']}")
    print(f"nonreuters plain mean={rt['plain']['means_p_real']:.4f} -> styled "
          f"{rt['styled']['mean_p_real']:.4f}  Δ={rt['add_delta_mean_p_real']:+.4f} "
          f"binary flips={rt['add_binary_flips']}/{rt['nonreuters_family_n']} "
          f"verdict flips={rt['add_verdict_flips']}/{rt['nonreuters_family_n']}")
    print("phase8 reference:", rt['phase8_reference'])
    print("\nwrote reports/forensic_diagnosis.json")


if __name__ == "__main__":
    main()