# Phase 9 — Source-format leakage in frozen Candidate D (robustness report)

**Date:** 2026-09-06 · **Model:** Candidate D (ISOT + BuzzFeed-v02, TF-IDF +
LogisticRegression), **FROZEN** · Full machine-readable data:
`reports/phase9_robustness_report.json`

## What was discovered

A fabricated article wrapped in a `LONDON (Reuters) - ` dateline is scored
**REAL** by the frozen detector even though the plain claim is scored FAKE.
Example (phone-charger claim): p(real) **0.203 → 0.825** with only the dateline
added.  This is a *model-level* ML robustness defect (source-format / style
leakage), **not** a deployment failure — the served artifacts are identical to
the frozen candidate.

## Root cause (why it happens)

| factor | value |
|---|---|
| `reuter` coefficient | **+21.05** |
| `reuter` in training vocab | yes (index 37 088; min_df=2) |
| training doc-frequency of `reuter` | 5 858 / 31 152 docs (18.8 %) |
| ... in REAL training docs | 5 685 (97 % of `reuter` docs) |
| ... as % of REAL training docs | **33.49 %** |
| all-OOV base rate | p(real) = 0.204 (intercept −1.362) |

The training corpus was cleaned to strip *leading* `CITY (Reuters) - `
datelines, but the token `reuter` still appears inline in a third of REAL
articles ("Reporting by X; editing by Y (Reuters)", "the Reuters poll", …).
Logistic regression therefore treats "reuter" as a near-perfect REAL signature
(coef +21.05 is among the largest in the model). Any input that contains that
token in a dateline position inherits a huge real-push that the model's
intercept otherwise dispenses sparingly.

## Adversarial evaluation (frozen model, raw pipeline)

- **Claims:** 5 fabricated claims (miracle cure, election decree, mega-storm,
  phone charger, climate scandal).
- **Formats:** plain (reference) + 11 wrappers = **55 adversarial examples**.
- **Verdict flips: 16 / 55** · **max |Δp| = 0.7137**
  (storm claim + "Reuters cited two officials" inline → 0.077 → 0.790).

| format | mean Δp | flips |
|---|---|---|
| reuters_london_dateline | +0.483 | 4 |
| reuters_dateline | +0.316 | 3 |
| reuters_inline_cite | +0.577 | 4 |
| ministry_said | +0.300 | 2 |
| publication_meta | +0.155 | 1 |
| ap_dateline | −0.081 | 1 |
| bbc_byline | −0.059 | 1 |
| cnn_banner | +0.077 | 0 |
| generic_newsroom | +0.099 | 0 |
| journalist_byline | −0.013 | 0 |
| officials_lang | −0.044 | 0 |

**Reuters-specific shift** (the discovered attack): `LONDON (Reuters) - `
mean |Δp| = **0.483**, max **0.622**; `WASHINGTON (Reuters) - ` mean +0.316;
7 dateline flips total.

Note: the direction is outlet-dependent (AP/BBC bylines carry rare tokens that
sometimes push *away* from real). The robust invariant is that **every**
stamped variant deviates from the plain-content score, and removing the stamp
restores it.

## Source-marker isolation (claim kept identical)

For the phone-charger claim: plain 0.203 · full dateline 0.825 ·
dateline effect +0.237 · `reuter`-token contribution +0.253 ·
removing the whole dateline ≡ plain (by construction). i.e. the dateline
city contributes a secondary real-push; the `reuter` token dominates.

## Preprocessing mitigation (ADOPTED — NO retraining)

`app.preprocessing.normalize_news_markers()` strips only unambiguous honour
stamps: outlet datelines (`CITY (Reuters) - `, `(CNN) — `, `CITY (AP) — `),
wire prefixes, journalist bylines, publication/date lines, and inline
`(Reuters)`-style parentheticals. Ordinary prose — even text that contains
"Reuters", "officials" or "ministry" — is never consumed.

**Status: ENABLED unconditionally in the production inference path**
(`app.model.ModelService.predict`, before cleaning). Locked in
`reports/release_manifest.json` (`inference_preprocessing` block pins the
module SHA-256 alongside the artifact hashes so model ↔ preprocessing cannot
drift apart silently).

- **All 10 format-level flips are neutralised** (Reuters datelines ×3,
  London datelines ×4, AP ×1, BBC byline ×1, publication meta ×1) — every
  stamped example returns to the plain-content verdict.
- **Verdict flips: 16 / 55 before → 6 / 55 after** (the 6 remaining are the
  documented prose-level evasions). **False-positive (REAL) verdicts among
  stamped fabricated samples: 18 → 16.**
- **Exact shift metrics:** max |Δp| **0.7137** · mean Reuters-dateline
  ΔP(real) **+0.3997** (n=10, both dateline formats) · max **+0.6218**.
- **Production probes unchanged:** CBN 0.9776→0.9776 · coral 0.4784→0.4784 ·
  fabricated-claim 0.0589→0.0589 · `fabricated_reuters_style` 0.4956→**0.0589**.
- **Reuters legit family (10/10) still REAL**, mean p 0.991→0.973.
- **OOD / non-Reuters unchanged** (identical verdicts, same 70 % real).
- **ISOT test:** acc 0.9898→0.9890 (3 borderline REAL rows lose the dateline
  signal and flip to FAKE), macro-F1 0.9897→0.9889. **BuzzFeed test,
  API parity, deployment & baseline tests: zero changes** (0 prediction
  changes; max |Δp| 0.004).
- **Locked precision re-locked on adoption:** `REUTERS_REAL`
  **0.999993 → 0.999827** (still REAL) — as predicted.
- **Full test suite: 192 passed, 0 failed** (173 prior + 16 robustness +
  3 new production-wiring tests proving `ModelService.predict` normalizes).

## Boundary cases (not solved by preprocessing)

Prose-level evasions that imitate news *language* rather than *format* are not
stamps and are **not** stripped:
- `Reuters cited two officials familiar with the matter.` → 4 flips remain.
- `The ministry said …` → 2 flips remain.

These are content-level leakage. They are not fixable by input preprocessing
and would be the target of a retraining/augmentation strategy (e.g. adding
wire-style attributional FAKE contrafactuals to the training mix and
re-isolating the vectorizer). **This is NOT implemented in Phase 9.**

## Conclusion

- **Why the attack works:** `reuter` (and to a lesser extent dateline cities)
  are near-perfect REAL signatures learned from 33 % of real training articles.
- **Preprocessing viability: YES for the discovered (stamp-level) attack** —
  adopted; removes it without retraining, with a measured −0.08pp accuracy
  collateral on ISOT test and no effect on BF/OOD/probes.
- **Retraining necessary: NO for the dateline family** (preprocessing fixes
  it). Retraining is only justified *if* the prose-level evasive-language
  family (inline "Reuters cited", "the ministry said") is in scope — not
  demonstrated to be required by this phase.
- **No thresholds, labels, coefficients, training data or artifact hashes were
  changed.** All four frozen artifacts verified byte-identical before and
  after (hashes recorded in the JSON report and matching
  `reports/release_manifest.json`).
- **Ready for PR review.** Normaliser is live in production; one locked
  precision re-locked (0.999993→0.999827, still REAL); full suite 192/192
  green; artifacts byte-identical; preprocessing pinned in the release
  manifest.

## Files changed (Phase 9)

| file | change |
|---|---|
| `app/preprocessing.py` | `normalize_news_markers()` (narrowly-scoped stamp stripping) |
| `app/model.py` | production inference path applies the normalizer before cleaning (ADOPTED) |
| `reports/release_manifest.json` | pinned `inference_preprocessing` module SHA-256 (anti-drift) |
| `scripts/phase9_robustness.py` | read-only robustness/report generator (exact-metrics block) |
| `reports/phase9_robustness_report.json` | machine-readable report (exact metrics incl. before/after flips, FP counts, Reuters dateline deltas) |
| `reports/phase9_robustness_report.md` | this document |
| `tests/test_robustness.py` | 16 leak/mitigation tests + 3 production-wiring tests that the normalizer is enabled |
| `tests/test_candidate_d.py` | re-locked `REUTERS_REAL` 0.999993→0.999827; offline oracle mirrors production normalization |