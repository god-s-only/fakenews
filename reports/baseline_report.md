# Baseline Report — Original Production Model

Status: **immutable baseline**. `artifacts/baseline/my_model.h5` + `artifacts/baseline/countvectorizer.pkl` are byte-identical copies of the original production artifacts (SHA-256 in `baseline_sha256.txt`).

## Provenance
- Dataset: ISOT Fake News Dataset (official `True.csv` 21,417 real + `Fake.csv` 23,481 fake = 44898)
- Model: `fake_news.ipynb` — CountVectorizer(max_features=40000) on the porter-stemmed, stopword-filtered text, Dense(12,relu)x3 → Dense(1, sigmoid), adam/binary_crossentropy, 10 epochs, batch 32.
- Split: 80/20 `random_state=42`, **unstratified** (as in the notebook).
- Note: the notebook fitted the CountVectorizer on the **full corpus before**
        the split (feature-set leakage) — reproduced here deliberately.

## Metrics on held-out ISOT test set (n = 8980)
- Accuracy: **0.99276** (notebook recorded 0.9927616926503341)
- Macro F1: **0.9928**
- ROC-AUC: 0.9978
- REAL precision/recall/F1: 0.9922 / 0.9928 / 0.9925
- FAKE precision/recall/F1: 0.9933 / 0.9927 / 0.9930
- Predicted REAL % / FAKE %: 48.25% / 51.75%   (actual: 48.22% / 51.78%)


Confusion (rows=true, cols=pred)
              predicted FAKE   predicted REAL
true FAKE           4616             34
true REAL             31           4299

## Why the high score is partly an artifact
- Of 21417 REAL articles, 18131 (84.66%) begin with a `CITY (Reuters) -` dateline.
- The vocabulary only contains a single REAL writing style (Reuters). The FAKE set spans many non-verified outlets. The model therefore tends to latch onto the Reuters signature rather than a general notion of trustworthy journalism.
- Confirmed empirically: a hand-written sentence such as `NEW YORK (Reuters) - The central bank lowered its benchmark rate on Thursday.` is scored REAL at P≈0.999, while structurally identical non-Reuters text is scored FAKE at P≈0.001.

## Constraints honoured
No probabilities manipulated, no labels inverted, no keyword rules, `my_model.h5` and `countvectorizer.pkl` untouched.