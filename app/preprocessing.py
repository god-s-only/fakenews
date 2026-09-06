"""Text preprocessing for the fake news detector.

The preprocessing steps mirror exactly the steps used to train the model
(see ``fake_news.ipynb``): strip non-alphabetic characters, lowercase,
remove stopwords (keeping ``not`` so negations are preserved) and stem.
"""

from __future__ import annotations

import re
from typing import Iterable

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

_NON_ALPHA = re.compile(r"[^a-zA-Z]")
_STOPWORDS_CACHE: set[str] | None = None

# --------------------------------------------------------------------------- #
# News source-marker normalisation (Phase 9 — ADOPTED)
#
# Source-format stamps (datelines, bylines, publication/date lines and inline
# outlet tags) are *formatting artefacts*, not content. They leak into the
# trained model as a source-format shortcut (e.g. the ``reuter`` token carries
# coefficient +21.05 and a fabricated article wrapped in a ``CITY (Reuters) -``
# dateline is pushed from FAKE to REAL — see reports/phase9_robustness_report.md).
#
# ``normalize_news_markers`` strips these stamps before the standard cleaning.
# It is deliberately CONSERVATIVE: whole lines / leading prefixes are only
# removed when they match strict, recognisable stamp patterns, and ordinary
# prose is never consumed (words such as "Reuters", "officials" or "ministry"
# inside the body are left untouched). The production inference path applies it
# unconditionally (app/model.py). No retraining is involved.
# --------------------------------------------------------------------------- #

_NEWS_OUTLET = (
    "Reuters|AP|AFP|CNN|BBC|Bloomberg|DPA|EFE|UPI|Xinhua|ANI|PTI"
)
_MONTHS = (
    "January|February|March|April|May|June|July|August|September|"
    "October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
)
_DOWS = "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"

# Dateline sharing its line, e.g. "WASHINGTON (Reuters) - ", "(CNN) — ",
# "NEW YORK (AP) — ". The CITY part is optional and may contain spaces/slashes.
_DATELINE_RE = re.compile(
    r"^\s*(?:[^()\n]{0,80}?\s*)?\(\s*(?:" + _NEWS_OUTLET + r")\s*\)"
    r"\s*(?:[-–—:.·]+\s*|\s*\n|\s*$)"
)
# Bare wire prefix, e.g. "REUTERS - " or "AFP: ".
_OUTLET_PREFIX_RE = re.compile(r"^\s*(?:" + _NEWS_OUTLET + r")\s*[-–—:]+\s*")
# Inline parenthetical outlet stamps anywhere in the text, e.g. " (Reuters) ".
_PAREN_OUTLET_RE = re.compile(r"\s*\(\s*(?:" + _NEWS_OUTLET + r")\s*\)\s*")
# Whole line that is only an outlet name, e.g. "Reuters" / "The Associated Press".
_OUTLET_LINE_RE = re.compile(
    r"^\s*(?:" + _NEWS_OUTLET + r"|The Associated Press|Press Trust of India"
    r"|Agence France-Presse)\s*$"
)
# Journalist byline, e.g. "By JANE SMITH, BBC News" / "BY TOM WREN" /
# "By Reuters Staff". Every token must be Capitalised and the whole line is
# consumed, so ordinary prose like "By many measures, ..." is never matched.
_BYLINE_LINE_RE = re.compile(
    r"^\s*[Bb][Yy]\s+(?:[A-Z][A-Za-z0-9.'&]+[,]?\s+)+[A-Z][A-Za-z0-9.'&]+[,.]?\s*$"
)
# Publication/date stamps on their own line(s).
_DATE_LINE_RE = re.compile(
    r"^\s*(?:"
    r"(?:" + _DOWS + r")[,\s]+\d{1,2}(?:st|nd|rd|th)?\s+(?:" + _MONTHS + r")[,]?\s+\d{4}"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:" + _MONTHS + r")[,]?\s+\d{4}"
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
    r"|\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
    r")\s*[\.,]?\s*$"
)
_TIME_LINE_RE = re.compile(
    r"^\s*\d{1,2}:\d{2}\s*(?:[APap][Mm])?\s*(?:GMT|UTC|ET|PT|CET|IST|AEST)?"
    r"[,\s]+\d{1,2}(?:st|nd|rd|th)?\s+(?:" + _MONTHS + r")[,]?\s+\d{4}\s*$"
)
_STAMP_LINE_RE = re.compile(
    r"^\s*(?:Published|Updated|Edited|Last updated|Posted|Refile|Correction)"
    r"\b[:.]?\s*.*$"
)
_COPYRIGHT_LINE_RE = re.compile(r"^\s*(?:Copyright|\(C\))\b.*$", re.IGNORECASE)
# Short Title-Case line that sits directly above a byline/date stamp (e.g. a
# publication or section name). Only stripped when followed by a stamp line.
_TITLE_LINE_RE = re.compile(r"^\s*[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9']+){1,3}\s*$")
# Generic em/en-dash city dateline, e.g. "LONDON — ", "London - ".
_CITY_DATELINE_RE = re.compile(r"^\s*([A-Z][A-Za-z0-9 ,.'\-]{0,60}?)\s*[-–—]\s+")

# Major cities/capitals used in plain "CITY — " datelines. The normaliser only
# removes an ALL-CAPS "CITY —" opening when the city is on this list, protecting
# all-caps headlines; Title-Case "City —" openings are always treated as marks.
_DATELINE_CITIES = frozenset({
    "Washington", "London", "New York", "Paris", "Berlin", "Moscow", "Beijing",
    "Tokyo", "Rome", "Madrid", "Lisbon", "Ankara", "Cairo", "Jerusalem",
    "Tehran", "Baghdad", "Kabul", "New Delhi", "Mumbai", "Hong Kong", "Seoul",
    "Singapore", "Bangkok", "Jakarta", "Manila", "Sydney", "Melbourne",
    "Toronto", "Ottawa", "Montreal", "Vancouver", "Mexico City", "Buenos Aires",
    "Sao Paulo", "Rio de Janeiro", "Santiago", "Lima", "Bogota", "Karachi",
    "Islamabad", "Lagos", "Abuja", "Nairobi", "Johannesburg", "Cape Town",
    "Accra", "Addis Ababa", "Casablanca", "Dubai", "Doha", "Geneva", "Brussels",
    "Vienna", "Stockholm", "Oslo", "Copenhagen", "Helsinki", "Amsterdam",
    "Munich", "Frankfurt", "Dublin", "Warsaw", "Prague", "Budapest", "Athens",
    "Bucharest", "Kyiv", "Sofia", "Belgrade", "Zagreb", "Tallinn", "Riga",
    "Vilnius", "Abu Dhabi", "Riyadh", "Kuwait City", "Kuala Lumpur", "Taipei",
})


def _allow_generic_dateline(prefix: str) -> bool:
    """True if the pre-dateline text looks like a city, not a headline."""
    prefix = prefix.strip()
    if not prefix:
        return False
    if prefix == prefix.upper():
        return prefix.capitalize() in _DATELINE_CITIES
    return prefix.split()[0].capitalize() in _DATELINE_CITIES


def normalize_news_markers(text: str) -> str:
    """Strip recognisable news source-format markers from the leading block.

    Removes outlet datelines (``CITY (Reuters) -`` / ``(CNN) — ``), wire
    prefixes, journalist bylines, publication/date stamps and inline
    parenthetical outlet tags. The underlying claim is otherwise untouched.
    Conservative by design: patterns must be unambiguous stamps; ordinary
    prose is never consumed. Returns text with any leading markers removed.
    """
    if not text:
        return text
    lines = text.splitlines()
    i = 0
    n = len(lines)
    guard = 0
    while i < n and guard < 16:
        guard += 1
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        new = _DATELINE_RE.sub("", lines[i], count=1)
        if new == lines[i]:
            new = _OUTLET_PREFIX_RE.sub("", lines[i], count=1)
        if new != lines[i]:
            lines[i] = new
            if not lines[i].strip() or _BYLINE_LINE_RE.match(lines[i].strip()):
                i += 1
            continue
        m = _CITY_DATELINE_RE.match(lines[i])
        if m and _allow_generic_dateline(m.group(1)):
            lines[i] = lines[i][m.end():]
            if not lines[i].strip():
                i += 1
            continue
        if (_BYLINE_LINE_RE.match(s) or _DATE_LINE_RE.match(s)
                or _TIME_LINE_RE.match(s) or _STAMP_LINE_RE.match(s)
                or _OUTLET_LINE_RE.match(s) or _COPYRIGHT_LINE_RE.match(s)):
            i += 1
            continue
        if _TITLE_LINE_RE.match(s) and i + 1 < n:
            nxt = lines[i + 1].strip()
            if (_BYLINE_LINE_RE.match(nxt) or _DATE_LINE_RE.match(nxt)
                    or _TIME_LINE_RE.match(nxt) or _OUTLET_LINE_RE.match(nxt)):
                i += 1
                continue
        break
    out = "\n".join(lines[i:]).strip()
    # Strip any remaining parenthetical outlet stamps (e.g. " (Reuters) "
    # mid-text or inside datelines the dateline regex already handled).
    return _PAREN_OUTLET_RE.sub(" ", out).strip()


def _get_stopwords() -> set[str]:
    """Return the cached set of stopwords with ``not`` removed."""
    global _STOPWORDS_CACHE
    if _STOPWORDS_CACHE is not None:
        return _STOPWORDS_CACHE
    try:
        words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = set(stopwords.words("english"))
    words.discard("not")
    _STOPWORDS_CACHE = words
    return _STOPWORDS_CACHE


def ensure_stopwords_available() -> None:
    """Pre-download NLTK stopwords if they are not yet present."""
    _get_stopwords()


def clean_single_text(text: str) -> str:
    """Preprocess a single raw string into a cleaned, space-joined corpus row.

    Returns the cleaned text string ready to be fed to the vectorizer.
    """
    cleaned = _NON_ALPHA.sub(" ", text)
    cleaned = cleaned.lower()
    tokens = cleaned.split()

    if not tokens:
        return ""

    stopwords_set = _get_stopwords()
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens if token not in stopwords_set]
    return " ".join(tokens)


def clean_corpus(texts: Iterable[str]) -> list[str]:
    """Preprocess an iterable of raw texts into a cleaned corpus list."""
    return [clean_single_text(text) for text in texts]


def tokenize(text: str) -> list[str]:
    """Split a raw string into its (unstemmed) lowercased tokens."""
    cleaned = _NON_ALPHA.sub(" ", text)
    return cleaned.lower().split()
