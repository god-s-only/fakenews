"""Phase 9 regression tests — source-format (dateline) leakage.

Locks in the discovered ML robustness defect and its mitigation so the
weakness cannot silently return (or be silently "hidden"):

* A fabricated claim wrapped in a ``CITY (Reuters) -  `` dateline is pushed
  from FAKE to REAL by the FROZEN model (this must keep failing — i.e. the
  raw pipeline must still exhibit the documented leak, so a silent regression
  is caught).
* The opt-in news-marker normaliser
  (``app.preprocessing.normalize_news_markers``) removes the dateline so the
  same claim is scored off its underlying content and returns to FAKE.
* The normaliser must leave ordinary prose (including the CBN / coral
  production probes) byte-for-byte untouched.
* The frozen Candidate D artifacts must never be silently replaced (same
  hashes as the release manifest).

These tests do NOT retrain, tune thresholds/labels, or touch the model files.
"""

import json
import pickle
from pathlib import Path

import pytest

from app import preprocessing
from app.config import settings

ROOT = Path(__file__).resolve().parent.parent


def _promoted_available() -> bool:
    return settings.model_file.exists() and settings.vectorizer_file.exists()


pytestmark = pytest.mark.skipif(
    not _promoted_available(),
    reason="Promoted model/vectorizer artifacts are not available",
)

CLAIM = (
    "Engineers at an unnamed startup have demonstrated a phone charger that "
    "ends the electricity shortage in their country. The device has "
    "apparently been kept secret so that power companies can continue "
    "overcharging customers."
)
DATELINE_WRAPPED = f"LONDON (Reuters) - {CLAIM}"
BBC_BYLINE = f"By JANE SMITH, BBC News\n{CLAIM}"
CNN_BANNER = f"(CNN) — By TIM JONES, CNN\nUpdated 09:45 GMT, 12 March 2026\n{CLAIM}"
AP_DATELINE = f"NEW YORK (AP) — {CLAIM}"
PUB_META = f"The National Ledger\nFriday, 12 March 2026\nBY TOM WREN\n{CLAIM}"


def _offline_probability(text: str, normalize: bool = False) -> float:
    with open(settings.model_file, "rb") as fh:
        model = pickle.load(fh)
    with open(settings.vectorizer_file, "rb") as fh:
        vec = pickle.load(fh)
    if normalize:
        text = preprocessing.normalize_news_markers(text) or text
    cleaned = preprocessing.clean_single_text(text)
    if not cleaned:
        return 0.5
    vector = vec.transform([cleaned]).toarray()
    if not vector.any():
        return 0.5
    return float(model.predict_proba(vector)[0][1])


# --------------------------------------------------------------------------- #
# 1. The documented leak must remain visible in the RAW (frozen) pipeline
# --------------------------------------------------------------------------- #
class TestLeakStillObservable:
    def test_fabricated_claim_is_fake_off_the_article(self):
        assert _offline_probability(CLAIM) == pytest.approx(0.2034, abs=1e-2)
        assert _offline_probability(CLAIM) < 0.5

    def test_reuters_dateline_flips_it_to_real(self):
        """The Phase-9 leak: LONDON (Reuters) -  dateline -> REAL 82.5%."""
        p = _offline_probability(DATELINE_WRAPPED)
        assert p == pytest.approx(0.8252, abs=1e-2)
        assert p > 0.6  # must remain the documented REAL misclassification

    def test_ap_and_bbc_stamps_pull_toward_fake_for_this_claim(self):
        # Format effect is directional per-outlet: for THIS claim the Reuters
        # dateline is a real-pusher while AP/BBC bylines contain rare tokens
        # that push the other way. Locking the observed direction.
        assert _offline_probability(AP_DATELINE) == pytest.approx(0.1101, abs=1e-2)
        assert _offline_probability(BBC_BYLINE) == pytest.approx(0.1403, abs=1e-2)
        assert _offline_probability(AP_DATELINE) < _offline_probability(CLAIM)

    def test_cnn_and_pubmeta_stamps_pull_toward_real(self):
        assert _offline_probability(CNN_BANNER) == pytest.approx(0.3209, abs=1e-2)
        assert _offline_probability(PUB_META) == pytest.approx(0.4352, abs=1e-2)
        assert _offline_probability(CNN_BANNER) > _offline_probability(CLAIM)
        assert _offline_probability(PUB_META) > _offline_probability(CLAIM)


# --------------------------------------------------------------------------- #
# 2. The normaliser must fully neutralise every source-format stamp
# --------------------------------------------------------------------------- #
class TestNormalizerStripsStamps:
    def test_reuters_dateline_stripped(self):
        norm = preprocessing.normalize_news_markers(DATELINE_WRAPPED)
        assert norm == CLAIM

    def test_bbc_byline_stripped(self):
        assert preprocessing.normalize_news_markers(BBC_BYLINE) == CLAIM

    def test_cnn_banner_and_updated_stamp_stripped(self):
        assert preprocessing.normalize_news_markers(CNN_BANNER) == CLAIM

    def test_ap_dateline_stripped(self):
        assert preprocessing.normalize_news_markers(AP_DATELINE) == CLAIM

    def test_publication_meta_stripped(self):
        assert preprocessing.normalize_news_markers(PUB_META) == CLAIM

    def test_normalizer_leaves_prose_untouched(self):
        prose = (
            "Researchers studying coral reefs have found that periods of "
            "unusually warm ocean temperatures can cause widespread coral "
            "bleaching."
        )
        assert preprocessing.normalize_news_markers(prose) == prose
        assert preprocessing.normalize_news_markers(
            "By many measures, this is ordinary prose."
        ) == "By many measures, this is ordinary prose."


# --------------------------------------------------------------------------- #
# 3. The normalised pipeline restores the FAKE verdict
# --------------------------------------------------------------------------- #
class TestMitigationNeutralizesAttack:
    def test_normalized_claim_returns_to_fake(self):
        p = _offline_probability(DATELINE_WRAPPED, normalize=True)
        assert p == pytest.approx(0.2034, abs=1e-2)
        assert p < 0.5

    def test_normalized_verdict_matches_plain_content(self):
        assert (
            _offline_probability(DATELINE_WRAPPED, normalize=True)
            == _offline_probability(CLAIM)
        )

    def test_legit_reuters_real_text_stays_real_after_normalization(self):
        reuters_real = (
            "WASHINGTON (Reuters) - The head of a conservative Republican "
            "faction in the U.S. Congress said on Thursday that party leaders "
            "were working to resolve differences over the fiscal measure."
        )
        p = _offline_probability(reuters_real, normalize=True)
        assert p > 0.5  # must remain REAL once dateline is stripped

    def test_cbn_and_guardian_probes_unchanged_by_normalization(self):
        cbn = (
            "The Central Bank of Nigeria said commercial banks will continue "
            "to operate under the existing cash withdrawal guidelines while "
            "customers are encouraged to use electronic payment channels. The "
            "bank said the policy is intended to improve the efficiency of "
            "the country's payment system and reduce reliance on physical "
            "cash."
        )
        assert preprocessing.normalize_news_markers(cbn) == cbn
        assert _offline_probability(cbn, normalize=True) == pytest.approx(0.9776, abs=1e-2)


# --------------------------------------------------------------------------- #
# 3b. Production inference must have the normalizer ENABLED
# --------------------------------------------------------------------------- #
class TestProductionInferenceEnablesNormalizer:
    def test_production_predict_neutralizes_dateline_leak(self):
        """The real ModelService.predict (app/model.py) must normalize, so a
        fabricated claim wrapped in a Reuters dateline returns to FAKE."""
        from app.model import ModelService

        svc = ModelService(settings.model_file, settings.vectorizer_file).load()
        pred = svc.predict(DATELINE_WRAPPED)
        assert pred.label == "fake"
        assert pred.probability_real == pytest.approx(0.2034, abs=1e-2)

    def test_production_predict_still_scores_legit_reuters_real(self):
        from app.model import ModelService

        svc = ModelService(settings.model_file, settings.vectorizer_file).load()
        reuters_real = (
            "WASHINGTON (Reuters) - The head of a conservative Republican "
            "faction in the U.S. Congress said on Thursday that party leaders "
            "were working to resolve differences over the fiscal measure."
        )
        pred = svc.predict(reuters_real)
        assert pred.label == "real"
        assert pred.probability_real == pytest.approx(0.999827, abs=1e-4)

    def test_production_predict_decision_matches_normalized_offline(self):
        from app.model import ModelService

        svc = ModelService(settings.model_file, settings.vectorizer_file).load()
        assert (
            svc.predict(DATELINE_WRAPPED).probability_real
            == _offline_probability(DATELINE_WRAPPED, normalize=True)
        )


# --------------------------------------------------------------------------- #
# 4. Frozen-artifact integrity (regression guard against silent retrains)
# --------------------------------------------------------------------------- #
class TestArtifactsFrozen:
    def test_production_artifacts_still_match_release_manifest(self):
        def sha(path: Path) -> str:
            digest = __import__("hashlib").sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        manifest = json.loads((ROOT / "reports/release_manifest.json").read_text())
        expected = manifest["new_artifacts"]
        assert sha(ROOT / "my_model_lr.pkl") == expected["my_model_lr.pkl"]["sha256"]
        assert (
            sha(ROOT / "my_tfidf_vectorizer.pkl")
            == expected["my_tfidf_vectorizer.pkl"]["sha256"]
        )


# --------------------------------------------------------------------------- #
# 5. The report itself is present and records the mitigation decision
# --------------------------------------------------------------------------- #
class TestReportPresent:
    def test_phase9_report_exists_and_records_decision(self):
        report = json.loads((ROOT / "reports/phase9_robustness_report.json").read_text())
        assert report["artifacts_byte_identical"] is True
        assert report["conclusion"]["preprocessing_viable"] is True
        assert report["conclusion"]["retraining_necessary_for_marker_attacks"] is False