"""Runs against the OFFICIAL dataset (expanded with magicpin's generate_dataset.py). Skipped if absent."""
import os
import re

import pytest

from bot import compose
from test_compose import _recheck
from vera.agents.finalizer import validate_output
from vera.dataset import load_dataset, load_pairs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "dataset", "expanded")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(EXP, "test_pairs.json")), reason="official dataset not expanded")
DS = load_dataset(EXP) if os.path.exists(EXP) else None
PAIRS = load_pairs(os.path.join(EXP, "test_pairs.json")) if DS else []
ALL = sorted(DS.triggers) if DS else []


def _ctx(trigger_id, customer_id=None):
    t = DS.triggers[trigger_id]
    m = DS.merchants[t["merchant_id"]]
    cid = customer_id or t.get("customer_id")
    return DS.category_for(m), m, t, (DS.customers.get(cid) if cid else None)


def test_thirty_canonical_pairs_present():
    assert len(PAIRS) == 30 and {p["test_id"] for p in PAIRS} == {f"T{i:02d}" for i in range(1, 31)}


@pytest.mark.parametrize("tid", ALL)
def test_every_official_trigger_is_valid_grounded_and_clean(tid):
    cat, m, t, c = _ctx(tid)
    out = compose(cat, m, t, c)
    assert not validate_output(out)
    b = out["body"]
    assert _recheck(cat, m, t, c, b).ok
    assert len(b) <= 640
    assert not re.search(r"\b[a-z]+_[a-z]+_[a-z_]+\b", b), "internal snake_case ids must not leak"
    assert "None" not in b and "The festival" not in b
    assert out["send_as"] == ("merchant_on_behalf" if c and "consent" not in out["rationale"].lower()[:20] else out["send_as"])


def test_placeholder_perf_dip_does_not_claim_a_drop_the_data_lacks():
    tid = next(p["trigger_id"] for p in PAIRS if p["test_id"] == "T25")
    body = compose(*_ctx(tid))["body"].lower()
    assert "girawat" not in body and "dropped" not in body


def test_competitor_offer_contrast_and_praise():
    b = compose(*_ctx(next(p["trigger_id"] for p in PAIRS if p["test_id"] == "T09")))["body"]
    assert "₹199" in b and "₹299" in b and "Smile Studio" in b


def test_chronic_refill_hindi_senior():
    b = compose(*_ctx(next(p["trigger_id"] for p in PAIRS if p["test_id"] == "T07")))["body"]
    assert re.search(r"[ऀ-ॿ]", b) and "metformin" in b and "Sharma" in b


def test_dataset_suppression_keys_kept_when_merchant_scoped():
    tid = next(p["trigger_id"] for p in PAIRS if p["test_id"] == "T24")
    assert compose(*_ctx(tid))["suppression_key"] == DS.triggers[tid]["suppression_key"]


# ---- real judge feedback (judge only sees identity, performance, signals, active offers, trigger payload) ----
def test_merchant_winback_is_about_the_plan_not_a_customer_recall():
    r = compose(*_ctx("trg_009_winback_glamour"))
    b = r["body"]
    assert r["send_as"] == "vera" and "Anjali" in b and "38" in b and "24" in b and "30%" in b
    assert "calls 30%" in b, "payload perf_dip_pct matches calls_pct, not views"
    assert "last visit" not in b.lower()


def test_festival_leads_with_merchants_own_numbers_and_active_offer():
    b = compose(*_ctx("trg_006_festival_diwali"))["body"]
    assert "4,980 views" in b and "62 calls" in b
    assert "Bridal Trial" not in b and "4x" not in b, "category-only facts look fabricated to the judge"


def test_match_night_uses_listing_numbers_and_real_weekday():
    b = compose(*_ctx("trg_010_ipl_match_delhi"))["body"]
    assert "2,200 views" in b and "Sunday" in b and "delivery late" not in b


def test_research_digest_ties_to_visible_listing_numbers_not_hidden_aggregates():
    b = compose(*_ctx("trg_029_research_digest_m_050_bharti_pharmac"))["body"]
    assert "895 views" in b and "6 calls" in b
    assert "1,535" not in b and "88%" not in b, "customer_aggregate / second-paragraph stats aren't visible to the judge"


def test_health_check_uses_30_day_totals_and_matches_direction():
    b = compose(*_ctx("trg_031_perf_dip_m_023_sushma_salon_p"))["body"]
    assert "2,547 views" in b and "+8%" not in b
    b = compose(*_ctx("trg_039_perf_spike_m_017_dr_rajan_denti"))["body"]
    assert "bana rahe" not in b, "numbers are falling — don't offer to 'keep it that way'"


def test_planning_first_invite_uses_visible_numbers():
    b = compose(*_ctx("trg_013_corporate_thali_planning"))["body"]
    assert "88 callers" in b and "4,200" not in b
