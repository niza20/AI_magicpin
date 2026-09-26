import copy
import re

import pytest

from bot import compose
from conftest import DS, ctx
from vera.agents.finalizer import validate_output
from vera.agents.trigger_analyst import classify_family
from vera.agents.validators import FactChecker
from vera.ledger import FactLedger
from vera.orchestrator import Orchestrator
from vera.types import Draft, StrategyPlan

ALL_TRIGGERS = sorted(DS.triggers)


def _recheck(cat, m, t, c, body):
    """Independent zero-fabrication check of a final body against a freshly built ledger."""
    orch = Orchestrator()
    w = orch.analyze(cat, m, t, c)
    d = Draft(segments=[("x", body)], cta="open_ended", plan=StrategyPlan("x", "", "", [], [], "", "", []))
    return FactChecker([]).run(d, w["ledger"], w["ta"], bool(c))


@pytest.mark.parametrize("tid", ALL_TRIGGERS)
def test_every_trigger_valid_grounded_and_deterministic(tid):
    cat, m, t, c = ctx(tid)
    out = compose(cat, m, t, c)
    assert not validate_output(out)
    assert set(out) == {"body", "cta", "send_as", "suppression_key", "rationale"}
    assert out == compose(*ctx(tid)), "compose must be deterministic"
    assert len(out["body"]) <= 640
    assert m["merchant_id"] in out["suppression_key"]
    if c and out["send_as"] == "merchant_on_behalf":
        assert c["customer_id"] in out["suppression_key"]
    chk = _recheck(cat, m, t, c, out["body"])
    assert chk.ok, [i.detail for i in chk.issues]
    taboos = (cat.get("voice") or {}).get("vocab_taboo", [])
    assert not any(re.search(r"(?<!\w)" + re.escape(x) + r"(?!\w)", out["body"].lower()) for x in taboos)


def test_all_five_categories_covered():
    slugs = {DS.merchants[t["merchant_id"]]["category_slug"] for t in DS.triggers.values()}
    assert slugs == {"dentists", "salons", "restaurants", "gyms", "pharmacies"}


@pytest.mark.parametrize("kind,family", [
    ("research_digest", "knowledge"), ("category_research_digest_release", "knowledge"), ("perf_spike", "perf_spike"),
    ("perf_dip", "perf_dip"), ("milestone_reached", "milestone"), ("dormant_with_vera", "dormant"),
    ("review_theme_emerged", "reputation"), ("festival_upcoming", "festival"), ("weather_heatwave", "weather"),
    ("local_news_event", "local_event"), ("regulation_change", "regulation"), ("competitor_opened", "competitor"),
    ("category_trend_movement", "trend"), ("scheduled_recurring", "recurring"), ("renewal_due", "account"),
    ("offer_expired", "offer"), ("totally_new_kind_xyz", "generic")])
def test_merchant_trigger_families(kind, family):
    assert classify_family(kind, "merchant") == family


@pytest.mark.parametrize("kind,family", [("recall_due", "customer_recall"), ("customer_lapsed_soft", "customer_recall"),
                                         ("appointment_tomorrow", "customer_appointment"), ("brand_new_customer_kind", "customer_recall")])
def test_customer_trigger_families(kind, family):
    assert classify_family(kind, "customer") == family


def test_research_digest_anchors_on_citation_and_merchant_cohort():
    out = compose(*ctx("trg_001_research_dentists"))
    b = out["body"]
    assert "JIDA" in b and "2,100" in b and "38%" in b
    assert b.startswith("Dr. Meera")
    assert "high risk adult" in b.lower()


def test_language_preferences():
    hinglish = compose(*ctx("trg_001_research_dentists"))["body"].lower()         # languages [en, hi]
    english = compose(*ctx("trg_003_perf_dip_smilecare"))["body"].lower()          # languages [en]
    assert re.search(r"\b(aapke|hai|doon|mein)\b", hinglish)
    assert not re.search(r"\b(aapke|hai|doon)\b", english)


def test_hindi_customer_gets_devanagari():
    cat, m, _, _ = ctx("trg_013_scheduled_ironcore")
    cust = copy.deepcopy(DS.customers["c_003_anita"])
    trig = {"id": "trg_t_anita", "scope": "customer", "kind": "membership_expiring", "merchant_id": m["merchant_id"],
            "customer_id": "c_003_anita", "payload": {"last_visit": "2026-08-01"}, "urgency": 2,
            "detected_at": "2026-09-26T09:00:00Z", "expires_at": "2026-10-10T00:00:00Z"}
    out = compose(cat, m, trig, cust)
    assert out["send_as"] == "merchant_on_behalf"
    assert re.search(r"[ऀ-ॿ]", out["body"])
    assert "Anita" in out["body"]


def test_customer_message_uses_real_slots_price_and_send_as():
    out = compose(*ctx("trg_014_recall_priya"))
    assert out["send_as"] == "merchant_on_behalf"
    assert "₹299" in out["body"] and "6pm" in out["body"] and "5pm" in out["body"]
    assert out["body"].startswith("Hi Priya")


def test_customer_without_consent_is_never_messaged():
    out = compose(*ctx("trg_017_recall_noconsent"))
    assert out["send_as"] == "vera"
    assert "consent" in out["rationale"].lower()
    assert "opt" in out["body"].lower()


# ------------------------------------------------------------- hidden-test resilience
def test_new_digest_item_is_used():
    cat, m, t, c = ctx("trg_001_research_dentists")
    cat["digest"].append({"id": "d_new_hidden", "kind": "research", "title": "Silver diamine fluoride halts 81% of early lesions",
                          "source": "Dental Tribune India, Nov 2026", "trial_n": 640, "summary": "SDF applied twice yearly."})
    t["payload"] = {"top_item_id": "d_new_hidden"}
    out = compose(cat, m, t, c)
    assert "Silver diamine fluoride" in out["body"] and "81%" in out["body"] and "640" in out["body"]
    assert "JIDA" not in out["body"]


def test_changed_metrics_flow_through():
    cat, m, t, c = ctx("trg_019_perf_dip_southspice")
    m["performance"].update({"calls": 77, "ctr": 0.012})
    t["payload"]["delta_pct"] = -0.35
    body = compose(cat, m, t, c)["body"]
    assert "35%" in body and "22%" not in body
    assert "77" in body or "1.2%" in body


def test_unknown_trigger_kind_uses_only_payload_facts():
    cat, m, t, c = ctx("trg_024_unknown_kind")
    out = compose(cat, m, t, c)
    assert "18,450" in out["body"] and "HDFC" in out["body"]
    assert _recheck(cat, m, t, c, out["body"]).ok


@pytest.mark.parametrize("mutate", [
    lambda cat, m, t: t.update(payload={}),
    lambda cat, m, t: t.pop("payload"),
    lambda cat, m, t: m.pop("performance"),
    lambda cat, m, t: m.update(offers=[], signals=[], customer_aggregate={}),
    lambda cat, m, t: cat.pop("voice"),
    lambda cat, m, t: cat.pop("peer_stats"),
    lambda cat, m, t: m["identity"].pop("owner_first_name"),
    lambda cat, m, t: t.pop("suppression_key") and t.pop("expires_at"),
])
@pytest.mark.parametrize("tid", ["trg_003_perf_dip_smilecare", "trg_008_diwali_pizza", "trg_001_research_dentists"])
def test_missing_optional_fields_still_valid(tid, mutate):
    cat, m, t, c = ctx(tid)
    mutate(cat, m, t)
    out = compose(cat, m, t, c)
    assert not validate_output(out)
    assert _recheck(cat, m, t, c, out["body"]).ok


def test_completely_empty_inputs_do_not_crash():
    out = compose({}, {}, {}, None)
    assert not validate_output(out)


def test_suppression_keys_deterministic_and_scoped():
    a = compose(*ctx("trg_001_research_dentists"))["suppression_key"]
    assert a == "research:dentists:m_001_drmeera:2026-W39"
    cat, m, t, c = ctx("trg_003_perf_dip_smilecare")
    t.pop("suppression_key", None)
    k1, k2 = compose(cat, m, t, c)["suppression_key"], compose(cat, m, t, c)["suppression_key"]
    assert k1 == k2 and re.fullmatch(r"perf:dentists:m_006_smilecare:\d{4}-W\d{2}", k1)


def test_no_repeat_of_previous_vera_message():
    cat, m, t, c = ctx("trg_003_perf_dip_smilecare")
    first = compose(cat, m, t, c)["body"]
    m["conversation_history"] = [{"from": "vera", "body": first, "engagement": "ignored"}]
    assert compose(cat, m, t, c)["body"] != first
