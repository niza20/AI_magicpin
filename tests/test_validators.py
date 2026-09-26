"""Adversarial tests: the deterministic validators must catch fabrication regardless of who wrote the draft."""
import pytest

import vera.llm as llm_mod
from bot import compose
from conftest import ctx
from vera.orchestrator import Orchestrator
from vera.agents.validators import FactChecker, PolicyChecker
from vera.types import Draft, StrategyPlan

PLAN = StrategyPlan("t", "", "", [], [], "effort", "", [])


def _w(tid="trg_001_research_dentists"):
    return Orchestrator().analyze(*ctx(tid))


def _fact(body, tid="trg_001_research_dentists"):
    w = _w(tid)
    return FactChecker([]).run(Draft([("x", body)], "binary_yes_stop", PLAN), w["ledger"], w["ta"], False)


def _policy(body, tid="trg_001_research_dentists", send_as="vera", prev=(), cta="binary_yes_stop"):
    w = _w(tid)
    return PolicyChecker([]).run(Draft([("x", body)], cta, PLAN), w["ledger"], w["prof"], w["ta"], w["cust"],
                                 send_as, w["lang"].language, list(prev))


def codes(res):
    return {i.code for i in res.issues if i.severity == "error"}


def test_grounded_draft_passes():
    r = _fact("Dr. Meera, JIDA Oct 2026, p.14: “3-mo fluoride recall cuts caries 38% better than 6-mo” (n=2,100). "
              "Your CTR is 2.1% vs 3%. Want me to draft a post with your Dental Cleaning @ ₹299 offer? Reply YES.")
    assert r.ok, [i.detail for i in r.issues]


def test_1_invented_discount():
    assert "invented_discount" in codes(_fact("Dr. Meera, run a Flat 20% off on cleaning this week? Reply YES."))


def test_2_invented_competitor():
    r = _fact("Dr. Meera, your competitor Smile Dental opened next door. Want me to respond? Reply YES.")
    assert {"invented_competitor", "unsupported_entity"} & codes(r)


def test_3_changed_metric():
    assert "unsupported_number" in codes(_fact("Dr. Meera, your CTR is 3.4% this month. Want a fix? Reply YES."))


def test_wrong_price():
    assert "unsupported_number" in codes(_fact("Dr. Meera, promote Cleaning @ ₹199 today? Reply YES."))


def test_invented_citation_outside_digest_trigger():
    r = _fact("Dr. Arjun, a new study shows clinics lose calls on weekends. Want help? Reply YES.", "trg_003_perf_dip_smilecare")
    assert "invented_citation" in codes(r)


def test_fake_social_proof_count():
    assert {"unsupported_number", "unsupported_social_proof"} & codes(_fact("Dr. Meera, 7 dentists in Lajpat Nagar did this last month. Reply YES."))


def test_fake_quote_and_date():
    r = _fact("Dr. Meera, patients said “best clinic ever” on Friday 13 Mar. Reply YES.")
    assert {"unsupported_quote", "unsupported_date"} <= codes(r) | {"unsupported_number"}


@pytest.mark.parametrize("body,code", [
    ("Dr. Meera, this treatment is guaranteed to work. Want it? Reply YES.", "taboo"),
    ("Dr. Meera, AMAZING DEAL for your clinic!!! Reply YES.", "hype"),
    ("Dr. Meera, I hope you're doing well. I am reaching out about posts. Reply YES.", "preamble"),
    ("Dr. Meera, Reply YES for posts. Also reply YES for offers.", "multi_cta"),
    ("Dr. Meera, want me to draft it? Reply YES. Anyway, your clinic looks good.", "buried_cta"),
    ("Dr. Meera, call Priya at 9876543210 to book. Reply YES.", "privacy"),
    ("Hi, this is Vera from magicpin. Want me to help? Reply YES.", "self_intro"),
])
def test_policy_violations(body, code):
    assert code in codes(_policy(body))


def test_policy_send_as_and_repeat():
    body = "Dr. Meera, want me to draft a post? Reply YES."
    assert "send_as" in codes(_policy(body, send_as="merchant_on_behalf"))
    assert "repeat" in codes(_policy(body, prev=[body]))


# ---------------------------------------------------------------- LLM boundary
class FakeLLM:
    def __init__(self, body):
        self.enabled, self.model, self._body = True, "fake", body
        self.stats = {}

    def complete_json(self, agent, system, user, deadline=None, max_tokens=0):
        if agent == "message_composer":
            return {"body": self._body, "used_fact_ids": []}
        return None


def test_llm_fabrication_is_rejected(monkeypatch):
    fake = FakeLLM("Dr. Meera, JIDA says 3-month recall is 90% better and 12 clinics near you use it. Reply YES.")
    monkeypatch.setattr(llm_mod, "_LLM", fake)
    out = compose(*ctx("trg_001_research_dentists"))
    assert "90%" not in out["body"] and "12 clinics" not in out["body"]


def test_llm_grounded_draft_can_win(monkeypatch):
    good = ("Dr. Meera, JIDA Oct 2026, p.14 just landed — “3-mo fluoride recall cuts caries 38% better than 6-mo” (n=2,100). "
            "Relevant for your high risk adult patients. Your CTR is 2.1% vs 3% for Delhi solo practices. "
            "Want me to draft a patient WhatsApp and a Google post with your Dental Cleaning @ ₹299 offer? Reply YES.")
    monkeypatch.setattr(llm_mod, "_LLM", FakeLLM(good))
    res = Orchestrator().compose(*ctx("trg_001_research_dentists"))
    assert res.output["body"] == good          # validated + scored ≥ deterministic best → adopted


def test_llm_failure_falls_back(monkeypatch):
    class Broken(FakeLLM):
        def complete_json(self, *a, **k):
            return None
    monkeypatch.setattr(llm_mod, "_LLM", Broken(""))
    out = compose(*ctx("trg_003_perf_dip_smilecare"))
    assert "41%" in out["body"]
