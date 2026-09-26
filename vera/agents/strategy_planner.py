"""AGENT 5 — Strategy Planner. Decides WHAT THE MESSAGE SHOULD ACHIEVE.

Produces 2-3 candidate plans (different lever mixes / CTA shapes) so the critic can compare real
alternatives instead of polishing one draft. Each plan uses the strongest 1-3 levers only.
Lever availability is fact-gated: e.g. social proof is only offered when a peer stat exists,
loss aversion only when a loss fact (gap, dip, expiry, lapse) exists.
"""
from __future__ import annotations

from typing import Optional

from ..llm import get_llm
from ..types import CategoryProfile, CustomerPlan, IntentResult, Personalization, StrategyPlan, TriggerAnalysis
from .base import Agent

# family -> list of (variant, levers, cta_key)
VARIANTS: dict[str, list[tuple[str, list[str], str]]] = {
    "knowledge": [("digest_effort", ["specificity", "reciprocity", "effort_externalization"], "effort"),
                  ("digest_curiosity", ["specificity", "curiosity"], "curiosity")],
    "regulation": [("compliance_effort", ["specificity", "loss_aversion", "effort_externalization"], "effort"),
                   ("compliance_curiosity", ["specificity", "curiosity"], "curiosity")],
    "perf_dip": [("dip_loss_effort", ["specificity", "loss_aversion", "effort_externalization"], "effort"),
                 ("dip_social_proof", ["specificity", "social_proof", "effort_externalization"], "effort"),
                 ("dip_ask", ["specificity", "asking_merchant"], "ask")],
    "perf_spike": [("spike_effort", ["specificity", "loss_aversion", "effort_externalization"], "effort"),
                   ("spike_curiosity", ["specificity", "curiosity"], "curiosity")],
    "milestone": [("milestone_proof", ["specificity", "social_proof", "effort_externalization"], "effort"),
                  ("milestone_ask", ["specificity", "reciprocity", "asking_merchant"], "ask")],
    "competitor": [("competitor_proof", ["specificity", "social_proof", "effort_externalization"], "effort"),
                   ("competitor_loss", ["specificity", "loss_aversion", "effort_externalization"], "effort"),
                   ("competitor_curiosity", ["specificity", "curiosity"], "curiosity")],
    "trend": [("trend_effort", ["specificity", "social_proof", "effort_externalization"], "effort"),
              ("trend_ask", ["specificity", "asking_merchant"], "ask")],
    "festival": [("festival_effort", ["specificity", "loss_aversion", "effort_externalization"], "effort"),
                 ("festival_ask", ["specificity", "asking_merchant"], "ask")],
    "weather": [("weather_effort", ["specificity", "reciprocity", "effort_externalization"], "effort"),
                ("weather_ask", ["specificity", "asking_merchant"], "ask")],
    "local_event": [("event_effort", ["specificity", "loss_aversion", "effort_externalization"], "effort"),
                    ("event_ask", ["specificity", "asking_merchant"], "ask")],
    "reputation": [("reviews_effort", ["specificity", "loss_aversion", "effort_externalization"], "effort"),
                   ("reviews_ask", ["specificity", "asking_merchant"], "ask")],
    "dormant": [("dormant_ask", ["reciprocity", "asking_merchant"], "ask"),
                ("dormant_effort", ["specificity", "loss_aversion", "effort_externalization"], "effort")],
    "recurring": [("recurring_ask", ["curiosity", "asking_merchant"], "ask"),
                  ("recurring_effort", ["specificity", "effort_externalization"], "effort")],
    "account": [("account_loss", ["specificity", "loss_aversion", "single_binary_commitment"], "effort"),
                ("account_value", ["specificity", "reciprocity", "single_binary_commitment"], "effort")],
    "profile": [("profile_effort", ["specificity", "loss_aversion", "effort_externalization"], "effort"),
                ("profile_proof", ["specificity", "social_proof", "effort_externalization"], "effort")],
    "offer": [("offer_effort", ["specificity", "loss_aversion", "effort_externalization"], "effort"),
              ("offer_curiosity", ["specificity", "curiosity"], "curiosity")],
    "customer_recall": [("recall_slots", ["specificity", "single_binary_commitment"], "slot"),
                        ("recall_binary", ["specificity", "single_binary_commitment"], "confirm")],
    "customer_promo": [("promo_confirm", ["specificity", "single_binary_commitment"], "confirm")],
    "customer_appointment": [("appointment_confirm", ["specificity", "single_binary_commitment"], "confirm")],
    "generic": [("generic_effort", ["specificity", "effort_externalization"], "effort"),
                ("generic_curiosity", ["specificity", "curiosity"], "curiosity")],
}

CTA_VALUE = {"effort": "binary_yes_stop", "confirm": "binary_yes_stop", "slot": "open_ended",
             "curiosity": "open_ended", "ask": "open_ended", "info": "none"}


class StrategyPlanner(Agent):
    name = "strategy_planner"
    uses_llm = True

    def run(self, ta: TriggerAnalysis, intent: IntentResult, pers: Personalization, pz: dict,
            prof: CategoryProfile, cust: Optional[CustomerPlan], lang: str,
            deadline: Optional[float] = None) -> list[StrategyPlan]:
        variants = VARIANTS.get(ta.family, VARIANTS["generic"])
        facts = list(dict.fromkeys(ta.key_fact_ids + pers.personalization_facts))
        has_peer = any(k in pz for k in ("peer_ctr", "peer_rating", "peer_reviews"))
        has_loss = any(k in pz for k in ("ctr_gap", "stale_days", "lapsed", "expired_offer")) or \
            ta.family in ("perf_dip", "regulation", "account", "festival", "competitor", "reputation", "local_event")
        plans = []
        for name, levers, cta in variants:
            lv = [l for l in levers
                  if not (l == "social_proof" and not has_peer and ta.family not in ("trend", "milestone"))
                  and not (l == "loss_aversion" and not has_loss)]
            if len(lv) < len(levers) and "effort_externalization" not in lv and cta == "effort":
                lv.append("effort_externalization")
            plans.append(StrategyPlan(
                variant=name, objective=ta.primary_goal,
                hook=f"why-now: {ta.why_now}", facts_to_use=facts, engagement_levers=lv[:3],
                cta_strategy=cta, tone=f"{prof.register}/{lang}",
                message_structure=["hook(why-now + salutation)", "specific fact", "why it matters (lever)", "one low-friction CTA"]))
        plans = self._llm_rank(plans, ta, pz, deadline)
        self.log(f"{len(plans)} candidate plans: " + "; ".join(f"{p.variant}{p.engagement_levers}->{p.cta_strategy}" for p in plans))
        return plans

    def _llm_rank(self, plans: list[StrategyPlan], ta: TriggerAnalysis, pz: dict, deadline) -> list[StrategyPlan]:
        """Optional: let the LLM reason about which plan best fits this merchant; order only (validated)."""
        llm = get_llm()
        if not llm.enabled or len(plans) < 2:
            return plans
        out = llm.complete_json(
            "strategy_planner",
            "You are the strategy planner in a merchant-engagement agent team. Pick which plan most likely earns a reply. JSON only.",
            f"Trigger: {ta.why_now} (family {ta.family}).\nMerchant facts: " +
            "; ".join(f"{k}={v['text']}" for k, v in pz.items() if k not in ('last_touch',)) +
            "\nPlans:\n" + "\n".join(f"- {p.variant}: levers={p.engagement_levers}, cta={p.cta_strategy}" for p in plans) +
            '\nReturn {"order": ["variant", ...], "reason": "..."}', deadline=deadline, max_tokens=600)
        order = out.get("order") if isinstance(out, dict) else None
        if isinstance(order, list):
            rank = {v: i for i, v in enumerate(order)}
            plans = sorted(plans, key=lambda p: rank.get(p.variant, 99))
        return plans
