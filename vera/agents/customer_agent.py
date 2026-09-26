"""AGENT 8 — Customer Agent (only runs when CustomerContext exists).

Checks consent against the purpose of this message, picks the minimum customer facts needed,
and sets send_as=merchant_on_behalf. Never exposes phone numbers or other customers' data.
"""
from __future__ import annotations

from typing import Optional

from ..tools import ContextTools
from ..types import CustomerPlan, TriggerAnalysis
from ..util import first_present, humanize
from .base import Agent

PURPOSE_SCOPES = {
    "customer_recall": ("recall", "reminder", "winback", "win_back", "re_engagement", "reengagement", "follow_up", "service_reminders"),
    "customer_appointment": ("appointment", "booking", "reminder"),
}
PROMO_SCOPES = ("promotion", "promotions", "promotional", "marketing", "offers", "offer", "all")


class CustomerAgent(Agent):
    name = "customer_agent"

    def run(self, tools: ContextTools, ta: TriggerAnalysis) -> Optional[CustomerPlan]:
        if not tools.has_customer:
            return None
        ident = tools.get_customer_fact("identity") or {}
        name = str(first_present(ident, ["first_name", "name"], "") or "").split()[0] if first_present(ident, ["first_name", "name"]) else ""
        consent = tools.get_customer_fact("consent") or {}
        ok, why = self._consent(consent, ta.family, ta.trigger_type)
        rel = tools.get_customer_fact("relationship") or {}
        facts = []
        for k in ("last_visit", "visits_total", "services_received"):
            if rel.get(k):
                facts.append(f"customer.relationship.{k}")
        prefs = tools.get_customer_fact("preferences") or {}
        if prefs.get("preferred_slots"):
            facts.append("customer.preferences.preferred_slots")
        goal = {"customer_recall": "book the due visit in a preferred slot",
                "customer_appointment": "confirm (or reschedule) the appointment"}.get(ta.family, "re-engage with a relevant service")
        plan = CustomerPlan(customer_personalization=facts, customer_goal=goal, consent_ok=ok, consent_reason=why,
                            first_name=name)
        self.log(f"customer={name or '?'} consent_ok={ok} ({why}); goal={goal}")
        return plan

    @staticmethod
    def _consent(consent: dict, family: str, kind: str) -> tuple[bool, str]:
        if not consent:
            return False, "no consent record"
        if consent.get("opted_out") or consent.get("revoked") or str(consent.get("status", "")).lower() in ("revoked", "opted_out"):
            return False, "customer opted out"
        scope = consent.get("scope")
        if scope is None:
            return bool(consent.get("opted_in_at") or consent.get("opted_in")), "opt-in present, no scope restriction"
        scopes = [str(s).lower() for s in (scope if isinstance(scope, list) else [scope])]
        wanted = PURPOSE_SCOPES.get(family, ()) + tuple(kind.lower().split("_"))
        if any(any(w in s for w in wanted if len(w) > 3) for s in scopes):
            return True, f"scope covers purpose ({', '.join(scopes)})"
        if any(s in PROMO_SCOPES for s in scopes):
            return True, "promotional scope"
        return False, f"scope {scopes} does not cover {humanize(kind)}"
