"""AGENT 8 — Customer Agent (only runs when CustomerContext exists).

Checks consent against the purpose of this message, picks the minimum customer facts needed,
and sets send_as=merchant_on_behalf. Never exposes phone numbers or other customers' data.
"""
from __future__ import annotations

import re
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
        name = display_name(first_present(ident, ["first_name", "name"], ""))
        consent = tools.get_customer_fact("consent") or {}
        ok, why = self._consent(consent, ta.family, ta.trigger_type, tools.get_customer_fact("preferences") or {})
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
    def _consent(consent: dict, family: str, kind: str, prefs: dict) -> tuple[bool, str]:
        if not consent or not (consent.get("opted_in_at") or consent.get("opted_in") or consent.get("scope")):
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
        reminder_ok = prefs.get("reminder_opt_in") is True
        transactional = family in ("customer_appointment", "customer_recall")
        if family == "customer_promo":
            if any(any(p in s for p in ("promot", "offer", "marketing", "special", "updates")) for s in scopes):
                return True, f"promotional consent ({', '.join(scopes)})"
            return False, f"consent covers only {', '.join(scopes) or 'nothing'}, not promotional messages"
        if transactional and reminder_ok and scopes:
            return True, f"opted in ({', '.join(scopes)}) with reminder_opt_in=true"
        if family == "customer_recall" and any(any(p in s for p in ("promot", "offer", "marketing")) for s in scopes):
            return True, f"win-back covered by promotional consent ({', '.join(scopes)})"
        return False, f"scope {scopes} does not cover {humanize(kind)}"


def display_name(raw) -> str:
    """'Mr. Sharma' → 'Sharma' is kept with honorific; '(walk-in, no profile)' → ''."""
    s = str(raw or "").strip()
    if not s or s.startswith("(") or "no profile" in s.lower() or s.lower() in ("unknown", "anonymous"):
        return ""
    if re.match(r"^(mr|mrs|ms|miss|dr|shri|smt)\.?\s+\S", s, re.I):
        return s
    return s.split()[0]
