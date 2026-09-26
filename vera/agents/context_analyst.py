"""AGENT 1 — Context Analyst. Builds the FACT LEDGER and a bucketed fact sheet. Never infers."""
from __future__ import annotations

from typing import Optional

from ..ledger import FactLedger
from ..tools import ContextTools
from ..types import ContextFacts
from .base import Agent

_PERF_LABELS = {"views", "calls", "directions", "ctr", "leads", "searches", "clicks", "bookings", "orders",
                "views_pct", "calls_pct", "ctr_pct", "directions_pct", "leads_pct"}
_PEER_PREFIX = "category.peer_stats"


class ContextAnalyst(Agent):
    name = "context_analyst"

    def run(self, category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> tuple[FactLedger, ContextTools, ContextFacts]:
        ledger = FactLedger.build(category, merchant, trigger, customer)
        tools = ContextTools(category, merchant, trigger, customer, ledger)
        sheet = ContextFacts()
        for f in ledger.facts.values():
            entry = f.to_dict()
            p = f.path
            if p.startswith("merchant.performance"):
                sheet.performance_facts.append(entry)
            elif p.startswith("merchant.offers") or p.startswith("category.offer_catalog"):
                sheet.offer_facts.append(entry)
            elif p.startswith(_PEER_PREFIX):
                sheet.peer_facts.append(entry)
            elif p.startswith("merchant.conversation_history"):
                sheet.conversation_facts.append(entry)
            elif f.layer == "category":
                sheet.category_facts.append(entry)
            elif f.layer == "merchant":
                sheet.merchant_facts.append(entry)
            elif f.layer == "trigger":
                sheet.trigger_facts.append(entry)
            elif f.layer == "customer":
                sheet.customer_facts.append(entry)
        sheet.available_actions = self._actions(tools)
        self.log(f"ledger built: {len(ledger)} facts "
                 f"(perf={len(sheet.performance_facts)}, offers={len(sheet.offer_facts)}, "
                 f"peer={len(sheet.peer_facts)}, trigger={len(sheet.trigger_facts)}, customer={len(sheet.customer_facts)})")
        return ledger, tools, sheet

    @staticmethod
    def _actions(tools: ContextTools) -> list[str]:
        """What Vera can actually do for this merchant, based only on what exists in context."""
        acts = ["draft_google_post", "reply_to_reviews_draft"]
        if tools.get_active_offers() or tools.get_catalog_offers():
            acts.append("feature_offer_in_post")
        if tools.get_inactive_offers():
            acts.append("reactivate_offer")
        if tools.content_library():
            acts.append("share_customer_content")
        if tools.get_category_fact("digest"):
            acts.append("send_digest_summary")
        if tools.get_merchant_fact("customer_aggregate"):
            acts.append("customer_winback_campaign")
        if tools.get_merchant_fact("subscription"):
            acts.append("subscription_renewal")
        if tools.has_customer:
            acts.append("send_customer_message_on_behalf")
        return acts
