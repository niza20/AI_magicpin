"""SUPPRESSION AGENT and AGENT 15 — Finalizer."""
from __future__ import annotations

import re
from typing import Optional

from ..tools import ContextTools
from ..types import Draft, TriggerAnalysis
from ..util import iso_week, parse_date, stable_hash
from .base import Agent

ALLOWED_CTA = {"binary_yes_stop", "open_ended", "none"}
ALLOWED_SEND_AS = {"vera", "merchant_on_behalf"}
FAMILY_PREFIX = {
    "knowledge": "research", "regulation": "regulation", "perf_dip": "perf", "perf_spike": "perf",
    "milestone": "milestone", "competitor": "competitor", "trend": "trend", "festival": "festival",
    "weather": "weather", "local_event": "local", "reputation": "reviews", "dormant": "dormant",
    "recurring": "recurring", "account": "account", "profile": "profile", "offer": "offer",
    "customer_recall": "recall", "customer_appointment": "appointment", "generic": "event",
}


class SuppressionAgent(Agent):
    """Deterministic dedup key: <family>:<category>:<merchant>[:<customer>]:<window>. Never random."""
    name = "suppression"

    def run(self, tools: ContextTools, ta: TriggerAnalysis) -> str:
        mid = str(tools.get_merchant_fact("merchant_id") or tools.get_trigger_fact("merchant_id") or "m_unknown")
        cid = tools.get_customer_fact("customer_id") or tools.get_trigger_fact("customer_id")
        slug = tools.category_slug or "general"
        given = tools.get_trigger_fact("suppression_key")
        if isinstance(given, str) and given.strip():
            parts = given.strip().split(":")
            if mid not in given:
                # merchant-scope keys like "research:dentists:2026-W17" are shared across merchants → scope them
                ins = len(parts) - 1 if len(parts) > 1 and re.search(r"\d{4}", parts[-1]) else len(parts)
                parts.insert(ins, mid)
            if cid and str(cid) not in ":".join(parts):
                parts.insert(len(parts) - 1 if re.search(r"\d{4}", parts[-1]) else len(parts), str(cid))
            key = ":".join(parts)
        else:
            ref = tools.reference_date() or parse_date(tools.get_trigger_fact("expires_at"))
            window = iso_week(ref) if ref else "t" + stable_hash([tools.get_trigger_fact("id"), ta.trigger_type])[:8]
            key = ":".join(x for x in [FAMILY_PREFIX.get(ta.family, "event"), slug, mid, str(cid) if cid else "", window] if x)
        self.log(key)
        return key


class Finalizer(Agent):
    name = "finalizer"

    def run(self, draft: Draft, send_as: str, suppression_key: str, rationale: str) -> dict:
        out = {"body": draft.body, "cta": draft.cta, "send_as": send_as,
               "suppression_key": suppression_key, "rationale": rationale}
        problems = validate_output(out)
        if problems:
            raise ValueError(f"finalizer schema violation: {problems}")
        self.log(f"final ({draft.source}, {len(out['body'])} chars, cta={out['cta']})")
        return out


def validate_output(out: dict) -> list[str]:
    p = []
    for k in ("body", "cta", "send_as", "suppression_key", "rationale"):
        if not isinstance(out.get(k), str) or not out.get(k).strip():
            p.append(f"{k} missing/empty")
    if out.get("cta") not in ALLOWED_CTA:
        p.append(f"cta '{out.get('cta')}' invalid")
    if out.get("send_as") not in ALLOWED_SEND_AS:
        p.append(f"send_as '{out.get('send_as')}' invalid")
    return p
