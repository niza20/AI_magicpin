"""AGENT 9 — Language Agent. Decides en / hi-en / hi from (in order of authority):
latest real reply in the conversation → customer language_pref (customer-facing) → merchant identity.languages."""
from __future__ import annotations

from typing import Optional

from ..tools import ContextTools
from ..types import CategoryProfile, LanguagePlan
from .base import Agent
from .intent_router import detect_language


def normalise_pref(pref) -> Optional[str]:
    if pref is None:
        return None
    items = pref if isinstance(pref, list) else [pref]
    s = " ".join(str(x).lower() for x in items)
    has_hi = "hi" in s.replace("chi", "") or "hindi" in s or "hinglish" in s
    has_en = "en" in s or "english" in s or "mix" in s or "hinglish" in s
    if has_hi and has_en:
        return "hi-en"
    if has_hi:
        return "hi"
    if has_en:
        return "en"
    return None


class LanguageAgent(Agent):
    name = "language_agent"

    def run(self, tools: ContextTools, prof: CategoryProfile, customer_facing: bool,
            latest_reply: Optional[str] = None) -> LanguagePlan:
        source = "default"
        lang: Optional[str] = None
        if latest_reply:
            lang, source = detect_language(latest_reply), "latest reply"
        if not lang and customer_facing:
            lang, source = normalise_pref(tools.get_customer_fact("identity.language_pref")
                                          or tools.get_customer_fact("identity.language")), "customer language_pref"
        if not lang and not customer_facing:
            for h in reversed(tools.get_conversation_history(6)):
                if str(h.get("from", "")).lower() in ("merchant", "mx", "user") and (h.get("body") or h.get("text")):
                    lang, source = detect_language(h.get("body") or h.get("text")), "merchant's last message"
                    break
        if not lang:
            lang = normalise_pref(tools.get_merchant_fact("identity.languages")
                                  or tools.get_merchant_fact("identity.language_pref"))
            source = "merchant identity.languages"
        if not lang:
            lang, source = "en", "default"
        if lang == "hi" and not customer_facing:
            lang = "hi-en"   # Roman Hinglish reads most naturally for merchants on WhatsApp
        rules = {
            "en": ["plain Indian English", "short sentences"],
            "hi-en": ["natural Hinglish (Roman script), English for numbers/technical terms",
                      "don't translate brand/service names", "aap-form, respectful"],
            "hi": ["Devanagari Hindi; keep prices, names, service titles as in context"],
        }[lang]
        tone = {"clinical": "peer", "trust": "calm-precise", "coach": "coach", "operator": "operator", "warm": "warm"}.get(prof.register, "peer")
        plan = LanguagePlan(language=lang, tone=tone, style_rules=rules)
        self.log(f"language={lang} (from {source}), tone={tone}")
        return plan
