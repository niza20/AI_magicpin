"""AGENT 6 — Category Expert. Derives voice/register, vocabulary, taboos and offer style from the
supplied CategoryContext. Slug-based defaults exist only as a fallback when the context is silent."""
from __future__ import annotations

import re

from ..tools import ContextTools
from ..types import CategoryProfile, TriggerAnalysis
from ..util import humanize
from .base import Agent

_REGISTER_KEYWORDS = [
    ("clinical", ("clinical", "medical", "technical", "doctor", "peer_clinical", "evidence")),
    ("trust", ("trust", "precise", "pharma", "careful", "compliance", "accurate")),
    ("coach", ("coach", "motivat", "energetic", "fitness", "trainer")),
    ("operator", ("operator", "business", "hustle", "numbers", "owner")),
    ("warm", ("warm", "friendly", "beauty", "stylist", "cheerful", "caring")),
]
_SLUG_DEFAULTS = {
    # slug: (register, noun, customer noun)
    "dentists": ("clinical", "clinic", "patients"),
    "doctors": ("clinical", "clinic", "patients"),
    "salons": ("warm", "salon", "clients"),
    "restaurants": ("operator", "restaurant", "diners"),
    "gyms": ("coach", "gym", "members"),
    "pharmacies": ("trust", "pharmacy", "customers"),
}
_GLOBAL_TABOOS = ["guaranteed", "guarantee", "100%", "cure", "miracle", "best in town", "no.1", "number one"]


class CategoryExpert(Agent):
    name = "category_expert"

    def run(self, tools: ContextTools, ta: TriggerAnalysis) -> CategoryProfile:
        slug = tools.category_slug or "business"
        voice = tools.get_category_fact("voice") or {}
        tone = str(voice.get("tone") or voice.get("style") or "")
        register = self._register(tone, slug)
        _, noun, cust = _SLUG_DEFAULTS.get(slug, (register, "business", "customers"))
        noun = str(tools.get_category_fact("business_noun") or noun)
        cust = str(tools.get_category_fact("customer_noun") or cust)
        vocab = self._list(voice, ("vocab_allowed", "allowed_vocabulary", "vocabulary", "preferred_terms"))
        taboos = self._list(voice, ("taboos", "vocab_taboo", "taboo_words", "banned", "avoid"))
        taboos = list(dict.fromkeys([t for t in taboos + _GLOBAL_TABOOS if t]))
        offers = [t for t, _ in tools.get_catalog_offers()]
        peer_scope = tools.peer_scope()
        peer_label = self._peer_label(peer_scope, noun, tools)
        hype_forbidden = register in ("clinical", "trust") or bool(re.search(r"no hype|not promotional|peer", tone.lower()))
        rules = [f"register={register}", "no ALL-CAPS or '!!!' hype", "service+price over % discounts",
                 f"refer to end-customers as '{cust}'"]
        if register == "clinical":
            rules += ["clinical-peer voice; technical terms OK", "no outcome claims ('cure', 'guaranteed')"]
        elif register == "trust":
            rules += ["precise and calm; no health claims", "cite exact product/regulation names only if in context"]
        elif register == "coach":
            rules += ["energetic coaching voice, but numbers-first"]
        elif register == "operator":
            rules += ["operator-to-operator: covers, orders, footfall, margins"]
        elif register == "warm":
            rules += ["warm, practical, friendly; emoji sparingly"]
        prof = CategoryProfile(
            slug=slug, register=register, tone=tone or register, vocab_allowed=vocab, taboos=taboos,
            hype_forbidden=hype_forbidden, offer_examples=offers, peer_label=peer_label, noun_singular=noun,
            customer_noun=cust, style_rules=rules,
            relevant_digest=[d for d, _ in tools.search_category_digest(terms=tools.signals() + [ta.trigger_type])[:3]],
            relevant_beats=[b for b, _ in tools.seasonal_beats()],
            relevant_trends=[t for t, _ in tools.trend_signals()][:3],
        )
        tools.ledger.allow_words(w for v in vocab for w in re.findall(r"[A-Za-z]+", v))
        self.log(f"slug={slug}, register={register}, taboos={len(taboos)}, vocab={len(vocab)}, peer='{peer_label}'")
        return prof

    @staticmethod
    def _register(tone: str, slug: str) -> str:
        t = tone.lower()
        for reg, keys in _REGISTER_KEYWORDS:
            if any(k in t for k in keys):
                return reg
        return _SLUG_DEFAULTS.get(slug, ("neutral",))[0]

    @staticmethod
    def _list(voice: dict, keys: tuple[str, ...]) -> list[str]:
        out: list[str] = []
        for k in keys:
            v = voice.get(k)
            if isinstance(v, list):
                out += [str(x) for x in v]
            elif isinstance(v, str):
                out += [s.strip() for s in v.split(",")]
        return out

    @staticmethod
    def _peer_label(scope, noun: str, tools: ContextTools) -> str:
        city = str(tools.get_merchant_fact("identity.city") or "")
        plural = noun[:-1] + "ies" if noun.endswith("y") else noun + ("es" if noun.endswith("s") else "s")
        def cap_city(label: str) -> str:
            w = label.split()
            return " ".join([w[0].title()] + w[1:]) if w and scope and scope.split("_")[0].isalpha() and len(w) > 1 else label
        if isinstance(scope, str) and scope.strip() and city and city.lower() not in scope.lower() \
                and "_" in scope and scope.split("_")[0].isalpha() and scope.split("_")[0].lower() not in ("all", "india", "national", "similar", "solo", "independent"):
            # benchmark is scoped to another city → don't imply it's this merchant's local market
            return f"the {noun} benchmark ({cap_city(humanize(scope))})"
        if isinstance(scope, str) and scope.strip():
            s = humanize(scope)
            s = re.sub(r"\bpractices?\b", "practices", s)
            words = s.split()
            city = str(tools.get_merchant_fact("identity.city") or "")
            return " ".join(w.title() if w.lower() == city.lower() else w for w in words)
        plural = noun[:-1] + "ies" if noun.endswith("y") else noun + ("es" if noun.endswith("s") else "s")
        return f"similar {plural}"
