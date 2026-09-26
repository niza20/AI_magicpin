"""AGENT 13 — Engagement Critic (+ the internal evaluation simulator it is built on).

Scores a draft 0-10 on the judge's five dimensions using observable, deterministic signals, then
names concrete weaknesses the Rewriter can act on. Optionally asks the LLM for a qualitative note
(never overrides the deterministic scores, only adds weaknesses).
"""
from __future__ import annotations

import re
from typing import Optional

from ..ledger import FactLedger, extract_numbers
from ..llm import get_llm
from ..types import CategoryProfile, CheckResult, Critique, Draft, TriggerAnalysis
from .base import Agent

LEVER_MARKERS = {
    "loss_aversion": r"(dropp|slipping|gap|lose|losing|missing|before|fade|too late|girawat|gap close|pehle|nahi|chips away|lapse)",
    "social_proof": r"(peer|average|benchmark|vs |head start|similar|above the)",
    "effort_externalization": r"(want me to|main .{0,60}(kar doon|bana doon|laga doon|bhej doon|draft)|i'll take it|sambhal|i'?ll)",
    "curiosity": r"(want to see|dekhna|want the|side-by-side|what's driving|kya wajah)",
    "reciprocity": r"(flagging|noticed|wanted to flag|thought you|share kar rahi|relevant)",
    "asking_merchant": r"(what's the one|what do you|what are you|do you offer|did anything|kya plan|kya aap|kis cheez|is something)",
    "single_binary_commitment": r"(reply yes|yes reply|reply 1 or 2|1 ya 2)",
}


class EngagementCritic(Agent):
    name = "engagement_critic"
    uses_llm = True

    def run(self, draft: Draft, ledger: FactLedger, ta: TriggerAnalysis, prof: CategoryProfile, pz: dict,
            language: str, fact: CheckResult, policy: CheckResult, deadline: Optional[float] = None,
            use_llm: bool = False) -> Critique:
        body = draft.body
        low = body.lower()
        s: dict[str, float] = {}
        w: list[str] = []

        # 1. Specificity — verifiable anchors (numbers, quoted headlines, sources, dates)
        nums = {n for n in extract_numbers(re.sub(r"\b[1-5]-min\b|\breply [12]\b", "", low)) if n >= 1 or 0 < n < 1}
        quoted = len(re.findall(r"[“\"][^”\"]{4,}[”\"]", body))
        src = 1 if any(ta.anchor.get(k, {}).get("text", "\0") in body for k in ("source", "date", "effective")) else 0
        anchors = len(nums) + quoted + src
        s["specificity"] = min(10.0, 3 + 1.6 * anchors)
        if re.search(r"\b\d+\s*%\s*off\b", low) and not ledger.text_supported("% off"):
            s["specificity"] -= 3
        if anchors < 2:
            w.append("specificity")

        # 2. Category fit — voice + vocabulary + no taboo/hype
        cat = 7.0
        vocab_hits = sum(1 for v in prof.vocab_allowed if v and v.lower() in low)
        cat += min(2, vocab_hits)
        if any(o.lower() in low for o in prof.offer_examples) or (pz.get("offer") and pz["offer"]["text"].lower() in low):
            cat += 1
        if any(i.code in ("taboo", "hype") and i.severity == "error" for i in policy.issues):
            cat -= 5
        if any(i.code == "generic_copy" for i in policy.issues):
            cat -= 2
        customer_facing = ta.family.startswith("customer_")
        if prof.register == "clinical" and not customer_facing and "dr." not in low:
            cat -= 1
        s["category_fit"] = max(0.0, min(10.0, cat))
        if s["category_fit"] < 7:
            w.append("category_fit")

        # 3. Merchant fit — this merchant's name/numbers/offers + language honoured
        mf = 2.0
        for k, pts in (("salutation", 2), ("name", 1), ("locality", 1), ("ctr", 1.5), ("views", 1.5), ("calls", 1), ("directions", 1.5),
                       ("offer", 1.5), ("rating", 1), ("reviews", 1), ("stale_days", 1), ("segment_match", 2),
                       ("lapsed", 1.5), ("customers_total", 1)):
            v = pz.get(k)
            if v and str(v["text"]).lower() in low:
                mf += pts
        if not any(i.code == "language" for i in policy.issues):
            mf += 1.5
        if ta.family.startswith("customer_"):
            # customer-facing: personalization is the customer's own relationship facts
            for k in ("last_service", "months", "last_visit", "slot1", "service", "when"):
                v = ta.anchor.get(k)
                if v and str(v.get("text", "")).lower() in low:
                    mf += 1.2
            if re.search(r"^(hi|namaste|नमस्ते) \w+", low):
                mf += 1.5
        s["merchant_fit"] = min(10.0, mf)
        if s["merchant_fit"] < 7:
            w.append("merchant_fit")

        # 4. Trigger relevance — trigger facts present + why-now framing
        trig_hits = sum(1 for k, v in ta.anchor.items() if not k.startswith("_") and v.get("text") and str(v["text"]).lower()[:40] in low)
        why_now = bool(re.search(r"\b(today|this week|just|days away|days left|now|abhi|aaj|is hafte|yesterday|week-over-week|"
                                 r"heads-up|reminder|due|crossed|opened|landed|mention|up \d|dropped|jumped|since we last spoke|check-in|expired|days ago)\b", low))
        s["trigger_relevance"] = min(10.0, 3 + 2 * min(3, trig_hits) + (1.5 if why_now else 0))
        if s["trigger_relevance"] < 7:
            w.append("trigger_relevance")

        # 5. Engagement compulsion — levers present, single clear CTA at the end, brevity
        levers = [k for k, rx in LEVER_MARKERS.items() if re.search(rx, low)]
        eng = 3 + 1.2 * min(4, len(levers))
        last = re.split(r"(?<=[.!?।])\s+", body.strip())[-1].lower()
        if draft.cta != "none" and re.search(r"\?|reply|batayein|bataiye", last):
            eng += 1.5
        if draft.cta == "binary_yes_stop" or re.search(r"reply 1 or 2|1 ya 2", low):
            eng += 0.5
        if "asking_merchant" in levers and last.endswith(("?", ".")) and draft.cta == "open_ended":
            eng += 1.5      # brief §10: asking the merchant is the most under-used, highest-yield lever
        if len(body) > 520:
            eng -= 1.5
        if any(i.code in ("multi_cta", "buried_cta") for i in policy.issues):
            eng -= 3
        s["engagement"] = max(0.0, min(10.0, eng))
        if s["engagement"] < 7:
            w.append("engagement")

        if not fact.ok:
            for k in s:
                s[k] = max(0.0, s[k] - 4)
            w.append("fabrication")
        notes = [f"levers detected: {levers}", f"anchors={anchors}", f"trigger facts used={trig_hits}"]
        if use_llm:
            notes += self._llm_notes(body, ta, deadline)
        crit = Critique(scores={k: round(v, 1) for k, v in s.items()}, weaknesses=w, notes=notes)
        self.log(f"total={crit.total}/50 {crit.scores} weak={w}")
        return crit

    def _llm_notes(self, body: str, ta: TriggerAnalysis, deadline) -> list[str]:
        llm = get_llm()
        if not llm.enabled:
            return []
        out = llm.complete_json("engagement_critic",
                                "You are a strict critic of merchant WhatsApp messages. JSON only.",
                                f"Why-now: {ta.why_now}\nMessage: {body}\nName the single biggest weakness in <=15 words. "
                                'Return {"weakness": "..."}', deadline=deadline, max_tokens=300)
        return [f"llm: {out['weakness']}"] if out and isinstance(out.get("weakness"), str) else []
