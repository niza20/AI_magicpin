"""AGENT 11 — Fact Checker and AGENT 12 — Policy/Constraint Checker. Pure deterministic Python.

Fact Checker compares every number, price, percentage, date/weekday/month, capitalised name,
quoted headline, URL, discount, competitor and research claim in the draft against the FACT LEDGER.
Anything it cannot trace → FAIL (error). No LLM is involved.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Optional

from ..ledger import FactLedger, extract_numbers
from ..types import CategoryProfile, CheckResult, CustomerPlan, Draft, Issue, TriggerAnalysis
from ..util import MONTHS, MONTHS_FULL, WEEKDAYS, WEEKDAYS_FULL
from .base import Agent

# Structural numbers that are not data claims: effort estimates ("2-min"), slot-choice keys ("Reply 1 or 2").
_SAFE_NUMBER_SPANS = [
    re.compile(r"\b[1-5][- ]?(min|mins|minute|minutes|page|pager)\b", re.I),
    re.compile(r"\b(reply|press|send|bhejein|भेजें)\s+[12]\b", re.I),
    re.compile(r"\b[12]\s+(ya|or|या)\s+[12]\b", re.I),
    re.compile(r"\b1\s+(for|ke liye)\b", re.I),
    re.compile(r"\b2\s+(for|ke liye)\b", re.I),
]
_COMMON = set("""a an the and or but if so to of in on at by for with from as is are was were be been being it its this that
these those your you yours we our us i me my he she they them their there here what which who whom whose when where why
how all any both each few more most other some such no nor not only own same than too very can will just don should now
yes stop reply ok okay hi hello dear team google whatsapp magicpin vera ctr gbp seo yoy sms pdf faq doctor dr mr ms mrs
monday tuesday wednesday thursday friday saturday sunday quick worth key heads up one two""".split())
_MONTH_WORDS = {m.lower() for m in MONTHS + MONTHS_FULL}
_WEEKDAY_WORDS = {w.lower() for w in WEEKDAYS + WEEKDAYS_FULL}


@lru_cache(maxsize=1)
def template_lexicon() -> frozenset:
    """Every word the deterministic realizers can emit (they are the only non-ledger text source)."""
    here = os.path.dirname(__file__)
    words: set[str] = set()
    for fn in ("composer.py", "../conversation.py"):
        try:
            with open(os.path.join(here, fn), encoding="utf-8") as f:
                src = f.read()
        except OSError:
            continue
        for lit in re.findall(r"(?:f?\"(?:[^\"\\\n]|\\.)*\")|(?:f?'(?:[^'\\\n]|\\.)*')", src):
            lit = re.sub(r"\{[^}]*\}", " ", lit)
            words.update(w.lower() for w in re.findall(r"[A-Za-zऀ-ॿ]+", lit))
    return frozenset(words - _MONTH_WORDS - _WEEKDAY_WORDS)


class FactChecker(Agent):
    name = "fact_checker"

    def run(self, draft: Draft, ledger: FactLedger, ta: TriggerAnalysis, has_customer: bool) -> CheckResult:
        body = draft.body
        issues: list[Issue] = []
        # 1. numbers / prices / percentages
        scrub = body
        for rx in _SAFE_NUMBER_SPANS:
            scrub = rx.sub(" ", scrub)
        scrub = re.sub(r"\bn=", " ", scrub)
        for n in extract_numbers(scrub):
            if not ledger.number_supported(n, draft.allowed_extra_numbers):
                issues.append(Issue("unsupported_number", f"{n:g} not traceable to any context fact"))
        # 1b. social-proof counts ("7 dentists near you did X") need a matching count fact, not just any number
        for m in re.finditer(r"\b(\d+)\s+(?:other\s+|more\s+|nearby\s+)?(dentists|doctors|clinics|salons|restaurants|gyms|"
                             r"pharmacies|chemists|merchants|businesses|practices|shops|stores|owners|peers|outlets)\b", body, re.I):
            n = float(m.group(1))
            backed = ledger.text_supported(m.group(0)) or any(
                isinstance(f.value, (int, float)) and not isinstance(f.value, bool) and float(f.value) == n
                and f.layer == "trigger" and re.search(r"count|num|n$|peers|merchants|nearby", f.label)
                for f in ledger.facts.values())
            if not backed:
                issues.append(Issue("unsupported_social_proof", f"'{m.group(0)}' has no count fact in context"))
        # 2. discounts must exist verbatim in context
        for m in re.finditer(r"(\d+)\s*%\s*(off|discount)", body, re.I):
            if not ledger.text_supported(m.group(0)):
                issues.append(Issue("invented_discount", f"'{m.group(0)}' is not an offer in context"))
        if re.search(r"\b(flat|upto|up to)\s+\d+\s*%", body, re.I) and not ledger.text_supported("%"):
            issues.append(Issue("invented_discount", "percentage discount framing not in context"))
        # 3. quoted headlines / quotes must be verbatim context
        for q in re.findall(r"[“\"]([^”\"]{4,})[”\"]", body):
            if not ledger.text_supported(q.strip().rstrip(".…")):
                issues.append(Issue("unsupported_quote", f"quoted text not in context: {q[:60]}"))
        # 4. capitalised names, months, weekdays
        lex = template_lexicon()
        for sent in re.split(r"(?<=[.!?:;—–\-\n(])\s+|[“”\"]", body):
            toks = re.findall(r"[A-Za-z][A-Za-z'’\.]*", sent)
            for i, tok in enumerate(toks):
                w = re.split(r"['’]", tok.strip("."))[0].lower().strip(".")
                if not w:
                    continue
                if (w in _MONTH_WORDS or w in _WEEKDAY_WORDS) and tok[0].isupper():
                    if w == "may" and not re.search(r"\d\s*May\b|\bMay\s*\d", sent):
                        continue
                    if not ledger.word_known(w):
                        issues.append(Issue("unsupported_date", f"'{tok}' is not a date present in context"))
                    continue
                if i == 0 or not tok[0].isupper():
                    continue
                if not (ledger.word_known(w) or w in lex or w in _COMMON):
                    issues.append(Issue("unsupported_entity", f"'{tok}' is not a name/term found in context"))
        # 5. competitor claims only when a competitor fact exists
        if re.search(r"\b(competitor|rival|competition)\b", body, re.I) and ta.family != "competitor" \
                and not ledger.find(label="competitor_name") and not ledger.text_supported("competitor"):
            issues.append(Issue("invented_competitor", "mentions a competitor but none exists in context"))
        # 6. research/citation claims only when a digest item backs them
        verbatim_scrubbed = body
        for f in ledger.facts.values():
            if isinstance(f.value, str) and len(f.value) >= 4 and f.value in verbatim_scrubbed:
                verbatim_scrubbed = verbatim_scrubbed.replace(f.value, " ")
        if re.search(r"\b(study|trial|journal|research shows|according to|published)\b", verbatim_scrubbed, re.I) \
                and ta.family not in ("knowledge", "regulation") and not self._has_cited_digest(body, ledger):
            issues.append(Issue("invented_citation", "research claim without a digest item in context"))
        # 7. URLs must come from context
        for u in re.findall(r"https?://\S+", body):
            if not ledger.text_supported(u.rstrip(".,)")):
                issues.append(Issue("unsupported_url", u))
        # 8. social-proof counts like "3 dentists in your area" are caught by (1); names by (4).
        res = CheckResult(ok=not any(i.severity == "error" for i in issues), issues=_dedup(issues))
        self.log(("PASS" if res.ok else "FAIL") + f" ({len(res.issues)} issues)", [f"{i.code}: {i.detail}" for i in res.issues][:6])
        return res

    @staticmethod
    def _has_cited_digest(body: str, ledger: FactLedger) -> bool:
        for f in ledger.find(prefix="category.digest"):
            if f.label in ("title", "source") and isinstance(f.value, str) and f.value.lower()[:25] in body.lower():
                return True
        return False


_HYPE_OK = {"YES", "STOP", "CHANGE", "CONFIRM", "SEND", "DONE", "EDIT", "HELP", "CTR", "GBP", "SEO", "ORS", "YOY", "OK", "NO", "WA", "FAQ", "PDF", "SMS", "UPI", "GST", "AC", "BP", "ENT"}
_PREAMBLE = re.compile(r"(hope (you('| a)re|this finds you) (doing )?(well|good)|i am reaching out|i'?m reaching out|"
                       r"greetings (from|of the day)|i am writing to|allow me to introduce)", re.I)
_SELF_INTRO = re.compile(r"\b(i am vera|i'?m vera|this is vera|vera here|vera se bol rahi|main vera)\b", re.I)
_GENERIC = re.compile(r"\b(boost your (sales|business)|increase your sales|grow your business|take your business to the next level|"
                      r"best deals?|amazing (deal|offer)s?|don'?t miss out)\b", re.I)
_CTA_MARK = re.compile(r"(\?|reply\b|batayein|bataiye|भेजें|बताएं|reply karein)", re.I)


class PolicyChecker(Agent):
    name = "policy_checker"

    def run(self, draft: Draft, ledger: FactLedger, prof: CategoryProfile, ta: TriggerAnalysis,
            cust: Optional[CustomerPlan], send_as: str, language: str,
            previous_bodies: list[str], customer_phone: Optional[str] = None, max_len: int = 640) -> CheckResult:
        body = draft.body
        low = body.lower()
        issues: list[Issue] = []
        expected = "merchant_on_behalf" if cust else "vera"
        if send_as != expected:
            issues.append(Issue("send_as", f"send_as={send_as} but expected {expected}"))
        if cust and not cust.consent_ok:
            issues.append(Issue("no_consent", cust.consent_reason))
        for t in prof.taboos:
            if t and re.search(r"(?<!\w)" + re.escape(t.lower()) + r"(?!\w)", low):
                issues.append(Issue("taboo", f"taboo term '{t}'"))
        caps = [w for w in re.findall(r"\b[A-Z]{4,}\b", body) if w not in _HYPE_OK and not ledger.word_known(w)]
        if "!!" in body or (caps and prof.hype_forbidden):
            issues.append(Issue("hype", f"promotional hype ({caps[:3] or '!!'})"))
        elif caps:
            issues.append(Issue("hype", f"all-caps words {caps[:3]}", "warn"))
        if prof.hype_forbidden and "!" in body:
            issues.append(Issue("hype", "exclamation in clinical/trust register", "warn"))
        if _PREAMBLE.search(body):
            issues.append(Issue("preamble", "long preamble"))
        if _SELF_INTRO.search(body):
            issues.append(Issue("self_intro", "re-introduces Vera"))
        if _GENERIC.search(body):
            issues.append(Issue("generic_copy", "generic marketing phrasing", "warn"))
        n_reply_yes = len(re.findall(r"reply (yes|haan)|yes reply|yes भेजें", low))
        if n_reply_yes > 1:
            issues.append(Issue("multi_cta", "more than one YES ask"))
        if re.search(r"reply\s+\w+\s+for\b.*reply\s+\w+\s+for\b.*reply\s+\w+\s+for\b", low):
            issues.append(Issue("multi_cta", "three+ reply options"))
        if body.count("?") > 2:
            issues.append(Issue("multi_cta", "too many questions", "warn"))
        if draft.cta != "none":
            last = re.split(r"(?<=[.!?।])\s+", body.strip())[-1]
            if not _CTA_MARK.search(last):
                issues.append(Issue("buried_cta", "CTA is not in the final sentence"))
        if len(body) > max_len:
            issues.append(Issue("too_long", f"{len(body)} chars > {max_len}"))
        elif len(body) > 520:
            issues.append(Issue("long", f"{len(body)} chars", "warn"))
        if re.search(r"(?<![\d-])(\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?![\d-])|\d{11,}", body) or (customer_phone and customer_phone in body):
            issues.append(Issue("privacy", "phone number in body"))
        if re.search(r"[\w.]+@[\w.]+\.\w+", body):
            issues.append(Issue("privacy", "email address in body"))
        norm = re.sub(r"\W+", " ", low).strip()
        if any(norm == re.sub(r"\W+", " ", (p or "").lower()).strip() for p in previous_bodies):
            issues.append(Issue("repeat", "identical to a previous message"))
        hing = len(re.findall(r"\b(aap|aapke|aapka|aapki|hai|hain|kar|doon|mein|ke|ki|ka|se|ho|bhej|karein|liye)\b", low))
        if language == "hi-en" and hing < 2:
            issues.append(Issue("language", "merchant prefers Hinglish but message is English", "warn"))
        if language == "en" and hing >= 4:
            issues.append(Issue("language", "English merchant received Hinglish", "warn"))
        if len(body.strip()) < 30:
            issues.append(Issue("too_short", "empty/too short"))
        res = CheckResult(ok=not any(i.severity == "error" for i in issues), issues=_dedup(issues))
        self.log(("PASS" if res.ok else "FAIL") + f" ({len(res.issues)} issues)", [f"{i.code}: {i.detail}" for i in res.issues][:6])
        return res


def _dedup(issues: list[Issue]) -> list[Issue]:
    seen, out = set(), []
    for i in issues:
        k = (i.code, i.detail)
        if k not in seen:
            seen.add(k)
            out.append(i)
    return out
