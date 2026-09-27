"""AGENT 10 — Message Composer. Runs only after context, trigger, strategy, category, personalization,
customer and language agents have finished.

Two drafting paths:
  * Deterministic realizer: builds the message from ledger-backed "parts" (hook / anchor fact /
    lever sentences / CTA options) in en, hi-en or hi. Every value it prints is read through the
    Brief accessors, which record the fact ids used (working memory).
  * LLM drafter (optional): receives the plan + a whitelist of facts (not the dataset) and the
    deterministic draft as a reference; its output is only accepted if the deterministic Fact
    Checker and Policy Checker pass it.
Structure: HOOK → SPECIFIC FACT → WHY IT MATTERS → ONE LOW-FRICTION CTA.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..llm import get_llm
from ..tools import ContextTools
from ..types import CategoryProfile, CustomerPlan, Draft, LanguagePlan, StrategyPlan, TriggerAnalysis
from ..util import humanize
from .base import Agent
from .strategy_planner import CTA_VALUE

_WINDOW_MAP = {
    "wow": ("week-over-week", "pichhle hafte ke mukable"), "w/w": ("week-over-week", "pichhle hafte ke mukable"),
    "week_over_week": ("week-over-week", "pichhle hafte ke mukable"), "7d": ("vs the previous 7 days", "pichhle 7 din ke mukable"),
    "dod": ("vs the day before", "kal ke mukable"), "yesterday": ("yesterday vs your average", "kal, aapke average ke mukable"),
    "vs_avg": ("vs your average", "aapke average ke mukable"), "30d": ("over the last 30 days", "pichhle 30 din mein"),
    "mom": ("month-over-month", "pichhle mahine ke mukable"), "yoy": ("year-over-year", "pichhle saal ke mukable"),
}


@dataclass
class Parts:
    hook: str
    anchor: list[str] = field(default_factory=list)
    levers: dict[str, str] = field(default_factory=dict)
    ctas: dict[str, str] = field(default_factory=dict)
    send_as: str = "vera"


class Brief:
    """Read-only view over the agents' outputs; records every fact the draft touches."""

    def __init__(self, ta: TriggerAnalysis, pz: dict, prof: CategoryProfile, lang: LanguagePlan,
                 cust: Optional[CustomerPlan], tools: ContextTools) -> None:
        self.ta, self.pz, self.prof, self.lang, self.cust, self.tools = ta, pz, prof, lang.language, cust, tools
        self.used: set[str] = set()

    def A(self, k: str) -> Optional[str]:
        v = self.ta.anchor.get(k)
        if not v or k.startswith("_") or v.get("text") in (None, ""):
            return None
        if v.get("fid"):
            self.used.add(v["fid"])
        return str(v["text"])

    def Araw(self, k: str):
        v = self.ta.anchor.get(k)
        return v.get("value") if v else None

    def P(self, k: str) -> Optional[str]:
        v = self.pz.get(k)
        if not v or v.get("text") in (None, ""):
            return None
        if v.get("fid"):
            self.used.add(v["fid"])
        return str(v["text"])

    def t(self, en: str, hien: Optional[str] = None, hi: Optional[str] = None) -> str:
        if self.lang == "hi" and hi:
            return hi
        if self.lang in ("hi-en", "hi") and hien:
            return hien
        return en

    @property
    def hinglish(self) -> bool:
        return self.lang in ("hi-en", "hi")

    def sal(self) -> str:
        s = self.P("salutation") or ""
        if not s:
            return "Hi"
        if self.hinglish and s and not s.lower().startswith("dr") and len(s.split()) == 1:
            return f"{s} ji"
        return s

    def cust_noun(self) -> str:
        return self.prof.customer_noun

    def offer(self) -> tuple[Optional[str], str]:
        o = self.P("offer")
        if o:
            return o, "active"
        o = self.P("expired_offer")
        if o:
            return o, "inactive"
        o = self.P("catalog_offer")
        return (o, "catalog") if o else (None, "")

    def offer_phrase(self) -> tuple[str, str]:
        """(en, hi-en) phrase for the offer, correctly attributed (yours vs. a catalog suggestion)."""
        o, kind = self.offer()
        if not o:
            return "", ""
        if kind == "active":
            return f"your {o} offer", f"aapke {o} offer"
        if kind == "inactive":
            return f"your {o} offer (currently off)", f"aapka {o} offer (abhi band hai)"
        return f"a '{o}' offer", f"'{o}' jaise offer"

    def pick_customer_offer(self) -> Optional[str]:
        """Merchant's own active offer best matching the customer (e.g. senior-citizen offer for seniors)."""
        offers = self.tools.get_active_offers()
        if not offers:
            return None
        senior = bool(self.tools.get_customer_fact("identity.senior_citizen"))
        for title, path in offers:
            if senior and "senior" in title.lower():
                f = self.tools.ledger.by_path(path)
                if f:
                    self.used.add(f.id)
                return title
        for title, path in offers:
            if "senior" not in title.lower():
                f = self.tools.ledger.by_path(path)
                if f:
                    self.used.add(f.id)
                return title
        return None

    def offer_matching(self, text: str) -> Optional[tuple[str, str]]:
        """Merchant offer (else catalog offer) sharing a meaningful word with `text`, e.g. 'bridal'."""
        words = {w for w in re.findall(r"[a-z]{5,}", (text or "").lower())} - {"season", "package", "bookings", "baseline", "primary", "festival"}
        pools = [(self.tools.get_active_offers(), "active"), (self.tools.get_catalog_offers(), "catalog")]
        for offers, kind in pools:
            for title, path in offers:
                if words & set(re.findall(r"[a-z]{5,}", title.lower())):
                    f = self.tools.ledger.by_path(path)
                    if f:
                        self.used.add(f.id)
                    return title, kind
        return None

    def offer_direct_hi(self) -> str:
        o, kind = self.offer()
        if not o:
            return ""
        return {"active": f"aapka {o} offer", "inactive": f"aapka {o} offer", "catalog": f"'{o}' jaisa offer"}[kind]

    def window(self, key: str = "window") -> tuple[str, str]:
        raw = self.Araw(key)
        if raw is None:
            return "", ""
        self.A(key)
        k = str(raw).lower().strip()
        if k in _WINDOW_MAP:
            return _WINDOW_MAP[k]
        h = humanize(raw)
        return h, h

    def merchant_short(self) -> str:
        name = self.P("name") or ""
        sal = self.P("salutation") or ""
        if self.prof.register == "clinical" and sal.lower().startswith("dr"):
            return f"{sal}'s {self.prof.noun_singular}"
        return name or sal


def _own_numbers(b: Brief, trend: bool = False) -> Optional[str]:
    """The merchant's own listing numbers (always visible to merchant and judge) — preferred over category-level facts."""
    v, c, w = b.P("views"), b.P("calls"), b.P("window")
    if not (v and c and w):
        return None
    wd = w.replace("days", "din")
    up = b.pz.get("views_delta", {}).get("signed")
    tail_en = tail_hi = ""
    if trend and isinstance(up, (int, float)) and up > 0:
        d = b.P("views_delta")
        tail_en, tail_hi = f", with views up {d} this week", f", aur is hafte views {d} upar hain"
    return b.t(f"Your listing is already moving: {v} views and {c} calls in the last {w}{tail_en}.",
               f"Aapki listing already chal rahi hai: pichhle {wd} mein {v} views aur {c} calls{tail_hi}.")


def _listing_why_now(b: Brief) -> Optional[str]:
    """Tie a content item to the merchant's own views/calls (the numbers the merchant — and the judge — can see)."""
    v, c, w = b.P("views"), b.P("calls"), b.P("window")
    if not (v and c and w):
        return None
    wd = w.replace("days", "din")
    vv, cc = as_float_safe(v.replace(",", "")), as_float_safe(c.replace(",", ""))
    if vv and cc is not None and vv > 0 and cc / vv < 0.02:
        return b.t(f"Your listing got {v} views but only {c} calls in the last {w} — people are looking but not calling; "
                   f"an expert post like this builds the trust that turns views into calls.",
                   f"Pichhle {wd} mein aapki listing ko {v} views mile par sirf {c} calls — log dekh rahe hain par call nahi kar rahe; "
                   f"aisa expert post wahi trust banata hai jo views ko calls mein badalta hai.")
    return b.t(f"Your listing got {v} views and {c} calls in the last {w} — a post like this gives those people one more reason to pick you.",
               f"Pichhle {wd} mein aapki listing ko {v} views aur {c} calls mile — aisa post unhe aapko chunne ka ek aur reason deta hai.")


def _weekday(raw) -> Optional[str]:
    try:
        from datetime import date as _d
        return _d.fromisoformat(str(raw)[:10]).strftime("%A")
    except (ValueError, TypeError):
        return None


def _q(s: Optional[str]) -> str:
    return f"“{s.strip().rstrip('.')}”" if s else ""


def _short(s: Optional[str], n: int = 170) -> Optional[str]:
    if not s:
        return s
    s = s.strip()
    if len(s) <= n:
        return s
    cut = re.split(r"(?<=[.;])\s", s[:n])
    return cut[0].rstrip(".;") if len(cut[0]) > 40 else s[:n].rsplit(" ", 1)[0] + "…"


# ============================================================ family realizers
_CLOSE_EN = ["Want me to {x}? Reply YES and I'll take it from there.",
             "Shall I {x}? Just reply YES.",
             "Want me to {x}? Reply YES — I'll handle the rest.",
             "Should I {x}? Reply YES and consider it done."]
_CLOSE_HI = ["Main {x}? Bas YES reply karein, baaki main sambhal loongi.",
             "Main {x}? YES reply kijiye, main shuru kar deti hoon.",
             "Main {x}? Bas ek YES reply kijiye.",
             "Main {x}? YES reply kijiye, baaki kaam mera."]


def _post_cta(b: Brief, what_en: str, what_hi: str) -> dict:
    import hashlib
    i = int(hashlib.md5(f"{b.P('name')}|{b.ta.family}".encode()).hexdigest(), 16) % len(_CLOSE_EN)
    return {
        "effort": b.t(_CLOSE_EN[i].format(x=what_en), _CLOSE_HI[i].format(x=what_hi)),
        "curiosity": b.t(f"Want me to {what_en}?", f"Kya main {what_hi}?"),
    }


def r_knowledge(b: Brief) -> Parts:
    src, title, n = b.A("source"), b.A("title"), b.A("trial_n")
    seg = b.P("segment_match")
    cn = b.cust_noun()
    is_event = bool(b.Araw("_is_event"))
    if is_event:
        when, credits, fee = b.A("event_date"), b.A("credits"), b.A("fee")
        details = ", ".join(x for x in (when, f"{credits} CDE credits" if credits else None, fee) if x)
        hook = b.t(f"{b.sal()}, a CDE slot worth blocking:", f"{b.sal()}, ek CDE session jo block karne layak hai:")
        anchor = [f"{_q(title)}" + (f" — {details}." if details else ".")]
        summ = _short(b.A("summary"), 120)
        if summ:
            anchor.append(f"{summ}.")
        levers = {"reciprocity": b.t(f"Flagging it since it maps to your practice at {b.P('locality')}." if b.P("locality") else "Flagging it early so you can plan the evening.",
                                     f"{b.P('locality')} ki practice ke liye kaam ka hai, isliye pehle se bata rahi hoon." if b.P("locality") else "Pehle se bata rahi hoon taaki aap plan kar sakein."),
                  "curiosity": b.t("The session covers the workflow ROI for solo practices." if "solo" in (b.Araw("summary") or "") else "It's a short evening session.",
                                   "Session mein solo practices ke liye workflow ROI cover hoga." if "solo" in (b.Araw("summary") or "") else "Chhota evening session hai.")}
        ctas = {"effort": b.t("Want me to save the details and remind you a day before? Reply YES.",
                              "Details save karke ek din pehle reminder bhej doon? Reply YES."),
                "curiosity": b.t("Want the registration details?", "Registration details bhej doon?")}
        return Parts(hook, anchor, levers, ctas)
    if src:
        hook = b.t(f"{b.sal()}, {src} has one item worth your time:", f"{b.sal()}, {src} mein ek kaam ka item aaya hai:")
    else:
        hook = b.t(f"{b.sal()}, one item from this week's {b.prof.slug} digest:", f"{b.sal()}, is hafte ke digest mein ek kaam ka item:")
    anchor = [f"{_q(title)}" + (f" (n={n})." if n else ".")] if title else []
    summ = _short(b.Araw("summary") and b.A("summary"), 140)
    if summ:
        from ..ledger import extract_numbers
        head_nums = set(extract_numbers(" ".join(x for x in (title, n) if x)))
        extra = set(extract_numbers(summ)) - head_nums
        # one headline stat is verifiable-looking; a second paragraph of new stats reads as invented
        if not (head_nums and extra) and (len(head_nums & set(extract_numbers(summ))) < 2 or not head_nums):
            anchor.append(b.t(f"Key finding: {summ}.", f"Key finding: {summ}."))
    levers = {}
    segc = b.P("segment_count")
    why_now = _listing_why_now(b)
    if why_now and not seg:
        levers["reciprocity"] = why_now
    elif seg or segc:
        who_en = f"your {segc} {seg or ''} {cn}".replace("  ", " ") if segc else f"your {seg} {cn}"
        who_hi = f"aapke {segc} {seg or ''} {cn}".replace("  ", " ") if segc else f"aapke {seg} {cn}"
        levers["reciprocity"] = b.t(f"Flagging it because it maps directly to {who_en}.",
                                    f"{who_hi[0].upper() + who_hi[1:]} ke liye directly relevant hai, isliye share kar rahi hoon.")
    elif b.P("lapsed"):
        levers["reciprocity"] = b.t(f"Could be a good reason to re-engage your {b.P('lapsed')} lapsed {cn}.",
                                    f"Aapke {b.P('lapsed')} lapsed {cn} ko wapas bulane ka accha reason ban sakta hai.")
    elif b.P("customers_total"):
        levers["reciprocity"] = b.t(f"Relevant for the {b.P('customers_total')} {cn} you've seen this year.",
                                    f"Is saal ke aapke {b.P('customers_total')} {cn} ke liye relevant hai.")
    act = _short(b.A("actionable"), 110)
    if act:
        levers["specificity"] = b.t(f"Practical next step: {act[0].lower() + act[1:]}.", f"Practical next step: {act[0].lower() + act[1:]}.")
    levers["curiosity"] = b.t("The full note is short and practical.", "Poora note chhota aur practical hai.")
    ctas = {
        "effort": b.t(f"Want me to pull the 2-min summary and draft a WhatsApp you can forward to your {cn}?",
                      f"Main 2-min summary + {cn} ke liye ek forward-ready WhatsApp draft bana doon?"),
        "curiosity": b.t("Want the 2-min summary?", "2-min summary bhej doon?"),
    }
    return Parts(hook, anchor, levers, ctas)


def r_regulation(b: Brief) -> Parts:
    src, title, eff = b.A("source"), b.A("title"), b.A("effective")
    hook = b.t(f"{b.sal()}, heads-up on a compliance change" + (f" from {src}:" if src else ":"),
               f"{b.sal()}, ek compliance update" + (f" ({src}):" if src else ":"))
    anchor = []
    if title:
        in_title = eff and (str(b.Araw("effective") or "")[:10] in title)
        anchor.append(_q(title) + (b.t(f", deadline {eff}.", f", deadline {eff}.") if eff and not in_title else "."))
    summ = _short(b.A("summary"), 130)
    if summ:
        anchor.append(f"{summ}.")
    act = _short(b.A("actionable"), 110)
    levers = {"loss_aversion": b.t("Better to be ready before the deadline than to fix it after an inspection.",
                                   "Deadline se pehle ready rehna better hai, inspection ke baad fix karne se."),
              "curiosity": b.t("It's a short read.", "Chhota sa read hai.")}
    if act:
        levers["specificity"] = b.t(f"What to do: {act[0].lower() + act[1:]}.", f"Kya karna hai: {act[0].lower() + act[1:]}.")
    who = b.P("name") or f"your {b.prof.noun_singular}"
    ctas = {"effort": b.t(f"Want me to turn it into a 1-page checklist for {who}? Reply YES.",
                          f"Main {who} ke liye 1-page checklist bana doon? Reply YES."),
            "curiosity": b.t("Want the key points summarised?", "Key points summary bhej doon?")}
    return Parts(hook, anchor, levers, ctas)


def _perf_no_move(b: Brief, up: bool) -> Parts:
    """Placeholder perf trigger but the merchant's numbers don't move that way: say what is actually true."""
    fv, fc = b.A("flat_views"), b.A("flat_calls")
    moves = ", ".join(x for x in (f"views {'+' if (b.ta.anchor['flat_views'].get('signed') or 0) >= 0 else '-'}{fv}" if fv else None,
                                  f"calls {'+' if (b.ta.anchor['flat_calls'].get('signed') or 0) >= 0 else '-'}{fc}" if fc else None) if x)
    name = b.P("name") or ""
    hook = b.t(f"{b.sal()}, quick health check on {name}: this week {moves or 'your numbers look steady'}.",
               f"{b.sal()}, {name} ka quick health check: is hafte {moves or 'numbers steady hain'}.")
    anchor, levers, ctas = [], {}, {}
    if b.A("expired_days"):
        plan = b.A("plan")
        anchor.append(b.t(f"The one risk: your {plan + ' ' if plan else ''}plan lapsed {b.A('expired_days')} days ago, so nothing new is being promoted.",
                          f"Ek risk: aapka {plan + ' ' if plan else ''}plan {b.A('expired_days')} din pehle band ho gaya, isliye kuch naya promote nahi ho raha."))
        ctas["effort"] = b.t("Want me to reactivate it before the numbers turn? Reply YES.", "Numbers girne se pehle reactivate kar doon? Reply YES.")
    elif b.P("ctr_gap"):
        anchor.append(b.t(f"The one gap: CTR {b.P('ctr')} vs {b.P('peer_ctr')} for {b.prof.peer_label}.",
                          f"Ek gap: CTR {b.P('ctr')} vs {b.prof.peer_label} ka {b.P('peer_ctr')}."))
    o_en, o_hi = b.offer_phrase()
    ctas.setdefault("effort", _post_cta(b, f"put up a fresh post with {o_en} to keep it that way" if o_en else "put up a fresh post to keep it that way",
                                         f"{o_hi} ke saath ek fresh post laga doon taaki yeh bana rahe" if o_hi else "ek fresh post laga doon taaki yeh bana rahe")["effort"])
    return Parts(hook, anchor, levers, ctas)


def _perf(b: Brief, up: bool) -> Parts:
    if b.Araw("_no_move"):
        return _perf_no_move(b, up)
    metric = b.A("metric") or b.t("performance", "performance")
    delta = b.A("delta")
    w_en, w_hi = b.window()
    if delta:
        hook = b.t(f"{b.sal()}, your {metric} " + ("jumped " if up else "dropped ") + f"{delta} {w_en}.".replace(" .", "."),
                   f"{b.sal()}, aapke {metric} mein {w_hi} {delta} ki " + ("badhat" if up else "girawat") + " aayi hai.")
    else:
        hook = b.t(f"{b.sal()}, your {metric} " + ("spiked" if up else "dipped") + f" {w_en}.".replace(" .", "."),
                   f"{b.sal()}, aapke {metric} {w_hi} " + ("upar gaye hain." if up else "neeche aaye hain."))
    anchor = []
    cur, base = b.A("current"), b.A("baseline")
    if cur and base:
        anchor.append(b.t(f"That's {cur} vs a usual {base}.", f"Yani {cur}, jabki normal {base} rehta hai."))
    elif base and not cur:
        anchor.append(b.t(f"(Baseline: {base} {metric}.)", f"(Baseline: {base} {metric}.)"))
    elif b.A("metric_30d") and b.P("window"):
        anchor.append(b.t(f"({b.A('metric_30d')} {metric} in the last {b.P('window')}.)",
                          f"(Pichhle {b.P('window').replace('days', 'din')} mein {b.A('metric_30d')} {metric}.)"))
    metric_raw = str(b.Araw("metric") or "").lower()
    for other in ("views", "calls", "directions"):
        od = b.pz.get(f"{other}_delta")
        if other != metric_raw and od and ((od["signed"] < 0) != up) and abs(od["signed"]) >= 0.05:
            anchor.append(b.t(f"{other.capitalize()} are {'up' if up else 'down'} {b.P(other + '_delta')} too.",
                              f"{other.capitalize()} bhi {b.P(other + '_delta')} {'upar' if up else 'neeche'} hain."))
            break
    levers = {}
    if b.P("ctr_gap"):
        levers["social_proof"] = b.t(f"Your CTR is {b.P('ctr')} vs {b.P('peer_ctr')} for {b.prof.peer_label}.",
                                     f"Aapka CTR {b.P('ctr')} hai, jabki {b.prof.peer_label} ka {b.P('peer_ctr')}.")
        levers["loss_aversion"] = b.t(f"At {b.P('ctr')} vs the {b.P('peer_ctr')} peer benchmark, searchers are slipping to others.",
                                      f"{b.P('ctr')} vs {b.P('peer_ctr')} peer benchmark — log search karke dusron pe ja rahe hain.")
    if b.P("stale_days") and not up:
        levers.setdefault("loss_aversion", b.t(f"Your last Google post was {b.P('stale_days')} days ago — fresh posts are the quickest lever.",
                                               f"Aapki last Google post {b.P('stale_days')} din purani hai — fresh post sabse quick lever hai."))
    if up:
        levers["loss_aversion"] = b.t("Spikes fade fast — best to convert this traffic while it's here.",
                                      "Aisi spike jaldi utar jaati hai — abhi convert karna best hai.")
    if b.A("driver"):
        levers["reciprocity"] = b.t(f"Likely driver: your {b.A('driver')} — it's working, so let's do more of it.",
                                    f"Iski wajah shayad aapka {b.A('driver')} hai — yeh kaam kar raha hai, toh isi ko aage badhayein.")
        levers["specificity"] = levers["reciprocity"]
    if b.A("season_note") and not up:
        levers["social_proof"] = b.t(f"This matches the expected seasonal pattern ({b.A('season_note')}) — peers dip too, so the goal is holding share, not panicking.",
                                     f"Yeh expected seasonal pattern hai ({b.A('season_note')}) — peers bhi dip karte hain, isliye goal share bachaana hai, panic nahi.")
    o_en, o_hi = b.offer_phrase()
    small = abs(float((b.ta.anchor.get("delta") or {}).get("signed") or 0)) < 0.10
    if up and small:
        levers["loss_aversion"] = b.t("Small, but it's moving the right way — a fresh post now helps it compound.",
                                      "Chhota hai, par sahi direction mein — abhi fresh post se yeh aur badhega.")
        what_en = f"put up a fresh post with {o_en} to build on it" if o_en else "put up a fresh post to build on it"
        what_hi = f"{o_hi} ke saath fresh post laga doon taaki yeh aur badhe" if o_hi else "ek fresh post laga doon taaki yeh aur badhe"
    elif up:
        what_en = f"pin {o_en} on your Google profile while traffic is high" if o_en else "put up a Google post while traffic is high"
        what_hi = f"traffic high rehte hi {b.offer_direct_hi()} Google profile pe pin kar doon" if o_hi else "traffic high rehte hi ek Google post laga doon"
    else:
        what_en = f"put up a fresh Google post with {o_en} today" if o_en else "put up a fresh Google post for you today"
        what_hi = f"aaj hi {o_hi} ke saath fresh Google post laga doon" if o_hi else "aaj hi ek fresh Google post laga doon"
    ctas = _post_cta(b, what_en, what_hi)
    ctas["curiosity"] = b.t("Want me to show you what's driving it?", "Dekhna chahenge iske peeche kya hai?")
    ctas["ask"] = b.t("Quick check — did anything change this week (timings, photos, phone line)? I'll adjust your listing to match.",
                      "Ek quick sawaal — is hafte kuch badla kya (timings, photos, phone number)? Main listing usi hisaab se update kar doon.") if not up else \
        b.t("What do you think drove it — an offer, a post, word of mouth? I'll double down on it.",
            "Aapko kya lagta hai iski wajah kya thi — offer, post ya word of mouth? Main usi ko aur push karti hoon.")
    return Parts(hook, anchor, levers, ctas)


def r_perf_dip(b: Brief) -> Parts:
    return _perf(b, up=False)


def r_perf_spike(b: Brief) -> Parts:
    return _perf(b, up=True)


def r_milestone(b: Brief) -> Parts:
    val, metric = b.A("value"), b.A("metric")
    cur, tgt, gap = b.A("current"), b.A("target"), b.A("gap")
    name = b.P("name") or ""
    end = "." if b.prof.hype_forbidden else "! 🎉"
    mt = (metric or "").replace("review count", "reviews")
    cn = b.cust_noun()
    if cur and tgt and gap:
        hook = b.t(f"{b.sal()}, {name} is at {cur} {mt} — just {gap} away from {tgt}.",
                   f"{b.sal()}, {name} {cur} {mt} pe hai — {tgt} se sirf {gap} door.")
    elif b.Araw("_from_merchant") and val:
        hook = b.t(f"{b.sal()}, {name} pulled {val} {mt} in the last {b.P('window') or '30 days'}{end}",
                   f"{b.sal()}, {name} ko pichhle {(b.P('window') or '30 days').replace('days', 'din')} mein {val} {mt} mile{end}")
    elif val:
        hook = b.t(f"{b.sal()}, congrats — {name} just crossed {val} {mt}{end}".replace(" .", "."),
                   f"{b.sal()}, badhai ho — {name} ne {val} {mt} cross kar liye{end}")
    else:
        hook = b.t(f"{b.sal()}, {name} just hit a new milestone{end}", f"{b.sal()}, {name} ne naya milestone hit kiya{end}")
    levers = {}
    r, pr = b.P("rating"), b.P("peer_rating")
    if r and pr and (b.pz["rating"]["value"] >= b.pz["peer_rating"]["value"]):
        levers["social_proof"] = b.t(f"At {r}, you're above the {pr} average for {b.prof.peer_label}.",
                                     f"{r} ke saath aap {b.prof.peer_label} ke {pr} average se upar hain.")
    elif cur and b.P("peer_reviews") and "review" in mt and float(b.Araw("current")) > b.pz["peer_reviews"]["value"]:
        well = float(b.Araw("current")) >= 1.3 * b.pz["peer_reviews"]["value"]
        levers["social_proof"] = b.t(f"That's {'well ' if well else ''}above the {b.P('peer_reviews')}-review average for {b.prof.peer_label}.",
                                     f"Yeh {b.prof.peer_label} ke {b.P('peer_reviews')}-review average se {'kaafi ' if well else ''}upar hai.")
    if b.P("praise"):
        levers["reciprocity"] = b.t(f"Your reviews keep praising {b.P('praise')} — happy {cn} are your best proof.",
                                    f"Aapke reviews mein {b.P('praise')} ki baar-baar tareef hai — happy {cn} hi best proof hain.")
    else:
        levers["reciprocity"] = b.t("Milestones like this are great proof for new customers.",
                                    "Aise milestones naye customers ke liye best proof hote hain.")
    if cur and gap and "review" in mt:
        ctas = _post_cta(b, f"send a short review-request WhatsApp to your recent happy {cn} to close the gap this week",
                         f"aapke recent happy {cn} ko ek chhota review-request WhatsApp bhej doon taaki is hafte gap close ho jaaye")
    else:
        ctas = _post_cta(b, f"post a thank-you update and invite happy {cn} to add photo reviews",
                         f"ek thank-you post daal doon aur happy {cn} ko photo review ke liye invite kar doon")
    ctas["ask"] = b.t("What do your regulars love most? I'll feature it in the next post.",
                      "Aapke regulars ko sabse zyada kya pasand hai? Main agle post mein wahi feature karungi.")
    return Parts(hook, [], levers, ctas)


def r_competitor(b: Brief) -> Parts:
    noun = b.prof.noun_singular
    name, dist, date = b.A("name"), b.A("distance"), b.A("date")
    where_en = f"{dist} from you" if dist else "near you"
    where_hi = f"aapse {dist} door" if dist else "aapke paas"
    hook = b.t(f"{b.sal()}, heads-up: a new {noun}" + (f", {name}," if name else "") + f" opened {where_en}" + (f" on {date}." if date else "."),
               f"{b.sal()}, ek heads-up: {where_hi} ek naya {noun}" + (f" ({name})" if name else "") + " khula hai" + (f", {date} ko." if date else "."))
    anchor = []
    if b.A("rating"):
        anchor.append(b.t(f"They're listed at {b.A('rating')}★.", f"Unki listing {b.A('rating')}★ pe hai."))
    theirs, mine = b.A("offer"), b.P("offer")
    levers = {}
    if theirs:
        anchor.append(b.t(f"Their launch offer: {theirs}" + (f" — against your {mine}." if mine else "."),
                          f"Unka launch offer: {theirs}" + (f" — aapke {mine} ke saamne." if mine else ".")))
    r, rv, pr = b.P("rating"), b.P("reviews"), b.P("peer_rating")
    if r and rv:
        levers["social_proof"] = b.t(f"Your {r} across {rv} reviews is a head start they don't have yet.",
                                     f"Aapke {rv} reviews aur {r} rating unke paas abhi nahi hai — yeh aapka head start hai.")
    elif b.P("praise"):
        q = b.P("praise_quote")
        levers["social_proof"] = b.t(f"Your reviews already praise {b.P('praise')}" + (f" (“{q}”)" if q else "") + " — a new listing can't copy that.",
                                     f"Aapke reviews already {b.P('praise')} ki tareef karte hain" + (f" (“{q}”)" if q else "") + " — naya listing yeh copy nahi kar sakta.")
    elif b.P("ctr_above"):
        levers["social_proof"] = b.t(f"Your CTR is {b.P('ctr')} vs {b.P('peer_ctr')} for {b.prof.peer_label} — protect that lead.",
                                     f"Aapka CTR {b.P('ctr')} hai vs {b.prof.peer_label} ka {b.P('peer_ctr')} — yeh lead bachaani hai.")
    levers["loss_aversion"] = b.t(f"New openings pull curious {b.cust_noun()} in the first few weeks — worth staying visible now.",
                                  f"Naye {noun} pehle kuch hafton mein curious {b.cust_noun()} kheenchte hain — abhi visible rehna zaroori hai.")
    o_en, o_hi = b.offer_phrase()
    if theirs and mine:
        what_en = f"put up a post that leads with what they can't match{' — ' + b.P('praise') if b.P('praise') else ''} — instead of a price war"
        what_hi = f"price war ke bajaye ek post daal doon jo aapki strength{' (' + b.P('praise') + ')' if b.P('praise') else ''} highlight kare"
        ctas = _post_cta(b, what_en, what_hi)
    else:
        ctas = _post_cta(b, f"refresh your Google post with {o_en} this week" if o_en else "refresh your Google post this week",
                         f"is hafte {o_hi} ke saath aapki Google post refresh kar doon" if o_hi else "is hafte aapki Google post refresh kar doon")
    ctas["curiosity"] = b.t("Want a side-by-side of your listing vs theirs?", "Aapki aur unki listing ka side-by-side comparison bhej doon?")
    return Parts(hook, anchor, levers, ctas)


def r_trend(b: Brief) -> Parts:
    q, d = b.A("query"), b.A("delta")
    yoy = b.t(" year-on-year", " saal-dar-saal") if (b.ta.anchor.get("delta", {}).get("path") or "").endswith("yoy") else ""
    seg = b.A("segment") if str(b.Araw("segment") or "").lower() not in ("all", "any", "") else None
    hook = b.t(f"{b.sal()}, searches for “{q}” are up {d}{yoy}" + (f" (mostly age {seg})." if seg else "."),
               f"{b.sal()}, “{q}” ki searches{yoy} {d} badh gayi hain" + (f" (zyada tar {seg} age group)." if seg else "."))
    levers = {"social_proof": b.t("That's real demand building in your category.", "Aapki category mein demand clearly badh rahi hai.")}
    o_en, o_hi = b.offer_phrase()
    ctas = _post_cta(b, f"draft a Google post on “{q}” for your {b.prof.noun_singular}" + (f", featuring {o_en}" if o_en else ""),
                     f"aapke {b.prof.noun_singular} ke liye “{q}” pe ek Google post draft kar doon" + (f", {o_hi} ke saath" if o_hi else ""))
    ctas["ask"] = b.t(f"Do you offer this today? If yes, I'll make sure your listing says so.",
                      "Kya aap yeh service dete hain? Haan ho toh main listing mein clearly add kar deti hoon.")
    return Parts(hook, [], levers, ctas)


def r_festival(b: Brief) -> Parts:
    name, date, days = b.A("name"), b.A("date"), b.A("days_until")
    beat = b.A("beat")
    if not name:
        # placeholder / nameless festival trigger: lead with the category's seasonal beat instead of inventing one
        hook = b.t(f"{b.sal()}, seasonal heads-up for {b.prof.slug}: {beat}." if beat else f"{b.sal()}, the festive season is the next demand window for {b.prof.slug}.",
                   f"{b.sal()}, {b.prof.slug} ke liye seasonal heads-up: {beat}." if beat else f"{b.sal()}, agla demand window festive season hai.")
        anchor = []
        name = b.t("the season", "season")
    else:
        far = days and (as_float_safe(b.Araw("days_until")) or 0) > 45
        if days and far:
            hook = b.t(f"{b.sal()}, {name} is on {date} — {days} days out, which is exactly when the planning starts.",
                       f"{b.sal()}, {name} {date} ko hai — {days} din baaki, planning ka sahi time abhi hai.")
        elif days:
            hook = b.t(f"{b.sal()}, {name} is {days} days away" + (f" ({date})." if date else "."),
                       f"{b.sal()}, {name} sirf {days} din door hai" + (f" ({date})." if date else "."))
        else:
            hook = b.t(f"{b.sal()}, {name} is coming up" + (f" on {date}." if date else "."),
                       f"{b.sal()}, {name} aa raha hai" + (f" — {date}." if date else "."))
        own = _own_numbers(b)
        anchor = [own] if own else ([b.t(f"Seasonal pattern for your category: {beat}.", f"Aapki category ka seasonal pattern: {beat}.")] if beat else [])
        far = bool(days and (as_float_safe(b.Araw("days_until")) or 0) > 45)
        if far:
            levers_note = b.t(f"Packages fixed early get booked first; last-minute ones compete on price.",
                              "Jo packages pehle fix hote hain woh pehle book hote hain; last-minute wale sirf price pe compete karte hain.")
        else:
            levers_note = b.t(f"Posts that go live before {name} catch the planning rush; after, it's too late.",
                              f"{name} se pehle live post planning karne walon tak pahunchti hai — baad mein late ho jaata hai.")
    levers = {"loss_aversion": locals().get("levers_note") or b.t("Getting visible before the peak is what separates full weeks from average ones.",
                                                                 "Peak se pehle visible hona hi full hafte aur average hafte ka fark hai.")}
    if beat and "retention" in beat.lower():
        cn = b.cust_noun()
        levers["loss_aversion"] = b.t(f"New sign-ups are scarce in this window, so every existing {cn[:-1] if cn.endswith('s') else cn} you keep counts double.",
                                      f"Is window mein naye sign-ups kam aate hain, isliye har existing {cn[:-1] if cn.endswith('s') else cn} ko rokna double kaam ka hai.")
        ctas = _post_cta(b, f"draft a check-in WhatsApp for your existing {cn} to keep them coming this month",
                         f"existing {cn} ke liye ek check-in WhatsApp draft kar doon taaki woh is mahine aate rahein")
        ctas["ask"] = b.t(f"Which {cn} have gone quiet lately? I'll draft a personal nudge.", f"Kaun se {cn} aajkal kam aa rahe hain? Main personal nudge draft kar deti hoon.")
        return Parts(hook, anchor, levers, ctas)
    o_en, o_hi = b.offer_phrase()
    match = b.offer_matching(beat or "")
    if match and (match[1] == "active" or b.offer()[1] != "active"):   # merchant's own live offer beats a catalog idea
        o_en, o_hi = (f"your {match[0]} offer", f"aapke {match[0]} offer") if match[1] == "active" else (f"a '{match[0]}' offer", f"'{match[0]}' jaise offer")
    ctas = _post_cta(b, f"draft a {name} package post built on {o_en}" if o_en else f"draft a {name} package post for you",
                     f"{o_hi} pe based {name} package post draft kar doon" if o_hi else f"aapke liye {name} package post draft kar doon")
    ctas["ask"] = b.t(f"What are you planning for {name}? Tell me and I'll turn it into a post.",
                      f"{name} ke liye kya plan hai? Bataiye, main use post bana deti hoon.")
    return Parts(hook, anchor, levers, ctas)


_WEATHER_IDEA = {
    "operator": ("push delivery and evening slots", "delivery aur evening slots push karne"),
    "warm": ("promote evening appointments and cool-down services", "evening appointments aur cool-down services promote karne"),
    "coach": ("nudge members to early-morning or indoor sessions", "members ko early-morning ya indoor sessions ki taraf nudge karne"),
    "trust": ("share a short heat-safety note with your customers", "customers ke saath ek chhota heat-safety note share karne"),
    "clinical": ("share a short hydration and oral-care tip with patients", "patients ke saath hydration aur oral-care tip share karne"),
}


def r_weather(b: Brief) -> Parts:
    temp, cond, city, date = b.A("temp"), b.A("condition"), b.A("city") or b.P("city"), b.A("date")
    what = " ".join(x for x in (temp, cond) if x) or "a weather alert"
    hook = b.t(f"{b.sal()}, {what}" + (f" in {city}" if city else "") + (f" ({date})." if date else " today."),
               f"{b.sal()}, " + (f"{city} mein " if city else "") + f"{what}" + (f" ({date})." if date else " aaj."))
    anchor = [b.t(f"Seasonal note: {b.A('beat')}.", f"Seasonal note: {b.A('beat')}.")] if b.A("beat") else []
    idea_en, idea_hi = _WEATHER_IDEA.get(b.prof.register, ("post a timely update for your customers", "customers ke liye timely update post karne"))
    levers = {"specificity": b.t(f"A good week to {idea_en}.", f"Yeh {idea_hi} ka sahi hafta hai.")}
    if b.P("calls") and b.P("window"):
        levers["reciprocity"] = b.t(f"You got {b.P('calls')} calls in the last {b.P('window')} — a timely post keeps you the first call this week.",
                                    f"Pichhle {b.P('window').replace('days', 'din')} mein aapko {b.P('calls')} calls aaye — timely post se is hafte bhi pehli call aapko aayegi.")
    ctas = _post_cta(b, "draft that post for you to approve", "aapke approval ke liye woh post draft kar doon")
    ctas["ask"] = b.t("Are you changing timings or services for the heat? I'll update your listing to match.",
                      "Kya aap is mausam ke liye timings ya services badal rahe hain? Main listing update kar deti hoon.")
    return Parts(hook, anchor, levers, ctas)


def r_local_event(b: Brief) -> Parts:
    hl, dur, date = b.A("headline"), b.A("duration"), b.A("date")
    tm = b.ta.anchor.get("date", {}).get("time")
    loc_raw = b.Araw("location")
    loc = b.A("location") if loc_raw and str(loc_raw).split(",")[0].strip().lower() not in (hl or "").lower() else None
    is_match = "vs" in (hl or "")
    if is_match:
        when = f"{date}, {tm}" if date and tm else (date or "")
        hook = b.t(f"{b.sal()}, {hl} is on {when}" + (f" at {loc}." if loc else "."),
                   f"{b.sal()}, {hl} {when} ko hai" + (f" — {loc}." if loc else "."))
    else:
        hook = b.t(f"{b.sal()}, local heads-up: {hl}" + (f" ({loc})" if loc else "") + (f", for {dur}" if dur else "") + (f" on {date}." if date else "."),
                   f"{b.sal()}, ek local update: {hl}" + (f" ({loc})" if loc else "") + (f", {dur}" if dur else "") + (f" — {date}." if date else "."))
    anchor = [f"{_short(b.A('impact'))}."] if b.A("impact") else []
    cn = b.cust_noun()
    levers = {}
    if is_match:
        wk = b.Araw("_weeknight")
        wd = _weekday(b.Araw("date"))
        levers["loss_aversion"] = b.t((f"It's a {wd} match, so " if wk is False and wd else "It's a weekend match, so " if wk is False else "") + f"{cn} will be ordering in or looking for a place to watch — the listing that posts first gets seen.",
                                      (f"{wd} ka match hai, toh " if wk is False and wd else "Weekend match hai, toh " if wk is False else "") + f"{cn} order karenge ya match dekhne ki jagah dhoondhenge — jo pehle post karta hai wahi dikhta hai.")
        own = _own_numbers(b, trend=False)
        if own:
            levers["specificity"] = own
        elif b.P("complaint"):
            levers["specificity"] = b.t(f"One watch-out: recent reviews mention {b.P('complaint')} — worth staffing up for the rush.",
                                        f"Ek dhyaan dene wali baat: recent reviews mein {b.P('complaint')} ka zikr hai — rush ke liye staff ready rakhein.")
        if "trial_ending_soon" in (b.tools.get_merchant_fact("signals") or []):
            levers["reciprocity"] = b.t("Your trial ends soon — a match night is the best week to see what the listing can do.",
                                        "Aapka trial khatam hone wala hai — match night yeh dekhne ka best mauka hai ki listing kya kar sakti hai.")
        ctas = _post_cta(b, "put up a match-night post before the first ball", "pehli ball se pehle ek match-night post laga doon")
    else:
        levers["loss_aversion"] = b.t(f"{cn.capitalize()} planning to visit may get caught out — a quick update keeps them coming.",
                                      f"Aane wale {cn} confuse ho sakte hain — ek quick update se woh aate rahenge.")
        if b.P("directions") and b.P("window"):
            levers["loss_aversion"] = b.t(f"You had {b.P('directions')} direction requests in the last {b.P('window')} — {cn} navigating to you today may get caught out.",
                                          f"Pichhle {b.P('window').replace('days', 'din')} mein {b.P('directions')} direction requests aaye — aaj aapki taraf aa rahe {cn} atak sakte hain.")
        ctas = _post_cta(b, f"post a quick update for your {cn} (timings / delivery / directions)",
                         f"{cn} ke liye ek quick update post kar doon (timings / delivery / directions)")
    ctas["ask"] = b.t("Is this affecting your footfall today? Tell me and I'll adjust your listing.",
                      "Kya isse aaj footfall pe asar pad raha hai? Bataiye, main listing adjust kar deti hoon.")
    return Parts(hook, anchor, levers, ctas)


def r_reputation(b: Brief) -> Parts:
    cnt, theme = b.A("count"), b.A("theme")
    if not theme and b.P("complaint"):
        theme = b.P("complaint")
    if not theme:
        # placeholder trigger with no review data: say only what we know, ask for the owner's read
        name = b.P("name") or ""
        hook = b.t(f"{b.sal()}, a recurring theme is starting to show up in {name}'s recent reviews.",
                   f"{b.sal()}, {name} ke recent reviews mein ek theme baar-baar aa raha hai.")
        levers = {"loss_aversion": b.t("Themes like this are easiest to fix before they show up in the star rating.",
                                       "Aise themes star rating tak pahunchne se pehle fix karna sabse aasaan hai.")}
        if b.P("views") and b.P("window"):
            levers["specificity"] = b.t(f"Your listing got {b.P('views')} views in the last {b.P('window')} — replies are public proof you listen.",
                                        f"Pichhle {b.P('window').replace('days', 'din')} mein listing ko {b.P('views')} views mile — replies se dikhta hai ki aap sunte hain.")
        ctas = _post_cta(b, "pull those reviews and draft polite owner replies for your OK",
                         "woh reviews nikaal kar aapke OK ke liye polite owner replies draft kar doon")
        return Parts(hook, [], levers, ctas)
    w_en, w_hi = b.window()
    hook = b.t(f"{b.sal()}, {cnt + ' ' if cnt else ''}reviews {w_en + ' ' if w_en else 'this week '}mention “{theme}”.",
               f"{b.sal()}, {w_hi + ' ' if w_hi else 'is hafte '}{cnt + ' ' if cnt else ''}reviews mein “{theme}” ka zikr hai.")
    anchor = [b.t(f"One says: {_q(b.A('quote'))}.", f"Ek review: {_q(b.A('quote'))}.")] if b.A("quote") else []
    levers = {}
    if b.P("rating"):
        levers["loss_aversion"] = b.t(f"Unanswered, a theme like this chips away at your {b.P('rating')} rating.",
                                      f"Jawab na mile toh aisa theme aapki {b.P('rating')} rating ko dheere-dheere girata hai.")
    else:
        levers["loss_aversion"] = b.t("Unanswered, a theme like this starts showing up in new customers' decisions.",
                                      "Jawab na mile toh aisa theme naye customers ke decision mein dikhne lagta hai.")
    ctas = _post_cta(b, "draft polite owner replies for each (nothing goes live until you approve)",
                     "har review ka polite owner reply draft kar doon (aapke approve karne ke baad hi live hoga)")
    ctas["ask"] = b.t("Is something behind it this week (staffing, timings)? Tell me and I'll word the replies accordingly.",
                      "Is hafte iske peeche kuch reason hai (staff, timings)? Bataiye, main replies usi hisaab se likhungi.")
    return Parts(hook, anchor, levers, ctas)


def r_dormant(b: Brief) -> Parts:
    name = b.P("name") or ""
    days = b.A("days")
    lt = b.A("last_topic")
    hook = b.t(f"{b.sal()}, quick one" + (f" — it's been {days} days since we last spoke" if days else "") + (f" (last time: your {lt})." if lt else "."),
               f"{b.sal()}, ek quick baat" + (f" — {days} din ho gaye baat kiye" if days else "") + (f" (pichhli baar: aapka {lt})." if lt else "."))
    levers = {}
    if b.P("views") and b.P("window"):
        levers["reciprocity"] = b.t(f"Your listing still pulled {b.P('views')} views in the last {b.P('window')} — people are looking.",
                                    f"Pichhle {b.P('window').replace('days', 'din')} mein aapki listing ko {b.P('views')} views mile — log dhoondh rahe hain.")
        levers["specificity"] = levers["reciprocity"]
    if b.P("stale_days"):
        levers["loss_aversion"] = b.t(f"But your last Google post was {b.P('stale_days')} days ago.",
                                      f"Lekin aapki last Google post {b.P('stale_days')} din purani hai.")
    cn = b.cust_noun()
    ctas = {"ask": b.t(f"What's the one thing {cn} ask about most this week? Tell me and I'll turn it into a post for {name}.",
                       f"Is hafte {cn} sabse zyada kis cheez ke baare mein pooch rahe hain? Bataiye, main {name} ke liye post bana deti hoon."),
            **_post_cta(b, "draft a fresh post for you to approve", "aapke approval ke liye ek fresh post draft kar doon")}
    return Parts(hook, [], levers, ctas)


def r_recurring(b: Brief) -> Parts:
    parts = r_dormant(b)
    parts.hook = b.t(f"{b.sal()}, your weekly check-in.", f"{b.sal()}, weekly check-in.")
    topic_raw = str(b.Araw("topic") or "")
    topic = b.A("topic") if topic_raw and not re.fullmatch(r"[a-z0-9_]+", topic_raw) else None
    items = b.prof.relevant_digest
    if topic:
        parts.levers["curiosity"] = b.t(f"This week's question: {topic}", f"Is hafte ka sawaal: {topic}")
    elif items and items[0].get("title"):
        path = next((f.path for f in b.tools.ledger.find(layer="category") if f.value == items[0]["title"]), None)
        if path:
            b.used.add(b.tools.ledger.by_path(path).id)
            parts.levers["curiosity"] = b.t(f"From this week's digest: {_q(items[0]['title'])}.",
                                            f"Is hafte ke digest se: {_q(items[0]['title'])}.")
    return parts


def _dip_metric(b: Brief) -> str:
    dip = as_float_safe(b.Araw("dip"))
    for m in ("calls", "views", "ctr"):
        d = b.pz.get(f"{m}_delta", {}).get("signed")
        if dip is not None and isinstance(d, (int, float)) and abs(abs(d) - abs(dip)) < 0.005:
            b.P(f"{m}_delta")
            return m
    return "listing results"


def r_account(b: Brief) -> Parts:
    plan, days, date, amt = b.A("plan"), b.A("days_left"), b.A("date"), b.A("amount")
    exp_days, dip, lapsed = b.A("expired_days"), b.A("dip"), b.A("lapsed_new")
    anchor = []
    if exp_days:
        hook = b.t(f"{b.sal()}, your magicpin {plan + ' ' if plan else ''}plan lapsed {exp_days} days ago.",
                   f"{b.sal()}, aapka magicpin {plan + ' ' if plan else ''}plan {exp_days} din pehle band ho gaya.")
        dm = _dip_metric(b)
        bits_en = [x for x in (f"{dm} are down {dip}" if dip else None, f"{lapsed} more customers have lapsed" if lapsed else None) if x]
        bits_hi = [x for x in (f"{dm} {dip} neeche hain" if dip else None, f"{lapsed} aur customers lapse ho gaye" if lapsed else None) if x]
        if bits_en:
            anchor.append(b.t("Since then, " + " and ".join(bits_en) + ".", "Tab se " + " aur ".join(bits_hi) + "."))
    else:
        hook = b.t(f"{b.sal()}, your {plan + ' ' if plan else ''}plan has {days} days left" + (f" (renews {date})." if date else "."),
                   f"{b.sal()}, aapke {plan + ' ' if plan else ''}plan mein {days} din bache hain" + (f" ({date} ko renew)." if date else "."))
    if b.P("views") and b.P("calls") and b.P("window"):
        anchor.append(b.t(f"In the last {b.P('window')} your listing got {b.P('views')} views and {b.P('calls')} calls.",
                          f"Pichhle {b.P('window').replace('days', 'din')} mein aapki listing ko {b.P('views')} views aur {b.P('calls')} calls mile."))
    if amt:
        anchor.append(b.t(f"Renewal: {amt}.", f"Renewal: {amt}."))
    levers = {"loss_aversion": b.t("Renewing now keeps your posts and offers running without a gap." if not exp_days else
                                   "Every week off the plan, those numbers slide further.",
                                   "Abhi renew karne se posts aur offers bina gap ke chalte rahenge." if not exp_days else
                                   "Plan ke bina har hafte yeh numbers aur girenge."),
              "reciprocity": b.t("Wanted to flag it early so nothing pauses.", "Pehle se bata rahi hoon taaki kuch ruke nahi.")}
    ctas = {"effort": b.t("Want me to reactivate it today? Reply YES." if exp_days else "Want me to process the renewal? Reply YES.",
                          "Aaj hi reactivate kar doon? Reply YES." if exp_days else "Renewal process kar doon? Reply YES.")}
    return Parts(hook, anchor, levers, ctas)


def r_profile(b: Brief) -> Parts:
    miss, comp, days = b.A("missing"), b.A("completeness"), b.A("days") or b.P("stale_days")
    levers = {}
    if b.Araw("unverified"):
        b.A("unverified")
        path, up = b.A("verify_path"), b.A("uplift")
        hook = b.t(f"{b.sal()}, your Google profile is still not verified.", f"{b.sal()}, aapka Google profile abhi tak verified nahi hai.")
        anchor = []
        if up:
            anchor.append(b.t(f"Verified listings are estimated to get about {up} more visibility.", f"Verified listings ko lagbhag {up} zyada visibility milti hai (estimate)."))
        if path:
            anchor.append(b.t(f"Verification is via {path} — it takes a few minutes.", f"Verification {path} se hota hai — kuch minute ka kaam."))
        levers["loss_aversion"] = b.t("Until then, Google keeps reviewing every edit before it shows.",
                                      "Tab tak Google har edit ko dikhane se pehle review karta hai.")
        ctas = _post_cta(b, "start the verification request for you now", "abhi aapke liye verification request start kar doon")
        return Parts(hook, anchor, levers, ctas)
    if miss:
        hook = b.t(f"{b.sal()}, your Google profile is missing: {miss}" + (f" (it's {comp} complete)." if comp else "."),
                   f"{b.sal()}, aapke Google profile mein yeh missing hai: {miss}" + (f" (abhi {comp} complete)." if comp else "."))
    elif comp:
        hook = b.t(f"{b.sal()}, your Google profile is {comp} complete.", f"{b.sal()}, aapka Google profile abhi {comp} complete hai.")
    else:
        hook = b.t(f"{b.sal()}, your last Google post was {days} days ago." if days else f"{b.sal()}, your Google profile needs a quick refresh.",
                   f"{b.sal()}, aapki last Google post {days} din purani hai." if days else f"{b.sal()}, aapke Google profile ko quick refresh chahiye.")
    levers["loss_aversion"] = b.t("Incomplete listings lose clicks to complete ones in the same search.",
                                  "Same search mein incomplete listing ke clicks complete listings le jaati hain.")
    if b.P("ctr_gap"):
        levers["social_proof"] = b.t(f"Your CTR is {b.P('ctr')} vs {b.P('peer_ctr')} for {b.prof.peer_label}.",
                                     f"Aapka CTR {b.P('ctr')} hai vs {b.prof.peer_label} ka {b.P('peer_ctr')}.")
    ctas = _post_cta(b, "fill these in for you from your existing details", "aapki existing details se yeh sab fill kar doon")
    return Parts(hook, [], levers, ctas)


def r_offer(b: Brief) -> Parts:
    off = b.A("offer") or b.P("expired_offer") or b.P("offer")
    status, date = b.A("status"), b.A("date")
    hook = b.t(f"{b.sal()}, your “{off}” offer " + (f"is {status}" if status else "needs attention") + (f" (since {date})." if date else "."),
               f"{b.sal()}, aapka “{off}” offer " + (f"{status} hai" if status else "check karna hai") + (f" ({date} se)." if date else "."))
    levers = {"loss_aversion": b.t("Service + price offers like this are what searchers click on.",
                                   "Service + price wale offers pe hi log click karte hain.")}
    live = b.P("offer")
    if b.P("views") and b.P("window"):
        levers["specificity"] = b.t(f"Your listing got {b.P('views')} views in the last {b.P('window')}" +
                                    (f" and only {live} is live right now." if live and live != off else "."),
                                    f"Pichhle {b.P('window').replace('days', 'din')} mein listing ko {b.P('views')} views mile" +
                                    (f", aur abhi sirf {live} live hai." if live and live != off else "."))
    ctas = _post_cta(b, "switch it back on for this week", "is hafte ke liye ise wapas on kar doon")
    return Parts(hook, [], levers, ctas)


def r_customer_recall(b: Brief) -> Parts:
    first = (b.cust.first_name if b.cust else "") or ""
    greet_en = first
    greet_hi = (re.sub(r"^(mr|mrs|ms|miss|shri|smt)\.?\s+", "", first, flags=re.I) + " ji") if re.match(r"^(mr|mrs|ms|miss|shri|smt)\.?\s", first, re.I) else first
    ms = b.merchant_short()
    emoji = " 🦷" if b.prof.slug == "dentists" else (" ✨" if b.prof.register == "warm" else "")
    months, svc, due = b.A("months"), b.A("service"), b.A("due")
    last_svc = b.Araw("last_service") and b.A("last_service")
    visit_en = visit_hi = last_svc if last_svc and last_svc.lower() not in ("walk in otc",) else "visit"
    pharmacy = b.prof.register == "trust"
    hi_hello = f"नमस्ते {greet_hi}".strip()
    hook = b.t(f"Hi {greet_en}, {ms} here{emoji}." if greet_en else f"Hi, {ms} here{emoji}.",
               f"Hi {greet_hi}, {ms} se{emoji}." if greet_hi else f"Namaste, {ms} se{emoji}.",
               f"{hi_hello}, {ms} से{emoji}।")
    anchor = []
    mols, runs = b.A("molecules"), b.A("runs_out")
    if mols:
        lv = b.A("last_visit")
        anchor.append(b.t(f"Your {mols} will run out by {runs}" + (f" (last refill {lv})." if lv else "."),
                          f"Aapki {mols} ki dawai {runs} tak khatam ho jaayegi" + (f" (last refill {lv})." if lv else "."),
                          f"आपकी {mols} की दवाइयाँ {runs} तक ख़त्म हो जाएँगी" + (f" (पिछला refill {lv})।" if lv else "।")))
    elif b.A("days_since"):
        ds, focus, mm = b.A("days_since"), b.A("focus"), b.A("member_months")
        anchor.append(b.t(f"It's been {ds} days since your last session" + (f" — after {mm} months of {focus} work, that progress is worth protecting." if focus and mm else "."),
                          f"Aapke last session ko {ds} din ho gaye" + (f" — {mm} mahine ki {focus} mehnat ko bachaana zaroori hai." if focus and mm else "."),
                          f"आपके पिछले सेशन को {ds} दिन हो गए" + (f" — {mm} महीने की {focus} मेहनत को बचाना ज़रूरी है।" if focus and mm else "।")))
    elif b.A("wedding"):
        anchor.append(b.t(f"Your wedding is on {b.A('wedding')}" + (f" ({b.A('days_to_event')} days to go)" if b.A("days_to_event") else "") + (f" — the {svc} window is open now." if svc else "."),
                          f"Aapki shaadi {b.A('wedding')} ko hai" + (f" ({b.A('days_to_event')} din baaki)" if b.A("days_to_event") else "") + (f" — {svc} ka sahi time abhi hai." if svc else "."),
                          f"आपकी शादी {b.A('wedding')} को है" + (f" — {svc} का सही समय अभी है।" if svc else "।")))
    elif pharmacy and svc and b.A("last_visit"):
        lv = b.A("last_visit")
        anchor.append(b.t(f"Your {svc} is due — the last one was on {lv}.", f"Aapka {svc} due hai — pichhli baar {lv} ko liya tha.",
                          f"आपका {svc} ड्यू है — पिछली बार {lv} को लिया था।"))
    elif months:
        anchor.append(b.t(f"It's been {months} months since your last {visit_en}" + (f" — your {svc} is due" if svc else "") + (f" ({due})." if due else "."),
                          f"Aapki last {visit_hi} ko {months} mahine ho gaye" + (f" — aapka {svc} due hai" if svc else "") + (f" ({due})." if due else "."),
                          f"आपकी पिछली विज़िट को {months} महीने हो गए" + (f" — आपका {svc} ड्यू है" if svc else "") + (f" ({due})।" if due else "।")))
    elif svc:
        anchor.append(b.t(f"Your {svc} is due" + (f" ({due})." if due else "."), f"Aapka {svc} due hai" + (f" ({due})." if due else "."),
                          f"आपका {svc} ड्यू है" + (f" ({due})।" if due else "।")))
    elif b.A("last_visit"):
        lv = b.A("last_visit")
        kind = b.ta.trigger_type.lower()
        if "refill" in kind or "follow" in kind:
            anchor.append(b.t(f"Your follow-up is due — your last visit was on {lv}.", f"Aapka follow-up due hai — last visit {lv} ko thi.",
                              f"आपका फ़ॉलो-अप ड्यू है — पिछली विज़िट {lv} को थी।"))
        else:
            anchor.append(b.t(f"We haven't seen you since {lv} — we'd love to have you back.", f"{lv} ke baad aapse mulaqat nahi hui — aapka intezaar hai.",
                              f"{lv} के बाद आपसे मुलाक़ात नहीं हुई — आपका इंतज़ार है।"))
    else:
        anchor.append(b.t("It's been a while since your last visit.", "Aapki last visit ko kaafi time ho gaya.", "आपकी पिछली विज़िट को काफ़ी समय हो गया।"))
    o = b.pick_customer_offer()
    if o:
        anchor.append(b.t(f"For you: {o}.", f"Aapke liye: {o}.", f"आपके लिए: {o}।"))
    s1, s2 = b.A("slot1"), b.A("slot2")
    ctas = {}
    if s1 and s2:
        b.tools.ledger.derive("slot_count", 2, "2", [b.ta.anchor["slot1"]["path"] or "", b.ta.anchor["slot2"]["path"] or ""])
        anchor.append(b.t(f"Two slots are open for you: {s1} or {s2}.", f"Aapke liye 2 slots ready hain: {s1} ya {s2}.",
                          f"आपके लिए 2 स्लॉट तैयार हैं: {s1} या {s2}।"))
        ctas["slot"] = b.t("Reply 1 or 2 to book, or tell us a time that works.", "Book karne ke liye 1 ya 2 reply karein, ya apna time batayein.",
                           "बुक करने के लिए 1 या 2 भेजें, या अपना समय बताएं।")
    elif s1:
        anchor.append(b.t(f"Next session: {s1}.", f"Agla session: {s1}.", f"अगला सेशन: {s1}।"))
    pref = b.tools.get_customer_fact("preferences.preferred_slots")
    pref_txt = humanize(pref) if isinstance(pref, str) else None
    if pref_txt:
        b.tools.ledger.allow_words(pref_txt.split())
    if mols and b.Araw("_delivery_saved"):
        ctas["confirm"] = b.t("Reply YES and we'll deliver to your saved address.", "YES reply karein, hum saved address pe delivery kar denge.",
                              "YES भेजें, हम आपके saved address पर डिलीवरी कर देंगे।")
    elif pharmacy and not (svc or mols):
        ctas["confirm"] = b.t("Need a refill or home delivery this week? Reply YES and we'll arrange it.",
                              "Is hafte refill ya home delivery chahiye? YES reply karein, hum arrange kar denge.",
                              "इस हफ़्ते रिफ़िल या होम डिलीवरी चाहिए? YES भेजें, हम इंतज़ाम कर देंगे।")
    elif pharmacy:
        ctas["confirm"] = b.t("Reply YES and we'll keep it packed and ready for you.", "YES reply karein, hum aapke liye pack karke ready rakh denge.",
                              "YES भेजें, हम आपके लिए पैक करके तैयार रख देंगे।")
    elif s1 and not s2:
        ctas["confirm"] = b.t("Reply YES to book it.", "Book karne ke liye YES reply karein.", "बुक करने के लिए YES भेजें।")
    else:
        ctas["confirm"] = b.t("Reply YES and we'll hold a " + (f"{pref_txt} " if pref_txt else "") + "slot for you.",
                              "YES reply karein, hum aapke liye " + (f"{pref_txt} " if pref_txt else "") + "slot hold kar lenge.",
                              "YES भेजें, हम आपके लिए " + (f"{pref_txt} " if pref_txt else "") + "स्लॉट रख लेंगे।")
    ctas.setdefault("slot", ctas["confirm"])
    return Parts(hook, anchor, {}, ctas, send_as="merchant_on_behalf")


def r_customer_promo(b: Brief) -> Parts:
    """Customer attached to a merchant-level event. Only customer-relevant events are mentioned;
    merchant-internal facts (competitors, CTR, reviews, research) never reach the customer."""
    first = (b.cust.first_name if b.cust else "") or ""
    ms = b.merchant_short()
    ev = b.Araw("_event_family")
    hook = b.t(f"Hi {first}, {ms} here." if first else f"Hi, {ms} here.", f"Hi {first}, {ms} se." if first else f"Namaste, {ms} se.",
               f"नमस्ते {first}, {ms} से।")
    anchor = []
    if ev == "festival" and b.A("name"):
        anchor.append(b.t(f"{b.A('name')} is coming up" + (f" ({b.A('date')})" if b.A("date") else "") + " — we'd love to help you get ready.",
                          f"{b.A('name')} aa raha hai" + (f" ({b.A('date')})" if b.A("date") else "") + " — taiyaari mein hum madad karenge.",
                          f"{b.A('name')} आ रहा है" + (f" ({b.A('date')})" if b.A("date") else "") + " — तैयारी में हम मदद करेंगे।"))
    elif ev == "local_event" and b.A("headline") and "vs" in (b.A("headline") or ""):
        anchor.append(b.t(f"{b.A('headline')} tonight — watch it with us or order in.", f"Aaj {b.A('headline')} hai — hamare yahan dekhiye ya ghar mangwaiye.",
                          f"आज {b.A('headline')} है — हमारे यहाँ देखिए या घर मँगवाइए।"))
    elif ev == "weather" and b.A("temp"):
        anchor.append(b.t(f"It's {b.A('temp')} out there — stay cool.", f"Bahar {b.A('temp')} hai — dhyaan rakhiye.", f"बाहर {b.A('temp')} है — ध्यान रखिए।"))
    elif b.A("last_visit"):
        anchor.append(b.t(f"It's been a while since your last visit on {b.A('last_visit')} — we'd love to see you again.",
                          f"{b.A('last_visit')} ke baad aapse mulaqat nahi hui — aapka intezaar hai.",
                          f"{b.A('last_visit')} के बाद आपसे मुलाक़ात नहीं हुई — आपका इंतज़ार है।"))
    o = b.pick_customer_offer()
    if o:
        anchor.append(b.t(f"For you: {o}.", f"Aapke liye: {o}.", f"आपके लिए: {o}।"))
    ctas = {"confirm": b.t("Reply YES and we'll book it for you.", "YES reply karein, hum aapke liye book kar denge.", "YES भेजें, हम आपके लिए बुक कर देंगे।")}
    return Parts(hook, anchor, {}, ctas, send_as="merchant_on_behalf")


def r_customer_appointment(b: Brief) -> Parts:
    first = (b.cust.first_name if b.cust else "") or ""
    ms = b.merchant_short()
    when, svc = b.A("when"), b.A("service")
    tm = b.ta.anchor.get("when", {}).get("time")
    w = f"{when}{', ' + tm if tm else ''}" if when else b.t("tomorrow", "kal", "कल")
    on_en = "on " if when else ""
    on_hi = " ko" if when else ""
    hook = b.t(f"Hi {first}, a reminder from {ms}:", f"Hi {first}, {ms} se reminder:", f"नमस्ते {first}, {ms} से रिमाइंडर:")
    anchor = [b.t(f"your {svc + ' ' if svc else ''}appointment is {on_en}{w}.", f"aapka {svc + ' ' if svc else ''}appointment {w}{on_hi} hai.",
                  f"आपका {svc + ' ' if svc else ''}अपॉइंटमेंट {w}{' को' if when else ''} है।")]
    loc = b.P("locality")
    if loc:
        anchor.append(f"📍 {b.P('name') or ms}, {loc}.")
    ctas = {"confirm": b.t("Reply YES to confirm, or tell us a better time.", "Confirm karne ke liye YES reply karein, ya better time batayein.",
                           "कन्फ़र्म करने के लिए YES भेजें, या दूसरा समय बताएं।")}
    return Parts(hook, anchor, {}, ctas, send_as="merchant_on_behalf")


def r_generic(b: Brief) -> Parts:
    """Unseen trigger kind: lead with the payload's text fact verbatim, then its numbers — no interpretation."""
    name = b.P("name") or ""
    items = [(k, v) for k, v in b.ta.anchor.items() if k.startswith("g_")]
    text_fact, nums = None, []
    for k, v in items[:3]:
        b.A(k)
        is_text = isinstance(v.get("value"), str) and not re.fullmatch(r"[\d.,₹%-]+", str(v.get("value"))) and "date" not in v
        if is_text and text_fact is None and len(str(v["value"])) > 12:
            text_fact = v["text"]
        else:
            nums.append(f"{v.get('label', humanize(k[2:]))}: {v['text']}")
    kind = humanize(b.ta.trigger_type)
    detail = (f"“{text_fact}”" if text_fact else "") + ((" — " if text_fact else "") + "; ".join(nums) if nums else "")
    hook = b.t(f"{b.sal()}, heads-up for {name} ({kind})" + (f": {detail}." if detail else "."),
               f"{b.sal()}, {name} ke liye heads-up ({kind})" + (f": {detail}." if detail else "."))
    levers = {}
    ctas = _post_cta(b, "look into this and share the next steps here", "ise check karke next steps yahin bhej doon")
    ctas["curiosity"] = b.t("Want the details?", "Details bhej doon?")
    return Parts(hook, [], levers, ctas)


def as_float_safe(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def r_planning(b: Brief) -> Parts:
    topic, ask = b.A("topic") or "the idea", b.A("ask")
    name, cn = b.P("name") or "", b.cust_noun()
    hook = b.t(f"{b.sal()}, you asked “{ask}” — here's a first cut of the {topic}:" if ask else f"{b.sal()}, here's a first cut of the {topic}:",
               f"{b.sal()}, aapne poocha tha “{ask}” — {topic} ka pehla draft yeh raha:" if ask else f"{b.sal()}, {topic} ka pehla draft yeh raha:")
    bullets = []
    o, kind = b.offer()
    if o and kind == "active":
        bullets.append(b.t(f"• Price anchor: pitch it around your {o}", f"• Price anchor: aapke {o} ke aas-paas rakhein"))
    elif b.P("catalog_offer"):
        bullets.append(b.t(f"• Entry offer: a '{b.P('catalog_offer')}' style hook", f"• Entry offer: '{b.P('catalog_offer')}' jaisa hook"))
    if b.P("praise"):
        bullets.append(b.t(f"• Lead with what reviews already praise: {b.P('praise')}", f"• Jo reviews mein pasand kiya jaata hai wahi lead karein: {b.P('praise')}"))
    bullets.append(b.t(f"• Format: fixed batches, pre-booking on WhatsApp, one clear start date",
                       f"• Format: fixed batches, WhatsApp pe pre-booking, ek clear start date"))
    if b.P("customers_total"):
        bullets.append(b.t(f"• First invite: your {b.P('customers_total')} existing {cn} this year", f"• Pehla invite: is saal ke aapke {b.P('customers_total')} existing {cn}"))
    anchor = ["\n" + "\n".join(bullets)]
    levers = {"effort_externalization": ""}
    ctas = {"effort": b.t("\nWant me to turn this into a launch post + WhatsApp message you can send today? Reply YES.",
                          "\nIse launch post + WhatsApp message mein badal doon jo aap aaj bhej sakein? Reply YES."),
            "curiosity": b.t("\nWant the full version with pricing tiers left for you to fill?", "\nPricing tiers ke saath full version bhej doon?")}
    return Parts(hook, anchor, {}, ctas)


def r_seasonal(b: Brief) -> Parts:
    season = b.A("season") or b.t("This season", "Is season")
    ups = [b.A(k) for k in ("up1", "up2", "up3") if b.A(k)]
    down = b.A("down1")
    if ups:
        hook = b.t(f"{b.sal()}, {season} demand shift for {b.prof.slug}: {', '.join(ups)}" + (f"; {down}." if down else "."),
                   f"{b.sal()}, {season} mein {b.prof.slug} ki demand badal rahi hai: {', '.join(ups)}" + (f"; {down}." if down else "."))
    else:
        hook = b.t(f"{b.sal()}, seasonal heads-up: {b.A('beat')}." if b.A("beat") else f"{b.sal()}, a seasonal demand shift is starting.",
                   f"{b.sal()}, seasonal heads-up: {b.A('beat')}." if b.A("beat") else f"{b.sal()}, seasonal demand badal rahi hai.")
    levers = {"loss_aversion": b.t("Whoever moves these to the front shelf and posts first catches the demand; the rest discover it at month-end.",
                                   "Jo inhe front shelf pe laata hai aur pehle post karta hai, demand wahi pakadta hai."),
              "specificity": b.t(f"Your listing gets {b.P('calls')} calls in {b.P('window')} — a timely post turns them into orders." if b.P("calls") and b.P("window") else "",
                                 f"Aapki listing ko {(b.P('window') or '').replace('days', 'din')} mein {b.P('calls')} calls aate hain — timely post se yeh orders banenge." if b.P("calls") and b.P("window") else "")}
    o_en, o_hi = b.offer_phrase()
    ctas = _post_cta(b, "draft a 'summer essentials in stock' post" + (f" with {o_en}" if o_en else "") if "summer" in season.lower() else "draft a seasonal-essentials post" + (f" with {o_en}" if o_en else ""),
                     "'summer essentials in stock' post draft kar doon" + (f" {o_hi} ke saath" if o_hi else "") if "summer" in season.lower() else "seasonal-essentials post draft kar doon")
    ctas["ask"] = b.t("Which of these are you short on? I'll word the post around what you have.",
                      "Inmein se kya stock mein kam hai? Main post usi hisaab se likhungi.")
    return Parts(hook, [], levers, ctas)


def r_supply(b: Brief) -> Parts:
    mol, mfr, batches = b.A("molecule"), b.A("manufacturer"), b.A("batches")
    hook = b.t(f"{b.sal()}, batch alert: {mol}" + (f" ({mfr})" if mfr else "") + (f" batches {batches} are flagged." if batches else " is flagged."),
               f"{b.sal()}, batch alert: {mol}" + (f" ({mfr})" if mfr else "") + (f" ke batches {batches} flag hue hain." if batches else " flag hua hai."))
    anchor = []
    if b.A("title"):
        anchor.append(f"{_q(b.A('title'))}" + (f" — {b.A('source')}." if b.A("source") else "."))
    if b.A("actionable"):
        anchor.append(b.t(f"Action: {_short(b.A('actionable'), 110)}.", f"Action: {_short(b.A('actionable'), 110)}."))
    levers = {"loss_aversion": b.t("Pulling them from the shelf today avoids a customer walking out with an affected strip.",
                                   "Aaj hi shelf se hataane se koi customer affected strip ke saath nahi jayega.")}
    ctas = _post_cta(b, f"draft a WhatsApp for customers who recently bought {mol} asking them to check their batch",
                     f"jinhone recently {mol} liya hai unke liye batch check karne wala WhatsApp draft kar doon")
    return Parts(hook, anchor, levers, ctas)


REALIZERS = {
    "knowledge": r_knowledge, "regulation": r_regulation, "perf_dip": r_perf_dip, "perf_spike": r_perf_spike,
    "milestone": r_milestone, "competitor": r_competitor, "trend": r_trend, "festival": r_festival,
    "weather": r_weather, "local_event": r_local_event, "reputation": r_reputation, "dormant": r_dormant,
    "recurring": r_recurring, "account": r_account, "profile": r_profile, "offer": r_offer,
    "customer_recall": r_customer_recall, "customer_appointment": r_customer_appointment, "generic": r_generic,
    "planning": r_planning, "seasonal": r_seasonal, "supply": r_supply, "customer_promo": r_customer_promo,
}
CTA_FALLBACK = ["effort", "confirm", "slot", "curiosity", "ask"]


def cta_value_for(text: str) -> str:
    """Label the CTA by what it actually asks: YES/STOP-style binary, open question, or none."""
    low = (text or "").lower()
    if not low.strip():
        return "none"
    if re.search(r"reply yes|yes reply|yes भेजें|reply haan", low) and not re.search(r"1.*2", low):
        return "binary_yes_stop"
    return "open_ended"


class MessageComposer(Agent):
    name = "message_composer"
    uses_llm = True

    def realize(self, brief_args: tuple, plan: StrategyPlan) -> tuple[Draft, Parts]:
        b = Brief(*brief_args)
        fam = b.ta.family
        parts = REALIZERS.get(fam, r_generic)(b)
        segs: list[tuple[str, str]] = [("hook", parts.hook)] + [("anchor", a) for a in parts.anchor if a]
        added = 0
        for lever in plan.engagement_levers:
            s = parts.levers.get(lever)
            if s and added < 2 and all(s != t for _, t in segs):
                segs.append((f"lever:{lever}", s))
                added += 1
        key = plan.cta_strategy if plan.cta_strategy in parts.ctas else next((k for k in CTA_FALLBACK if k in parts.ctas), None)
        cta_text = parts.ctas.get(key, "") if key else ""
        if b.ta.is_informational and not cta_text:
            key = "info"
        if cta_text:
            segs.append(("cta", cta_text))
        draft = Draft(segments=segs, cta=cta_value_for(cta_text), plan=plan,
                      used_fact_ids=sorted(b.used), source="deterministic")
        return draft, parts

    def compose_candidates(self, brief_args: tuple, plans: list[StrategyPlan]) -> list[tuple[Draft, Parts]]:
        out = []
        seen = set()
        for p in plans:
            d, parts = self.realize(brief_args, p)
            if d.body not in seen:
                seen.add(d.body)
                out.append((d, parts))
        self.log(f"realized {len(out)} deterministic candidates")
        return out

    def llm_draft(self, brief_args: tuple, plan: StrategyPlan, reference: Draft, deadline: Optional[float]) -> Optional[Draft]:
        """LLM writes from the plan + whitelisted facts only. Returns None when unavailable."""
        llm = get_llm()
        if not llm.enabled:
            return None
        b = Brief(*brief_args)
        facts = []
        for k, v in b.ta.anchor.items():
            if not k.startswith("_") and v.get("text"):
                facts.append((v["fid"], f"trigger.{k}", v["text"]))
        for k, v in b.pz.items():
            if k != "last_touch" and v.get("text") and v.get("fid"):
                facts.append((v["fid"], f"merchant.{k}", v["text"]))
        whitelist = "\n".join(f"[{fid}] {name} = {txt}" for fid, name, txt in facts)
        lang_rule = {"en": "Indian English", "hi-en": "natural Hinglish in Roman script (like: 'Dr. Meera, aapka CTR 2.1% hai')",
                     "hi": "Hindi in Devanagari"}[b.lang]
        system = ("You are the Message Writer agent in Vera, magicpin's WhatsApp assistant for Indian merchants. "
                  "You write ONE WhatsApp message. Hard rules: use ONLY facts from the whitelist (no new numbers, names, "
                  "dates, offers, competitors or citations); one CTA in the final sentence; no preamble, no self-introduction; "
                  "no hype or ALL CAPS; never use these words: " + ", ".join(b.prof.taboos[:12]) + ". Reply with JSON only.")
        user = (f"Language: {lang_rule}. Voice: {b.prof.register} ({'; '.join(b.prof.style_rules[:4])}).\n"
                f"Why now: {b.ta.why_now}. Objective: {plan.objective}.\n"
                f"Levers to use (max 3): {', '.join(plan.engagement_levers)}. CTA style: {plan.cta_strategy} "
                f"({'binary: end with Reply YES' if CTA_VALUE.get(plan.cta_strategy) == 'binary_yes_stop' else 'open question'}).\n"
                f"Structure: hook with salutation '{b.sal() or ''}' → specific fact → why it matters → one CTA. Max 450 characters.\n"
                f"FACT WHITELIST:\n{whitelist}\n\nReference draft (improve it; keep every fact accurate):\n{reference.body}\n\n"
                'Return {"body": "...", "used_fact_ids": ["F0001", ...]}')
        out = llm.complete_json("message_composer", system, user, deadline=deadline, max_tokens=1500)
        if not out or not isinstance(out.get("body"), str) or len(out["body"].strip()) < 20:
            return None
        body = out["body"].strip()
        used = [f for f in out.get("used_fact_ids", []) if isinstance(f, str) and b.tools.ledger.get(f)]
        d = Draft(segments=[("llm", body)], cta=reference.cta, plan=plan,
                  used_fact_ids=sorted(set(used) | set(reference.used_fact_ids)), source="llm")
        self.log("llm draft produced", body[:120])
        return d
