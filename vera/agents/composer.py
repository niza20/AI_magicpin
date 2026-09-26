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
        return f"a '{o}' offer", f"'{o}' jaisa offer"

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
def _post_cta(b: Brief, what_en: str, what_hi: str) -> dict:
    return {
        "effort": b.t(f"Want me to {what_en}? Reply YES and I'll take it from there.",
                      f"Main {what_hi}? Bas YES reply karein, baaki main sambhal loongi."),
        "curiosity": b.t(f"Want me to {what_en}?", f"Kya main {what_hi}?"),
    }


def r_knowledge(b: Brief) -> Parts:
    src, title, n = b.A("source"), b.A("title"), b.A("trial_n")
    seg = b.P("segment_match")
    cn = b.cust_noun()
    if src:
        hook = b.t(f"{b.sal()}, {src} has one item worth your time:", f"{b.sal()}, {src} mein ek kaam ka item aaya hai:")
    else:
        hook = b.t(f"{b.sal()}, one item from this week's {b.prof.slug} digest:", f"{b.sal()}, is hafte ke digest mein ek kaam ka item:")
    anchor = [f"{_q(title)}" + (f" (n={n})." if n else ".")] if title else []
    summ = _short(b.Araw("summary") and b.A("summary"), 140)
    if summ:
        from ..ledger import extract_numbers
        head_nums = set(extract_numbers(" ".join(x for x in (title, n) if x)))
        overlap = len(head_nums & set(extract_numbers(summ)))
        if overlap < 2:
            anchor.append(b.t(f"Key takeaway: {summ}.", f"Key takeaway: {summ}."))
    levers = {}
    if seg:
        levers["reciprocity"] = b.t(f"Flagging it because it maps to your {seg} {cn}.",
                                    f"Aapke {seg} {cn} ke liye directly relevant hai, isliye share kar rahi hoon.")
    elif b.P("lapsed"):
        levers["reciprocity"] = b.t(f"Could be a good reason to re-engage your {b.P('lapsed')} lapsed {cn}.",
                                    f"Aapke {b.P('lapsed')} lapsed {cn} ko wapas bulane ka accha reason ban sakta hai.")
    elif b.P("customers_total"):
        levers["reciprocity"] = b.t(f"Relevant for the {b.P('customers_total')} {cn} you've seen this year.",
                                    f"Is saal ke aapke {b.P('customers_total')} {cn} ke liye relevant hai.")
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
        anchor.append(_q(title) + (b.t(f", effective {eff}.", f", {eff} se lagu.") if eff else "."))
    summ = _short(b.A("summary"))
    if summ:
        anchor.append(f"{summ}.")
    levers = {"loss_aversion": b.t("Better to be ready before the deadline than to fix it after an inspection.",
                                   "Deadline se pehle ready rehna better hai, inspection ke baad fix karne se."),
              "curiosity": b.t("It's a short read.", "Chhota sa read hai.")}
    who = b.P("name") or f"your {b.prof.noun_singular}"
    ctas = {"effort": b.t(f"Want me to turn it into a 1-page checklist for {who}? Reply YES.",
                          f"Main {who} ke liye 1-page checklist bana doon? Reply YES."),
            "curiosity": b.t("Want the key points summarised?",
                             "Key points summary bhej doon?")}
    return Parts(hook, anchor, levers, ctas)


def _perf(b: Brief, up: bool) -> Parts:
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
    o_en, o_hi = b.offer_phrase()
    if up:
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
    name = b.P("name") or ""
    end = "." if b.prof.hype_forbidden else "! 🎉"
    if val:
        hook = b.t(f"{b.sal()}, congrats — {name} just crossed {val} {metric or ''}{end}".replace(" .", ".").replace("  ", " "),
                   f"{b.sal()}, badhai ho — {name} ne {val} {metric or ''} cross kar liye{end}".replace("  ", " "))
    else:
        hook = b.t(f"{b.sal()}, congrats — {name} just hit a {metric or 'new'} milestone{end}",
                   f"{b.sal()}, badhai ho — {name} ne naya {metric or ''} milestone hit kiya{end}")
    levers = {}
    r, pr = b.P("rating"), b.P("peer_rating")
    if r and pr and (b.pz["rating"]["value"] >= b.pz["peer_rating"]["value"]):
        levers["social_proof"] = b.t(f"At {r}, you're above the {pr} average for {b.prof.peer_label}.",
                                     f"{r} ke saath aap {b.prof.peer_label} ke {pr} average se upar hain.")
    elif b.P("reviews") and b.P("peer_reviews"):
        levers["social_proof"] = b.t(f"That's {b.P('reviews')} reviews vs a {b.P('peer_reviews')} peer average.",
                                     f"Aapke {b.P('reviews')} reviews vs {b.P('peer_reviews')} peer average.")
    levers["reciprocity"] = b.t("Milestones like this are great proof for new customers.",
                                "Aise milestones naye customers ke liye best proof hote hain.")
    cn = b.cust_noun()
    ctas = _post_cta(b, f"post a thank-you update and invite happy {cn} to add photo reviews",
                     f"ek thank-you post daal doon aur happy {cn} ko photo review ke liye invite kar doon")
    ctas["ask"] = b.t(f"What do your regulars love most? I'll feature it in the thank-you post.",
                      "Aapke regulars ko sabse zyada kya pasand hai? Main thank-you post mein wahi feature karungi.")
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
    if b.A("offer"):
        anchor.append(b.t(f"Their launch offer: {b.A('offer')}.", f"Unka launch offer: {b.A('offer')}."))
    levers = {}
    r, rv, pr = b.P("rating"), b.P("reviews"), b.P("peer_rating")
    if r and rv:
        levers["social_proof"] = b.t(f"Your {r} across {rv} reviews is a head start they don't have yet.",
                                     f"Aapke {rv} reviews aur {r} rating unke paas abhi nahi hai — yeh aapka head start hai.")
    elif r and pr:
        levers["social_proof"] = b.t(f"Your {r} vs the {pr} peer average is your edge.", f"Aapka {r} vs {pr} peer average — yahi aapka edge hai.")
    levers["loss_aversion"] = b.t(f"New openings pull curious {b.cust_noun()} in the first few weeks — worth staying visible now.",
                                  f"Naye {noun} pehle kuch hafton mein curious {b.cust_noun()} kheenchte hain — abhi visible rehna zaroori hai.")
    o_en, o_hi = b.offer_phrase()
    ctas = _post_cta(b, f"refresh your Google post with {o_en} this week" if o_en else "refresh your Google post this week",
                     f"is hafte {o_hi} ke saath aapki Google post refresh kar doon" if o_hi else "is hafte aapki Google post refresh kar doon")
    ctas["curiosity"] = b.t("Want a side-by-side of your listing vs theirs?", "Aapki aur unki listing ka side-by-side comparison bhej doon?")
    return Parts(hook, anchor, levers, ctas)


def r_trend(b: Brief) -> Parts:
    q, d = b.A("query"), b.A("delta")
    yoy = b.t(" year-on-year", " saal-dar-saal") if (b.ta.anchor.get("delta", {}).get("path") or "").endswith("yoy") else ""
    seg = b.A("segment")
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
    name, date, days = b.A("name") or "The festival", b.A("date"), b.A("days_until")
    if days:
        hook = b.t(f"{b.sal()}, {name} is {days} days away" + (f" ({date})." if date else "."),
                   f"{b.sal()}, {name} sirf {days} din door hai" + (f" ({date})." if date else "."))
    else:
        hook = b.t(f"{b.sal()}, {name} is coming up" + (f" on {date}." if date else "."),
                   f"{b.sal()}, {name} aa raha hai" + (f" — {date}." if date else "."))
    anchor = []
    if b.A("beat"):
        anchor.append(b.t(f"Seasonal pattern for your category: {b.A('beat')}.", f"Aapki category ka seasonal pattern: {b.A('beat')}."))
    levers = {"loss_aversion": b.t(f"Posts that go live before {name} catch the planning rush; after, it's too late.",
                                   f"{name} se pehle live post planning karne walon tak pahunchti hai — baad mein late ho jaata hai.")}
    o_en, o_hi = b.offer_phrase()
    ctas = _post_cta(b, f"set up a {name} post featuring {o_en}" if o_en else f"set up a {name} post for you",
                     f"{o_hi} ke saath {name} post ready kar doon" if o_hi else f"aapke liye {name} post ready kar doon")
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
    loc_raw = b.Araw("location")
    loc = b.A("location") if loc_raw and str(loc_raw).split(",")[0].strip().lower() not in (hl or "").lower() else None
    hook = b.t(f"{b.sal()}, local heads-up: {hl}" + (f" ({loc})" if loc else "") + (f", for {dur}" if dur else "") + (f" on {date}." if date else "."),
               f"{b.sal()}, ek local update: {hl}" + (f" ({loc})" if loc else "") + (f", {dur}" if dur else "") + (f" — {date}." if date else "."))
    anchor = [f"{_short(b.A('impact'))}."] if b.A("impact") else []
    cn = b.cust_noun()
    levers = {"loss_aversion": b.t(f"{cn.capitalize()} planning to visit may get caught out — a quick update keeps them coming.",
                                   f"Aane wale {cn} confuse ho sakte hain — ek quick update se woh aate rahenge.")}
    if b.P("directions") and b.P("window"):
        levers["loss_aversion"] = b.t(f"You had {b.P('directions')} direction requests in the last {b.P('window')} — {cn} navigating to you today may get caught out.",
                                      f"Pichhle {b.P('window').replace('days', 'din')} mein {b.P('directions')} direction requests aaye — aaj aapki taraf aa rahe {cn} atak sakte hain.")
    elif b.P("calls") and b.P("window"):
        levers["loss_aversion"] = b.t(f"With {b.P('calls')} calls in the last {b.P('window')}, some {cn} will ask — a quick update saves the back-and-forth.",
                                      f"Pichhle {b.P('window').replace('days', 'din')} mein {b.P('calls')} calls aaye — ek quick update se {cn} confuse nahi honge.")
    ctas = _post_cta(b, f"post a quick update for your {cn} (timings / delivery / directions)",
                     f"{cn} ke liye ek quick update post kar doon (timings / delivery / directions)")
    ctas["ask"] = b.t("Is this affecting your footfall today? Tell me and I'll adjust your listing.",
                      "Kya isse aaj footfall pe asar pad raha hai? Bataiye, main listing adjust kar deti hoon.")
    return Parts(hook, anchor, levers, ctas)


def r_reputation(b: Brief) -> Parts:
    cnt, theme = b.A("count"), b.A("theme")
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
    hook = b.t(f"{b.sal()}, quick one" + (f" — it's been {days} days since we last spoke." if days else "."),
               f"{b.sal()}, ek quick baat" + (f" — {days} din ho gaye baat kiye." if days else "."))
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
    topic = b.A("topic")
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


def r_account(b: Brief) -> Parts:
    plan, days, date, amt = b.A("plan"), b.A("days_left"), b.A("date"), b.A("amount")
    hook = b.t(f"{b.sal()}, your {plan + ' ' if plan else ''}plan has {days} days left" + (f" (renews {date})." if date else "."),
               f"{b.sal()}, aapke {plan + ' ' if plan else ''}plan mein {days} din bache hain" + (f" ({date} ko renew)." if date else "."))
    anchor = []
    if b.P("views") and b.P("calls") and b.P("window"):
        anchor.append(b.t(f"In the last {b.P('window')} your listing got {b.P('views')} views and {b.P('calls')} calls.",
                          f"Pichhle {b.P('window').replace('days', 'din')} mein aapki listing ko {b.P('views')} views aur {b.P('calls')} calls mile."))
    if amt:
        anchor.append(b.t(f"Renewal: {amt}.", f"Renewal: {amt}."))
    levers = {"loss_aversion": b.t("Renewing now keeps your posts and offers running without a gap.",
                                   "Abhi renew karne se posts aur offers bina gap ke chalte rahenge."),
              "reciprocity": b.t("Wanted to flag it early so nothing pauses.", "Pehle se bata rahi hoon taaki kuch ruke nahi.")}
    ctas = {"effort": b.t("Want me to process the renewal? Reply YES.", "Renewal process kar doon? Reply YES.")}
    return Parts(hook, anchor, levers, ctas)


def r_profile(b: Brief) -> Parts:
    miss, comp, days = b.A("missing"), b.A("completeness"), b.A("days") or b.P("stale_days")
    if miss:
        hook = b.t(f"{b.sal()}, your Google profile is missing: {miss}" + (f" (it's {comp} complete)." if comp else "."),
                   f"{b.sal()}, aapke Google profile mein yeh missing hai: {miss}" + (f" (abhi {comp} complete)." if comp else "."))
    elif comp:
        hook = b.t(f"{b.sal()}, your Google profile is {comp} complete.", f"{b.sal()}, aapka Google profile abhi {comp} complete hai.")
    else:
        hook = b.t(f"{b.sal()}, your last Google post was {days} days ago." if days else f"{b.sal()}, your Google profile needs a quick refresh.",
                   f"{b.sal()}, aapki last Google post {days} din purani hai." if days else f"{b.sal()}, aapke Google profile ko quick refresh chahiye.")
    levers = {"loss_aversion": b.t("Incomplete listings lose clicks to complete ones in the same search.",
                                   "Same search mein incomplete listing ke clicks complete listings le jaati hain.")}
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
    ms = b.merchant_short()
    emoji = " 🦷" if b.prof.slug == "dentists" else (" ✨" if b.prof.register == "warm" else "")
    months, svc, due = b.A("months"), b.A("service"), b.A("due")
    last_svc = b.Araw("last_service") and b.A("last_service")
    visit_en, visit_hi = (last_svc, last_svc) if last_svc else ("visit", "visit")
    pharmacy = b.prof.register == "trust"
    hook = b.t(f"Hi {first}, {ms} here{emoji}.", f"Hi {first}, {ms} se{emoji}.", f"नमस्ते {first}, {ms} से{emoji}।")
    anchor = []
    if pharmacy and svc and b.A("last_visit"):
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
    else:
        anchor.append(b.t("It's been a while since your last visit.", "Aapki last visit ko kaafi time ho gaya.", "आपकी पिछली विज़िट को काफ़ी समय हो गया।"))
    o = b.P("offer")
    if o:
        anchor.append(b.t(f"Current offer: {o}.", f"Abhi offer: {o}.", f"अभी का ऑफ़र: {o}।"))
    s1, s2 = b.A("slot1"), b.A("slot2")
    ctas = {}
    if s1 and s2:
        b.tools.ledger.derive("slot_count", 2, "2", [b.ta.anchor["slot1"]["path"] or "", b.ta.anchor["slot2"]["path"] or ""])
        anchor.append(b.t(f"Two slots are open for you: {s1} or {s2}.", f"Aapke liye 2 slots ready hain: {s1} ya {s2}.",
                          f"आपके लिए 2 स्लॉट तैयार हैं: {s1} या {s2}।"))
        ctas["slot"] = b.t("Reply 1 or 2 to book, or tell us a time that works.", "Book karne ke liye 1 ya 2 reply karein, ya apna time batayein.",
                           "बुक करने के लिए 1 या 2 भेजें, या अपना समय बताएं।")
    pref = b.tools.get_customer_fact("preferences.preferred_slots")
    pref_txt = humanize(pref) if isinstance(pref, str) else None
    if pref_txt:
        b.tools.ledger.allow_words(pref_txt.split())
    if pharmacy:
        ctas["confirm"] = b.t("Reply YES and we'll keep it packed and ready for you.", "YES reply karein, hum aapke liye pack karke ready rakh denge.",
                              "YES भेजें, हम आपके लिए पैक करके तैयार रख देंगे।")
    else:
        ctas["confirm"] = b.t("Reply YES and we'll hold a " + (f"{pref_txt} " if pref_txt else "") + "slot for you.",
                              "YES reply karein, hum aapke liye " + (f"{pref_txt} " if pref_txt else "") + "slot hold kar lenge.",
                              "YES भेजें, हम आपके लिए " + (f"{pref_txt} " if pref_txt else "") + "स्लॉट रख लेंगे।")
    ctas.setdefault("slot", ctas["confirm"])
    return Parts(hook, anchor, {}, ctas, send_as="merchant_on_behalf")


def r_customer_appointment(b: Brief) -> Parts:
    first = (b.cust.first_name if b.cust else "") or ""
    ms = b.merchant_short()
    when, svc = b.A("when"), b.A("service")
    tm = b.ta.anchor.get("when", {}).get("time")
    w = f"{when}{', ' + tm if tm else ''}" if when else b.t("tomorrow", "kal", "कल")
    hook = b.t(f"Hi {first}, a reminder from {ms}:", f"Hi {first}, {ms} se reminder:", f"नमस्ते {first}, {ms} से रिमाइंडर:")
    anchor = [b.t(f"your {svc + ' ' if svc else ''}appointment is on {w}.", f"aapka {svc + ' ' if svc else ''}appointment {w} ko hai.",
                  f"आपका {svc + ' ' if svc else ''}अपॉइंटमेंट {w} को है।")]
    ctas = {"confirm": b.t("Reply YES to confirm, or tell us a better time.", "Confirm karne ke liye YES reply karein, ya better time batayein.",
                           "कन्फ़र्म करने के लिए YES भेजें, या दूसरा समय बताएं।")}
    return Parts(hook, anchor, {}, ctas, send_as="merchant_on_behalf")


def r_generic(b: Brief) -> Parts:
    name = b.P("name") or ""
    items = [(k, v) for k, v in b.ta.anchor.items() if k.startswith("g_")]
    facts = []
    for k, v in items[:3]:
        b.A(k)
        facts.append(f"{v.get('label', humanize(k[2:]))}: {v['text']}")
    kind = humanize(b.ta.trigger_type)
    hook = b.t(f"{b.sal()}, a quick {kind} update for {name}" + (f" — {'; '.join(facts)}." if facts else "."),
               f"{b.sal()}, {name} ke liye ek quick {kind} update" + (f" — {'; '.join(facts)}." if facts else "."))
    levers = {}
    ctas = _post_cta(b, "look into this and share the next steps here", "ise check karke next steps yahin bhej doon")
    ctas["curiosity"] = b.t("Want the details?", "Details bhej doon?")
    return Parts(hook, [], levers, ctas)


REALIZERS = {
    "knowledge": r_knowledge, "regulation": r_regulation, "perf_dip": r_perf_dip, "perf_spike": r_perf_spike,
    "milestone": r_milestone, "competitor": r_competitor, "trend": r_trend, "festival": r_festival,
    "weather": r_weather, "local_event": r_local_event, "reputation": r_reputation, "dormant": r_dormant,
    "recurring": r_recurring, "account": r_account, "profile": r_profile, "offer": r_offer,
    "customer_recall": r_customer_recall, "customer_appointment": r_customer_appointment, "generic": r_generic,
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
