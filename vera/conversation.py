"""MULTI-TURN MEMORY + reply state machine.

Memory separation:
  * SHORT-TERM  — ConversationState: turns, intents, CTAs used, facts already mentioned, auto-reply
                  status, actions requested, previous strategy, exit state.
  * LONG-TERM   — the merchant/category/trigger/customer contexts supplied by the dataset/judge
                  (referenced, never mutated; the LLM can't write persistent facts).
  * WORKING     — the per-turn analysis bundle (ledger, anchor, personalization) rebuilt from
                  long-term context; discarded after the turn.

Reply flow: Incoming → Auto-Reply Detector → Intent Router → state update → action/response
strategy → deterministic realizer → Fact/Policy validators → finalize ({send|wait|end}).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from .agents.composer import Brief
from .agents.intent_router import AutoReplyDetector, IntentRouter, confidential_kind, detect_language, language_signal
from .agents.validators import FactChecker, PolicyChecker
from .orchestrator import Orchestrator
from .types import Draft, LanguagePlan, StrategyPlan, TraceStep
from .util import humanize

CURVEBALLS = {
    "identity": ("_cb_identity", "binary_yes_stop", "Merchant asked who Vera is → brief honest intro, then back to the one action."),
    "trust": ("_cb_trust", "binary_yes_stop", "Trust/scam question → show where the facts come from; never invented numbers."),
    "callback": ("_cb_callback", "none", "Merchant wants a human → hand off to the magicpin team, keep the draft ready."),
    "billing": ("_cb_billing", "binary_yes_stop", "Billing/commission is out of Vera's authority → say so honestly, offer what she can do."),
    "outcome": ("_cb_outcome", "binary_yes_stop", "Asked for guaranteed results → refuse to invent a number, cite real listing numbers."),
    "glossary": ("_cb_glossary", "binary_yes_stop", "Explained the term with this merchant's own value."),
    "delegate": ("_cb_delegate", "binary_yes_stop", "Someone else manages the page → offer forwardable draft / send to them."),
    "already_done": ("_cb_already", "binary_yes_stop", "Merchant already did something similar → acknowledge, reuse it on Google."),
}
_WHEN = re.compile(r"\b(tomorrow(?: morning| evening| afternoon| night)?|kal(?: subah| shaam| raat)?|tonight|this evening|this weekend|"
                   r"(?:on )?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|at \d{1,2}(?::\d{2})?\s?(?:am|pm)?)\b", re.I)


_DAY = r"(today|aaj|tonight|tomorrow|kal|parso|day after tomorrow|this weekend|weekend|next week|monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)"
_PART = r"(morning|afternoon|evening|night|subah|dopahar|shaam|sham|raat)"
_CLOCK = r"(\d{1,2}(?::\d{2})?\s*(?:am|pm|baje)|\d{1,2}:\d{2})"
_DAY_EN = {"aaj": "today", "kal": "tomorrow", "parso": "day after tomorrow", "mon": "Monday", "tue": "Tuesday", "wed": "Wednesday",
           "thu": "Thursday", "fri": "Friday", "sat": "Saturday", "sun": "Sunday", "weekend": "this weekend"}
_PART_EN = {"subah": "morning", "dopahar": "afternoon", "shaam": "evening", "sham": "evening", "raat": "night"}


def cust_time(message: str) -> Optional[str]:
    """A day and/or time the customer proposed ("kal shaam 6 baje" → "Tomorrow evening, 6pm"); None if there's none."""
    low = (message or "").lower()
    d = re.search(rf"\b{_DAY}\b", low)
    p = re.search(rf"\b{_PART}\b", low)
    c = re.search(rf"\b{_CLOCK}", low)
    if not (d or c) and not (p and re.search(r"\b(book|slot|come|aa|aaunga|aaungi|chalega|works)\b", low)):
        return None
    day = _DAY_EN.get(d.group(1), d.group(1)) if d else ""
    part = _PART_EN.get(p.group(1), p.group(1)) if p else ""
    clock = re.sub(r"\s*baje", "", c.group(1)).replace(" ", "") if c else ""
    if clock and not re.search(r"am|pm", clock):
        h = int(clock.split(":")[0])
        clock += "pm" if (part in ("evening", "night", "afternoon") or 1 <= h <= 8) else "am"
    head = " ".join(x for x in (day, part) if x).strip()
    out = (head[:1].upper() + head[1:]) if head else ""
    return (f"{out}, {clock}" if out and clock else out or clock) or None


def loc_time(t: Optional[str], lang: str) -> str:
    """'Tomorrow evening, 6pm' → 'kal shaam, 6pm' for Hinglish/Hindi conversations (Hindi script comes from translit)."""
    if not t or lang not in ("hi-en", "hi"):
        return t or ""
    out = t
    for en, hi in (("Day after tomorrow", "parso"), ("day after tomorrow", "parso"), ("Tomorrow", "kal"), ("tomorrow", "kal"),
                   ("Today", "aaj"), ("today", "aaj"), ("This weekend", "is weekend"), ("this weekend", "is weekend"),
                   ("morning", "subah"), ("afternoon", "dopahar"), ("evening", "shaam"), ("night", "raat")):
        out = re.sub(rf"\b{en}\b", hi, out)
    return out


_ITEM_REQ = re.compile(r"\b(?:bring|add|send|include|deliver|also need|need|want|chahiye|laana|lana|bhej(?:na|do)?)\s+(?:me\s+|us\s+|some\s+|a\s+|an\s+|the\s+|ek\s+)?"
                       r"([a-z][a-z0-9+ -]{2,30}?)(?=\s+(?:along|with|too|as well|also|bhi|saath|please|pls)\b|\s*[?.!]|\s*$)", re.I)
_ITEM_STOP = {"it", "this", "that", "them", "slot", "booking", "appointment", "reminder", "details", "time", "delivery", "help", "info", "more", "photo"}


def _when(message: str) -> Optional[str]:
    m = _WHEN.search(message or "")
    return m.group(0) if m else None


POST_FAMILIES = {"perf_dip", "perf_spike", "milestone", "competitor", "trend", "festival", "weather", "local_event",
                 "dormant", "recurring", "profile", "offer", "generic"}


@dataclass
class ConversationState:
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    trigger_id: Optional[str] = None
    # long-term context (references)
    category: dict = field(default_factory=dict)
    merchant: dict = field(default_factory=dict)
    trigger: dict = field(default_factory=dict)
    customer: Optional[dict] = None
    # short-term memory
    messages: list[dict] = field(default_factory=list)          # {"from": "vera"|"merchant"|"customer", "body": str}
    intents: list[str] = field(default_factory=list)
    facts_mentioned: set = field(default_factory=set)
    ctas_used: list[str] = field(default_factory=list)
    offers_mentioned: set = field(default_factory=set)
    merchant_sentiment: str = "neutral"
    merchant_intent: str = "none"
    auto_reply_count: int = 0
    clarify_attempted: bool = False
    actions_requested: list[str] = field(default_factory=list)
    previous_strategy: Optional[str] = None
    exit_state: Optional[str] = None                             # None | "ended" | "waiting"
    language: Optional[str] = None
    language_locked: bool = False                                # user explicitly chose a language → don't auto-switch
    hostile_count: int = 0
    off_topic_count: int = 0
    objection_count: int = 0
    unclear_count: int = 0
    later_count: int = 0
    merchant_memory: dict = field(default_factory=dict)          # shared per-merchant memory (e.g. auto-reply texts)
    attachments: list = field(default_factory=list)              # photos the merchant shared in this chat
    pending_deal: Optional[str] = None                           # catalog deal Vera proposed to put live
    last_draft: Optional[str] = None                             # the artifact Vera last delivered (post text, WhatsApp, ...)
    draft_live: bool = False                                     # merchant confirmed it with GO
    ideas: list = field(default_factory=list)                    # numbered next-step ideas Vera offered
    promise_override: Optional[tuple] = None                     # (kind, offer) chosen from the ideas menu
    delivered: list = field(default_factory=list)                # (kind, offer) of every artifact already delivered
    cust_stage: str = ""                                         # customer booking flow: "" | "ask_time" | "booked"
    booked: Optional[str] = None                                 # the slot the customer booked
    partial_day: Optional[str] = None                            # customer gave a day but no time yet ("kal")
    fulfil: Optional[str] = None                                 # pharmacies: "delivery" | "pickup"
    order_items: list = field(default_factory=list)              # extra items the customer asked to add

    def bot_bodies(self) -> list[str]:
        return [m["body"] for m in self.messages if m["from"] == "vera"]

    def their_msgs(self) -> list[str]:
        return [m["body"] for m in self.messages if m["from"] != "vera"]

    def record_bot(self, body: str, cta: str) -> None:
        self.messages.append({"from": "vera", "body": body, "ts": time.time()})
        self.ctas_used.append(cta)


_PROMISES = [  # (kind, pattern over Vera's last offer) — first match wins; order matters
    ("thanks_post", r"thank-you post|thank you post"),
    ("review_request", r"review[- ]request|photo review|review ke liye invite|invite .{0,40}review|reviews? (maang|request)"),
    ("review_replies", r"owner repl|review repl"),
    ("checklist", r"checklist"),
    ("summary", r"2-min summary|summary"),
    ("renewal", r"renew|reactivat"),
    ("verification", r"verification|verify"),
    ("reminder", r"reminder bhej|remind you|reminder a day|ek din pehle reminder"),
    ("customer_msg", r"check-in whatsapp|whatsapp draft|nudge|batch check|wapas bula|win-?back"),
    ("pin", r"\bpin\b"),
    ("program", r"pehla draft|first draft|launch post"),
    ("post", r"\bpost\b"),
]
_ACK = re.compile(r"^\s*(thanks?|thank you|thx|ty|shukriya|dhanyavaad|dhanyawad|great|nice|cool|awesome|super|perfect|good|"
                  r"got it|noted|okay noted|ok noted|theek hai|thik hai|accha|achha)\b[\w\s!.🙏👍]*$", re.I)


_OTHER_PEOPLE = re.compile(r"\b(which|who|kaun|kis|how many)\b[^.?!]{0,25}\b(other|else|aur|baaki|dusre|doosre)\b[^.?!]{0,20}\b(customers?|patients?|clients?|members?|people|log)\b"
                           r"|\b(other|baaki|dusre|doosre)\s+(customers?|patients?|clients?|members?|people|log)\b[^.?!]{0,30}\b(name|number|coming|visit|book|appointment|details?|kaun)", re.I)


_IDEAS_Q = re.compile(r"\b((other|more|aur|next|new|koi aur|dusr\w*|doosr\w*)\s+(marketing\s+)?(strateg\w*|ideas?|suggestions?|options?|tips?|ways?|tareeke|tarike)"
                      r"|marketing\s+(strateg\w*|ideas?|tips?|plan)|what\s+else\b|aur\s+kya\b|kya\s+aur\b|how\s+(can|do)\s+i\s+(grow|get\s+more|increase|improve)"
                      r"|suggest\s+(something|more|me)|any\s+(other\s+)?(ideas?|suggestions?)|next\s+step|^\s*ideas?\s*[!.?]*\s*$)", re.I)
_DRAFT_Q = re.compile(r"\b(where|show|send|see|share|resend|kahan|kaha|dikhao|dikha\s*do|bhejo|bhej\s*do)\b[^.?!]{0,25}\b(draft|preview|post|message|msg)\b"
                      r"|\b(draft|preview)\b[^.?!]{0,15}\b(where|kahan|kaha|nahi\s+(aaya|mila|dikha)|not\s+(here|shown|visible|received)|missing)\b"
                      r"|\b(i\s+)?(can'?t|cannot|don'?t)\s+see\s+(the\s+|any\s+)?(draft|preview|post)", re.I)
_NO_ONLY = re.compile(r"^\s*(no|nope|nah|nahi|nahin|na|no thanks|no changes?|nothing|kuch nahi|all good|sab theek|theek hai)[\s!.,🙂👍]*$", re.I)
_PICK = re.compile(r"^\s*(?:option\s*|idea\s*|no\.?\s*)?([1-3])\b|^\s*(first|second|third|pehla|pehli|doosra|doosri|dusra|teesra|teesri)\b", re.I)
_ORD = {"first": 1, "pehla": 1, "pehli": 1, "second": 2, "doosra": 2, "doosri": 2, "dusra": 2, "third": 3, "teesra": 3, "teesri": 3}


def draft_core(body: str) -> str:
    """The artifact inside a delivery message: lines after '↓', minus the trailing GO/edits instruction."""
    lines = (body or "").split("\n")
    idx = [i for i, ln in enumerate(lines) if "↓" in ln]
    if not idx:
        return body.strip()
    core = lines[idx[0] + 1:]
    while core and re.search(r"\bGO\b|edits|bhej dijiye|reply", core[-1], re.I):
        core = core[:-1]
    return "\n".join(core).strip() or body.strip()


def promised(state) -> tuple[str, Optional[str]]:
    """What Vera last offered to do, read from its own words (so a YES delivers exactly that)."""
    if state.promise_override:
        return state.promise_override
    for body in reversed(state.bot_bodies()):
        asks = [x for x in re.split(r"(?<=[.!?])\s+", body) if "?" in x or re.search(r"\bYES\b", x)]
        text = " ".join(asks[-2:]) if asks else ""
        if not text:
            continue
        low = text.lower()
        for kind, rx in _PROMISES:
            if re.search(rx, low):
                # the offer Vera named — matched against real offer titles, never parsed free-form
                titles = [str(o.get("title")) for o in (state.merchant or {}).get("offers", []) if isinstance(o, dict) and o.get("title")]
                titles += [str(o.get("title")) for o in (state.category or {}).get("offer_catalog", []) if isinstance(o, dict) and o.get("title")]
                named = sorted((t for t in titles if t in text), key=len, reverse=True)
                return kind, (named[0] if named else None)
    return "post", None


class ReplyEngine:
    def __init__(self) -> None:
        self.orch = Orchestrator()

    # ------------------------------------------------------------------ main
    def respond(self, state: ConversationState, message: str, from_role: str = "merchant") -> dict:
        trace: list[TraceStep] = []
        if state.exit_state == "ended":
            return {"action": "end", "rationale": "Conversation already closed; not re-engaging."}
        prior_theirs = state.their_msgs()
        state.messages.append({"from": from_role, "body": message, "ts": time.time()})

        mm = state.merchant_memory
        prior_merchant_auto = mm.get("auto_reply_texts", [])
        verdict = AutoReplyDetector(trace).run(message, prior_theirs + prior_merchant_auto, max(state.auto_reply_count, mm.get("auto_reply_count", 0)))
        if verdict.is_auto_reply:
            state.auto_reply_count += 1
            mm["auto_reply_count"] = mm.get("auto_reply_count", 0) + 1
            mm.setdefault("auto_reply_texts", []).append(message)
            state.intents.append("auto_reply")
            state.merchant_intent = "auto_reply"
            if verdict.recommended_behavior == "clarify_once" and not state.clarify_attempted:
                state.clarify_attempted = True
                w = self._working(state, None)
                if state.actions_requested:
                    b = w["brief"]
                    sal = "" if b.sal() == "Hi" else f" {b.sal()},"
                    body = b.t(f"Looks like an auto-reply 🙂{sal} your draft is ready above — just reply GO to publish it, or send edits.",
                               f"Lagta hai yeh auto-reply hai 🙂{sal} aapka draft upar ready hai — publish karne ke liye bas GO reply karein, ya edits bhejein.")
                    return self._send(state, w, body, "binary_yes_stop",
                                      f"Auto-reply detected ({verdict.reason}) after the action was delivered; reminding of the pending GO instead of re-pitching.")
                return self._send(state, w, self._clarify_auto(w), "binary_yes_stop",
                                  f"Auto-reply detected ({verdict.reason}); one owner-directed nudge, will exit if it repeats.")
            state.exit_state = "ended"
            return {"action": "end", "rationale": f"Auto-reply again ({verdict.reason}); exiting instead of burning turns."}

        router = IntentRouter(trace)
        intent = router.classify(message)
        state.intents.append(intent.merchant_intent)
        state.merchant_intent = intent.merchant_intent
        lang_now = language_signal(message)
        if lang_now and not state.language_locked:
            state.language = lang_now                  # a merchant who writes Devanagari gets Devanagari back
        w = self._working(state, message)
        mi = intent.merchant_intent

        conf = confidential_kind(message)
        if not conf and (from_role == "customer" or state.customer) and _OTHER_PEOPLE.search(message):
            conf = "customer_pii"          # a customer asking about other customers' visits/bookings
        if conf and (from_role == "customer" or state.customer):
            return self._send(state, w, self._confidential_customer(w, conf), "open_ended",
                              f"Confidentiality ({conf}): customer asked for data that isn't theirs to see — declined, offered help.")
        if from_role == "customer" or state.customer:
            return self._customer_reply(state, w, message, mi)
        if conf:
            state.intents[-1] = f"confidential:{conf}"
            return self._send(state, w, self._confidential(state, w, conf), "binary_yes_stop" if conf != "own_data" else "open_ended",
                              f"Confidentiality ({conf}): shared only what this merchant is entitled to see, then back to the one useful step.",
                              allow=("self_intro",) if conf == "internal" else ())

        b = w["brief"]
        last_vera = (state.bot_bodies() or [""])[-1] if len(state.bot_bodies()) else ""
        if re.search(r"[\u0900-\u097F]", message):
            from .translit import to_roman
            message = to_roman(message)            # "कोई और आइडिया?" → "koi aur ideas?" for the checks below
        # a number / "first" right after the ideas menu → deliver that idea as a ready draft
        pick = _PICK.match(message) if state.ideas else None
        if pick:
            k = int(pick.group(1)) if pick.group(1) else _ORD.get(pick.group(2).lower(), 1)
            if not 1 <= k <= len(state.ideas):
                return self._send(state, w, self._ideas_menu(state, w), "open_ended", "Picked a number that isn't on the list → show the options again.")
            if 1 <= k <= len(state.ideas):
                idea = state.ideas[k - 1]
                state.ideas, state.promise_override = [], (idea["kind"], idea.get("offer"))
                w["promise"] = state.promise_override
                state.actions_requested.append(idea["kind"])
                body = self._act(w, False)
                state.delivered.append((idea["kind"], idea.get("offer")))
                if idea["kind"] != "photos":                   # photos: nothing to publish until they arrive
                    state.last_draft, state.draft_live = body[0], False
                return self._send(state, w, body, "open_ended", f"Merchant picked idea {k} ({idea['kind']}) → delivered it as a ready draft.")
        # "other strategies / what else / aur kya" → 3 concrete ideas from this merchant's data
        if _IDEAS_Q.search(message):
            return self._send(state, w, self._ideas_menu(state, w), "open_ended",
                              "Merchant asked for more ideas → 3 numbered, data-backed next steps; a number delivers the draft.")
        # "where's the draft / show me" → show the actual draft (or make it now)
        if _DRAFT_Q.search(message):
            if state.last_draft:
                core = draft_core(state.last_draft)
                tail = b.t("Reply GO and it goes live, or send any edits.", "GO reply karein toh live kar doon, ya edits bhej dijiye.") if not state.draft_live \
                    else b.t("This one is already live ✅ Want 3 more ideas? Reply YES.", "Yeh already live hai ✅ 3 aur ideas chahiye? YES reply karein.")
                return self._send(state, w, [b.t(f"Here it is ↓\n{core}\n{tail}", f"Yeh raha ↓\n{core}\n{tail}"),
                                             b.t(f"Sure — here's the draft ↓\n{core}\n{tail}", f"Zaroor — draft yeh raha ↓\n{core}\n{tail}")], "binary_yes_stop",
                                  "Merchant asked for the draft → re-showed the exact artifact.")
            mi = "explicit_action"                     # no draft yet → make it now
        # "No" after "anything to change / anything else?" means no changes — not an opt-out
        if _NO_ONLY.match(message) and state.actions_requested and (state.last_draft or re.search(
                r"changed|change\b|edits?|anything else|aur kuch|badlaav|kuch aur", last_vera, re.I)):
            if state.last_draft and not state.draft_live:
                if re.search(r"changed|change\b|edits?|badlaav", last_vera, re.I):
                    return self._send(state, w, [b.t("Great — no changes then. Reply GO and it goes live today.",
                                                     "Badhiya — koi badlaav nahi. GO reply karein toh aaj hi live kar doon.")],
                                      "binary_yes_stop", "'No' = no changes to the draft → one step from live.")
                return self._send(state, w, [b.t("No problem — I'll hold it, nothing goes live without your OK. Want 3 other ideas instead? Reply YES.",
                                                 "Koi baat nahi — main ise rok ke rakhti hoon, aapke OK ke bina kuch live nahi hoga. Iske bajaye 3 aur ideas chahiye? YES reply karein.")],
                                  "open_ended", "'No' to a pending draft = hold it (not an opt-out) → offer alternatives.")
            name = b.P("name") or "your business"
            state.exit_state = "waiting"
            return self._send(state, w, [b.t(f"All set 🙂 I'll message you when there's something useful for {name}. Ask me for ideas anytime.",
                                             f"Sab set hai 🙂 {name} ke liye kuch useful hoga toh message karungi. Aur ideas chahiye toh kabhi bhi poochiye.")],
                              "none", "'No' = nothing else needed → friendly close, conversation stays open.")
        if mi == "not_interested":
            state.exit_state = "ended"
            state.merchant_sentiment = "negative"
            return {"action": "end", "rationale": "Merchant opted out / not interested — closing politely, no further nudges."}
        if mi == "hostile":
            state.hostile_count += 1
            state.merchant_sentiment = "negative"
            if state.hostile_count >= 2:
                state.exit_state = "ended"
                return {"action": "end", "rationale": "Repeated hostility — exiting gracefully."}
            off = bool(re.search(r"gst|tax|loan|itr", message.lower()))
            return self._send(state, w, self._hostile(w, off), "open_ended",
                              "Hostile reply: apologise once, restate the single useful thing, offer STOP.")
        if mi == "off_topic":
            state.off_topic_count += 1
            if state.off_topic_count >= 2:
                state.exit_state = "waiting"
                return {"action": "wait", "wait_seconds": 86400, "rationale": "Second off-topic ask — backing off for a day."}
            return self._send(state, w, self._off_topic(w, message), "binary_yes_stop",
                              "Off-topic request: decline politely, stay on mission, re-offer the one action.")
        if mi == "later":
            state.later_count += 1
            state.exit_state = "waiting"
            secs = IntentRouter.wait_seconds(message) * (2 if state.later_count > 1 else 1)
            return {"action": "wait", "wait_seconds": secs, "rationale": f"Merchant asked for time — backing off {secs // 60} min."}
        if mi in CURVEBALLS:
            body_fn, cta, why = CURVEBALLS[mi]
            if mi == "callback":
                state.actions_requested.append("callback")
            return self._send(state, w, getattr(self, body_fn)(state, w, message), cta, why,
                              allow=("self_intro",) if mi == "identity" else ())
        if mi == "explicit_action" and state.pending_deal:
            deal, state.pending_deal = state.pending_deal, None
            state.actions_requested.append("deal")
            b = w["brief"]
            name = b.P("name") or ""
            return self._send(state, w, [b.t(f"Done ✅ “{deal}” is set up as a new magicpin deal for {name}. It goes live as soon as you reply GO — and I'll feature it in your next Google post too.",
                                             f"Ho gaya ✅ “{deal}” {name} ke liye naye magicpin deal ke roop mein set hai. GO reply karte hi live ho jaayega — aur agle Google post mein bhi feature kar dungi.")],
                              "binary_yes_stop", "Merchant accepted the proposed magicpin deal → set it up immediately.")
        if mi == "explicit_action" and state.actions_requested and state.actions_requested[-1] in ("photo", "deal") and re.search(r"\bgo\b", message.lower()):
            b = w["brief"]
            what = "post with your photo" if state.actions_requested[-1] == "photo" else "deal"
            return self._send(state, w, [b.t(f"Scheduled ✅ Your {what} is queued for publishing — you'll see it on your listing shortly. Anything else you want to add this week?",
                                             f"Schedule ho gaya ✅ Aapka {'photo wala post' if what != 'deal' else 'deal'} publish queue mein hai — thodi der mein listing pe dikhega. Is hafte aur kuch add karna hai?")],
                              "open_ended", "Merchant confirmed with GO → publish and offer the next step.")
        if mi == "explicit_action" and state.actions_requested and re.search(
                r"^\s*(go|go ahead|send|send it|publish|post it|bhej do|daal do|live kar do)\b", message.lower()):
            b = w["brief"]
            kind = (w.get("promise") or ("post", None))[0]
            cn = b.cust_noun()
            if kind == "photos" and not state.attachments:
                return self._send(state, w, [b.t("Ready when you are 📸 Attach the photos with the 📎 button and I'll draft the post with them.",
                                                 "Main ready hoon 📸 📎 button se photos bhejiye, main unke saath post draft kar dungi."),
                                             b.t("Still waiting for the photos 📸 — the 📎 button is next to the message box.",
                                                 "Photos ka intezaar hai 📸 — 📎 button message box ke paas hai.")],
                                  "open_ended", "GO before any photo arrived → ask for the photos, don't claim anything is scheduled.")
            done = {
                "review_request": (f"Sent ✅ The review request is going out to your recent happy {cn} today. I'll let you know as new reviews come in.",
                                   f"Bhej diya ✅ Review request aaj aapke recent happy {cn} ko ja rahi hai. Naye reviews aate hi bataungi."),
                "thanks_post": (f"Live ✅ The thank-you post is scheduled and the review request goes out to your recent {cn} today.",
                                f"Live ✅ Thank-you post schedule ho gaya aur review request aaj aapke recent {cn} ko ja rahi hai."),
                "customer_msg": (f"Sent ✅ The WhatsApp is going out to your {cn} today. I'll share replies and bookings here.",
                                 f"Bhej diya ✅ WhatsApp aaj aapke {cn} ko ja raha hai. Replies aur bookings yahin share karungi."),
                "review_replies": ("Posted ✅ The owner replies go up on those reviews today.", "Post ho gaya ✅ Owner replies aaj un reviews pe lag jaayenge."),
                "pin": ("Pinned ✅ The offer is now at the top of your Google profile.", "Pin ho gaya ✅ Offer ab aapke Google profile ke top pe hai."),
                "summary": (f"Done ✅ The forward-ready note is formatted — share it with your {cn} whenever you like.",
                            f"Ho gaya ✅ Forward-ready note format ho gaya — jab chahein {cn} ke saath share kijiye."),
            }.get(kind, ("Scheduled ✅ Your post is queued and goes live on your listing today.",
                         "Schedule ho gaya ✅ Aapka post queue mein hai aur aaj listing pe live ho jaayega."))
            state.actions_requested.append("go")
            state.draft_live = True
            return self._send(state, w, [b.t(done[0] + " Want 3 more ideas for this week? Reply YES.", done[1] + " Is hafte ke liye 3 aur ideas chahiye? YES reply karein."),
                                         b.t("That's already live ✅ Want 3 more ideas for this week? Reply YES.", "Yeh already live hai ✅ Is hafte ke liye 3 aur ideas chahiye? YES reply karein.")],
                              "open_ended", f"Merchant confirmed with GO → '{kind}' executed; offer the next step.")
        if mi == "explicit_action" and re.search(r"3 (more|other|aur) ideas", last_vera):
            return self._send(state, w, self._ideas_menu(state, w), "open_ended", "Merchant said yes to more ideas → 3 numbered next steps.")
        if mi == "explicit_action":
            state.exit_state = None
            state.merchant_sentiment = "positive"
            if state.last_draft and not state.draft_live:
                core = draft_core(state.last_draft)
                return self._send(state, w, [b.t(f"Your draft is ready ↓\n{core}\nReply GO and it goes live today, or send any edits.",
                                                 f"Aapka draft ready hai ↓\n{core}\nGO reply karein toh aaj hi live kar doon, ya edits bhej dijiye."),
                                             b.t(f"Here's the draft once more ↓\n{core}\nJust reply GO to publish it — or tell me one thing to change (the offer, timings or wording).",
                                                 f"Draft ek baar phir ↓\n{core}\nPublish karne ke liye bas GO reply karein — ya ek cheez bataiye jo badalni hai (offer, timings ya wording).")],
                                  "binary_yes_stop", "Go-ahead while a draft is pending → re-show the exact draft, one step from live.")
            if state.draft_live:
                return self._send(state, w, self._ideas_menu(state, w), "open_ended",
                                  "Go-ahead after the last draft went live → offer the next 3 concrete ideas.")
            state.actions_requested.append(w["ta"].family)
            body = self._act(w, False, price=router.is_price_question(message), when=_when(message))
            state.last_draft, state.draft_live = body[0], False
            state.delivered.append(tuple(w.get("promise") or ("post", None)))
            return self._send(state, w, body, "open_ended",
                              "Explicit go-ahead → ACT immediately (no re-qualification): delivered the artifact + next step.")
        if _ACK.match(message) and mi not in ("explicit_action",):
            b = w["brief"]
            if state.draft_live:
                return self._send(state, w, [b.t("Glad it helps 🙂 Want 3 more ideas for this week? Reply YES.",
                                                 "Khushi hui 🙂 Is hafte ke liye 3 aur ideas chahiye? YES reply karein.")],
                                  "open_ended", "Acknowledgement after going live → offer next ideas.")
            if state.actions_requested:
                return self._send(state, w, [b.t("Glad it helps 🙂 Reply GO whenever you want it live — or send any edits.",
                                                 "Khushi hui 🙂 Jab chahein GO reply kar dijiye — ya edits bhej dijiye.")],
                                  "binary_yes_stop", "Acknowledgement after delivery → confirm the one pending step, no re-pitch.")
            a_en, a_hi = self._action_phrase(w)
            return self._send(state, w, [b.t(f"Happy to help 🙂 Whenever you're ready, I can {a_en} — just reply YES.",
                                             f"Khushi hui 🙂 Jab ready hon, main {a_hi.rstrip('?')} — bas YES reply kijiye.")],
                              "binary_yes_stop", "Acknowledgement → warm, one low-pressure CTA.")
        if mi == "question":
            if router.is_price_question(message):
                return self._send(state, w, self._price(w), "binary_yes_stop", "Price question answered only from context; no invented pricing.")
            return self._send(state, w, self._answer(state, w, message), "binary_yes_stop",
                              "Question answered with a not-yet-used context fact, then one CTA.")
        if mi == "objection":
            state.objection_count += 1
            if state.objection_count >= 2:
                state.exit_state = "ended"
                return {"action": "end", "rationale": "Second objection — respecting it and exiting."}
            return self._send(state, w, self._objection(state, w), "binary_yes_stop", "Objection handled once with a fact + smallest next step.")
        if mi == "interested":
            return self._send(state, w, self._interested(state, w), "binary_yes_stop", "Interest → concrete proposal with one binary CTA.")
        state.unclear_count += 1
        if state.unclear_count >= 2:
            state.exit_state = "waiting"
            return {"action": "wait", "wait_seconds": 3600, "rationale": "Still unclear after one clarification — waiting."}
        return self._send(state, w, self._clarify(w, bool(state.actions_requested)), "binary_yes_stop", "Unclear reply → one binary clarification.")

    # ------------------------------------------------------------- plumbing
    def _working(self, state: ConversationState, latest: Optional[str]) -> dict:
        trig = state.trigger or {"id": "trg_conversation", "kind": "dormant_with_vera", "scope": "merchant", "payload": {}}
        merchant = state.merchant or {"merchant_id": state.merchant_id or "m_unknown", "identity": {}}
        category = state.category or {"slug": merchant.get("category_slug", "general")}
        w = self.orch.analyze(category, merchant, trig, state.customer, latest_reply=latest)
        if state.language:
            w["lang"] = LanguagePlan(language=state.language, tone=w["lang"].tone, style_rules=w["lang"].style_rules)
        w["brief"] = Brief(w["ta"], w["pz"], w["prof"], w["lang"], w["cust"], w["tools"])
        w["promise"] = promised(state)
        return w

    def _send(self, state: ConversationState, w: dict, bodies, cta: str, rationale: str, allow: tuple = ()) -> dict:
        options = bodies if isinstance(bodies, list) else [bodies]
        prev = state.bot_bodies()
        # echoing the other side's own words/times ("tomorrow 10am", "Instagram") is not fabrication
        theirs = " ".join(state.their_msgs()[-1:])
        w["ledger"].allow_words(re.findall(r"[A-Za-z]+", theirs))
        from .ledger import extract_numbers
        their_nums = set(extract_numbers(theirs))
        chosen = None
        for body in options:
            body = re.sub(r"[ \t]+", " ", body).strip()
            if body in prev:
                continue
            d = Draft(segments=[("reply", body)], cta=cta, plan=StrategyPlan("reply", "", "", [], [], cta, "", []),
                      allowed_extra_numbers=their_nums)
            f = FactChecker([]).run(d, w["ledger"], w["ta"], bool(state.customer))
            p = PolicyChecker([]).run(d, w["ledger"], w["prof"], w["ta"], w["cust"],
                                      "merchant_on_behalf" if state.customer else "vera", w["lang"].language, prev, max_len=900)
            blocking = [i for i in f.errors + p.errors if i.code not in ("buried_cta",) + tuple(allow)]
            if not blocking:
                chosen = body
                break
        if chosen is None:
            b = w["brief"]
            if state.customer:
                first = (w["cust"].first_name if w["cust"] else "") or ""
                chosen = b.t(f"Thanks {first}! We'll get back to you on this chat shortly.",
                             f"Thanks {first}! Hum isi chat pe jaldi reply karte hain.")
            else:
                chosen = b.t(f"Noted, {b.sal()}. I'll keep things moving on my side and update you here.",
                             f"Noted, {b.sal()}. Main apni taraf se kaam aage badhati hoon aur yahin update deti hoon.")
            if chosen in prev:
                chosen += " 🙂"
            rationale += " (validated fallback phrasing)"
        state.record_bot(chosen, cta)
        state.previous_strategy = rationale.split(":")[0]
        return {"action": "send", "body": chosen, "cta": cta, "rationale": rationale}

    # ------------------------------------------------------------ responses
    @staticmethod
    def _hook_line(w: dict) -> str:
        """The trigger's why-now line (hook + first anchor when the hook alone is thin), minus salutation."""
        from .agents.composer import REALIZERS, r_generic
        b = Brief(w["ta"], w["pz"], w["prof"], w["lang"], w["cust"], w["tools"])
        parts = REALIZERS.get(w["ta"].family, r_generic)(b)
        hook = parts.hook
        sal = b.sal()
        if sal and hook.startswith(sal):
            hook = hook[len(sal):].lstrip(", ").strip()
        extra = (parts.anchor[:1] or [v for v in parts.levers.values() if re.search(r"\d", v)][:1])
        if extra and (hook.endswith(":") or len(hook) < 40):
            hook = f"{hook.rstrip(':.')}: {extra[0]}" if hook.endswith(":") else f"{hook} {extra[0]}"
        if hook and len(hook) > 1 and hook[1].islower():
            hook = hook[0].lower() + hook[1:]
        return hook

    def _action_phrase(self, w: dict) -> tuple[str, str]:
        fam = w["ta"].family
        b = w["brief"]
        kind = (w.get("promise") or ("", None))[0]
        cn = b.cust_noun()
        by_promise = {
            "review_request": (f"send the review-request WhatsApp to your recent happy {cn}", f"recent happy {cn} ko review-request WhatsApp bhej doon"),
            "customer_msg": (f"draft the WhatsApp for your {cn}", f"{cn} ke liye WhatsApp draft kar doon"),
            "thanks_post": ("put up the thank-you post and invite reviews", "thank-you post daal doon aur reviews ke liye invite kar doon"),
            "pin": ("pin the offer on your Google profile", "offer ko Google profile pe pin kar doon"),
            "verification": ("start the verification request", "verification request start kar doon"),
            "reminder": ("save the details and remind you a day before", "details save karke ek din pehle reminder bhej doon"),
            "review_replies": ("draft the review replies", "review replies draft kar doon"),
            "checklist": ("put together the checklist", "checklist bana doon"),
            "summary": ("send you the 2-min summary + a forward-ready draft", "2-min summary + forward-ready draft bhej doon"),
            "renewal": ("process the renewal", "renewal process kar doon"),
        }
        if kind in by_promise:
            return by_promise[kind]
        return {
            "knowledge": ("send you the 2-min summary + a forward-ready draft", "2-min summary + forward-ready draft bhej doon"),
            "regulation": ("put together the compliance checklist", "compliance checklist bana doon"),
            "reputation": ("draft the review replies", "review replies draft kar doon"),
            "account": ("process the renewal", "renewal process kar doon"),
            "profile": ("fill in the missing profile details", "profile ki missing details fill kar doon"),
        }.get(fam, ("draft the Google post for your OK", "aapke OK ke liye Google post draft kar doon"))

    def _clarify_auto(self, w: dict) -> list[str]:
        b = w["brief"]
        hook = self._hook_line(w)
        a_en, a_hi = self._action_phrase(w)
        sal = "" if b.sal() == "Hi" else f" {b.sal()},"
        return [b.t(f"Looks like an auto-reply 🙂{sal} when you get a moment: {hook} Want me to {a_en}? Reply YES.",
                    f"Lagta hai yeh auto-reply hai 🙂{sal} jab time mile: {hook} Main {a_hi}? YES reply karein.")]

    def _hostile(self, w: dict, off_topic: bool) -> list[str]:
        b = w["brief"]
        name = b.P("name") or "your business"
        extra_en = " GST/tax filing is outside what I can help with." if off_topic else ""
        extra_hi = " GST/tax filing mein main madad nahi kar sakti." if off_topic else ""
        return [b.t(f"Sorry for the bother, {b.sal()}.{extra_en} I only message about things that help {name} get more customers — "
                    f"reply STOP and I won't message again.",
                    f"Pareshani ke liye sorry, {b.sal()}.{extra_hi} Main sirf {name} ke customers badhane wali cheezon ke liye message karti hoon — "
                    f"STOP reply karein toh aage message nahi aayega.")]

    def _off_topic(self, w: dict, message: str) -> list[str]:
        b = w["brief"]
        name = b.P("name") or "your business"
        m = re.search(r"\b(gst|income tax|itr|tax|loan|visa|passport|insurance|cricket|stock|crypto|aadhaar|pan card)\b", message.lower())
        topic = (m.group(1).upper() if m and len(m.group(1)) <= 3 else (m.group(1) if m else "that"))
        a_en, a_hi = self._action_phrase(w)
        return [b.t(f"{topic} is outside what I can help with, {b.sal()} — I handle {name}'s Google listing, posts, offers and customer messages. "
                    f"Meanwhile, want me to {a_en}? Reply YES.",
                    f"{topic} mein main madad nahi kar paungi, {b.sal()} — main {name} ki Google listing, posts, offers aur customer messages sambhalti hoon. "
                    f"Tab tak, main {a_hi}? Reply YES.")]

    def _post_text(self, w: dict, promised_offer: Optional[str] = None) -> tuple[str, str]:
        b = w["brief"]
        name, loc = b.P("name") or "", b.P("locality")
        where = f"{name}, {loc}" if loc else name
        offer, kind = (promised_offer, "active") if promised_offer else b.offer()
        fam = w["ta"].family
        lead = ""
        if fam == "festival" and b.A("name"):
            lead = f"This {b.A('name')}, "
        elif fam == "seasonal":
            items = [re.sub(r"\s*[+−-]?\d+%.*$", "", b.A(k) or "").strip() for k in ("up1", "up2", "up3")]
            items = [x for x in items if x]
            if items:
                season = (b.Araw("season") or "")
                label = "Summer essentials" if "summer" in str(season).lower() else "Seasonal essentials"
                lead = f"{label} in stock: {', '.join(items[:-1]) + ' & ' + items[-1] if len(items) > 1 else items[0]}. "
        elif fam == "local_event" and b.A("headline") and "vs" in (b.A("headline") or ""):
            lead = f"{b.A('headline')} tonight? "
        elif fam == "weather" and b.A("condition"):
            lead = f"{str(b.A('condition')).capitalize()} today? "
        elif fam == "milestone":
            lead = "Thank you to every customer who made this possible! "
        praise = b.P("praise")
        if praise and fam not in ("reputation",):
            lead += f"Loved for {praise}. "
        rating = f" Rated {b.P('rating')} by {b.P('reviews')} customers." if b.P("rating") and b.P("reviews") else ""
        if offer and kind in ("active", "inactive"):
            post = f"{lead}{offer} at {where}.{rating} Message us on WhatsApp to book."
        elif fam == "trend" and b.A("query"):
            post = f"Searching for {b.A('query')}? Talk to the team at {where}.{rating} Message us on WhatsApp."
        else:
            post = f"{lead}Visit {where}.{rating} Message us on WhatsApp to book."
        return post[0].upper() + post[1:], post[0].upper() + post[1:]

    def _act(self, w: dict, already: bool, price: bool = False, when: Optional[str] = None) -> list[str]:
        b, ta = w["brief"], w["ta"]
        fam = ta.family
        cn = b.cust_noun()
        name = b.P("name") or ""
        if already:
            return [b.t("On it ✅ Everything's queued — you'll get the preview here for a final OK. Anything you'd like changed?",
                        "Kaam chalu hai ✅ Sab queue mein hai — final OK ke liye preview yahin aayega. Kuch badlaav chahiye?"),
                    b.t("Done on my side ✅ Preview is next — just reply with edits if you want any.",
                        "Meri taraf se ho gaya ✅ Agla step preview hai — edits ho toh reply kar dijiye.")]
        kind, promised_offer = w.get("promise") or ("post", None)
        done = self._deliver_promise(w, kind, promised_offer, when)
        if done:
            return done
        if fam == "knowledge":
            title, src, summ = b.A("title"), b.A("source"), b.A("summary")
            return [b.t(f"Sending it now 📄 Summary: {summ or title}" + (f" — {src}." if src else ".") +
                        f"\nDraft for your {cn} ↓\nQuick update from {name}: {title}. Ask us about it at your next visit.\nReply GO and I'll format it for forwarding.",
                        f"Abhi bhej rahi hoon 📄 Summary: {summ or title}" + (f" — {src}." if src else ".") +
                        f"\n{cn.capitalize()} ke liye draft ↓\nQuick update from {name}: {title}. Ask us about it at your next visit.\nGO reply karein, main forward-ready format kar dungi.")]
        if fam == "regulation":
            title, src, eff, summ = b.A("title"), b.A("source"), b.A("effective"), b.A("summary")
            lines = [f"• What changed: {title}" + (f" ({src})" if src else "")]
            if summ:
                lines.append(f"• What to do: {summ}")
            if eff:
                lines.append(f"• Deadline: {eff}")
            lines.append(f"• Owner: assign one person at {name} to keep the register")
            return [b.t("Here's your checklist ✅\n" + "\n".join(lines) + "\nWant a reminder a week before the deadline? Reply YES.",
                        "Yeh raha aapka checklist ✅\n" + "\n".join(lines) + "\nDeadline se ek hafte pehle reminder chahiye? Reply YES.")]
        if fam == "reputation":
            theme, cnt = b.A("theme"), b.A("count")
            reply = f"Thank you for the feedback about {theme}. We've shared it with the team and are working on it — hope to see you again soon. — {name}"
            return [b.t(f"Drafted ✅ Owner reply for the {theme} reviews ↓\n{reply}\nReply GO and I'll post it on " + (f"all {cnt}." if cnt else "each one."),
                        f"Draft ready ✅ {theme} wale reviews ke liye owner reply ↓\n{reply}\nGO reply karein, main " + (f"sabhi {cnt} pe post kar dungi." if cnt else "har review pe post kar dungi."))]
        if fam == "account":
            plan = b.A("plan") or "current"
            return [b.t(f"Done ✅ I've raised the renewal for your {plan} plan — the magicpin team will confirm it on this chat. Your posts and offers keep running meanwhile.",
                        f"Ho gaya ✅ Aapke {plan} plan ka renewal raise kar diya hai — magicpin team isi chat pe confirm karegi. Tab tak posts aur offers chalte rahenge.")]
        if fam == "profile":
            miss = b.A("missing") or "the missing details"
            return [b.t(f"On it ✅ I'll fill in {miss} from your existing details and share a preview here. Send your opening hours in one line and I'll add them too.",
                        f"Kaam shuru ✅ {miss} aapki existing details se fill karke preview yahin bhejti hoon. Opening hours ek line mein bhej dijiye, woh bhi add kar dungi.")]
        post_en, post_hi = self._post_text(w, promised_offer)
        extra_en = " Pricing for the post itself isn't in my records here, so I'll have the magicpin team confirm — no guesses." if price else ""
        extra_hi = " Post ki pricing mere records mein nahi hai, magicpin team confirm karegi — main guess nahi karungi." if price else ""
        live_en = f"I'll schedule it for {when} — reply GO to confirm, or send any edits." if when else "Reply GO and it goes live today, or send any edits."
        live_hi = f"Main ise {when} ke liye schedule kar dungi — confirm karne ke liye GO reply karein, ya edits bhej dijiye." if when else "GO reply karein toh aaj hi live kar doon, ya edits bhej dijiye."
        return [b.t(f"Done ✅ Here's the draft post for {name} ↓\n{post_en}\n{live_en}{extra_en}",
                    f"Ho gaya ✅ {name} ke liye draft post ↓\n{post_hi}\n{live_hi}{extra_hi}")]

    def _ideas_menu(self, state: ConversationState, w: dict) -> list[str]:
        """3 concrete next steps, each backed by this merchant's own data; replying 1/2/3 delivers the draft."""
        b = w["brief"]
        cn = b.cust_noun()
        name = b.P("name") or "your business"
        done_kinds = set(state.actions_requested)
        o, okind = b.offer()
        cands = []
        if o:
            what = "your" if okind == "active" else "a"
            q = o if okind == "active" else f"'{o}'"
            views = f" — your listing got {b.P('views')} views in {b.P('window')}" if b.P("views") and b.P("window") else ""
            views_hi = f" — pichhle {b.P('window').replace('days', 'din')} mein {b.P('views')} views" if b.P("views") and b.P("window") else ""
            cands.append({"kind": "post", "offer": o, "en": f"A fresh Google post featuring {what} {q} offer{views}",
                          "hi": f"{q} offer ke saath ek fresh Google post{views_hi}"})
            cands.append({"kind": "pin", "offer": o, "en": f"Pin {q} to the top of your Google profile so every visitor sees it first",
                          "hi": f"{q} ko Google profile ke top pe pin karna, taaki har visitor pehle wahi dekhe"})
        if b.P("complaint"):
            cands.append({"kind": "review_replies", "en": f"Polite owner replies to the reviews mentioning {b.P('complaint')}",
                          "hi": f"{b.P('complaint')} wale reviews pe polite owner replies"})
        rv = f" — you're at {b.P('reviews')} reviews" if b.P("reviews") else ""
        cands.append({"kind": "review_request", "en": f"A review-request WhatsApp to your recent happy {cn}{rv}",
                      "hi": f"Recent happy {cn} ko review-request WhatsApp" + (f" — abhi {b.P('reviews')} reviews hain" if b.P("reviews") else "")})
        lp = b.P("lapsed")
        cands.append({"kind": "customer_msg", "offer": o,
                      "en": f"A comeback WhatsApp to your {lp + ' ' if lp else ''}lapsed {cn}" + (f" with {o}" if o else ""),
                      "hi": f"Aapke {lp + ' ' if lp else ''}lapsed {cn} ko comeback WhatsApp" + (f" {o} ke saath" if o else "")})
        thing = {"restaurants": "dishes", "salons": "work (before/after)", "gyms": "sessions", "dentists": "clinic", "pharmacies": "store"}.get(b.prof.slug, "business")
        thing_hi = {"restaurants": "dishes", "salons": "kaam (before/after)", "gyms": "sessions", "dentists": "clinic", "pharmacies": "store"}.get(b.prof.slug, "business")
        cands.append({"kind": "photos", "en": f"Fresh photos of your {thing} — send 3-5 here and I'll turn them into listing photos + a post",
                      "hi": f"Aapke {thing_hi} ki fresh photos — 3-5 yahan bhejiye, main listing photos + post bana dungi"})
        done = {(k, o) for k, o in state.delivered} | {(k, None) for k, _ in state.delivered}
        fresh = [c for c in cands if (c["kind"], c.get("offer")) not in done and (c["kind"], None) not in done and c["kind"] not in done_kinds]
        ideas = (fresh or [c for c in cands if (c["kind"], c.get("offer")) not in done] or cands)[:3]
        state.ideas = ideas
        lines_en = "\n".join(f"{i + 1}. {c['en']}" for i, c in enumerate(ideas))
        lines_hi = "\n".join(f"{i + 1}. {c['hi']}" for i, c in enumerate(ideas))
        n = len(ideas)
        nums_en = " or ".join(str(i + 1) for i in range(n)) if n <= 2 else "1, 2 or 3"
        nums_hi = " ya ".join(str(i + 1) for i in range(n)) if n <= 2 else "1, 2 ya 3"
        return [b.t(f"Here are {n} ideas for {name} this week:\n{lines_en}\nReply {nums_en} and I'll have it ready in a minute.",
                    f"{name} ke liye is hafte {n} ideas:\n{lines_hi}\n{nums_hi} reply karein — ek minute mein ready."),
                b.t(f"A few more things I can do for {name}:\n{lines_en}\nReply {nums_en}.", f"{name} ke liye aur kya kar sakti hoon:\n{lines_hi}\n{nums_hi} reply karein.")]

    def _deliver_promise(self, w: dict, kind: str, offer: Optional[str], when: Optional[str]) -> Optional[list[str]]:
        """Deliver exactly what the opening message offered (review request, customer WhatsApp, pin, ...)."""
        b = w["brief"]
        cn = b.cust_noun()
        name = b.P("name") or ""
        go_en = f"Reply GO and I'll send it{' ' + when if when else ''}, or send any edits."
        go_hi = f"GO reply karein, main{' ' + when if when else ''} bhej dungi — ya edits bhej dijiye."
        if kind in ("review_request", "thanks_post"):
            msg = f"Hi! Thank you for choosing {name} 🙏 If you enjoyed your visit, a quick Google review would mean a lot to us — it takes 30 seconds. See you again soon!"
            post = f"Thank-you post + review request ready ↓\nPost: Thank you to every customer who made this possible 🙏 — team {name}\n"
            pre_en = "Done ✅ " + (post if kind == "thanks_post" else "")
            pre_hi = "Ho gaya ✅ " + (post if kind == "thanks_post" else "")
            return [b.t(f"{pre_en}Review-request WhatsApp for your recent happy {cn} ↓\n{msg}\n(Your Google review link is added automatically.) {go_en}",
                        f"{pre_hi}Recent happy {cn} ke liye review-request WhatsApp ↓\n{msg}\n(Google review link apne aap jud jaayega.) {go_hi}")]
        if kind == "customer_msg":
            off = offer or b.offer()[0]
            tail = f" {off} is available this month." if off else ""
            msg = f"Hi! It's been a while — we'd love to see you at {name} this week.{tail} Reply here to book a slot. — {name}"
            return [b.t(f"Draft ready ✅ WhatsApp for your {cn} ↓\n{msg}\n{go_en}", f"Draft ready ✅ {cn} ke liye WhatsApp ↓\n{msg}\n{go_hi}")]
        if kind == "pin":
            off = offer or b.offer()[0] or "your top offer"
            return [b.t(f"Done ✅ “{off}” is set to be pinned at the top of your Google profile while traffic is high. Reply GO to confirm, or tell me a different offer.",
                        f"Ho gaya ✅ “{off}” ko Google profile ke top pe pin karne ke liye set kar diya hai. Confirm karne ke liye GO reply karein, ya koi aur offer bataiye.")]
        if kind == "photos":
            return [b.t(f"Great 📸 Attach 3-5 photos here with the 📎 button (one at a time is fine). I'll crop them for your Google listing and draft a post with the best one — nothing goes live without your OK.",
                        f"Badhiya 📸 📎 button se 3-5 photos yahan bhejiye (ek-ek karke bhi chalega). Main unhe Google listing ke liye crop karke best photo ke saath post draft kar dungi — aapke OK ke bina kuch live nahi hoga.")]
        if kind == "review_replies" and w["ta"].family != "reputation":
            theme = b.P("complaint") or "your recent feedback"
            reply = f"Thank you for the feedback about {theme}. We've shared it with the team and are working on it — hope to see you again soon. — {name}"
            return [b.t(f"Drafted ✅ Owner reply for the {theme} reviews ↓\n{reply}\n{go_en}", f"Draft ready ✅ {theme} wale reviews ke liye owner reply ↓\n{reply}\n{go_hi}")]
        if kind == "verification":
            return [b.t(f"Started ✅ I've raised the Google verification request for {name}. Google usually confirms by postcard or a phone call — reply here when you get the code and I'll finish it.",
                        f"Shuru kar diya ✅ {name} ke liye Google verification request raise kar di hai. Google postcard ya phone call se confirm karta hai — code aate hi yahan reply kijiye, baaki main kar dungi.")]
        if kind == "reminder":
            return [b.t("Saved ✅ I'll send you the details and a reminder the day before. Anything else for this week?",
                        "Save kar liya ✅ Details aur ek din pehle reminder bhej dungi. Is hafte aur kuch?")]
        return None

    # ------------------------------------------------------- confidentiality
    def _own_numbers(self, w: dict) -> tuple[str, str]:
        b = w["brief"]
        parts_en, parts_hi = [], []
        if b.P("views") and b.P("window"):
            parts_en.append(f"{b.P('views')} views" + (f" and {b.P('calls')} calls" if b.P("calls") else "") + f" in the last {b.P('window')}")
            parts_hi.append(f"pichhle {b.P('window').replace('days', 'din')} mein {b.P('views')} views" + (f" aur {b.P('calls')} calls" if b.P("calls") else ""))
        if b.P("ctr"):
            parts_en.append(f"CTR {b.P('ctr')}" + (f" (anonymised {b.prof.peer_label} average: {b.P('peer_ctr')})" if b.P("peer_ctr") else ""))
            parts_hi.append(f"CTR {b.P('ctr')}" + (f" (anonymised {b.prof.peer_label} average: {b.P('peer_ctr')})" if b.P("peer_ctr") else ""))
        o, kind = b.offer()
        if o and kind == "active":
            parts_en.append(f"active offer: {o}")
            parts_hi.append(f"active offer: {o}")
        return "; ".join(parts_en), "; ".join(parts_hi)

    def _confidential(self, state: ConversationState, w: dict, kind: str) -> list[str]:
        """Merchant asked for something confidential — share only what's theirs, never other people's data."""
        b = w["brief"]
        name = b.P("name") or "your business"
        cn = b.cust_noun()
        a_en, a_hi = self._action_phrase(w)
        own_en, own_hi = self._own_numbers(w)
        if kind == "customer_pii":
            return [b.t(f"I can't share {cn}' phone numbers, addresses or names, {b.sal()} — that stays private, even from me in chat. "
                        f"What I can do: send your message to them through magicpin (only to those who opted in), so you reach them without seeing their details. "
                        f"Want me to draft that WhatsApp? Reply YES.",
                        f"{b.sal()}, {cn} ke phone number, address ya naam main share nahi kar sakti — woh private rehte hain. "
                        f"Main kya kar sakti hoon: aapka message magicpin ke through unhe bhej sakti hoon (sirf opted-in {cn} ko), bina unki details dikhaye. "
                        f"WhatsApp draft kar doon? Reply YES.")]
        if kind == "other_business":
            comp = ""
            if w["ta"].family == "competitor" and b.A("name"):
                bits = [x for x in (b.A("distance") and f"{b.A('distance')} away", b.A("offer") and f"launch offer {b.A('offer')}") if x]
                comp = f"{b.A('name')}" + (f" ({', '.join(bits)})" if bits else "")
            pub_en = f" What's public: {comp}." if comp else ""
            pub_hi = f" Jo public hai: {comp}." if comp else ""
            peer_en = f" For comparison I only use anonymised averages — your CTR is {b.P('ctr')} vs {b.P('peer_ctr')} for {b.prof.peer_label}." if b.P("ctr") and b.P("peer_ctr") else ""
            peer_hi = f" Comparison ke liye main sirf anonymised average use karti hoon — aapka CTR {b.P('ctr')} hai vs {b.prof.peer_label} ka {b.P('peer_ctr')}." if b.P("ctr") and b.P("peer_ctr") else ""
            return [b.t(f"That's another business's private data, {b.sal()}, so I can't share it — just as I'd never share {name}'s numbers with anyone.{pub_en}{peer_en} "
                        f"Want me to {a_en}? Reply YES.",
                        f"{b.sal()}, yeh doosre business ka private data hai, isliye share nahi kar sakti — jaise {name} ke numbers bhi kisi ko nahi deti.{pub_hi}{peer_hi} "
                        f"Main {a_hi.rstrip('?')}? Reply YES.")]
        if kind == "internal":
            return [b.t(f"That's internal to magicpin, so I can't share it, {b.sal()}. In short: I'm Vera — I look at {name}'s listing data and suggest one useful step at a time, and nothing goes live without your OK. "
                        f"Want me to {a_en}? Reply YES.",
                        f"{b.sal()}, yeh magicpin ka internal hai, isliye share nahi kar sakti. Short mein: main Vera hoon — {name} ke listing data se ek-ek useful step suggest karti hoon, aur aapke OK ke bina kuch live nahi hota. "
                        f"Main {a_hi.rstrip('?')}? Reply YES.")]
        if kind == "share_concern":
            return [b.t(f"No — {name}'s numbers stay private to you and the magicpin team, {b.sal()}. Other merchants never see them; comparisons only use anonymised averages across many businesses. "
                        f"Want me to {a_en}? Reply YES.",
                        f"Nahi — {name} ke numbers sirf aapke aur magicpin team ke paas rehte hain, {b.sal()}. Doosre merchants unhe kabhi nahi dekhte; comparison sirf kai businesses ke anonymised average se hota hai. "
                        f"Main {a_hi.rstrip('?')}? Reply YES.")]
        # own_data — the merchant is entitled to their own numbers
        body_en = f"Here's what I have for {name}: {own_en}." if own_en else f"I only use {name}'s own listing data (views, calls, offers, reviews)."
        body_hi = f"{name} ke liye mere paas yeh hai: {own_hi}." if own_hi else f"Main sirf {name} ka apna listing data use karti hoon (views, calls, offers, reviews)."
        return [b.t(f"{body_en} Only you and the magicpin team see this — never other merchants. Want me to {a_en}?",
                    f"{body_hi} Yeh sirf aap aur magicpin team dekhte hain — doosre merchants nahi. Main {a_hi.rstrip('?')}?")]

    def _confidential_customer(self, w: dict, kind: str) -> list[str]:
        b = w["brief"]
        name = b.P("name") or "the team"
        first = (w["cust"].first_name if w.get("cust") else "") or ""
        return [b.t(f"Sorry {first}, we can't share other customers' or staff members' personal details. For anything about your own visit, just reply here and {name} will help.".replace("Sorry , ", "Sorry, "),
                    f"Sorry {first}, doosre customers ya staff ki personal details hum share nahi kar sakte. Apni visit ke baare mein kuch bhi ho toh yahin reply kijiye, {name} madad karega.".replace("Sorry , ", "Sorry, "))]

    # ------------------------------------------------------- photos & deals
    def respond_photo(self, state: ConversationState, image_ref: str, caption: str = "") -> dict:
        """Merchant shared a photo (e.g. a dish). Vera turns it into listing photos + a post draft."""
        state.attachments.append(image_ref)
        state.messages.append({"from": "customer" if state.customer else "merchant", "body": caption or "[photo]", "ts": time.time()})
        w = self._working(state, caption or None)
        b = w["brief"]
        name, loc = b.P("name") or "", b.P("locality")
        where = f"{name}, {loc}" if loc else name
        cap = (caption or "").strip().rstrip(".")
        offer, kind = b.offer()
        food = b.prof.slug == "restaurants"
        headline_en = cap or ("Fresh from our kitchen" if food else "A look inside")
        headline_hi = cap or ("Fresh from our kitchen" if food else "Ek jhalak andar se")
        line = f"{headline_en} — " + (f"{offer} at {where}." if offer and kind == "active" else f"at {where}.") + " Message us on WhatsApp to order." if food else \
            f"{headline_en} — " + (f"{offer} at {where}." if offer and kind == "active" else f"{where}.") + " Message us on WhatsApp to book."
        peer_photos, _ = b.tools.get_peer_stat("photos")
        noun = b.prof.noun_singular + ("s" if not b.prof.noun_singular.endswith("y") else "").replace("ys", "ies")
        peer_en = f" Similar {b.prof.noun_singular}s average {int(peer_photos)} photos on Google, so every good one counts." if peer_photos else ""
        peer_hi = f" Similar {b.prof.noun_singular}s ke Google pe average {int(peer_photos)} photos hote hain, isliye har acchi photo kaam ki hai." if peer_photos else ""
        state.actions_requested.append("photo")
        if state.customer:
            body = [b.t(f"Thanks for sharing! 🙏 We've got your photo — our team at {name} will get back to you here shortly.",
                        f"Photo ke liye thanks! 🙏 {name} ki team yahin jaldi jawab degi.")]
            return self._send(state, w, body, "none", "Customer shared a photo → acknowledge; merchant team follows up.")
        body = [b.t(f"Got it 📸 {'That looks delicious! ' if food else ''}I'll add it to {name}'s Google photos and build a post around it.{peer_en}\n"
                    f"Draft post ↓\n{line}\nReply GO to publish, or send a different caption.",
                    f"Mil gayi 📸 {'Kya baat hai, bahut tasty lag raha hai! ' if food else ''}Main ise {name} ki Google photos mein add karke iske saath ek post bana deti hoon.{peer_hi}\n"
                    f"Draft post ↓\n{line.replace(headline_en, headline_hi)}\nPublish karne ke liye GO reply karein, ya naya caption bhejiye.")]
        return self._send(state, w, body, "binary_yes_stop",
                          "Photo received → added to listing photos + post draft built on the merchant's real offer (peer photo benchmark cited).")

    def deals(self, state: ConversationState) -> dict:
        """Notify about magicpin deals: what's live / off for this merchant, and one catalog deal worth adding."""
        w = self._working(state, None)
        b = w["brief"]
        tools = w["tools"]
        name = b.P("name") or ""
        live = [t for t, _ in tools.get_active_offers()]
        off = [(t, s) for t, s, _ in tools.get_inactive_offers()]
        have = {t.lower() for t in live} | {t.lower() for t, _ in off}

        def rank(t):
            tl = t.lower()
            return 0 if "@ ₹" in t else 1 if "free" in tl and "%" not in tl else 3 if "%" in tl else 2
        suggest = sorted([t for t, _ in tools.get_catalog_offers() if t.lower() not in have], key=rank)[:2]
        if state.customer:
            first = (w["cust"].first_name if w["cust"] else "") or ""
            if not (w["cust"] and w["cust"].consent_ok and any("promot" in str(s) or "offer" in str(s)
                                                            for s in (tools.get_customer_fact("consent.scope") or []))):
                return {"action": "end", "rationale": f"Not sent: {first or 'this customer'} hasn't consented to promotional messages (deal alerts need promotional consent)."}
            if not live:
                return {"action": "end", "rationale": "Not sent: the merchant has no live deals to share."}
            body = [b.t(f"Hi {first}, {b.merchant_short()} here 🏷 Deals live on magicpin this week: {'; '.join(live[:2])}. Reply YES and we'll hold one for you.",
                        f"Hi {first}, {b.merchant_short()} se 🏷 Is hafte magicpin pe deals: {'; '.join(live[:2])}. YES reply karein, hum aapke liye hold kar lenge.",
                        f"नमस्ते {first}, {b.merchant_short()} से 🏷 इस हफ़्ते magicpin पर deals: {'; '.join(live[:2])}। YES भेजें, हम आपके लिए रख लेंge।".replace("ge।", "गे।"))]
            return self._send(state, w, body, "binary_yes_stop", "Deal alert to a customer with promotional consent, using only the merchant's live deals.")
        lines_en, lines_hi = [], []
        if live:
            lines_en.append("• Live now: " + "; ".join(live[:3]))
            lines_hi.append("• Abhi live: " + "; ".join(live[:3]))
        if off:
            lines_en.append("• Not running: " + "; ".join(f"{t} ({s})" for t, s in off[:2]))
            lines_hi.append("• Band hai: " + "; ".join(f"{t} ({s})" for t, s in off[:2]))
        if suggest:
            lines_en.append("• Popular in your category, not on your listing yet: " + "; ".join(suggest))
            lines_hi.append("• Aapki category mein popular, par aapki listing pe nahi: " + "; ".join(suggest))
            state.pending_deal = suggest[0]
            ask_en = f"Want me to put “{suggest[0]}” live for {name} this week? Reply YES."
            ask_hi = f"Main {name} ke liye is hafte “{suggest[0]}” live kar doon? Reply YES."
        elif off:
            ask_en, ask_hi = f"Want me to switch “{off[0][0]}” back on? Reply YES.", f"“{off[0][0]}” wapas on kar doon? Reply YES."
            state.pending_deal = off[0][0]
        else:
            ask_en, ask_hi = "Want me to feature your live deal in this week's Google post? Reply YES.", "Is hafte ke Google post mein aapka live deal feature kar doon? Reply YES."
        body = [b.t(f"Quick magicpin deals check for {name} 🏷\n" + "\n".join(lines_en) + "\nService + price deals get picked up best by people browsing magicpin. " + ask_en,
                    f"{name} ke magicpin deals ka quick check 🏷\n" + "\n".join(lines_hi) + "\nmagicpin pe service + price wale deals sabse zyada chalte hain. " + ask_hi)]
        return self._send(state, w, body, "binary_yes_stop", "magicpin deals notification: live / not running / one catalog deal to add (service+price first).")

    # ---------------------------------------------------------- curveballs
    def _cb_identity(self, state, w, message):
        b = w["brief"]
        name = b.P("name") or "your business"
        a_en, a_hi = self._action_phrase(w)
        return [b.t(f"I'm Vera, magicpin's assistant for {name} — not Google, but I help you run your Google listing: posts, offers and review replies. "
                    f"Right now I'd suggest this: want me to {a_en}? Reply YES.",
                    f"Main Vera hoon, {name} ke liye magicpin ki assistant — Google nahi, par aapki Google listing sambhalne mein madad karti hoon: posts, offers, review replies. "
                    f"Abhi ke liye: main {a_hi}? Reply YES.")]

    def _cb_trust(self, state, w, message):
        b = w["brief"]
        a_en, a_hi = self._action_phrase(w)
        facts = self._facts_pool(w)[:2]
        f_en = "; ".join(f[1] for f in facts) or self._hook_line(w)
        f_hi = "; ".join(f[2] for f in facts) or self._hook_line(w)
        return [b.t(f"Fair question, {b.sal()}. Everything I mention comes from your own listing data and magicpin's category benchmarks — e.g. {f_en}. "
                    f"I never invent numbers, and nothing goes live without your OK. Want me to {a_en}? Reply YES.",
                    f"Sahi sawaal, {b.sal()}. Main jo bhi batati hoon woh aapki listing ke data aur magicpin ke category benchmarks se hai — jaise {f_hi}. "
                    f"Main koi number invent nahi karti, aur aapke OK ke bina kuch live nahi hota. Main {a_hi}? Reply YES.")]

    def _cb_callback(self, state, w, message):
        b = w["brief"]
        return [b.t(f"Sure, {b.sal()} — I'll ask the magicpin team to call you on this number. Meanwhile I'll keep the draft ready so the call is quick.",
                    f"Zaroor, {b.sal()} — main magicpin team ko isi number pe call karne ko bol deti hoon. Tab tak draft ready rakhti hoon taaki call jaldi ho jaaye.")]

    def _cb_billing(self, state, w, message):
        b = w["brief"]
        a_en, a_hi = self._action_phrase(w)
        return [b.t(f"I can't change commissions or billing myself, {b.sal()} — the magicpin team handles that, and I'll pass your request on. "
                    f"What I can do today: {a_en}. Want that? Reply YES.",
                    f"Commission ya billing main khud nahi badal sakti, {b.sal()} — woh magicpin team dekhti hai, main aapki request aage bhej dungi. "
                    f"Aaj main yeh kar sakti hoon: {a_hi.rstrip('?')}. Chahiye? Reply YES.")]

    def _cb_outcome(self, state, w, message):
        b = w["brief"]
        a_en, a_hi = self._action_phrase(w)
        facts = [f for f in self._facts_pool(w) if f[0] in ("views", "calls", "ctr")][:2]
        f_en = "; ".join(f[1] for f in facts)
        f_hi = "; ".join(f[2] for f in facts)
        return [b.t(f"Honestly, I can't promise a number — anyone who does is guessing. What I can tell you: {f_en or self._hook_line(w)}. "
                    f"A fresh post is the cheapest way to move those. Want me to {a_en}? Reply YES.",
                    f"Sach kahun toh exact number ka waada nahi kar sakti — jo karta hai woh guess karta hai. Jo pakka hai: {f_hi or self._hook_line(w)}. "
                    f"Fresh post in numbers ko badhane ka sabse sasta tareeka hai. Main {a_hi}? Reply YES.")]

    def _cb_glossary(self, state, w, message):
        b = w["brief"]
        low = message.lower()
        a_en, a_hi = self._action_phrase(w)
        if "ctr" in low or "click" in low or "benchmark" in low:
            mine = f" Yours is {b.P('ctr')} vs {b.P('peer_ctr')} for {b.prof.peer_label}." if b.P("ctr") and b.P("peer_ctr") else ""
            mine_hi = f" Aapka {b.P('ctr')} hai, {b.prof.peer_label} ka {b.P('peer_ctr')}." if b.P("ctr") and b.P("peer_ctr") else ""
            return [b.t(f"CTR = out of everyone who sees your Google listing, the % who act on it (call, ask directions, visit your site).{mine} "
                        f"Want me to {a_en}? Reply YES.",
                        f"CTR matlab: jitne log aapki Google listing dekhte hain, unmein se kitne % action lete hain (call, directions, website).{mine_hi} "
                        f"Main {a_hi}? Reply YES.")]
        if "gbp" in low or "google" in low or "verification" in low:
            return [b.t(f"GBP is your Google Business Profile — the listing people see on Google Search and Maps. Verified, complete profiles get more calls. Want me to {a_en}? Reply YES.",
                        f"GBP matlab aapka Google Business Profile — Google Search aur Maps pe dikhne wali listing. Verified aur complete profile ko zyada calls milte hain. Main {a_hi}? Reply YES.")]
        return [b.t(f"Good question — in short: {self._hook_line(w)} Want me to {a_en}? Reply YES.",
                    f"Short mein: {self._hook_line(w)} Main {a_hi}? Reply YES.")]

    def _cb_delegate(self, state, w, message):
        b = w["brief"]
        return [b.t(f"No problem, {b.sal()}. I'll keep the draft here so you can forward it, or share their WhatsApp number and I'll send it to them directly. Shall I draft it now? Reply YES.",
                    f"Koi baat nahi, {b.sal()}. Main draft yahin rakh deti hoon aap forward kar dijiye, ya unka WhatsApp number bhej dijiye, main seedha unhe bhej dungi. Abhi draft kar doon? Reply YES.")]

    def _cb_already(self, state, w, message):
        b = w["brief"]
        views = f" ({b.P('views')} views on your listing in the last {b.P('window')})" if b.P("views") and b.P("window") else ""
        views_hi = f" (pichhle {(b.P('window') or '').replace('days', 'din')} mein listing pe {b.P('views')} views)" if b.P("views") and b.P("window") else ""
        return [b.t(f"Nice — that helps. Google is a separate place people look{views}, so want me to reuse the same idea as a Google post? Reply YES.",
                    f"Badhiya — isse madad milti hai. Google alag jagah hai jahan log dhoondhte hain{views_hi}, toh wahi idea Google post ke roop mein bhi daal doon? Reply YES.")]

    def _price(self, w: dict) -> list[str]:
        b = w["brief"]
        offer, kind = b.offer()
        a_en, a_hi = self._action_phrase(w)
        off_en = f" Your live offer on the listing is {offer}." if offer and kind == "active" else ""
        off_hi = f" Listing pe aapka live offer {offer} hai." if offer and kind == "active" else ""
        plan = b.tools.get_merchant_fact("subscription.plan")
        plan_en = f" You're on the {plan} plan;" if plan else ""
        plan_hi = f" Aap {plan} plan pe hain;" if plan else ""
        return [b.t(f"Straight answer: I don't have a price for this in your account data, so I won't guess.{plan_en} "
                    f"the magicpin team can confirm exact charges.{off_en} Meanwhile, want me to {a_en}? Reply YES.",
                    f"Seedhi baat: iska price aapke account data mein nahi hai, isliye main guess nahi karungi.{plan_hi} "
                    f"exact charges magicpin team confirm kar degi.{off_hi} Tab tak, main {a_hi}? Reply YES.")]

    def _facts_pool(self, w: dict) -> list[tuple[str, str, str]]:
        b = w["brief"]
        pool = []
        if b.P("ctr") and b.P("peer_ctr"):
            pool.append(("ctr", f"your CTR is {b.P('ctr')} vs {b.P('peer_ctr')} for {b.prof.peer_label}",
                         f"aapka CTR {b.P('ctr')} hai vs {b.prof.peer_label} ka {b.P('peer_ctr')}"))
        if b.P("views") and b.P("window"):
            pool.append(("views", f"your listing got {b.P('views')} views in the last {b.P('window')}",
                         f"pichhle {b.P('window').replace('days', 'din')} mein aapki listing ko {b.P('views')} views mile"))
        if b.P("calls") and b.P("window"):
            pool.append(("calls", f"those turned into {b.P('calls')} calls", f"unmein se {b.P('calls')} calls aaye"))
        if b.P("rating") and b.P("reviews"):
            pool.append(("rating", f"you're at {b.P('rating')} across {b.P('reviews')} reviews",
                         f"aapki rating {b.P('rating')} hai, {b.P('reviews')} reviews ke saath"))
        if b.A("summary"):
            pool.append(("summary", f"the key point is: {b.A('summary')}", f"main baat: {b.A('summary')}"))
        return pool

    def _answer(self, state: ConversationState, w: dict, message: str) -> list[str]:
        b = w["brief"]
        a_en, a_hi = self._action_phrase(w)
        how = bool(re.search(r"\bhow\b|kaise|process|work", message.lower()))
        what = bool(re.search(r"\b(what|about|kya hai|kya baat|matlab|explain|samjha)", message.lower()))
        out = []
        low = message.lower()
        entity = []
        a = w["ta"].anchor
        if re.search(r"\b(which|who|kaun|kaunsa|name)\b", low) and w["ta"].family == "competitor" and "name" in a:
            bits = [b.A("name"), (b.A("distance") and b.t(f"{b.A('distance')} from you", f"aapse {b.A('distance')} door")),
                    (b.A("date") and b.t(f"opened {b.A('date')}", f"{b.A('date')} ko khula")),
                    (b.A("offer") and b.t(f"launch offer {b.A('offer')}", f"launch offer {b.A('offer')}"))]
            out.append(b.t(f"It's {', '.join(x for x in bits if x)}. Want me to {a_en}? Reply YES.",
                           f"Woh {', '.join(x for x in bits if x)} hai. Main {a_hi}? Reply YES."))
        elif re.search(r"\b(which|who|kaun|kaunsa|name)\b", low):
            entity = [k for k in ("name", "headline", "title", "query", "theme", "offer", "molecule") if k in a]
        elif re.search(r"\b(when|kab|date|deadline)\b", low):
            entity = [k for k in ("date", "effective", "event_date", "days_until", "due", "when") if k in a]
        elif re.search(r"\b(where|kahan|how far|distance|location)\b", low):
            entity = [k for k in ("distance", "location", "city") if k in a] + (["locality"] if b.P("locality") else [])
        if entity:
            parts = []
            for k in entity[:3]:
                txt = b.A(k) if k in a else b.P(k)
                if txt:
                    parts.append(txt)
            if parts:
                out.append(b.t(f"{' · '.join(parts)} — that's what I have from the context. Want me to {a_en}? Reply YES.",
                               f"{' · '.join(parts)} — yeh jaankari mere paas hai. Main {a_hi}? Reply YES."))
        if what and "hook" not in state.facts_mentioned:
            state.facts_mentioned.add("hook")
            out.append(b.t(f"In short: {self._hook_line(w)} Want me to {a_en}? Reply YES.",
                           f"Short mein: {self._hook_line(w)} Main {a_hi}? Reply YES."))
        if how:
            out.append(b.t(f"Simple: I draft it, you approve, it goes live on your Google profile — nothing is posted without your OK. Want me to {a_en}? Reply YES.",
                           f"Simple hai: main draft karti hoon, aap approve karte hain, phir woh Google profile pe live hota hai — aapke OK ke bina kuch post nahi hota. Main {a_hi}? Reply YES."))
        for key, en, hi in self._facts_pool(w):
            if key in state.facts_mentioned:
                continue
            state.facts_mentioned.add(key)
            out.append(b.t(f"Good question — {en}. That's why I flagged it. Want me to {a_en}? Reply YES.",
                           f"Accha sawaal — {hi}. Isliye flag kiya. Main {a_hi}? Reply YES."))
        out.append(b.t(f"Happy to explain more — the short version: {self._hook_line(w)} Want me to {a_en}? Reply YES.",
                       f"Short mein: {self._hook_line(w)} Main {a_hi}? Reply YES."))
        return out

    def _objection(self, state: ConversationState, w: dict) -> list[str]:
        b = w["brief"]
        a_en, a_hi = self._action_phrase(w)
        pool = [p for p in self._facts_pool(w) if p[0] not in state.facts_mentioned]
        fact_en, fact_hi = (pool[0][1], pool[0][2]) if pool else ("it takes me the work, not you", "kaam meri taraf se hoga, aapki taraf se nahi")
        if pool:
            state.facts_mentioned.add(pool[0][0])
        return [b.t(f"Fair point, {b.sal()}. For context, {fact_en}. Smallest step: I {a_en.replace('send you', 'send').replace('put together', 'prepare')} and you decide — no commitment. Reply YES if you'd like to see it.",
                    f"Sahi baat, {b.sal()}. Context ke liye, {fact_hi}. Sabse chhota step: main {a_hi.rstrip('?')} aur decision aapka — koi commitment nahi. Dekhna ho toh YES reply karein.")]

    def _interested(self, state: ConversationState, w: dict) -> list[str]:
        b = w["brief"]
        a_en, a_hi = self._action_phrase(w)
        pool = [p for p in self._facts_pool(w) if p[0] not in state.facts_mentioned]
        fact = (pool[0][1], pool[0][2]) if pool else None
        if pool:
            state.facts_mentioned.add(pool[0][0])
        lead_en = f"Quick context: {fact[0]}. " if fact else ""
        lead_hi = f"Quick context: {fact[1]}. " if fact else ""
        return [b.t(f"{lead_en}I can {a_en} right now — takes you 1 minute to approve. Reply YES.",
                    f"{lead_hi}Main abhi {a_hi.rstrip('?')} — approve karne mein aapka 1 minute lagega. Reply YES.")]

    def _clarify(self, w: dict, delivered: bool = False) -> list[str]:
        b = w["brief"]
        if delivered:
            return [b.t(f"Just to check, {b.sal()} — shall I go ahead with the draft above? Reply GO, or send any edits.",
                        f"Bas confirm karna tha, {b.sal()} — upar wala draft aage badha doon? GO reply karein, ya edits bhejein.")]
        a_en, a_hi = self._action_phrase(w)
        return [b.t(f"Just to check, {b.sal()} — should I go ahead and {a_en}? Reply YES or NO.",
                    f"Bas confirm karna tha, {b.sal()} — main {a_hi}? YES ya NO reply karein.")]

    # -------------------------------------------------------- customer side
    def _customer_reply(self, state: ConversationState, w: dict, message: str, mi: str) -> dict:
        """Customer ↔ merchant (Vera writes on the merchant's behalf): a small booking flow that always moves forward."""
        b = w["brief"]
        low = message.strip().lower()
        name = b.P("name") or "us"
        where = ", ".join(x for x in (b.P("locality"), b.P("city")) if x)
        first = (w["cust"].first_name if w.get("cust") else "") or ""
        slots = [b.A(k) for k in ("slot1", "slot2") if b.A(k)]
        o = b.P("offer") or b.offer()[0]
        say = lambda *opts: self._send(state, w, [o_[:1].upper() + o_[1:] for o_ in opts], "open_ended", why)   # noqa: E731

        def mid(t_):
            return t_[0].lower() + t_[1:] if t_ and re.match(r"(Today|Tomorrow|Day after|This weekend|Next week)", t_) else t_

        pharmacy = b.prof.slug == "pharmacies"
        L = lambda t_: loc_time(t_, b.lang)                                                  # noqa: E731

        def book_text(t_):
            if pharmacy and state.fulfil == "pickup":
                return (f"Ready for pickup ✅ {t_} at {name}" + (f", {where}" if where else "") + ". Reply CHANGE to move it.",
                        f"Pickup ready ✅ {L(t_)}, {name}" + (f", {where}" if where else "") + ". Time badalna ho toh CHANGE reply karein.")
            if pharmacy:
                extra = f" (with {', '.join(state.order_items)})" if state.order_items else ""
                extra_hi = f" ({', '.join(state.order_items)} ke saath)" if state.order_items else ""
                return (f"Order placed ✅ Your refill{extra} will be delivered {mid(t_)} to your saved address. Reply CHANGE to move it.",
                        f"Order ho gaya ✅ Aapka refill{extra_hi} {L(t_)} aapke saved address pe deliver hoga. Time badalna ho toh CHANGE reply karein.")
            return (f"Booked ✅ {t_} at {name}" + (f", {where}" if where else "") + ". We'll send a reminder before your visit. Reply CHANGE anytime to reschedule.",
                    f"Booked ✅ {L(t_)}, {name}" + (f", {where}" if where else "") + " mein. Visit se pehle reminder bhejenge. Reschedule karna ho toh CHANGE reply karein.")

        def slot_menu():
            if pharmacy and not state.fulfil:
                return b.t("Home delivery or pick up from the store? And which day and time suits you?",
                           "Home delivery chahiye ya store se pickup? Aur kaunsa din aur time theek rahega?")
            if len(slots) >= 2:
                return b.t(f"Pick one: 1) {slots[0]} or 2) {slots[1]} — reply 1 or 2, or send another day/time.",
                           f"Ek chuniye: 1) {slots[0]} ya 2) {slots[1]} — 1 ya 2 reply karein, ya koi aur din/time bhejiye.")
            return b.t("Which day works for you — today, tomorrow or the weekend? And morning or evening?",
                       "Aapke liye kaunsa din theek rahega — aaj, kal ya weekend? Aur subah ya shaam?")

        if mi == "not_interested":
            state.exit_state = "ended"
            return {"action": "end", "rationale": "Customer declined/opted out — no further messages on behalf of the merchant."}
        if mi == "later" and not cust_time(message):
            return {"action": "wait", "wait_seconds": IntentRouter.wait_seconds(message), "rationale": "Customer asked for later."}
        # 0a) pharmacy: delivery or pickup
        if pharmacy and re.search(r"\b(deliver\w*|home|ghar|bhej do|send it)\b", low):
            state.fulfil = "delivery"
        elif pharmacy and re.search(r"\b(pick ?up|collect|aake|aa ke|store se|shop se|khud)\b", low):
            state.fulfil = "pickup"
        # 0b) "can you bring paracetamol too?" → add to the order / pass to the team
        im = _ITEM_REQ.search(message)
        item = im.group(1).strip() if im else None
        if item and item.lower() not in _ITEM_STOP and not cust_time(item):
            state.order_items.append(item)
            status = b.t(f" Your {'delivery' if pharmacy and state.fulfil != 'pickup' else 'booking'} is set for {mid(state.booked)}." if state.booked else " " + slot_menu(),
                         f" Aapki {'delivery' if pharmacy and state.fulfil != 'pickup' else 'booking'} {L(state.booked)} ki hai." if state.booked else " " + slot_menu())
            why = "Customer asked to add an item → added to the order / passed to the team (no invented price or stock)."
            if pharmacy:
                return say(b.t(f"Sure ✅ I've added {item} to your order — the pharmacist will confirm the strength and price when packing it.{status}",
                               f"Zaroor ✅ {item} aapke order mein add kar diya — pharmacist packing ke waqt strength aur price confirm karenge.{status}"))
            return say(b.t(f"Noted ✅ I've passed '{item}' to the {name} team — they'll confirm on this chat.{status}",
                           f"Noted ✅ '{item}' ki request {name} team ko bhej di hai — woh isi chat pe confirm karenge.{status}"))
        # 1) picked a numbered slot
        pick = re.fullmatch(r"\s*([12])\s*[.!)]?\s*", low)
        if pick and len(slots) >= int(pick.group(1)):
            state.booked, state.cust_stage = slots[int(pick.group(1)) - 1], "booked"
            why = "Customer picked a slot → booked."
            en_, hi_ = book_text(state.booked)
            return say(b.t(en_, hi_))
        # 2) proposed a day / time → book it (a bare day asks for the time first)
        t = cust_time(message)
        if t and state.partial_day and not re.match(r"(Today|Tomorrow|Day after|This weekend|Next week|Mon|Tue|Wed|Thu|Fri|Sat|Sun)", t):
            t = f"{state.partial_day} {t[0].lower() + t[1:]}".replace(" ,", ",")
        elif not t and state.partial_day and re.search(r"\b(morning|afternoon|evening|night|subah|dopahar|shaam|sham|raat)\b", low):
            t = cust_time(f"{state.partial_day} {low}")
        bare_day = t and re.fullmatch(r"(Today|Tomorrow|Day after tomorrow|This weekend|Next week|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)", t)
        if bare_day:
            state.partial_day, state.cust_stage = t, "ask_time"
            why = "Customer gave a day without a time → confirm the day, ask morning/evening."
            return say(b.t(f"{t} works ✅ Morning or evening — or a specific time?", f"{L(t)} theek hai ✅ Subah ya shaam — ya koi exact time?"),
                       b.t(f"Great, {mid(t)}. What time suits you?", f"Badhiya, {L(t)}. Kaunsa time theek rahega?"))
        if t and not re.search(r"\b(timing|timings|open|khula|close|band)\b", low):
            state.booked, state.cust_stage, state.partial_day = t, "booked", None
            why = "Customer proposed a time → booked it and confirmed the details."
            en_, hi_ = book_text(t)
            return say(b.t(en_, hi_), b.t(f"Done ✅ All set for {mid(t)}.", f"Ho gaya ✅ {L(t)} ke liye sab set."))
        # 2b) "slot booked?" / "is it confirmed?" → direct status answer
        if re.search(r"\b(booked|confirmed?|confirm hua|book hua|ho gaya|hua kya|pakka)\b.*\?|^\s*(is|was)\s+(my|the)\s+(slot|appointment|booking)", low) \
                or re.search(r"\b(slot|appointment|booking)\b[^.?!]{0,20}\b(booked|confirmed?|done|pakka|hua)\b", low):
            why = "Customer asked for booking status → direct answer."
            if state.booked and pharmacy:
                kind_en = "pickup" if state.fulfil == "pickup" else "delivery"
                return say(b.t(f"Yes ✅ Your refill {kind_en} is confirmed for {mid(state.booked)}" + (f" (with {', '.join(state.order_items)})" if state.order_items else "") + ". Reply CHANGE to move it.",
                               f"Haan ✅ Aapka refill {kind_en} {L(state.booked)} ke liye confirm hai" + (f" ({', '.join(state.order_items)} ke saath)" if state.order_items else "") + ". Time badalna ho toh CHANGE reply karein."),
                           b.t(f"Confirmed ✅ {kind_en.capitalize()} {mid(state.booked)}.", f"Confirm ✅ {kind_en.capitalize()} {L(state.booked)}."))
            if state.booked:
                return say(b.t(f"Yes ✅ You're booked for {mid(state.booked)} at {name}" + (f", {where}" if where else "") + ". Reply CHANGE to reschedule.",
                               f"Haan ✅ Aapki booking {L(state.booked)}, {name} mein pakki hai. Reschedule ke liye CHANGE reply karein."),
                           b.t(f"Confirmed ✅ {state.booked} at {name}. We'll remind you before the visit.",
                               f"Confirm hai ✅ {L(state.booked)}, {name}. Visit se pehle reminder bhejenge."),
                           b.t(f"All good — your {mid(state.booked)} slot is locked in 🙂", f"Sab theek — aapka {L(state.booked)} ka slot pakka hai 🙂"))
            state.cust_stage = "ask_time"
            return say(b.t("Not yet — send me a day and time (for example 'tomorrow 5pm') and I'll book it right away.",
                           "Abhi nahi — ek din aur time bhej dijiye (jaise 'kal 5 baje'), main turant book kar dungi."),
                       b.t("Not booked yet — just tell me which day and time suits you.", "Abhi booking nahi hui — bas din aur time bata dijiye."))
        if pharmacy and state.fulfil and not state.booked and re.search(r"\b(deliver\w*|home|ghar|pick ?up|collect|store|shop)\b", low):
            why = "Pharmacy customer chose delivery/pickup → ask the time."
            what_en = "Home delivery" if state.fulfil == "delivery" else "Store pickup"
            what_hi = "Home delivery" if state.fulfil == "delivery" else "Store pickup"
            return say(b.t(f"{what_en} it is ✅ Which day and time suits you?", f"{what_hi} ✅ Kaunsa din aur time theek rahega?"))
        # 3) reschedule
        if re.search(r"\b(change|reschedule|another time|different time|dusra time|time badal|shift)\b", low):
            state.cust_stage = "ask_time"
            why = "Customer wants to reschedule → ask for the new time."
            return say(b.t("Sure — which day and time would suit you better?", "Zaroor — aapke liye kaunsa din aur time better rahega?"),
                       b.t("No problem. Send the new day and time and we'll move it.", "Koi baat nahi. Naya din aur time bhejiye, hum shift kar denge."))
        # 4) questions: timings / where / price / what is this
        if re.search(r"\b(timings?|time kya|hours|open|kab tak|kab khula|khula|closing|kitne baje)\b", low):
            why = "Customer asked about timings → answer from the slots we have, ask for a preference."
            booked = b.t(f" You're booked for {mid(state.booked)}.", f" Aapki booking {L(state.booked)} ki hai.") if state.booked else ""
            if slots:
                return say(b.t(f"Open slots right now: {' or '.join(slots)}.{booked} Reply 1 or 2, or tell us a time that suits you.",
                               f"Abhi ke slots: {' ya '.join(slots)}.{booked} 1 ya 2 reply karein, ya apna time bataiye."))
            return say(b.t(f"{name} will confirm the exact time for you.{booked} Tell us a day and whether morning or evening suits you, and we'll book the nearest free slot.",
                           f"{name} aapke liye exact time confirm karega.{booked} Bas din aur subah/shaam bata dijiye, hum sabse paas ka free slot book kar denge."),
                       b.t("Share any day and time (e.g. 'tomorrow 6pm') and we'll check it's free and confirm.",
                           "Koi bhi din aur time bhejiye (jaise 'kal 6 baje'), hum free hai ya nahi check karke confirm karenge."))
        if re.search(r"\b(where|address|location|kahan|kaha|pata|direction)\b", low):
            why = "Customer asked where → location from the merchant profile."
            return say(b.t(f"We're at {name}" + (f", {where}" if where else "") + ". " + ("You're booked for " + mid(state.booked) + "." if state.booked else slot_menu()),
                           f"Hum {name}" + (f", {where}" if where else "") + " mein hain. " + ("Aapki booking " + L(state.booked) + " ki hai." if state.booked else slot_menu())))
        if re.search(r"\b(price|cost|charges?|fees?|kitna|kitne ka|rate|offer|paise|₹)\b", low):
            why = "Customer asked the price → only the merchant's real offer, never an invented price."
            return say(b.t((f"Current offer: {o}. " if o else f"{name} will share the exact price when confirming. ") + slot_menu(),
                           (f"Abhi ka offer: {o}. " if o else f"{name} confirm karte waqt exact price bata dega. ") + slot_menu()))
        if re.search(r"\b(what|why|kya|kyu|kyun|kyaa|kis liye|matlab|about)\b", low) and not state.booked:
            opening = (state.bot_bodies() or [""])[0]
            sents = [x.strip() for x in re.split(r"(?<!\bDr)(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<=[.!?।])\s+", opening) if x.strip()]
            body_sents = [x for x in sents if not re.search(r"\b(here|se|से)\b\s*\W*$|^(hi|hello|namaste|नमस्ते)\b", x, re.I)
                          and not re.search(r"\bYES\b|reply|bhejein|भेजें", x, re.I)]
            core = body_sents[0] if body_sents else (sents[0] if sents else "")
            why = "Customer asked what this is → restate the reason from the opening, then the next step."
            return say(b.t(f"{core} " + slot_menu(), f"{core} " + slot_menu()),
                       b.t(f"Just a reminder from {name}: {core[0].lower() + core[1:] if core else 'your visit is due'} " + slot_menu(),
                           f"{name} se ek reminder: {core} " + slot_menu()))
        # 5) yes / krdo / ok / book
        if mi == "explicit_action" or re.fullmatch(r"(ok|okay|okk|haan|ha|ji|yes|sure)[\s!.]*", low):
            why = "Customer said yes → move to booking (offer slots or ask day + time)."
            if state.booked:
                return say(b.t(f"You're all set for {mid(state.booked)} 🙂 See you then!", f"Aapki booking {L(state.booked)} ki pakki hai 🙂 Milte hain!"),
                           b.t(f"All confirmed for {mid(state.booked)}. Reply CHANGE if you need a different time.", f"{L(state.booked)} ke liye sab confirm hai. Time badalna ho toh CHANGE reply karein."))
            if state.cust_stage == "ask_time":
                return say(b.t("Just send a day and time (for example 'tomorrow 6pm') and I'll book it for you.",
                               "Bas ek din aur time bhej dijiye (jaise 'kal shaam 6 baje'), main book kar dungi."),
                           b.t(f"Almost done{', ' + first if first else ''} — which day suits you, and morning or evening?",
                               f"Bas ek step{', ' + first if first else ''} — kaunsa din theek rahega, aur subah ya shaam?"),
                           b.t(f"Tell me any time that works and I'll hold it at {name}.", f"Jo bhi time theek ho bataiye, main {name} mein hold kar dungi."))
            state.cust_stage = "ask_time"
            return say(b.t(f"Great{', ' + first if first else ''}! " + slot_menu(), f"Badhiya{', ' + first if first else ''}! " + slot_menu()),
                       b.t("Perfect — " + slot_menu()[0].lower() + slot_menu()[1:], "Perfect — " + slot_menu()))
        # 6) thanks
        if _ACK.match(message):
            why = "Customer said thanks → warm close or the one next step."
            if state.booked and pharmacy:
                arrive_en = "is ready for pickup" if state.fulfil == "pickup" else "arrives"
                arrive_hi = "pickup ke liye ready hoga" if state.fulfil == "pickup" else "pahunch jaayega"
                return say(b.t(f"You're welcome! Your refill {arrive_en} {mid(state.booked)} 🙂", f"Shukriya! Aapka refill {L(state.booked)} {arrive_hi} 🙂"))
            if state.booked:
                return say(b.t(f"You're welcome! See you {mid(state.booked)} 🙂" if re.match(r"(Today|Tomorrow)", state.booked) else f"You're welcome! See you on {state.booked} 🙂", f"Shukriya! {L(state.booked)} milte hain 🙂"))
            return say(b.t("You're welcome! " + slot_menu(), "Shukriya! " + slot_menu()))
        # 7) "hello?", "?", anything unclear → the next step for where we are, never the same line twice
        why = "Unclear / nudge → restate the next step for the current stage."
        if state.booked:
            return say(b.t(f"You're booked for {mid(state.booked)} at {name}. Reply CHANGE to reschedule, or ask me anything about your visit.",
                           f"Aapki booking {L(state.booked)}, {name} mein hai. Reschedule ke liye CHANGE reply karein, ya visit ke baare mein kuch bhi poochiye."),
                       b.t(f"Hi{' ' + first if first else ''} 🙂 All set for {mid(state.booked)}. Anything else I can help with?",
                           f"Hi{' ' + first if first else ''} 🙂 {L(state.booked)} ke liye sab set hai. Aur kuch madad chahiye?"))
        return say(b.t(f"Hi{' ' + first if first else ''} 🙂 " + slot_menu(), f"Hi{' ' + first if first else ''} 🙂 " + slot_menu()),
                   b.t("Just send a day and time (for example 'tomorrow 6pm') and we'll book it for you.",
                       "Bas ek din aur time bhej dijiye (jaise 'kal shaam 6 baje'), hum book kar denge."),
                   b.t(f"We're here to help{', ' + first if first else ''}. Reply with a day and morning or evening, and we'll hold a slot at {name}.",
                       f"Hum madad ke liye yahin hain. Din aur subah/shaam bata dijiye, hum {name} mein slot hold kar denge."))
