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
from .agents.intent_router import AutoReplyDetector, IntentRouter, detect_language
from .agents.validators import FactChecker, PolicyChecker
from .orchestrator import Orchestrator
from .types import Draft, LanguagePlan, StrategyPlan, TraceStep
from .util import humanize

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
    hostile_count: int = 0
    off_topic_count: int = 0
    objection_count: int = 0
    unclear_count: int = 0
    later_count: int = 0
    merchant_memory: dict = field(default_factory=dict)          # shared per-merchant memory (e.g. auto-reply texts)

    def bot_bodies(self) -> list[str]:
        return [m["body"] for m in self.messages if m["from"] == "vera"]

    def their_msgs(self) -> list[str]:
        return [m["body"] for m in self.messages if m["from"] != "vera"]

    def record_bot(self, body: str, cta: str) -> None:
        self.messages.append({"from": "vera", "body": body, "ts": time.time()})
        self.ctas_used.append(cta)


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
                return self._send(state, w, self._clarify_auto(w), "binary_yes_stop",
                                  f"Auto-reply detected ({verdict.reason}); one owner-directed nudge, will exit if it repeats.")
            state.exit_state = "ended"
            return {"action": "end", "rationale": f"Auto-reply again ({verdict.reason}); exiting instead of burning turns."}

        router = IntentRouter(trace)
        intent = router.classify(message)
        state.intents.append(intent.merchant_intent)
        state.merchant_intent = intent.merchant_intent
        lang_now = detect_language(message)
        if lang_now and len(message.split()) >= 2:
            state.language = lang_now if not (lang_now == "hi" and not state.customer) else "hi-en"
        w = self._working(state, message)
        mi = intent.merchant_intent

        if from_role == "customer" or state.customer:
            return self._customer_reply(state, w, message, mi)

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
        if mi == "explicit_action":
            state.exit_state = None
            state.merchant_sentiment = "positive"
            already = bool(state.actions_requested)
            state.actions_requested.append(w["ta"].family)
            body = self._act(w, already, price=router.is_price_question(message))
            return self._send(state, w, body, "open_ended",
                              "Explicit go-ahead → ACT immediately (no re-qualification): delivered the artifact + next step.")
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
        return self._send(state, w, self._clarify(w), "binary_yes_stop", "Unclear reply → one binary clarification.")

    # ------------------------------------------------------------- plumbing
    def _working(self, state: ConversationState, latest: Optional[str]) -> dict:
        trig = state.trigger or {"id": "trg_conversation", "kind": "dormant_with_vera", "scope": "merchant", "payload": {}}
        merchant = state.merchant or {"merchant_id": state.merchant_id or "m_unknown", "identity": {}}
        category = state.category or {"slug": merchant.get("category_slug", "general")}
        w = self.orch.analyze(category, merchant, trig, state.customer, latest_reply=latest)
        if state.language:
            w["lang"] = LanguagePlan(language=state.language, tone=w["lang"].tone, style_rules=w["lang"].style_rules)
        w["brief"] = Brief(w["ta"], w["pz"], w["prof"], w["lang"], w["cust"], w["tools"])
        return w

    def _send(self, state: ConversationState, w: dict, bodies, cta: str, rationale: str) -> dict:
        options = bodies if isinstance(bodies, list) else [bodies]
        prev = state.bot_bodies()
        chosen = None
        for body in options:
            body = re.sub(r"[ \t]+", " ", body).strip()
            if body in prev:
                continue
            d = Draft(segments=[("reply", body)], cta=cta, plan=StrategyPlan("reply", "", "", [], [], cta, "", []))
            f = FactChecker([]).run(d, w["ledger"], w["ta"], bool(state.customer))
            p = PolicyChecker([]).run(d, w["ledger"], w["prof"], w["ta"], w["cust"],
                                      "merchant_on_behalf" if state.customer else "vera", w["lang"].language, prev, max_len=900)
            blocking = [i for i in f.errors + p.errors if i.code not in ("buried_cta",)]
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

    def _post_text(self, w: dict) -> tuple[str, str]:
        b = w["brief"]
        name, loc = b.P("name") or "", b.P("locality")
        where = f"{name}, {loc}" if loc else name
        offer, kind = b.offer()
        fam = w["ta"].family
        lead = ""
        if fam == "festival" and b.A("name"):
            lead = f"This {b.A('name')}, "
        rating = f" Rated {b.P('rating')} by {b.P('reviews')} customers." if b.P("rating") and b.P("reviews") else ""
        if offer and kind in ("active", "inactive"):
            post = f"{lead}{offer} at {where}.{rating} Message us on WhatsApp to book."
        elif fam == "trend" and b.A("query"):
            post = f"Searching for {b.A('query')}? Talk to the team at {where}.{rating} Message us on WhatsApp."
        else:
            post = f"{lead}Visit {where}.{rating} Message us on WhatsApp to book."
        return post[0].upper() + post[1:], post[0].upper() + post[1:]

    def _act(self, w: dict, already: bool, price: bool = False) -> list[str]:
        b, ta = w["brief"], w["ta"]
        fam = ta.family
        cn = b.cust_noun()
        name = b.P("name") or ""
        if already:
            return [b.t("On it ✅ Everything's queued — you'll get the preview here for a final OK. Anything you'd like changed?",
                        "Kaam chalu hai ✅ Sab queue mein hai — final OK ke liye preview yahin aayega. Kuch badlaav chahiye?"),
                    b.t("Done on my side ✅ Preview is next — just reply with edits if you want any.",
                        "Meri taraf se ho gaya ✅ Agla step preview hai — edits ho toh reply kar dijiye.")]
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
        post_en, post_hi = self._post_text(w)
        extra_en = " Pricing for the post itself isn't in my records here, so I'll have the magicpin team confirm — no guesses." if price else ""
        extra_hi = " Post ki pricing mere records mein nahi hai, magicpin team confirm karegi — main guess nahi karungi." if price else ""
        return [b.t(f"Done ✅ Here's the draft post for {name} ↓\n{post_en}\nReply GO and it goes live today, or send any edits.{extra_en}",
                    f"Ho gaya ✅ {name} ke liye draft post ↓\n{post_hi}\nGO reply karein toh aaj hi live kar doon, ya edits bhej dijiye.{extra_hi}")]

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

    def _clarify(self, w: dict) -> list[str]:
        b = w["brief"]
        a_en, a_hi = self._action_phrase(w)
        return [b.t(f"Just to check, {b.sal()} — should I go ahead and {a_en}? Reply YES or NO.",
                    f"Bas confirm karna tha, {b.sal()} — main {a_hi}? YES ya NO reply karein.")]

    # -------------------------------------------------------- customer side
    def _customer_reply(self, state: ConversationState, w: dict, message: str, mi: str) -> dict:
        b = w["brief"]
        low = message.strip().lower()
        if mi == "not_interested":
            state.exit_state = "ended"
            return {"action": "end", "rationale": "Customer declined/opted out — no further messages on behalf of the merchant."}
        pick = re.fullmatch(r"\s*([12])\s*[.!]?\s*", low)
        if pick and b.A(f"slot{pick.group(1)}"):
            slot = b.A(f"slot{pick.group(1)}")
            return self._send(state, w, [b.t(f"Booked ✅ {slot}. See you then! Reply CHANGE anytime to reschedule.",
                                             f"Booked ✅ {slot}. Milte hain! Reschedule karna ho toh CHANGE reply karein.",
                                             f"बुक हो गया ✅ {slot}. बदलना हो तो CHANGE भेजें।")], "none",
                              "Customer picked a slot → confirm booking immediately.")
        if mi == "explicit_action":
            return self._send(state, w, [b.t("Great ✅ We'll confirm your slot on this chat shortly.",
                                             "Badhiya ✅ Aapka slot isi chat pe jaldi confirm karte hain.",
                                             "बढ़िया ✅ आपका स्लॉट इसी चैट पर जल्दी कन्फ़र्म करेंगे।")], "none",
                              "Customer accepted → confirm; merchant's team finalises the time.")
        if mi == "later":
            return {"action": "wait", "wait_seconds": IntentRouter.wait_seconds(message), "rationale": "Customer asked for later."}
        o = b.P("offer")
        return self._send(state, w, [b.t("Happy to help! " + (f"Current offer: {o}. " if o else "") + "Reply with a day and time that suits you and we'll book it.",
                                         "Zaroor! " + (f"Abhi offer: {o}. " if o else "") + "Apna convenient din aur time bhejiye, hum book kar denge.",
                                         "ज़रूर! " + (f"अभी का ऑफ़र: {o}। " if o else "") + "अपना सुविधाजनक दिन और समय भेजें, हम बुक कर देंगे।")],
                          "open_ended", "Customer question → answer from merchant facts, ask for a time.")
