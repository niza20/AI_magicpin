"""AGENT 3 — Intent Router, and AGENT 4 — Auto-Reply Detector.

Rule-first (deterministic, instant, auditable); the LLM is consulted only when the rules return
`unclear`. CRITICAL RULE: explicit action intent overrides qualification → mode ACT.
"""
from __future__ import annotations

import re
from typing import Optional

from ..llm import get_llm
from ..types import AutoReplyVerdict, IntentResult, TriggerAnalysis
from .base import Agent

_HINGLISH = {"hai", "hain", "nahi", "nahin", "kya", "karo", "kar", "mujhe", "aap", "aapka", "aapki", "haan", "han",
             "theek", "thik", "chahiye", "kitna", "kitne", "bhai", "ji", "mein", "ka", "ki", "ko", "kaise", "abhi",
             "baad", "kab", "batao", "bolo", "accha", "achha", "acha", "karna", "karni", "hoga", "wala", "wali",
             "chalega", "sahi", "bhejo", "bhej", "mat", "band", "kyun", "kyu", "matlab", "samjha", "dijiye", "karein"}

_STOP = re.compile(r"\b(stop|unsubscribe|opt[\s-]?out|spam|don'?t (message|msg|text|contact|call)|do not (message|msg|text|contact)|"
                   r"stop (messaging|texting|sending)|band karo|mat bhejo|message mat|remove me|block)\b")
_ABUSE = re.compile(r"\b(idiot|stupid|useless|nonsense|bakwas|bakwaas|bewakoof|chutiya|pagal|shut up|chup|fraud|scam|"
                    r"harass\w*|irritat\w*|pareshan|faltu|rubbish|waste of time|go to hell)\b")
_NOT_INTERESTED = re.compile(r"\b(not interested|no interest|interest nahi|nahi chahiye|nahin chahiye|no thanks|no thank you|"
                             r"don'?t need|dont want|don'?t want|no need|zarurat nahi|zaroorat nahi|mat karo|not required|"
                             r"not for me|we'?re good|i'?m good)\b|^\s*(no|nope|nahi|nahin|na)\s*[.!]*\s*$")
_ACTION = re.compile(r"\b(yes|yess+|yeah|yep|yup|haan|han ji|haan ji|ha ji|ok+|okay|okk|sure|go ahead|go for it|do it|let'?s do( it)?|"
                     r"lets do( it)?|let'?s go|start( it)?|join|proceed|send( it| me| please)?|confirm|confirmed|please do|"
                     r"kar ?do|karo|kardo|kar dijiye|chalega|chalo|theek hai|thik hai|done|book( it)?|activate|"
                     r"update (my|the)|sounds good|go live|publish|post it|i want to join|judna hai|judrna hai|"
                     r"i'?m in|count me in|bhej do|bhejo|haan karo|agreed|approve|approved)\b|👍|✅")
_LATER = re.compile(r"\b(later|busy|baad me(in)?|kal|tomorrow|call me later|not now|abhi nahi|abhi nahin|next week|"
                    r"in a meeting|thodi der|some other time|remind me)\b")
_OFF_TOPIC = re.compile(r"\b(gst|income tax|itr|tax filing|file (my )?tax|loan|visa|passport|insurance claim|"
                        r"cricket|score|stock|share market|bitcoin|crypto|election|recipe|movie|aadhaar|pan card)\b")
_OBJECTION = re.compile(r"\b(too expensive|expensive|mehenga|mehnga|costly|no budget|budget nahi|doesn'?t work|does not work|"
                        r"not useful|no use|koi fayda nahi|already (have|doing|tried)|pehle se|tried before|"
                        r"didn'?t work|waste|no results|not convinced|trust nahi)\b")
_PRICE = re.compile(r"\b(how much|kitna|kitne|price|pricing|cost|charges?|fees?|rate|paisa|paise|rs\.?|₹)\b")
_QUESTION = re.compile(r"\?|^\s*(what|how|why|when|where|which|who|can you|could you|is it|are you|do you|kya|kaise|kyun|kab|"
                       r"kahan|kaun|kaunsa)\b")
_INTERESTED = re.compile(r"\b(interesting|tell me more|more details|details|batao|bataiye|sounds interesting|hmm+|achha|accha|"
                         r"acha|nice|good|great|cool|really|sach mein|go on)\b")
_WAIT_LONG = re.compile(r"\b(tomorrow|kal|next week)\b")

_AUTO_PATTERNS = re.compile(
    r"(thank(s| you) for (contacting|reaching|your (message|enquiry|inquiry))|we will (get back|respond|revert)|"
    r"will (get back|respond|revert) (to you )?(shortly|soon|asap)|our (team|executive|representative) will|"
    r"automated (assistant|message|reply|response)|auto[- ]?reply|out of (the )?office|currently (unavailable|away|closed)|"
    r"business hours|we are closed|working hours|aapki jaankari ke liye|team tak pahuncha|hum jald|sampark karne ke liye|"
    r"sampark ke liye dhanyavaad|welcome to [a-z]|how (can|may) (we|i) (help|assist) you|for (bookings|orders|appointments),? (please )?(call|visit|click)|"
    r"this is an automated|do not reply|main ek automated|kindly (wait|hold))", re.I)


def detect_language(text: str) -> Optional[str]:
    if not text or not text.strip():
        return None
    if re.search(r"[ऀ-ॿ]", text):
        return "hi"
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return None
    hits = sum(1 for w in words if w in _HINGLISH)
    if hits >= 2 or (hits >= 1 and len(words) <= 4):
        return "hi-en"
    return "en"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9ऀ-ॿ ]+", "", (s or "").lower()).strip()


class AutoReplyDetector(Agent):
    name = "auto_reply_detector"

    def run(self, message: str, prior_merchant_msgs: list[str], prior_auto_replies: int) -> AutoReplyVerdict:
        n = _norm(message)
        repeats = sum(1 for m in prior_merchant_msgs if _norm(m) == n and n)
        pattern = bool(_AUTO_PATTERNS.search(message or ""))
        if pattern and repeats:
            v = AutoReplyVerdict(True, 0.99, f"canned phrasing + repeated verbatim {repeats + 1}x", "exit")
        elif repeats >= 1:
            v = AutoReplyVerdict(True, 0.9 if repeats >= 2 else 0.75, f"identical message repeated {repeats + 1}x",
                                 "exit" if repeats >= 2 or prior_auto_replies >= 1 else "clarify_once")
        elif pattern:
            v = AutoReplyVerdict(True, 0.85, "canned WhatsApp Business auto-reply phrasing",
                                 "exit" if prior_auto_replies >= 1 else "clarify_once")
        else:
            v = AutoReplyVerdict(False, 0.9, "no canned/repetition signal", "continue")
        self.log(f"auto_reply={v.is_auto_reply} ({v.confidence}) → {v.recommended_behavior}: {v.reason}")
        return v


class IntentRouter(Agent):
    name = "intent_router"
    uses_llm = True

    def initial(self, ta: TriggerAnalysis) -> IntentResult:
        """First outbound: mode comes from the trigger family (no merchant utterance yet)."""
        from .trigger_analyst import FAMILY_GOALS
        mode = FAMILY_GOALS.get(ta.family, FAMILY_GOALS["generic"])[0]
        r = IntentResult(mode=mode, merchant_intent="none", signals=[f"family={ta.family}"])
        self.log(f"initial mode={mode}")
        return r

    def classify(self, message: str, deadline: Optional[float] = None) -> IntentResult:
        t = (message or "").strip()
        low = t.lower()
        lang = detect_language(t)
        sig: list[str] = []

        def res(mode: str, intent: str, conf: float = 0.9) -> IntentResult:
            r = IntentResult(mode=mode, merchant_intent=intent, confidence=conf, signals=sig, language=lang)
            self.log(f"'{t[:60]}' → intent={intent}, mode={mode}, lang={lang}", sig)
            return r

        if not low:
            return res("FOLLOW_UP", "unclear", 0.3)
        if _STOP.search(low):
            sig.append("stop/opt-out keyword")
            return res("EXIT", "not_interested", 0.97)
        if _NOT_INTERESTED.search(low) and not re.search(r"\b(yes|haan|ok)\b", low):
            sig.append("not-interested phrase")
            return res("EXIT", "not_interested", 0.9)
        abusive = bool(_ABUSE.search(low))
        off_topic = bool(_OFF_TOPIC.search(low))
        if abusive:
            sig.append("abusive language")
            if off_topic:
                sig.append("off-topic ask")
            return res("RECOVER", "hostile", 0.85)
        action = bool(_ACTION.search(low))
        negated = bool(re.search(r"\b(don'?t|do not|mat|nahi|not)\b", low)) and not re.search(r"\bnot now\b", low)
        if action and not negated and not off_topic:
            sig.append("explicit action phrase — overrides qualification")
            if _PRICE.search(low):
                sig.append("also asked price")
            return res("ACT", "explicit_action", 0.93)
        if off_topic:
            sig.append("off-topic request")
            return res("RECOVER", "off_topic", 0.85)
        if _LATER.search(low):
            sig.append("deferral")
            return res("FOLLOW_UP", "later", 0.85)
        if _OBJECTION.search(low):
            sig.append("objection")
            return res("RECOVER", "objection", 0.8)
        if _QUESTION.search(low) or _PRICE.search(low):
            sig.append("price question" if _PRICE.search(low) else "question")
            return res("INFORM", "question", 0.85)
        if _INTERESTED.search(low):
            sig.append("interest marker")
            return res("RECOMMEND", "interested", 0.7)
        llm = get_llm()
        if llm.enabled:
            out = llm.complete_json(
                "intent_router",
                "You classify a merchant's WhatsApp reply to a marketing assistant. Reply with JSON only.",
                "Classify this merchant reply into exactly one intent from: interested, explicit_action, question, objection, "
                "not_interested, auto_reply, off_topic, later, unclear.\n"
                f'Reply: "{t}"\nReturn {{"intent": "...", "confidence": 0.0}}', deadline=deadline, max_tokens=400)
            if out and out.get("intent") in {"interested", "explicit_action", "question", "objection", "not_interested",
                                             "auto_reply", "off_topic", "later", "unclear"}:
                intent = out["intent"]
                mode = {"explicit_action": "ACT", "not_interested": "EXIT", "question": "INFORM", "objection": "RECOVER",
                        "off_topic": "RECOVER", "later": "FOLLOW_UP", "interested": "RECOMMEND"}.get(intent, "FOLLOW_UP")
                sig.append("llm classification")
                return res(mode, intent, float(out.get("confidence", 0.6) or 0.6))
        return res("FOLLOW_UP", "unclear", 0.4)

    @staticmethod
    def wait_seconds(message: str) -> int:
        return 86400 if _WAIT_LONG.search((message or "").lower()) else 3600

    @staticmethod
    def is_price_question(message: str) -> bool:
        return bool(_PRICE.search((message or "").lower()))
