"""Structured contracts passed between agents. Every agent consumes and emits one of these."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class Fact:
    id: str
    path: str            # source path, e.g. "merchant.performance.ctr"
    value: Any
    text: str            # human rendering, e.g. "2.1%"
    layer: str           # category | merchant | trigger | customer | derived | conversation
    kind: str = "text"   # number | percent | money | date | text | offer | citation
    label: str = ""      # short semantic label, e.g. "ctr", "active_offer"

    def to_dict(self) -> dict:
        return {"fact": f"{self.label or self.path}: {self.text}", "source": self.path, "id": self.id}


@dataclass
class ContextFacts:
    """Output of the Context Analyst: facts bucketed by role, each carrying its source path."""
    category_facts: list[str] = field(default_factory=list)
    merchant_facts: list[str] = field(default_factory=list)
    trigger_facts: list[str] = field(default_factory=list)
    customer_facts: list[str] = field(default_factory=list)
    performance_facts: list[str] = field(default_factory=list)
    offer_facts: list[str] = field(default_factory=list)
    peer_facts: list[str] = field(default_factory=list)
    conversation_facts: list[str] = field(default_factory=list)
    language: str = "en"
    available_actions: list[str] = field(default_factory=list)


@dataclass
class TriggerAnalysis:
    trigger_type: str
    family: str
    urgency: int
    why_now: str
    primary_goal: str
    recommended_action: str
    expiration: Optional[str]
    key_fact_ids: list[str] = field(default_factory=list)
    anchor: dict = field(default_factory=dict)   # the resolved "most important trigger fact" bundle
    is_expired: bool = False
    is_informational: bool = False


@dataclass
class IntentResult:
    mode: str                 # DISCOVER | INFORM | RECOMMEND | ACT | FOLLOW_UP | RECOVER | EXIT
    merchant_intent: str = "none"   # interested | explicit_action | question | objection | not_interested | auto_reply | unclear | hostile | off_topic | later | none
    confidence: float = 1.0
    signals: list[str] = field(default_factory=list)
    language: Optional[str] = None


@dataclass
class AutoReplyVerdict:
    is_auto_reply: bool
    confidence: float
    reason: str
    recommended_behavior: str   # continue | clarify_once | exit


@dataclass
class CategoryProfile:
    slug: str
    register: str             # clinical | warm | operator | coach | trust | neutral
    tone: str
    vocab_allowed: list[str]
    taboos: list[str]
    hype_forbidden: bool
    offer_examples: list[str]
    peer_label: str
    noun_singular: str        # "clinic", "salon", ...
    customer_noun: str        # "patients", "clients", "guests", "members", "customers"
    relevant_digest: list[dict] = field(default_factory=list)
    relevant_beats: list[dict] = field(default_factory=list)
    relevant_trends: list[dict] = field(default_factory=list)
    style_rules: list[str] = field(default_factory=list)


@dataclass
class Personalization:
    personalization_facts: list[str]
    reason: str
    salutation: str = ""


@dataclass
class CustomerPlan:
    customer_personalization: list[str]
    customer_goal: str
    consent_ok: bool
    consent_reason: str
    send_as: str = "merchant_on_behalf"
    first_name: str = ""


@dataclass
class LanguagePlan:
    language: str             # en | hi-en | hi
    tone: str
    style_rules: list[str] = field(default_factory=list)


@dataclass
class StrategyPlan:
    variant: str
    objective: str
    hook: str
    facts_to_use: list[str]
    engagement_levers: list[str]
    cta_strategy: str         # binary_yes_stop | open_ended | slot_choice | none
    tone: str
    message_structure: list[str]


@dataclass
class Draft:
    segments: list[tuple[str, str]]          # ordered (role, text)
    cta: str
    plan: StrategyPlan
    used_fact_ids: list[str] = field(default_factory=list)
    source: str = "deterministic"
    allowed_extra_numbers: set = field(default_factory=set)

    @property
    def body(self) -> str:
        from .util import norm_space
        return norm_space(" ".join(t for _, t in self.segments if t))

    def replace(self, role: str, text: str) -> None:
        for i, (r, _) in enumerate(self.segments):
            if r == role:
                self.segments[i] = (role, text)
                return
        self.segments.append((role, text))


@dataclass
class Issue:
    code: str
    detail: str
    severity: str = "error"   # error blocks finalization; warn feeds the critic


@dataclass
class CheckResult:
    ok: bool
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]


@dataclass
class Critique:
    scores: dict[str, float]
    weaknesses: list[str]
    notes: list[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        return round(sum(self.scores.values()), 2)


@dataclass
class TraceStep:
    agent: str
    summary: str
    data: Any = None


def to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, set):
        return sorted(to_jsonable(x) for x in obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    return obj
