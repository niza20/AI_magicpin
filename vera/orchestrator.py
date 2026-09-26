"""ORCHESTRATOR — deterministic state machine that owns the workflow.

Agents never call each other; the orchestrator passes each one exactly the structured inputs it
needs and records every transition in a trace. Initial composition:

  CONTEXT → TRIGGER → INTENT → CATEGORY → PERSONALIZATION → [CUSTOMER] → LANGUAGE → STRATEGY
  → COMPOSE (N candidates, + optional LLM draft) → FACT_CHECK → POLICY_CHECK → CRITIC
  → REWRITE (≤2 cycles, re-validated) → SELECT → SUPPRESSION → FINALIZE   (or SAFE_FALLBACK)

(Strategy runs after Category/Personalization/Customer/Language because it consumes their outputs.)
Skips: CUSTOMER when customer is None; LLM steps when no LLM or no time budget; FINALIZE is
unreachable for a draft that failed fact or policy validation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .agents.category_expert import CategoryExpert
from .agents.composer import Brief, MessageComposer
from .agents.context_analyst import ContextAnalyst
from .agents.critic import EngagementCritic
from .agents.customer_agent import CustomerAgent
from .agents.finalizer import Finalizer, SuppressionAgent, validate_output
from .agents.intent_router import IntentRouter
from .agents.language_agent import LanguageAgent
from .agents.personalization import PersonalizationAgent
from .agents.rewriter import Rewriter
from .agents.strategy_planner import StrategyPlanner
from .agents.trigger_analyst import TriggerAnalyst
from .agents.validators import FactChecker, PolicyChecker
from .llm import get_llm
from .types import CheckResult, Critique, Draft, StrategyPlan, TraceStep, to_jsonable
from .util import owner_display

MAX_REWRITE_CYCLES = 2


@dataclass
class Candidate:
    draft: Draft
    parts: object
    fact: CheckResult
    policy: CheckResult
    crit: Critique

    @property
    def valid(self) -> bool:
        return self.fact.ok and self.policy.ok


@dataclass
class ComposeResult:
    output: dict
    trace: list[TraceStep]
    extras: dict = field(default_factory=dict)


class Orchestrator:
    def __init__(self, budget_s: float = 24.0) -> None:
        self.budget_s = budget_s

    def analyze(self, category: dict, merchant: dict, trigger: dict, customer: Optional[dict] = None,
                latest_reply: Optional[str] = None, trace: Optional[list] = None) -> dict:
        """Understanding-only pass (no drafting) reused by the multi-turn handler as working memory."""
        trace = trace if trace is not None else []
        ledger, tools, _ = ContextAnalyst(trace).run(category, merchant, trigger, customer)
        ta = TriggerAnalyst(trace).run(tools, ledger)
        prof = CategoryExpert(trace).run(tools, ta)
        pers, pz = PersonalizationAgent(trace).run(tools, ta, prof)
        cust = CustomerAgent(trace).run(tools, ta) if customer else None
        lang = LanguageAgent(trace).run(tools, prof, customer_facing=bool(customer), latest_reply=latest_reply)
        plans = StrategyPlanner(trace).run(ta, IntentRouter(trace).initial(ta), pers, pz, prof, cust, lang.language)
        return {"ledger": ledger, "tools": tools, "ta": ta, "prof": prof, "pz": pz, "cust": cust, "lang": lang,
                "plans": plans, "trace": trace}

    def compose(self, category: dict, merchant: dict, trigger: dict, customer: Optional[dict] = None,
                now: Optional[str] = None, previous_bodies: Optional[list[str]] = None,
                deadline: Optional[float] = None) -> ComposeResult:
        trace: list[TraceStep] = []
        deadline = deadline or (time.monotonic() + self.budget_s)
        state = "CONTEXT"
        log = lambda s, msg: trace.append(TraceStep("orchestrator", f"{s}: {msg}"))
        llm_on = get_llm().enabled

        ledger, tools, sheet = ContextAnalyst(trace).run(category, merchant, trigger, customer)
        state = "TRIGGER"
        ta = TriggerAnalyst(trace).run(tools, ledger, now)
        state = "INTENT"
        intent = IntentRouter(trace).initial(ta)
        state = "CATEGORY"
        prof = CategoryExpert(trace).run(tools, ta)
        state = "PERSONALIZATION"
        pers, pz = PersonalizationAgent(trace).run(tools, ta, prof)
        cust = None
        if customer:
            state = "CUSTOMER"
            cust = CustomerAgent(trace).run(tools, ta)
        else:
            log("CUSTOMER", "skipped (no CustomerContext)")
        state = "LANGUAGE"
        lang = LanguageAgent(trace).run(tools, prof, customer_facing=bool(customer))
        send_as = "merchant_on_behalf" if customer else "vera"

        if cust and not cust.consent_ok:
            # Policy gate before composing: never draft customer outreach without consent.
            return self._consent_block(trace, tools, ta, prof, pz, lang.language, cust, ledger)

        state = "STRATEGY"
        plans = StrategyPlanner(trace).run(ta, intent, pers, pz, prof, cust, lang.language, deadline)
        brief_args = (ta, pz, prof, lang, cust, tools)
        prev = list(previous_bodies or []) + [h.get("body", "") for h in tools.get_conversation_history(20)
                                              if str(h.get("from", "")).lower() in ("vera", "bot", "assistant")]

        state = "COMPOSE"
        composer = MessageComposer(trace)
        fc, pc, critic, rewriter = FactChecker(trace), PolicyChecker(trace), EngagementCritic(trace), Rewriter(trace)
        cphone = tools.get_customer_fact("identity.phone") if customer else None

        def evaluate(d: Draft, parts) -> Candidate:
            f = fc.run(d, ledger, ta, bool(customer))
            p = pc.run(d, ledger, prof, ta, cust, send_as, lang.language, prev, cphone if isinstance(cphone, str) else None)
            c = critic.run(d, ledger, ta, prof, pz, lang.language, f, p, deadline)
            return Candidate(d, parts, f, p, c)

        cands = [evaluate(d, parts) for d, parts in composer.compose_candidates(brief_args, plans)]

        state = "REWRITE"
        for cycle in range(MAX_REWRITE_CYCLES):
            improved = []
            for c in cands:
                if c.valid and not c.crit.weaknesses:
                    continue
                nd = rewriter.run(c.draft, c.parts, c.crit, c.fact, c.policy, prof.taboos, deadline)
                if nd is not None and nd.body not in {x.draft.body for x in cands + improved}:
                    improved.append(evaluate(nd, c.parts))
            log("REWRITE", f"cycle {cycle + 1}: {len(improved)} rewritten candidates")
            if not improved:
                break
            cands += improved
            if any(c.valid and not c.crit.weaknesses for c in cands):
                break

        valid = [c for c in cands if c.valid]
        best = max(valid, key=lambda c: c.crit.total, default=None)

        if llm_on and best and time.monotonic() < deadline - 6:
            state = "LLM_DRAFT"
            ld = composer.llm_draft(brief_args, best.draft.plan, best.draft, deadline)
            if ld is not None:
                lc = evaluate(ld, None)
                if not lc.valid:
                    nd = rewriter.run(ld, None, lc.crit, lc.fact, lc.policy, prof.taboos, deadline)
                    lc = evaluate(nd, None) if nd else lc
                if lc.valid and lc.crit.total >= best.crit.total:
                    log("SELECT", f"LLM draft adopted ({lc.crit.total} ≥ {best.crit.total})")
                    best = lc
                else:
                    log("SELECT", f"LLM draft rejected (valid={lc.valid}, score={lc.crit.total})")

        state = "SUPPRESSION"
        skey = SuppressionAgent(trace).run(tools, ta)
        if best is None:
            state = "SAFE_FALLBACK"
            log(state, "no candidate passed validation → deterministic safe message")
            return self._safe(trace, tools, ta, prof, pz, lang.language, send_as, skey, ledger, cust)

        state = "FINALIZE"
        rationale = self._rationale(ta, best, ledger, prof, lang.language, cust)
        out = Finalizer(trace).run(best.draft, send_as, skey, rationale)
        log(state, f"selected {best.draft.plan.variant} ({best.draft.source}) score={best.crit.total}")
        extras = {"template_name": f"vera_{ta.family}_v1",
                  "template_params": [pz.get("salutation", {}).get("text", ""), ta.why_now, best.draft.segments[-1][1] if best.draft.segments else ""],
                  "scores": best.crit.scores, "family": ta.family, "language": lang.language,
                  "facts_used": [ledger.get(f).path for f in best.draft.used_fact_ids if ledger.get(f)],
                  "candidates": [{"variant": c.draft.plan.variant, "source": c.draft.source, "valid": c.valid,
                                  "score": c.crit.total, "issues": [i.code for i in c.fact.issues + c.policy.issues]} for c in cands],
                  "is_expired": ta.is_expired, "urgency": ta.urgency, "consent_ok": (cust.consent_ok if cust else None)}
        return ComposeResult(out, trace, extras)

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _rationale(ta, best: Candidate, ledger, prof, lang, cust) -> str:
        paths = []
        for fid in best.draft.used_fact_ids:
            f = ledger.get(fid)
            if f and f.layer != "derived":
                paths.append(f.path.replace("category.", "cat.").replace("merchant.", "m.").replace("trigger.", "trg."))
        levers = ", ".join(l.replace("_", " ") for l in best.draft.plan.engagement_levers)
        who = f"customer-facing on behalf of merchant ({cust.consent_reason})" if cust else "merchant-facing"
        r = (f"Why now: {ta.why_now}. Goal: {ta.primary_goal}. Levers: {levers}. "
             f"Anchored on {', '.join(paths[:4]) or 'trigger facts'}. {prof.register} voice, {lang}, {who}. "
             f"All numbers/names verified against context; critic {best.crit.total:.0f}/50.")
        return r

    def _safe(self, trace, tools, ta, prof, pz, lang, send_as, skey, ledger, cust) -> ComposeResult:
        sal = pz.get("salutation", {}).get("text") or owner_display(tools.get_merchant_fact("") or {}, prof.slug)
        if cust:
            first = cust.first_name or ""
            body = {"en": f"Hi {first}, a quick note from {pz.get('name', {}).get('text', 'us')}. Reply YES if you'd like us to book your next visit.",
                    "hi-en": f"Hi {first}, {pz.get('name', {}).get('text', '')} se ek quick note. Agli visit book karni ho toh YES reply karein."}
        else:
            body = {"en": f"{sal}, something relevant to your {prof.noun_singular} came up this week. Want me to share the details? Reply YES.",
                    "hi-en": f"{sal}, is hafte aapke {prof.noun_singular} se related ek update aaya hai. Details bhej doon? Reply YES."}
        text = body.get(lang, body["en"])
        out = {"body": text, "cta": "binary_yes_stop", "send_as": send_as, "suppression_key": skey,
               "rationale": f"Safe fallback: no drafted candidate passed deterministic fact/policy validation for '{ta.trigger_type}'. "
                            f"Sent a minimal, fact-free nudge rather than risk fabrication."}
        assert not validate_output(out)
        trace.append(TraceStep("finalizer", "safe fallback emitted"))
        return ComposeResult(out, trace, {"template_name": "vera_safe_v1", "template_params": [sal], "fallback": True,
                                          "family": ta.family, "is_expired": ta.is_expired, "urgency": ta.urgency})

    def _consent_block(self, trace, tools, ta, prof, pz, lang, cust, ledger) -> ComposeResult:
        """Customer has no valid consent: address the merchant instead (Vera → merchant), never the customer."""
        skey = SuppressionAgent(trace).run(tools, ta) + ":consent"
        sal = pz.get("salutation", {}).get("text", "")
        first = cust.first_name or "this customer"
        body = {"en": f"{sal}, {first}'s follow-up is due, but they haven't opted in to messages for this. "
                      f"Want me to send them a one-time opt-in request from your number instead? Reply YES.",
                "hi-en": f"{sal}, {first} ka follow-up due hai, lekin unhone is tarah ke messages ke liye opt-in nahi kiya hai. "
                         f"Main aapke number se ek one-time opt-in request bhej doon? Reply YES."}
        out = {"body": body.get(lang if lang != "hi" else "hi-en", body["en"]), "cta": "binary_yes_stop", "send_as": "vera",
               "suppression_key": skey,
               "rationale": f"Consent gate: {cust.consent_reason}. Customer outreach blocked; asked merchant to request opt-in instead."}
        trace.append(TraceStep("policy_checker", f"consent blocked customer send: {cust.consent_reason}"))
        return ComposeResult(out, trace, {"template_name": "vera_consent_check_v1", "template_params": [sal, first],
                                          "consent_blocked": True, "family": ta.family, "is_expired": ta.is_expired,
                                          "urgency": ta.urgency, "consent_ok": False})


def trace_to_json(trace: list[TraceStep]) -> list[dict]:
    return [to_jsonable(t) for t in trace]
