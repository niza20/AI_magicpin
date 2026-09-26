"""AGENT 14 — Rewriter. Repairs ONLY the weak segments named by the critic/validators, using parts
the realizer already produced from ledger facts (so it can't introduce new facts). An LLM rewrite is
attempted only for LLM-sourced drafts and is re-validated like any other draft."""
from __future__ import annotations

import re
from typing import Optional

from ..llm import get_llm
from ..types import CheckResult, Critique, Draft
from .base import Agent
from .composer import cta_value_for


class Rewriter(Agent):
    name = "rewriter"
    uses_llm = True

    def run(self, draft: Draft, parts, crit: Critique, fact: CheckResult, policy: CheckResult,
            taboos: list[str], deadline: Optional[float] = None) -> Optional[Draft]:
        if draft.source == "llm":
            return self._llm_fix(draft, crit, fact, policy, deadline)
        if parts is None:
            return None
        d = Draft(segments=list(draft.segments), cta=draft.cta, plan=draft.plan,
                  used_fact_ids=list(draft.used_fact_ids), source="deterministic+rewrite")
        ops: list[str] = []
        codes = {i.code for i in fact.issues + policy.issues}
        present = {t for _, t in d.segments}
        levers_unused = [(k, v) for k, v in parts.levers.items() if v not in present]

        if "fabrication" in crit.weaknesses or not fact.ok:
            # drop the offending lever/anchor segments; keep hook + CTA
            bad = [i.detail for i in fact.errors]
            keep = []
            for role, text in d.segments:
                if role != "hook" and role != "cta" and any(_mentions(text, b) for b in bad):
                    ops.append(f"drop {role}")
                    continue
                keep.append((role, text))
            d.segments = keep
        if "specificity" in crit.weaknesses or "merchant_fit" in crit.weaknesses:
            numeric = [(k, v) for k, v in levers_unused if re.search(r"\d", v)]
            if numeric:
                k, v = numeric[0]
                self._insert_before_cta(d, (f"lever:{k}", v))
                ops.append(f"add {k} fact")
        if "engagement" in crit.weaknesses or "buried_cta" in codes or "multi_cta" in codes:
            for key in ("effort", "confirm", "slot"):
                if key in parts.ctas:
                    d.replace("cta", parts.ctas[key])
                    d.segments = [s for s in d.segments if s[0] != "cta"] + [("cta", parts.ctas[key])]
                    d.cta = cta_value_for(parts.ctas[key])
                    ops.append(f"cta→{key}")
                    break
        if "too_long" in codes or "long" in codes:
            levers = [i for i, (r, _) in enumerate(d.segments) if r.startswith("lever:")]
            for i in reversed(levers[1:]):
                d.segments.pop(i)
            ops.append("trim levers")
            if len(d.body) > 520:
                d.segments = [(r, t if r != "anchor" or len(t) < 120 else t[:117].rsplit(" ", 1)[0] + "…") for r, t in d.segments]
                ops.append("trim anchor")
        if "taboo" in codes or "hype" in codes:
            new = []
            for r, t in d.segments:
                for tb in taboos:
                    t = re.sub(r"(?i)(?<!\w)" + re.escape(tb) + r"(?!\w)", "", t)
                new.append((r, t.replace("!!", ".").replace("!", ".")))
            d.segments = new
            ops.append("strip taboo/hype")
        if not ops or d.body == draft.body:
            self.log("no applicable rewrite ops")
            return None
        self.log("ops: " + ", ".join(ops))
        return d

    @staticmethod
    def _insert_before_cta(d: Draft, seg) -> None:
        idx = next((i for i, (r, _) in enumerate(d.segments) if r == "cta"), len(d.segments))
        d.segments.insert(idx, seg)

    def _llm_fix(self, draft: Draft, crit: Critique, fact: CheckResult, policy: CheckResult, deadline) -> Optional[Draft]:
        llm = get_llm()
        if not llm.enabled:
            return None
        problems = [f"{i.code}: {i.detail}" for i in fact.issues + policy.issues] + [f"weak {w}" for w in crit.weaknesses]
        out = llm.complete_json(
            "rewriter",
            "You fix a WhatsApp message. Change only what the problem list requires. Never add numbers, names, dates or "
            "offers that are not already in the message. Keep one CTA as the final sentence. JSON only.",
            f"Message: {draft.body}\nProblems: {problems}\nReturn {{\"body\": \"...\"}}", deadline=deadline, max_tokens=1200)
        if not out or not isinstance(out.get("body"), str):
            return None
        self.log("llm rewrite applied")
        return Draft(segments=[("llm", out["body"].strip())], cta=draft.cta, plan=draft.plan,
                     used_fact_ids=draft.used_fact_ids, source="llm")


def _mentions(text: str, detail: str) -> bool:
    m = re.search(r"'([^']+)'|(\d[\d.,]*)", detail)
    token = (m.group(1) or m.group(2)) if m else None
    return bool(token and token.lower() in text.lower())
