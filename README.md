# Vera: an agentic merchant-engagement bot (magicpin AI Challenge)

`compose(category, merchant, trigger, customer=None) → {body, cta, send_as, suppression_key, rationale}`, plus the
5-endpoint judge contract (`uvicorn bot:app --port 8080`). This is not a single prompt. **15 specialised agents** run
under a deterministic orchestrator, and every word that reaches a merchant is checked by deterministic validators.

## Approach
```
CONTEXT ANALYST → TRIGGER ANALYST → INTENT ROUTER → CATEGORY EXPERT → PERSONALIZATION → [CUSTOMER] → LANGUAGE
   → STRATEGY PLANNER (2-3 lever plans) → COMPOSER (N candidates [+ LLM draft]) → FACT CHECKER → POLICY CHECKER
   → ENGAGEMENT CRITIC → REWRITER (≤2 cycles, re-validated) → SELECT → SUPPRESSION → FINALIZER  (or SAFE FALLBACK)
multi-turn: reply → AUTO-REPLY DETECTOR → INTENT ROUTER → ConversationState → response strategy → validators → send|wait|end
```
* **Fact ledger.** Every scalar in the 4 contexts becomes a fact with its source path
  (`merchant.performance.ctr = 0.021`). Derived facts such as "4 months since last visit" record the paths they came
  from. Drafts can only reference ledger facts.
* **Deterministic Fact Checker.** Every number, price, %, date/weekday/month, capitalised name, quoted headline, URL,
  discount, competitor, citation and "N peers did X" claim must trace back to the ledger, or the draft fails and
  never reaches the finalizer.
* **Policy Checker.** Checks consent scope versus message purpose, `send_as`, taboos from `voice`, hype, a single CTA
  placed last, preambles, self-intros, PII, length, verbatim repeats, and language match.
* **Strategy then composition.** The planner picks 1-3 levers per plan, and only when a fact supports them (social
  proof needs a peer stat; loss aversion needs a gap, dip, expiry or lapse). The composer realizes each plan in
  en / hi-en / hi. The critic scores all 5 judge dimensions and names weaknesses, and the rewriter repairs only
  those segments.
* **Multi-turn.** Explicit intent ("yes", "go ahead", "lets do it", "haan karo") triggers **ACT** immediately and
  delivers the artifact: a draft post, checklist, review replies or summary. Auto-replies get one owner-directed
  nudge, then exit; detection is tracked per merchant, so it also catches repeats across conversation ids. STOP and
  not-interested end the conversation. Hostile or off-topic replies get one polite redirect. A price question gets
  "I won't guess". "Later" returns `wait`. Language is re-detected every turn.
* **Hidden-test resilience.** Nothing is keyed on IDs, names or sample text. Trigger families are inferred from
  `kind` substrings, with a generic path for unseen kinds. Payload keys are read through alias lists, and digest
  items are resolved by id or by relevance.

## Run
```bash
pip install -r requirements.txt
cd dataset/seed && python generate_dataset.py --out ../expanded && cd ../..   # magicpin's own deterministic expander
python generate_submission.py --dataset dataset/expanded        # → submission.jsonl (the 30 canonical pairs T01-T30)
python -m pytest -q                                             # 239 tests (all 100 official triggers + adversarial + multi-turn + HTTP)
python evaluate.py --dataset dataset/expanded --show            # internal 5-dimension critic + replay scenarios
uvicorn bot:app --port 8080 && python scripts/run_judge_offline.py dataset/expanded all
```
**Results on the official data:** all 30 pairs, and all 100 triggers, compose with 0 safe-fallbacks and 0 crashes.
Each body is re-verified fact-by-fact against the contexts. magicpin's simulator passes auto-reply, intent-transition and
hostile, with ticks under 40 ms. About half the test pairs are *placeholder* triggers (payload `{"placeholder": true}`).
For those, Vera uses only the merchant's own data and never invents a festival, competitor or drop. For example, T25 is
a "perf_dip" whose deltas are actually positive, so it reports the real risk: the plan lapsed 39 days ago.

The LLM layer is optional (`ANTHROPIC_API_KEY`; model via `VERA_LLM_MODEL`, default `claude-opus-5`). It drafts,
ranks plans, classifies unclear intents and critiques. Its output is always re-validated by deterministic code. It
is cached by content hash so results are deterministic, deadline-bounded to stay under 30 s, and falls back to the
deterministic agents on any failure.

## Tradeoffs
* Deterministic realizers are the default because the judge penalises fabrication harder than it rewards flourish.
  Fixed phrasing is less varied, and the LLM path restores variety only where the validators allow it.
* The fact checker is strict. Unknown capitalised words and numbers not in the context are rejected. Occasional
  false rejections fall back to a safer candidate rather than ever shipping an invented fact.
* The critic is a heuristic proxy for the LLM judge. It is useful for ranking candidates and finding weak
  dimensions, but it is not ground truth.

## What would have helped most
Real payloads for the placeholder triggers (half the test set), actual review text for `review_theme_emerged`, open
slots for customer bookings, and Vera's own plan pricing so "how much?" could be answered instead of deferred.
A consistent "now" timestamp on triggers would also help: here it is inferred from `date − days_until` or the ISO week
in the suppression key. `dev_fixtures/` holds extra synthetic edge cases used by the tests.

**Verify against the brief:** `python scripts/verify_brief.py` runs 79 checks (HTTP contract, context injection, replay scenarios, anti-patterns, deliverables) on the official dataset.
