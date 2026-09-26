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
python -m pytest -q                                   # 134 tests: all categories/families, adversarial, multi-turn, HTTP
python evaluate.py --dataset dev_fixtures --pairs dev_fixtures/dev_pairs.json --show
python generate_submission.py                         # official: needs ./dataset with the canonical test-pair file
uvicorn bot:app --port 8080 && python scripts/run_judge_offline.py dev_fixtures all
```
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
The official dataset and the 30 canonical pairs. The data host was not reachable from the build environment, so
`dev_fixtures/` are schema-faithful synthetic stand-ins. The next most useful inputs would be real merchant slot
availability, Vera's own plan pricing (so "how much?" could be answered), and review text for richer reputation
triggers.
