# Architecture: agents, contracts, memory

| # | Agent (file) | Responsibility | Input → Output | LLM? |
|---|---|---|---|---|
| 1 | Context Analyst (`agents/context_analyst.py`) | Builds the **FactLedger** and a bucketed fact sheet (category / merchant / trigger / customer / performance / offer / peer / conversation facts, available actions). Never infers. | 4 raw contexts → `FactLedger`, `ContextTools`, `ContextFacts` | no |
| 2 | Trigger Analyst (`agents/trigger_analyst.py`) | *Why now?* Classifies `kind` into a family (unseen kinds → `generic`), resolves the **anchor** bundle (most important trigger facts, each with a ledger id), urgency, goal, action, expiry. | tools, ledger → `TriggerAnalysis` | no |
| 3 | Intent Router (`agents/intent_router.py`) | Initial mode (DISCOVER/INFORM/RECOMMEND/ACT…); per-turn merchant intent (explicit_action overrides qualification). | trigger / message → `IntentResult` | only for `unclear` |
| 4 | Auto-Reply Detector (`agents/intent_router.py`) | Canned phrasing (EN + Hinglish), verbatim repetition (conversation + merchant memory). Clarify once, then exit. | message, history → `AutoReplyVerdict` | no |
| 5 | Strategy Planner (`agents/strategy_planner.py`) | 2-3 candidate plans: objective, hook, facts, **1-3 fact-gated levers**, CTA shape, structure. | analysis → `[StrategyPlan]` | optional ranking |
| 6 | Category Expert (`agents/category_expert.py`) | Register from `voice.tone`, vocabulary, taboos (+ global legal taboos), offer style, peer label (never implies another city is local). | category → `CategoryProfile` | no |
| 7 | Personalization (`agents/personalization.py`) | Ranks merchant facts by family (CTR vs peer, offers, stale posts, cohort signals, lapsed counts…). | tools → `Personalization`, `pz` | no |
| 8 | Customer Agent (`agents/customer_agent.py`) | Runs only with CustomerContext. Checks consent scope vs purpose, uses minimal facts, `send_as=merchant_on_behalf`. | tools → `CustomerPlan` | no |
| 9 | Language Agent (`agents/language_agent.py`) | en / hi-en / hi from the latest reply → customer pref → merchant `languages`. | → `LanguagePlan` | no |
| 10 | Message Composer (`agents/composer.py`) | Realizes each plan (HOOK → FACT → WHY → CTA) from ledger-backed parts. Optional LLM draft from a fact **whitelist**. | plans → `[Draft]` | optional |
| 11 | Fact Checker (`agents/validators.py`) | Deterministic: numbers, ₹, %, dates, weekdays, names, quotes, URLs, discounts, competitors, citations, social-proof counts. | draft, ledger → `CheckResult` | **never** |
| 12 | Policy Checker (`agents/validators.py`) | Consent, send_as, taboos, hype, single last CTA, preamble, self-intro, PII, length, repeats, language. | draft → `CheckResult` | **never** |
| 13 | Engagement Critic (`agents/critic.py`) | Scores the 5 judge dimensions 0-10 and names weaknesses. | draft → `Critique` | optional note |
| 14 | Rewriter (`agents/rewriter.py`) | Fixes only weak/failed segments using existing ledger-backed parts; ≤2 cycles; re-validated. | draft + critique → `Draft` | LLM drafts only |
| 15 | Finalizer + Suppression (`agents/finalizer.py`) | Deterministic `family:category:merchant[:customer]:ISO-week` key (or the trigger's key, scoped to the merchant); final schema validation. | → output dict | no |

**Orchestrator** (`vera/orchestrator.py`): a state machine. It skips the Customer agent without a customer, skips
LLM steps without a key or time budget, can never finalize a draft that failed validation, and falls back to a
fact-free safe message if nothing passes. A consent failure short-circuits to a merchant-facing opt-in suggestion.

**Memory.** *Short-term* is `ConversationState` (turns, intents, CTAs used, facts mentioned, auto-reply count,
actions, exit state). *Long-term* is the supplied contexts, read-only. *Working* is the per-turn ledger, anchor and
personalization, rebuilt each turn. The LLM cannot create persistent facts.

**Tools** (`vera/tools.py`): `get_merchant_fact`, `get_category_fact`, `get_trigger_fact`, `get_customer_fact`,
`search_category_digest`, `get_peer_stat`, `get_active_offers`, `get_conversation_history`, `perf_metric`,
`perf_delta`. Each call is logged. Agents receive tools, not the dataset.

**HTTP tick policy** (`bot.py`): skip expired triggers, opted-out merchants, customers without consent, low-value
safe fallbacks and already-sent suppression keys. Sends at most one action per merchant or customer per tick and at
most 20 per tick. Composes in parallel under a 25 s deadline.
