"""Vera — agentic merchant-engagement bot for the magicpin AI challenge.

Two surfaces:
  1. `compose(category, merchant, trigger, customer=None) -> dict`   (submission contract)
  2. FastAPI app implementing the judge harness contract: /v1/context, /v1/tick, /v1/reply,
     /v1/healthz, /v1/metadata (+ optional /v1/teardown).
     Run: uvicorn bot:app --host 0.0.0.0 --port 8080

All composition goes through vera.orchestrator.Orchestrator (multi-agent pipeline).
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait as fwait
from datetime import datetime, timezone
from typing import Any, Optional

from vera.agents.finalizer import validate_output
from vera.conversation import ConversationState, ReplyEngine
from vera.llm import get_llm
from vera.orchestrator import Orchestrator
from vera.util import parse_datetime

_ORCH = Orchestrator(budget_s=24.0)


def compose(category: dict, merchant: dict, trigger: dict, customer: Optional[dict] = None) -> dict:
    """Submission entry point. Deterministic for identical inputs; always returns a valid object."""
    try:
        res = _ORCH.compose(category or {}, merchant or {}, trigger or {}, customer)
        out = res.output
    except Exception as e:  # last-resort guard: the contract says always return valid output
        name = ((merchant or {}).get("identity") or {}).get("name") or "there"
        out = {"body": f"Hi {name}, a quick update on your listing is ready. Want me to share it? Reply YES.",
               "cta": "binary_yes_stop", "send_as": "merchant_on_behalf" if customer else "vera",
               "suppression_key": f"{(trigger or {}).get('kind', 'event')}:{(merchant or {}).get('merchant_id', 'm')}:{(trigger or {}).get('id', 't')}",
               "rationale": f"Safe fallback after internal error ({type(e).__name__}); no context facts asserted."}
    assert not validate_output(out)
    return {k: out[k] for k in ("body", "cta", "send_as", "suppression_key", "rationale")}


# =============================================================================== HTTP service
try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError:  # compose() stays importable without the web stack
    FastAPI = None

START = time.time()
VALID_SCOPES = {"category", "merchant", "customer", "trigger"}


class Store:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.reset()

    def reset(self) -> None:
        self.ctx: dict[tuple[str, str], dict] = {}
        self.conversations: dict[str, ConversationState] = {}
        self.sent_keys: set[str] = set()
        self.merchant_memory: dict[str, dict] = {}

    def get(self, scope: str, cid: Optional[str]) -> Optional[dict]:
        if not cid:
            return None
        v = self.ctx.get((scope, cid))
        return v["payload"] if v else None

    def find_merchant(self, mid: Optional[str]) -> Optional[dict]:
        m = self.get("merchant", mid)
        if m or not mid:
            return m
        for (s, _), v in self.ctx.items():
            if s == "merchant" and v["payload"].get("merchant_id") == mid:
                return v["payload"]
        return None

    def category_for(self, merchant: dict) -> dict:
        slug = merchant.get("category_slug") or merchant.get("category")
        return self.get("category", slug) or {"slug": slug or "general"}


STORE = Store()
ENGINE = ReplyEngine()


def _trigger_ids(trg: dict) -> tuple[Optional[str], Optional[str]]:
    p = trg.get("payload") or {}
    return trg.get("merchant_id") or p.get("merchant_id"), trg.get("customer_id") or p.get("customer_id")


def _mm(mid: str) -> dict:
    return STORE.merchant_memory.setdefault(mid or "?", {})


def plan_tick(now: str, trigger_ids: list[str], deadline: float) -> list[dict]:
    """Decide which triggers deserve a send right now; compose them in parallel within budget."""
    now_dt = parse_datetime(now) or datetime.now(timezone.utc)
    best: dict[str, tuple[int, str, dict]] = {}
    for tid in trigger_ids:
        trg = STORE.get("trigger", tid)
        if not trg:
            continue
        mid, cid = _trigger_ids(trg)
        exp = parse_datetime(trg.get("expires_at"))
        if exp and now_dt > exp:
            continue                                   # stale trigger → restraint
        if _mm(mid).get("opted_out"):
            continue                                   # merchant said stop
        key = (mid or "?") + (f"/{cid}" if cid else "")
        urg = int(trg.get("urgency") or 1)
        if key not in best or urg > best[key][0]:
            best[key] = (urg, tid, trg)
    jobs = sorted(best.values(), key=lambda x: -x[0])[:20]

    def work(item):
        urg, tid, trg = item
        mid, cid = _trigger_ids(trg)
        merchant = STORE.find_merchant(mid)
        if not merchant:
            return None
        customer = STORE.get("customer", cid) if cid else None
        if cid and not customer:
            return None
        conv_id = f"conv_{mid}_{tid}" + (f"_{cid}" if cid else "")
        if conv_id in STORE.conversations:
            return None
        category = STORE.category_for(merchant)
        res = _ORCH.compose(category, merchant, trg, customer, now=now, deadline=deadline)
        out, ex = res.output, res.extras
        if ex.get("consent_blocked") or ex.get("fallback"):
            return None                                # don't spam low-value / non-consented sends
        if out["suppression_key"] in STORE.sent_keys:
            return None
        return {"conversation_id": conv_id, "merchant_id": mid, "customer_id": cid, "send_as": out["send_as"],
                "trigger_id": tid, "template_name": ex.get("template_name", "vera_generic_v1"),
                "template_params": [str(p) for p in ex.get("template_params", [])], "body": out["body"],
                "cta": out["cta"], "suppression_key": out["suppression_key"], "rationale": out["rationale"],
                "_ctx": (category, merchant, trg, customer)}

    actions = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(work, j) for j in jobs]
        done, _ = fwait(futs, timeout=max(1.0, deadline - time.monotonic()))
        for f in futs:
            if f in done and f.exception() is None and f.result():
                actions.append(f.result())
    with STORE.lock:
        final = []
        for a in actions:
            if a["suppression_key"] in STORE.sent_keys:
                continue
            STORE.sent_keys.add(a["suppression_key"])
            category, merchant, trg, customer = a.pop("_ctx")
            st = ConversationState(conversation_id=a["conversation_id"], merchant_id=a["merchant_id"],
                                   customer_id=a["customer_id"], trigger_id=a["trigger_id"], category=category,
                                   merchant=merchant, trigger=trg, customer=customer, merchant_memory=_mm(a["merchant_id"]))
            st.record_bot(a["body"], a["cta"])
            STORE.conversations[a["conversation_id"]] = st
            final.append(a)
    return final


def handle_reply(body: dict) -> dict:
    conv_id = body.get("conversation_id") or "conv_unknown"
    with STORE.lock:
        st = STORE.conversations.get(conv_id)
        if st is None:
            mid, cid = body.get("merchant_id"), body.get("customer_id")
            merchant = STORE.find_merchant(mid) or {"merchant_id": mid or "m_unknown", "identity": {}}
            trig = None
            cands = [v["payload"] for (s, _), v in STORE.ctx.items() if s == "trigger" and _trigger_ids(v["payload"])[0] == mid]
            if cands:
                trig = max(cands, key=lambda t: int(t.get("urgency") or 0))
            st = ConversationState(conversation_id=conv_id, merchant_id=mid, customer_id=cid,
                                   trigger_id=(trig or {}).get("id"), category=STORE.category_for(merchant),
                                   merchant=merchant, trigger=trig or {}, customer=STORE.get("customer", cid) if cid else None,
                                   merchant_memory=_mm(mid))
            STORE.conversations[conv_id] = st
    try:
        out = ENGINE.respond(st, body.get("message") or "", body.get("from_role") or "merchant")
    except Exception as e:
        out = {"action": "wait", "wait_seconds": 1800, "rationale": f"Internal error ({type(e).__name__}); backing off safely."}
    if out.get("action") == "end" and st.merchant_intent == "not_interested" and not st.customer:
        _mm(st.merchant_id).update(opted_out=True)
    if out.get("action") == "send" and not (out.get("body") or "").strip():
        out = {"action": "wait", "wait_seconds": 1800, "rationale": "Empty draft suppressed."}
    return out


if FastAPI is not None:
    app = FastAPI(title="Vera agentic bot")

    class CtxBody(BaseModel):
        scope: str
        context_id: str
        version: int
        payload: dict[str, Any]
        delivered_at: Optional[str] = None

    class TickBody(BaseModel):
        now: str
        available_triggers: list[str] = []

    class ReplyBody(BaseModel):
        conversation_id: str
        merchant_id: Optional[str] = None
        customer_id: Optional[str] = None
        from_role: str = "merchant"
        message: str = ""
        received_at: Optional[str] = None
        turn_number: Optional[int] = None

    @app.get("/v1/healthz")
    def healthz():
        counts = {s: 0 for s in VALID_SCOPES}
        for (s, _) in list(STORE.ctx):
            counts[s] = counts.get(s, 0) + 1
        return {"status": "ok", "uptime_seconds": int(time.time() - START), "contexts_loaded": counts}

    @app.get("/v1/metadata")
    def metadata():
        llm = get_llm()
        return {"team_name": "Vera Agentic", "team_members": ["Nishant"],
                "model": llm.model if llm.enabled else "deterministic-agents (LLM optional)",
                "approach": "15-agent orchestrated pipeline: fact-ledger context analyst, trigger analyst, intent router, "
                            "strategy planner, category expert, personalization, customer, language, composer (multi-candidate), "
                            "deterministic fact + policy checkers, engagement critic, rewriter, finalizer; stateful multi-turn with auto-reply detection",
                "contact_email": "nishantaws01@gmail.com", "version": "1.0.0", "submitted_at": "2026-09-26T00:00:00Z"}

    @app.post("/v1/context")
    def push_context(body: CtxBody):
        if body.scope not in VALID_SCOPES:
            return JSONResponse(status_code=400, content={"accepted": False, "reason": "invalid_scope", "details": body.scope})
        key = (body.scope, body.context_id)
        with STORE.lock:
            cur = STORE.ctx.get(key)
            if cur and cur["version"] > body.version:
                return JSONResponse(status_code=409, content={"accepted": False, "reason": "stale_version", "current_version": cur["version"]})
            if not (cur and cur["version"] == body.version):
                STORE.ctx[key] = {"version": body.version, "payload": body.payload}
                if body.scope == "merchant" and body.payload.get("merchant_id") and body.payload["merchant_id"] != body.context_id:
                    STORE.ctx[("merchant", body.payload["merchant_id"])] = STORE.ctx[key]
        return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}",
                "stored_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}

    @app.post("/v1/tick")
    def tick(body: TickBody):
        deadline = time.monotonic() + 25.0
        try:
            return {"actions": plan_tick(body.now, body.available_triggers, deadline)}
        except Exception:
            return {"actions": []}

    @app.post("/v1/reply")
    def reply(body: ReplyBody):
        return handle_reply(body.model_dump())

    @app.post("/v1/teardown")
    def teardown():
        with STORE.lock:
            STORE.reset()
        return {"ok": True}
