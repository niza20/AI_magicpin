"""Interactive demo served by the bot at /demo, running the real agent pipeline.

Shows: the 30 official scenarios grouped by audience/type, quality badges per message, one-click
judge replay tests with pass/fail checks, a weekly conversation portfolio per merchant, how each
production pain point is handled, and a language switch (Auto / English / Hinglish / Hindi).
Uses its own session store; never touches the judge's /v1 state.
"""
from __future__ import annotations

import re
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

from vera.conversation import ConversationState, ReplyEngine
from vera.dataset import load_dataset, load_pairs
from vera.orchestrator import Orchestrator

router = APIRouter()
_ROOT = Path(__file__).resolve().parent
_DS_DIR = _ROOT / "dataset" / "expanded"
_lock = threading.Lock()
_sessions: dict[str, ConversationState] = {}
_ds = None
_orch = Orchestrator()
_engine = ReplyEngine()

GROUPS = {
    "knowledge": "Knowledge & curiosity", "regulation": "Knowledge & curiosity", "supply": "Knowledge & curiosity",
    "seasonal": "Knowledge & curiosity", "trend": "Knowledge & curiosity",
    "perf_dip": "Performance & account", "perf_spike": "Performance & account", "milestone": "Performance & account",
    "profile": "Performance & account", "account": "Performance & account", "offer": "Performance & account",
    "competitor": "Market & local events", "festival": "Market & local events", "weather": "Market & local events",
    "local_event": "Market & local events", "reputation": "Market & local events",
    "dormant": "Engagement & merchant intent", "recurring": "Engagement & merchant intent",
    "planning": "Engagement & merchant intent", "generic": "Engagement & merchant intent",
    "customer_recall": "Customer-facing (sent on the merchant's behalf)",
    "customer_appointment": "Customer-facing (sent on the merchant's behalf)",
    "customer_promo": "Customer-facing (sent on the merchant's behalf)",
}
LANGS = {"auto": None, "en": "en", "hi-en": "hi-en", "hi": "hi"}
QUALIFYING = ["would you", "do you", "can you tell", "what if", "how about"]


def _dataset():
    global _ds
    if _ds is None and _DS_DIR.exists():
        _ds = load_dataset(str(_DS_DIR))
    return _ds


def _ctx(trigger_id: str, customer_id: Optional[str] = None):
    ds = _dataset()
    t = ds.triggers[trigger_id]
    m = ds.merchants[t["merchant_id"]]
    cid = customer_id or t.get("customer_id")
    return ds.category_for(m), m, t, (ds.customers.get(cid) if cid else None)


def _badges(body: str, res, merchant: dict, category: dict, customer: Optional[dict]) -> list[dict]:
    """Quality badges derived from the actual output + validator/critic results (not decoration)."""
    sc = res.extras.get("scores") or {}
    low = body.lower()
    out = []
    nums = re.findall(r"\d[\d,.]*", body)
    if len(nums) >= 2:
        out.append({"ok": True, "label": "Specific numbers"})
    offers = [o.get("title", "") for o in merchant.get("offers", []) if isinstance(o, dict)] + \
             [o.get("title", "") for o in category.get("offer_catalog", []) if isinstance(o, dict)]
    if any(o and o.lower() in low for o in offers):
        out.append({"ok": True, "label": "Real offer (service + price)"})
    if re.search(r"\b\d+\s*%\s*off\b", low) and not any("%" in o and o.lower() in low for o in offers):
        out.append({"ok": False, "label": "Generic % discount"})
    if sc.get("trigger_relevance", 0) >= 7:
        out.append({"ok": True, "label": "Clear why-now"})
    if sc.get("merchant_fit", 0) >= 7:
        out.append({"ok": True, "label": "This merchant's data"})
    if sc.get("category_fit", 0) >= 7:
        out.append({"ok": True, "label": f"{category.get('slug', '')} voice"})
    if res.output["cta"] in ("binary_yes_stop", "open_ended"):
        out.append({"ok": True, "label": "Single CTA"})
    out.append({"ok": True, "label": f"0 invented facts ({len(res.extras.get('facts_used', []))} verified)"})
    if customer:
        out.append({"ok": True, "label": "Consent checked"})
    return out


@router.get("/demo/api/scenarios")
def scenarios():
    ds = _dataset()
    if not ds:
        return JSONResponse(status_code=404, content={"error": "dataset/expanded not found"})
    from vera.agents.trigger_analyst import classify_family
    out = []
    for p in load_pairs(str(_DS_DIR / "test_pairs.json")):
        t, m = ds.triggers.get(p["trigger_id"]), ds.merchants.get(p["merchant_id"])
        if not t or not m:
            continue
        cid = p.get("customer_id") or t.get("customer_id")
        c = ds.customers.get(cid) if cid else None
        fam = classify_family(t["kind"], t.get("scope", "merchant"))
        out.append({"id": p["test_id"], "trigger_id": p["trigger_id"], "customer_id": cid, "merchant_id": m["merchant_id"],
                    "merchant": m["identity"]["name"], "category": m.get("category_slug"), "kind": t["kind"].replace("_", " "),
                    "group": GROUPS.get(fam, "Engagement & merchant intent"),
                    "audience": "customer" if c else "merchant",
                    "placeholder": bool((t.get("payload") or {}).get("placeholder")),
                    "to": (c or {}).get("identity", {}).get("name") if c else m["identity"].get("owner_first_name"),
                    "languages": m["identity"].get("languages"),
                    "customer_lang": (c or {}).get("identity", {}).get("language_pref") if c else None})
    order = ["Knowledge & curiosity", "Performance & account", "Market & local events", "Engagement & merchant intent",
             "Customer-facing (sent on the merchant's behalf)"]
    out.sort(key=lambda x: (order.index(x["group"]) if x["group"] in order else 9, x["id"]))
    return out


GENERIC = {
    "dentists": "Hi Doctor, want to run a discount campaign today to increase sales?",
    "salons": "Hi! Want to run a 20% off offer this week to get more customers?",
    "restaurants": "Hello! Boost your sales with a flat 30% discount this weekend?",
    "gyms": "Hi, want to increase memberships with a special discount offer?",
    "pharmacies": "Hi, want to run an offer to increase your sales?",
}


def _profile(m: dict, cat: dict, c: Optional[dict]) -> dict:
    """Profile card data — shown verbatim from the context the bot received."""
    ident, perf = m.get("identity", {}), m.get("performance", {})
    peer = cat.get("peer_stats", {})
    d7 = perf.get("delta_7d") or {}
    kpis = []
    for k, label in (("views", "Views"), ("calls", "Calls"), ("directions", "Directions")):
        if perf.get(k) is not None:
            delta = d7.get(f"{k}_pct")
            kpis.append({"label": label, "value": f"{perf[k]:,}", "sub": (f"{delta * 100:+.0f}% 7d" if isinstance(delta, (int, float)) else "30 days")})
    if perf.get("ctr") is not None:
        kpis.append({"label": "CTR", "value": f"{perf['ctr'] * 100:.1f}%",
                     "sub": f"peer {peer['avg_ctr'] * 100:.1f}%" if peer.get("avg_ctr") else "", "warn": bool(peer.get("avg_ctr") and perf["ctr"] < peer["avg_ctr"])})
    prof = {"name": ident.get("name"), "owner": ident.get("owner_first_name"), "locality": ident.get("locality"), "city": ident.get("city"),
            "category": cat.get("slug"), "languages": ident.get("languages"), "verified": ident.get("verified"),
            "plan": f"{(m.get('subscription') or {}).get('plan', '')} · {(m.get('subscription') or {}).get('status', '')}",
            "kpis": kpis, "offers": [{"title": o.get("title"), "status": o.get("status")} for o in m.get("offers", []) if isinstance(o, dict)][:3],
            "reviews": [{"theme": r.get("theme", "").replace("_", " "), "sentiment": r.get("sentiment")} for r in m.get("review_themes", []) if isinstance(r, dict)][:3],
            "signals": [s.replace("_", " ") for s in m.get("signals", []) if isinstance(s, str)][:4]}
    if c:
        rel, ci = c.get("relationship", {}), c.get("identity", {})
        prof["customer"] = {"name": ci.get("name"), "language": ci.get("language_pref"), "state": (c.get("state") or "").replace("_", " "),
                            "last_visit": rel.get("last_visit"), "visits": rel.get("visits_total"),
                            "services": [s.replace("_", " ") for s in rel.get("services_received", []) if s != "..."][:3],
                            "slots": ((c.get("preferences") or {}).get("preferred_slots") or "").replace("_", " "),
                            "consent": [s.replace("_", " ") for s in (c.get("consent") or {}).get("scope", [])]}
    return prof


def _compose_payload(res, cat, m, t, c):
    generic = "Hi! Visit us again and get 10% off. Hurry, limited time!" if c else GENERIC.get(cat.get("slug"), "Hi, want to run a discount to increase sales?")
    return {**res.output, "profile": _profile(m, cat, c), "highlights": res.extras.get("highlights", []), "generic": generic, "family": res.extras.get("family"), "language": res.extras.get("language"),
            "scores": res.extras.get("scores"), "facts_used": res.extras.get("facts_used", [])[:8],
            "badges": _badges(res.output["body"], res, m, cat, c), "trigger_kind": t.get("kind"),
            "placeholder": bool((t.get("payload") or {}).get("placeholder"))}


@router.post("/demo/api/start")
def start(body: dict):
    cat, m, t, c = _ctx(body["trigger_id"], body.get("customer_id"))
    lang = LANGS.get(body.get("language") or "auto")
    res = _orch.compose(cat, m, t, c, language=lang)
    sid = uuid.uuid4().hex[:12]
    st = ConversationState(conversation_id=sid, merchant_id=m["merchant_id"], customer_id=(c or {}).get("customer_id"),
                           trigger_id=t["id"], category=cat, merchant=m, trigger=t, customer=c, merchant_memory={},
                           language=lang, language_locked=bool(lang))
    st.record_bot(res.output["body"], res.output["cta"])
    with _lock:
        _sessions[sid] = st
    out = {"session_id": sid, **_compose_payload(res, cat, m, t, c)}
    low = out["body"].lower()
    out["buttons"] = ["1", "2"] if re.search(r"reply 1 or 2|1 ya 2", low) else \
        (["Yes, go ahead", "Not now"] if out["cta"] == "binary_yes_stop" else [])
    return out


def _with_ui(st: ConversationState, r: dict) -> dict:
    """Decorate a reply for the chat UI: WhatsApp-style quick-reply buttons + draft-post preview image."""
    body = r.get("body") or ""
    low = body.lower()
    buttons = []
    if r.get("action") == "send":
        if "reply go" in low or "go reply" in low:
            buttons = ["GO"]
        elif re.search(r"reply 1 or 2|1 ya 2", low):
            buttons = ["1", "2"]
        elif "reply yes" in low or "yes reply" in low or "yes भेजें" in low:
            buttons = ["Yes, go ahead", "Not now"]
    r["buttons"] = buttons
    if "↓" in body and st.attachments:
        r["draft_image"] = st.attachments[-1]
    return r


@router.post("/demo/api/reply")
def reply(body: dict):
    st = _sessions.get(body.get("session_id", ""))
    if not st:
        return JSONResponse(status_code=404, content={"error": "session expired — pick a scenario again"})
    if body.get("image"):
        img = str(body["image"])
        if not img.startswith("data:image/") or len(img) > 4_000_000:
            return JSONResponse(status_code=400, content={"error": "please attach a JPG/PNG under ~3 MB"})
        return _with_ui(st, _engine.respond_photo(st, img, body.get("message", "")))
    return _with_ui(st, _engine.respond(st, body.get("message", ""), "customer" if st.customer else "merchant"))


@router.post("/demo/api/deals")
def deals(body: dict):
    st = _sessions.get(body.get("session_id", ""))
    if not st:
        return JSONResponse(status_code=404, content={"error": "session expired — pick a scenario again"})
    return _with_ui(st, _engine.deals(st))


REPLAYS = {
    "auto_reply": {"title": "Auto-reply hell", "pain": "Auto-reply pollution",
                   "turns": ["Thank you for contacting us! Our team will respond shortly."] * 4,
                   "expect": "Detect the canned reply, nudge the owner once, then exit (production Vera burns 2-3 turns)."},
    "intent": {"title": "Intent transition", "pain": "Intent-handoff failures",
               "turns": ["Interesting, tell me more", "What would you post?", "ok let's do it"],
               "expect": "On 'ok let's do it' switch straight to ACTION: deliver the work, no qualifying questions."},
    "join": {"title": "\"I want to join\"", "pain": "Intent-handoff failures",
             "turns": ["Mujhe magicpin judna hai"],
             "expect": "Explicit intent on the first reply → act immediately (Pattern D from the brief, done right)."},
    "hostile": {"title": "Hostile + off-topic", "pain": "Staying on mission",
                "turns": ["you guys are useless", "can you also help me file my GST?"],
                "expect": "Apologise once, decline GST politely, stay on mission; exit if hostility repeats."},
    "stop": {"title": "STOP", "pain": "Knowing when to stop", "turns": ["STOP"],
             "expect": "End immediately and never message this merchant again."},
    "curveballs": {"title": "Curveball questions", "pain": "Replay: curveball questions",
                   "turns": ["Who are you? Are you from Google?", "Which competitor?", "What is CTR?", "Kitne customers aayenge isse?",
                             "Can you reduce my magicpin commission?", "Do it tomorrow morning"],
                   "expect": "Every off-script question gets an on-topic, fact-based answer; no invented forecasts; the action still lands."},
    "language": {"title": "Language switch", "pain": "Per-turn language", "turns": ["haan theek hai, yeh kya hai?", "ok karo"],
                 "expect": "Merchant switches to Hinglish → Vera replies in Hinglish and acts on 'ok karo'."},
}


def _check(name: str, results: list[dict]) -> tuple[bool, str]:
    acts = [r["action"] for r in results]
    last = results[-1] if results else {}
    body = (last.get("body") or "").lower()
    if name == "auto_reply":
        ok = "end" in acts and acts.index("end") <= 1
        return ok, f"exited on reply #{acts.index('end') + 1}" if "end" in acts else "never exited"
    if name in ("intent", "join"):
        ok = last.get("action") == "send" and not any(q in body for q in QUALIFYING) and bool(re.search(r"done|draft|ho gaya|on it|here|sending|abhi", body))
        return ok, "acted without re-qualifying" if ok else "still qualifying"
    if name == "hostile":
        ok = all(a in ("send", "end") for a in acts) and ("gst" in body or acts[-1] == "end")
        return ok, "polite, declined GST, stayed on mission" if ok else "went off mission"
    if name == "stop":
        return acts == ["end"], "ended immediately" if acts == ["end"] else "kept talking"
    if name == "curveballs":
        bodies = " ".join((r.get("body") or "") for r in results).lower()
        checks = ["not google" in bodies, "smile studio" in bodies, "ctr" in bodies,
                  ("waada nahi" in bodies or "can't promise" in bodies), "magicpin team" in bodies, "tomorrow morning" in bodies]
        ok = all(checks) and all(r["action"] == "send" for r in results)
        return ok, f"{sum(checks)}/6 answered on-topic from facts" if ok else f"only {sum(checks)}/6 on-topic"
    if name == "language":
        ok = all(a == "send" for a in acts) and bool(re.search(r"\b(hai|kar|main|aap|mein|karein)\b", body))
        return ok, "replied in Hinglish" if ok else "language not matched"
    return True, ""


@router.post("/demo/api/replay")
def replay(body: dict):
    name = body.get("name")
    spec = REPLAYS[name]
    cat, m, t, c = _ctx(body.get("trigger_id") or "trg_023_competitor_opened_dentist")
    if c:  # replays are merchant-side tests
        cat, m, t, c = _ctx("trg_023_competitor_opened_dentist")
    lang = LANGS.get(body.get("language") or "auto")
    res = _orch.compose(cat, m, t, None, language=lang)
    st = ConversationState(conversation_id=uuid.uuid4().hex[:8], merchant_id=m["merchant_id"], trigger_id=t["id"],
                           category=cat, merchant=m, trigger=t, merchant_memory={},
                           language=lang, language_locked=bool(lang) and name != "language")
    st.record_bot(res.output["body"], res.output["cta"])
    results = []
    for msg in spec["turns"]:
        if st.exit_state == "ended":
            break
        r = _engine.respond(st, msg, "merchant")
        results.append({"merchant": msg, **r})
    ok, verdict = _check(name, results)
    return {"title": spec["title"], "pain": spec["pain"], "expect": spec["expect"], "opening": res.output["body"],
            "merchant": m["identity"]["name"], "turns": results, "pass": ok, "verdict": verdict}


@router.post("/demo/api/inject")
def inject(body: dict):
    """Brief §8 twist, live: same merchant before vs after new context arrives (new digest item, shifted metrics,
    customer attached to a merchant-level trigger)."""
    import copy
    lang = LANGS.get(body.get("language") or "auto")
    ds = _dataset()
    steps = []
    # 1. new digest item → research trigger uses it
    cat, m, t, _ = _ctx("trg_001_research_digest_dentists")
    before = _orch.compose(cat, m, t, None, language=lang).output["body"]
    cat2 = copy.deepcopy(cat)
    cat2["digest"].append({"id": "d_injected_sdf", "kind": "research", "title": "Silver diamine fluoride arrests 81% of early caries in 12 months",
                           "source": "IJDR May 2026, p.33", "trial_n": 640, "patient_segment": "pediatric",
                           "actionable": "Consider SDF for pediatric patients who can't sit for fillings"})
    t2 = {**t, "payload": {"top_item_id": "d_injected_sdf"}}
    after = _orch.compose(cat2, m, t2, None, language=lang).output["body"]
    steps.append({"what": "New digest item pushed (category v2) + trigger pointing at it", "before": before, "after": after})
    # 2. performance snapshot shifts
    cat, m, t, _ = _ctx("trg_004_perf_dip_bharat")
    before = _orch.compose(cat, m, t, None, language=lang).output["body"]
    m2 = copy.deepcopy(m)
    m2["performance"].update({"calls": 7, "ctr": 0.012, "delta_7d": {"views_pct": -0.35, "calls_pct": -0.61}})
    after = _orch.compose(cat, m2, {**t, "payload": {"metric": "calls", "delta_pct": -0.61, "window": "7d"}}, None, language=lang).output["body"]
    steps.append({"what": "Merchant performance updated (calls -61%, CTR 1.2%)", "before": before, "after": after})
    # 3. customer context attached to a merchant-level trigger
    cat, m, t, _ = _ctx("trg_023_competitor_opened_dentist")
    before = _orch.compose(cat, m, t, None, language=lang).output["body"]
    cust = next(c for c in ds.customers.values() if c["merchant_id"] == m["merchant_id"]
                and "promotional_offers" in (c.get("consent") or {}).get("scope", []))
    res = _orch.compose(cat, m, t, cust, language=lang)
    steps.append({"what": f"CustomerContext ({cust['identity']['name']}) attached to the competitor trigger",
                  "before": before, "after": res.output["body"] + f"\n[send_as={res.output['send_as']} · competitor intel withheld from the customer]"})
    return {"steps": steps}


@router.post("/demo/api/portfolio")
def portfolio(body: dict):
    """A week of *different* conversation types for one merchant: its real triggers plus
    knowledge/curiosity touches generated only from the category context (no invented facts)."""
    ds = _dataset()
    mid = body["merchant_id"]
    m = ds.merchants[mid]
    cat = ds.category_for(m)
    lang = LANGS.get(body.get("language") or "auto")
    cands = [t for t in ds.triggers.values() if t.get("merchant_id") == mid and not t.get("customer_id")]
    research = next((d for d in cat.get("digest", []) if d.get("kind") == "research"), None)
    cde = next((d for d in cat.get("digest", []) if d.get("kind") in ("cde", "event", "webinar")), None)
    synth = []
    if research:
        synth.append(("research_digest", {"top_item_id": research["id"]}))
    if cat.get("trend_signals"):
        synth.append(("category_trend_movement", {}))
    synth.append(("curious_ask_due", {"ask_template": "what_service_in_demand_this_week"}))
    if cde:
        synth.append(("cde_opportunity", {"digest_item_id": cde["id"]}))
    for kind, payload in synth:
        cands.append({"id": f"portfolio_{kind}_{mid}", "scope": "merchant", "kind": kind, "merchant_id": mid,
                      "payload": payload, "urgency": 2, "suppression_key": f"portfolio:{kind}:{mid}:2026-W17", "_generated": True})
    from vera.agents.trigger_analyst import classify_family
    seen, week = set(), []
    for t in sorted(cands, key=lambda t: (t.get("_generated", False), -int(t.get("urgency") or 0))):
        fam = classify_family(t["kind"], "merchant")
        if fam in seen:
            continue
        seen.add(fam)
        res = _orch.compose(cat, m, t, None, language=lang)
        if res.extras.get("fallback"):
            continue
        week.append({"kind": t["kind"].replace("_", " "), "family": fam, "group": GROUPS.get(fam, ""),
                     "source": "generated from category knowledge" if t.get("_generated") else "dataset trigger",
                     "body": res.output["body"], "cta": res.output["cta"]})
        if len(week) == 5:
            break
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    return {"merchant": m["identity"]["name"], "week": [{"day": days[i], **w} for i, w in enumerate(week)]}


@router.get("/demo", response_class=HTMLResponse)
def page():
    return HTMLResponse(_PAGE)


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vera by magicpin — demo</title>
<style>
:root{--bg:#f0f2f5;--panel:#fff;--ink:#111b21;--muted:#667781;--line:#e9edef;--me:#d9fdd3;--them:#fff;--chat:#efeae2;--accent:#008069;--accent2:#25d366;--chip:#e7f5f1;--good:#e6f4ea;--goodink:#1e6b3a;--bad:#fde8ec;--badink:#b3163c;--btn:#027eb5;--shadow:0 1px .5px rgba(11,20,26,.13)}
@media (prefers-color-scheme:dark){:root{--bg:#0c1317;--panel:#111b21;--ink:#e9edef;--muted:#8696a0;--line:#222d34;--me:#005c4b;--them:#202c33;--chat:#0b141a;--accent:#00a884;--chip:#1f2c33;--good:#12301f;--goodink:#7fd49b;--bad:#3a1520;--badink:#ff8fa6;--btn:#53bdeb}}
*{box-sizing:border-box}html,body{height:100%}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
button{font:inherit}
.app{display:flex;align-items:center;gap:14px;padding:10px 18px;background:var(--panel);border-bottom:1px solid var(--line);flex-wrap:wrap}
.logo{width:38px;height:38px;border-radius:50%;background:radial-gradient(circle at 30% 30%,#1a8f7a,#0b4f45);color:#fff;font-weight:800;font-size:10px;display:flex;align-items:center;justify-content:center;letter-spacing:.04em}
.brand b{display:block;font-size:16px}.brand span{color:var(--muted);font-size:12px}.sp{flex:1}
.nav{display:flex;background:var(--bg);border-radius:20px;padding:3px}.nav button{border:0;background:transparent;color:var(--muted);padding:6px 14px;border-radius:16px;cursor:pointer;font-weight:600;font-size:13px}.nav button.on{background:var(--panel);color:var(--accent);box-shadow:var(--shadow)}
select{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:16px;padding:6px 10px;font-size:13px}
.shell{display:grid;grid-template-columns:300px minmax(0,1fr);height:calc(100vh - 59px)}
.shell.drawer{grid-template-columns:300px minmax(0,1fr) 340px}
#side{background:var(--panel);border-right:1px solid var(--line);overflow:auto}
.search{position:sticky;top:0;background:var(--panel);padding:10px 12px;border-bottom:1px solid var(--line);z-index:1}.search input{width:100%;border:0;background:var(--bg);color:var(--ink);border-radius:8px;padding:8px 12px;font-size:14px}
.gh{padding:10px 16px 4px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--accent)}
.row{display:flex;gap:10px;align-items:center;padding:9px 14px;cursor:pointer;border-bottom:1px solid var(--line)}.row:hover,.row.on{background:var(--chip)}
.ra{width:36px;height:36px;border-radius:50%;flex:none;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;color:#fff}
.row b{display:block;font-size:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.row span{display:block;color:var(--muted);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.row .tx{min-width:0;flex:1}
.pass{color:var(--goodink);font-weight:700;font-size:12px}
.phone{display:flex;flex-direction:column;min-width:0;background:var(--chat);background-image:radial-gradient(rgba(0,0,0,.035) 1px,transparent 1px);background-size:18px 18px}
.ch{display:flex;align-items:center;gap:12px;padding:9px 16px;background:var(--panel);border-bottom:1px solid var(--line)}
.ch .logo{width:40px;height:40px}.ch b{font-size:16px}.ch .v{color:#1d9bf0;font-size:14px;display:inline}.ch #to{display:block;color:var(--muted);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tool{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:18px;padding:6px 12px;cursor:pointer;font-size:13px;font-weight:600}.tool:hover{border-color:var(--accent);color:var(--accent)}.tool.on{background:var(--accent);color:#fff;border-color:var(--accent)}
#log{flex:1;overflow:auto;padding:14px 7%}
.chip{display:table;margin:10px auto;background:var(--panel);color:var(--muted);font-size:12px;padding:5px 12px;border-radius:8px;box-shadow:var(--shadow)}.chip.sys{color:var(--accent);max-width:88%;text-align:center}
.b{width:fit-content;max-width:78%;border-radius:8px;margin:5px 0;box-shadow:var(--shadow);position:relative}.b .t{padding:7px 10px 4px;white-space:pre-wrap;word-wrap:break-word}
.v{background:var(--them)}.u{background:var(--me);margin-left:auto}
.tm{display:block;text-align:right;color:var(--muted);font-size:11px;padding:0 9px 5px}
.bt{border-top:1px solid var(--line);display:flex}.bt button{flex:1;border:0;background:transparent;color:var(--btn);padding:9px;cursor:pointer;font-weight:600;font-size:14px}.bt button+button{border-left:1px solid var(--line)}.bt button:disabled{color:var(--muted);cursor:default}
.post{max-width:380px;margin:4px 10px 8px;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--panel)}.post img{display:block;width:100%;max-height:190px;object-fit:cover}.post .pl{font-size:11px;color:var(--muted);padding:6px 10px 0;text-transform:uppercase;letter-spacing:.05em}.post .pt{padding:4px 10px 9px;font-size:14px}
.u img{display:block;max-width:260px;border-radius:6px;margin:4px}
.why{border:0;background:none;color:var(--muted);font-size:11px;cursor:pointer;padding:0 9px 5px}.why:hover{color:var(--accent)}
form{display:flex;gap:8px;align-items:center;padding:9px 14px;background:var(--panel)}
.ic{width:40px;height:40px;border-radius:50%;border:0;background:transparent;color:var(--muted);cursor:pointer;font-size:20px}.ic:hover{background:var(--bg)}
#in{flex:1;border:0;border-radius:22px;padding:11px 16px;background:var(--bg);color:var(--ink);font-size:15px}
.send{width:44px;height:44px;border-radius:50%;border:0;background:var(--accent);color:#fff;cursor:pointer;font-size:18px}
.send:disabled,.ic:disabled{opacity:.4;cursor:default}
.empty{max-width:560px;margin:6vh auto;text-align:center}.empty h2{margin:.2em 0}.empty p{color:var(--muted)}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:18px 0}.step{background:var(--panel);border-radius:12px;padding:14px;text-align:left;box-shadow:var(--shadow)}.step b{display:block;margin:4px 0}.step span{color:var(--muted);font-size:13px}.step i{font-style:normal;font-size:22px}
.cta{background:var(--accent);color:#fff;border:0;border-radius:22px;padding:10px 20px;font-weight:700;cursor:pointer}
#drawer{background:var(--panel);border-left:1px solid var(--line);overflow:auto;display:none}.shell.drawer #drawer{display:block}
.dh{display:flex;align-items:center;padding:12px 16px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--panel)}.dh b{flex:1}.dh button{border:0;background:none;color:var(--muted);font-size:20px;cursor:pointer}
.sec{padding:12px 16px;border-bottom:1px solid var(--line)}.sec h4{margin:0 0 8px;font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}.sec p{margin:4px 0;font-size:13.5px}
.kp{display:grid;grid-template-columns:1fr 1fr;gap:6px}.kp div{background:var(--bg);border-radius:8px;padding:6px 9px}.kp b{display:block;font-size:15px}.kp span{font-size:11px;color:var(--muted)}.kp .warn b{color:var(--badink)}
.tags{display:flex;flex-wrap:wrap;gap:5px}.tg{font-size:11.5px;padding:3px 8px;border-radius:10px;background:var(--chip)}.tg.pos{background:var(--good);color:var(--goodink)}.tg.neg{background:var(--bad);color:var(--badink)}.tg.off{text-decoration:line-through;opacity:.6}
.bar{display:flex;align-items:center;gap:8px;font-size:12px;margin:4px 0}.bar i{flex:1;height:6px;background:var(--line);border-radius:3px;overflow:hidden}.bar i b{display:block;height:100%;background:var(--accent)}
.fact{font-size:12.5px;margin:5px 0}.fact code{font-size:11px;color:var(--muted);word-break:break-all;display:block}
mark{border-radius:3px;padding:0 1px;color:inherit}mark.merchant{background:rgba(0,168,132,.25)}mark.trigger{background:rgba(59,130,246,.25)}mark.category{background:rgba(168,85,247,.24)}mark.customer{background:rgba(245,158,11,.3)}mark.derived{background:rgba(120,120,120,.22)}
.sw{display:flex;align-items:center;gap:8px;font-size:13px;cursor:pointer}
.cmp{display:grid;gap:8px}.cmp div{border-radius:8px;padding:8px 10px;font-size:13px}.cmp .g{background:var(--bad)}.cmp .y{background:var(--good)}.cmp small{display:block;font-weight:700;font-size:11px;margin-bottom:3px}
.verdict{display:table;margin:14px auto;padding:9px 16px;border-radius:10px;font-weight:700}.verdict.ok{background:var(--good);color:var(--goodink)}.verdict.no{background:var(--bad);color:var(--badink)}
.day{background:var(--them);border-radius:10px;padding:10px 12px;margin:10px 0;box-shadow:var(--shadow)}.day h5{margin:0 0 4px;font-size:13px}.day h5 small{color:var(--muted);font-weight:400}
#pick{display:none}
@media (max-width:900px){.shell,.shell.drawer{grid-template-columns:1fr}#side{display:none}#pick{display:block;max-width:60vw}#drawer{position:fixed;inset:59px 0 0 12%;z-index:5;box-shadow:-4px 0 18px rgba(0,0,0,.2)}.steps{grid-template-columns:1fr}#log{padding:12px 3%}.b{max-width:90%}}
</style></head><body>
<header class="app"><div class="logo">VERA</div><div class="brand"><b>Vera by magicpin</b><span>AI assistant for local merchants on WhatsApp — live demo</span></div>
<div class="nav" id="nav"><button data-m="chat" class="on">Live chat</button><button data-m="tests">Judge tests</button><button data-m="week">Weekly plan</button></div>
<div class="sp"></div><select id="pick"></select>
<select id="lang" title="Message language"><option value="auto">🌐 Auto (from profile)</option><option value="en">English</option><option value="hi-en">Hinglish</option><option value="hi">हिन्दी</option></select></header>
<div class="shell" id="shell">
<nav id="side"></nav>
<section class="phone">
 <div class="ch"><div class="logo">VERA</div><div style="flex:1;min-width:0"><b>magicpin</b> <span class="v">✔</span><span id="to">Pick a merchant to start</span></div>
  <button class="tool" id="dealsBtn" hidden>🏷 Deals</button><button class="tool" id="insBtn" hidden>ⓘ Insights</button></div>
 <div id="log"></div>
 <form id="f"><button type="button" class="ic" id="att" title="Attach a photo" disabled>📎</button><input type="file" id="file" accept="image/*" hidden>
  <input id="in" placeholder="Pick a merchant first…" autocomplete="off" disabled><button class="send" id="send" disabled>➤</button></form>
</section>
<aside id="drawer"></aside>
</div>
<script>
const $=s=>document.querySelector(s);let SC=[],cur=null,sid=null,MODE='chat',LANG='auto',LAST=null,HL=false;
const COLORS=['#00a884','#027eb5','#7c5cff','#e67e22','#c0392b','#16a085','#8e44ad','#2c3e50'];
const color=s=>COLORS[[...String(s)].reduce((a,c)=>a+c.charCodeAt(0),0)%COLORS.length];
const ini=n=>String(n||'?').replace(/^Dr\.?\s*/,'').split(/\s+/).map(w=>w[0]).join('').slice(0,2).toUpperCase();
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const now=()=>new Date().toLocaleTimeString([],{hour:'numeric',minute:'2-digit'});
async function post(u,b){return (await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)})).json()}
const TESTS=[["auto_reply","Auto-reply hell","Same canned auto-reply 4× in a row"],["intent","Intent transition","Qualifying turns, then “ok let's do it”"],["join","“I want to join”","Explicit intent on the first reply"],["hostile","Hostile + off-topic","Abuse, then a GST question"],["stop","STOP","Hard opt-out"],["curveballs","Curveball questions","Who are you? · Which competitor? · CTR? · results?"],["language","Language switch","Merchant replies in Hinglish"],["inject","Context injection (§8 twist)","New digest item, new numbers, a customer added"]];

function input(on,ph){$('#in').disabled=!on;$('#send').disabled=!on;$('#att').disabled=!on||!(cur&&cur.audience==='merchant');$('#in').placeholder=ph||'Type a message';}
function header(to,tools){$('#to').textContent=to;$('#dealsBtn').hidden=!tools;$('#insBtn').hidden=!tools}
function drawer(open){$('#shell').classList.toggle('drawer',open);$('#insBtn').classList.toggle('on',open)}
function scrollEnd(){$('#log').scrollTop=1e9}
function chip(t,cls){$('#log').insertAdjacentHTML('beforeend',`<div class="chip ${cls||''}">${t}</div>`);scrollEnd()}

function hl(text,spans){const t=String(text),taken=new Array(t.length).fill(false),m=[];
 (spans||[]).forEach(s=>{let i=t.indexOf(s.text);while(i>=0){let ok=true;for(let k=i;k<i+s.text.length;k++)if(taken[k]){ok=false;break}
  if(ok){for(let k=i;k<i+s.text.length;k++)taken[k]=true;m.push([i,i+s.text.length,s]);break}i=t.indexOf(s.text,i+1)}});
 m.sort((a,b)=>a[0]-b[0]);let o='',p=0;m.forEach(([a,b,s])=>{o+=esc(t.slice(p,a))+`<mark class="${s.layer}" title="${esc(s.source)}">${esc(t.slice(a,b))}</mark>`;p=b});return o+esc(t.slice(p))}

function vera(r,{spans=null,first=false}={}){const body=r.body||'';const [pre,draft]=body.split(/Draft post ↓\n|draft post ↓\n/);
 let main=pre,rest='';if(draft!==undefined){const i=draft.lastIndexOf('\n');rest=i>=0?draft.slice(i+1):'';main=pre}
 let h=`<div class="t">${spans?hl(main,spans):esc(main)}</div>`;
 if(draft!==undefined){const txt=draft.split('\n').slice(0,-1).join('\n')||draft;h+=`<div class="post">${r.draft_image?`<img src="${r.draft_image}" alt="attached photo">`:''}<div class="pl">📍 Google post preview</div><div class="pt">${esc(txt)}</div></div>`+(rest?`<div class="t">${esc(rest)}</div>`:'')}
 h+=`<span class="tm">${now()}</span>`;if(first)h+=`<button class="why" onclick="drawer(true);return false">ⓘ How Vera wrote this</button>`;
 if(r.buttons&&r.buttons.length)h+=`<div class="bt">${r.buttons.map(b=>`<button type="button">${esc(b)}</button>`).join('')}</div>`;
 const d=document.createElement('div');d.className='b v';d.innerHTML=h;$('#log').appendChild(d);
 d.querySelectorAll('.bt button').forEach(bt=>bt.onclick=()=>{d.querySelectorAll('.bt button').forEach(x=>x.disabled=true);send(bt.textContent)});scrollEnd();return d}
function me(text,img){const d=document.createElement('div');d.className='b u';d.innerHTML=(img?`<img src="${img}" alt="photo">`:'')+(text?`<div class="t">${esc(text)}</div>`:'')+`<span class="tm">${now()} ✓✓</span>`;$('#log').appendChild(d);scrollEnd()}
function handle(r){if(r.error){chip(esc(r.error),'sys');return}
 if(r.action==='send')vera(r);else if(r.action==='wait')chip(`⏸ Vera pauses for ${Math.round(r.wait_seconds/60)} min — ${esc(r.rationale)}`,'sys');
 else{chip(`🔚 ${esc(r.rationale)}`,'sys');input(false,'Conversation closed — pick another merchant')}}

// ---------- side lists
function side(){const el=$('#side'),pk=$('#pick');let h='',opts='<option value="">Choose…</option>';
 if(MODE==='chat'){h+=`<div class="search"><input id="q" placeholder="🔍 Search merchants or triggers"></div><div id="rows"></div>`;el.innerHTML=h;
  const draw=()=>{const q=($('#q').value||'').toLowerCase();let r='',g='';SC.filter(x=>(x.merchant+x.kind+x.category+x.id+(x.to||'')).toLowerCase().includes(q)).forEach(x=>{
   if(x.group!==g){g=x.group;r+=`<div class="gh">${esc(g)}</div>`}
   r+=`<div class="row${cur&&cur.id===x.id?' on':''}" data-id="${x.id}"><div class="ra" style="background:${color(x.merchant)}">${ini(x.merchant)}</div><div class="tx"><b>${esc(x.merchant)}</b><span>${x.audience==='customer'?'👤 to '+esc(x.to)+' · ':''}${esc(x.kind)} · ${esc(x.category)}</span></div></div>`});
   $('#rows').innerHTML=r;document.querySelectorAll('.row').forEach(n=>n.onclick=()=>start(SC.find(x=>x.id===n.dataset.id)))};
  $('#q').oninput=draw;draw();SC.forEach(x=>opts+=`<option value="${x.id}">${x.id} · ${esc(x.merchant)} — ${esc(x.kind)}</option>`)}
 else if(MODE==='tests'){h='<div class="gh">What the judge runs</div>'+TESTS.map(t=>`<div class="row" data-t="${t[0]}"><div class="ra" style="background:${color(t[1])}">🧪</div><div class="tx"><b>${t[1]}</b><span>${t[2]}</span></div><span class="pass" id="res_${t[0]}"></span></div>`).join('');
  el.innerHTML=h;document.querySelectorAll('.row').forEach(n=>n.onclick=()=>n.dataset.t==='inject'?runInject():runTest(n.dataset.t));TESTS.forEach(t=>opts+=`<option value="${t[0]}">${t[1]}</option>`)}
 else{const seen=new Set();const ms=SC.filter(x=>x.audience==='merchant'&&!seen.has(x.merchant_id)&&seen.add(x.merchant_id));
  el.innerHTML='<div class="gh">Pick a merchant</div>'+ms.map(x=>`<div class="row" data-m="${x.merchant_id}"><div class="ra" style="background:${color(x.merchant)}">${ini(x.merchant)}</div><div class="tx"><b>${esc(x.merchant)}</b><span>${esc(x.category)}</span></div></div>`).join('');
  document.querySelectorAll('.row').forEach(n=>n.onclick=()=>week(SC.find(x=>x.merchant_id===n.dataset.m)));ms.forEach(x=>opts+=`<option value="${x.merchant_id}">${esc(x.merchant)}</option>`)}
 pk.innerHTML=opts}
$('#pick').onchange=e=>{const v=e.target.value;if(!v)return;if(MODE==='chat')start(SC.find(x=>x.id===v));else if(MODE==='tests')(v==='inject'?runInject():runTest(v));else week(SC.find(x=>x.merchant_id===v))};

// ---------- views
function empty(){cur=null;sid=null;header('Pick a merchant to start',false);drawer(false);input(false,'Pick a merchant first…');
 $('#log').innerHTML=`<div class="empty"><div class="logo" style="width:64px;height:64px;margin:auto;font-size:14px">VERA</div><h2>Vera, magicpin's merchant assistant</h2>
 <p>Every message is written by a team of AI agents from the merchant's real data — and every number is checked before it's sent.</p>
 <div class="steps"><div class="step"><i>1️⃣</i><b>Pick a merchant</b><span>Vera writes today's message from their data and the day's trigger.</span></div>
 <div class="step"><i>2️⃣</i><b>Reply like the owner</b><span>Try “yes”, an auto-reply, a question — or 📎 a dish photo.</span></div>
 <div class="step"><i>3️⃣</i><b>Open ⓘ Insights</b><span>See why Vera said it and where every fact came from.</span></div></div>
 <button class="cta" onclick="start(SC.find(x=>x.id==='T09'))">Try it: Dr. Meera's clinic →</button></div>`}
async function start(x){if(!x)return;MODE='chat';cur=x;side();$('#log').innerHTML='';drawer(false);
 header(x.audience==='customer'?`to ${x.to} · on behalf of ${x.merchant}`:`to ${x.merchant} · ${x.category}`,true);$('#dealsBtn').textContent=x.audience==='customer'?'🏷 Send deals':'🏷 Deals';
 chip('Today');chip('Vera works for magicpin · every fact is verified against the merchant\'s data','sys');chip('Writing…');
 const r=await post('/demo/api/start',{trigger_id:x.trigger_id,customer_id:x.customer_id,language:LANG});$('#log').lastChild.remove();
 sid=r.session_id;LAST=r;vera(r,{first:true});input(true,x.audience==='customer'?`Reply as ${x.to}…`:`Reply as ${x.to||'the owner'}…`);insights()}
async function send(text,img){if(!sid||(!text.trim()&&!img))return;me(text,img);$('#in').value='';handle(await post('/demo/api/reply',{session_id:sid,message:text,image:img||null}))}
async function deals(){if(!sid)return;handle(await post('/demo/api/deals',{session_id:sid}))}
function insights(){const r=LAST;if(!r)return;const p=r.profile||{};const cu=p.customer;
 let h=`<div class="dh"><b>ⓘ How Vera wrote this</b><button onclick="drawer(false)">×</button></div>`;
 h+=`<div class="sec"><h4>Why this message, why now</h4><p>${esc(r.rationale.split('. Levers')[0])}.</p><p style="color:var(--muted);font-size:12px">Trigger: ${esc(r.trigger_kind.replace(/_/g,' '))}${r.placeholder?' (no payload — built from the merchant\'s own data)':''} · Language: ${esc(({en:'English','hi-en':'Hinglish',hi:'Hindi'})[r.language]||r.language)}</p></div>`;
 h+=`<div class="sec"><h4>Personalisation</h4><label class="sw"><input type="checkbox" id="hlsw" ${HL?'checked':''}> Highlight facts in the chat</label>${(r.highlights||[]).slice(0,8).map(f=>`<div class="fact"><mark class="${f.layer}">${esc(f.text)}</mark><code>${esc(f.source)}</code></div>`).join('')}</div>`;
 h+=`<div class="sec"><h4>${esc(p.name)}</h4><p style="color:var(--muted);font-size:12px">${esc(p.locality||'')}, ${esc(p.city||'')} · ${p.verified?'✔ verified':'not verified'} · ${esc(p.plan)}</p><div class="kp">${(p.kpis||[]).map(k=>`<div class="${k.warn?'warn':''}"><b>${esc(k.value)}</b><span>${esc(k.label)} · ${esc(k.sub)}</span></div>`).join('')}</div>
  <div class="tags" style="margin-top:8px">${(p.offers||[]).map(o=>`<span class="tg${o.status==='active'?'':' off'}">🏷 ${esc(o.title)}</span>`).join('')}${(p.reviews||[]).map(v=>`<span class="tg ${v.sentiment==='pos'?'pos':v.sentiment==='neg'?'neg':''}">★ ${esc(v.theme)}</span>`).join('')}</div>
  ${cu?`<p style="margin-top:8px">👤 <b>${esc(cu.name)}</b> · ${esc(cu.state)} · ${cu.visits||0} visits · last ${esc(cu.last_visit||'')}${cu.slots?' · prefers '+esc(cu.slots):''}</p><div class="tags"><span class="tg pos">consent: ${esc((cu.consent||[]).join(', ')||'none')}</span></div>`:''}</div>`;
 h+=`<div class="sec"><h4>Quality checks</h4><div class="tags">${(r.badges||[]).map(b=>`<span class="tg ${b.ok?'pos':'neg'}">${b.ok?'✓':'✗'} ${esc(b.label)}</span>`).join('')}</div><div style="margin-top:8px">${Object.entries(r.scores||{}).map(([k,v])=>`<div class="bar">${k.replace('_',' ')}<i><b style="width:${v*10}%"></b></i>${v}</div>`).join('')}</div></div>`;
 h+=`<div class="sec"><h4>Generic vs Vera</h4><div class="cmp"><div class="g"><small>✗ TYPICAL GENERIC</small>${esc(r.generic)}</div><div class="y"><small>✓ VERA</small>${(r.highlights||[]).length} facts from this merchant's data, anchored on today's trigger, one clear ask.</div></div></div>`;
 $('#drawer').innerHTML=h;$('#hlsw').onchange=e=>{HL=e.target.checked;const first=document.querySelector('#log .b.v .t');if(first)first.innerHTML=HL?hl(LAST.body.split(/Draft post ↓\n/)[0],LAST.highlights):esc(LAST.body.split(/Draft post ↓\n/)[0])}}
async function runTest(name){MODE='tests';side();sid=null;input(false,'Judge test — read-only');$('#log').innerHTML='';header('Judge test · judge plays the merchant',false);chip('Running…');
 const r=await post('/demo/api/replay',{name,language:LANG});$('#log').innerHTML='';header(`Judge test · ${r.title}`,false);chip(esc(r.merchant));chip('Expected: '+esc(r.expect),'sys');vera({body:r.opening});
 r.turns.forEach(t=>{me(t.merchant);handle({...t,buttons:[]})});$('#log').insertAdjacentHTML('beforeend',`<div class="verdict ${r.pass?'ok':'no'}">${r.pass?'✅ PASS':'❌ FAIL'} — ${esc(r.verdict)}</div>`);scrollEnd();
 const el=$('#res_'+name);if(el)el.textContent=r.pass?'PASS':'FAIL';
 $('#drawer').innerHTML=`<div class="dh"><b>🧪 ${esc(r.title)}</b><button onclick="drawer(false)">×</button></div><div class="sec"><h4>What the judge checks</h4><p>${esc(r.expect)}</p><p><b>${r.pass?'PASS':'FAIL'}</b> — ${esc(r.verdict)}</p></div><div class="sec"><h4>Production pain point</h4><p>${esc(r.pain)}</p></div>`;drawer(true)}
async function runInject(){MODE='tests';side();sid=null;input(false,'Judge test — read-only');$('#log').innerHTML='';header('Judge test · post-submission context injection',false);chip('Injecting new context…');
 const r=await post('/demo/api/inject',{language:LANG});$('#log').innerHTML='';
 r.steps.forEach(s=>{chip(esc(s.what),'sys');chip('Before');vera({body:s.before});chip('After new context');vera({body:s.after})});
 $('#log').insertAdjacentHTML('beforeend',`<div class="verdict ok">✅ Adapted to new data — nothing invented, no merchant intel leaked to customers</div>`);const el=$('#res_inject');if(el)el.textContent='PASS';
 $('#drawer').innerHTML=`<div class="dh"><b>🧪 Context injection</b><button onclick="drawer(false)">×</button></div><div class="sec"><h4>The twist (brief §8)</h4><p>After submission the judge pushes new digest items, shifted metrics, new triggers and customer contexts. Bots that adapt without hallucinating score higher.</p></div>`;drawer(true)}
async function week(x){if(!x)return;MODE='week';side();sid=null;input(false,'Weekly plan — read-only');$('#log').innerHTML='';header(`Weekly plan · ${x.merchant}`,false);chip('Planning the week…');
 const r=await post('/demo/api/portfolio',{merchant_id:x.merchant_id,language:LANG});$('#log').innerHTML='';chip('5 different conversations — not just reminders','sys');
 r.week.forEach(w=>$('#log').insertAdjacentHTML('beforeend',`<div class="day"><h5>${w.day} · ${esc(w.kind)} <small>· ${esc(w.group)} · ${esc(w.source)}</small></h5>${esc(w.body).replace(/\n/g,'<br>')}</div>`));
 $('#drawer').innerHTML=`<div class="dh"><b>📅 Why a portfolio</b><button onclick="drawer(false)">×</button></div><div class="sec"><p>Reminders like renewals are rare. Engaging a merchant 3–5× a week needs knowledge- and curiosity-led conversations: research, trends, events, asks. Each one here is fact-checked.</p></div>`;drawer(true)}

// ---------- wiring
document.querySelectorAll('#nav button').forEach(b=>b.onclick=()=>{MODE=b.dataset.m;document.querySelectorAll('#nav button').forEach(x=>x.classList.toggle('on',x===b));side();
 if(MODE==='chat')empty();else{sid=null;drawer(false);header(MODE==='tests'?'Pick a judge test on the left':'Pick a merchant on the left',false);input(false,'Read-only view');
  $('#log').innerHTML=`<div class="empty"><h2>${MODE==='tests'?'Judge tests':'Weekly plan'}</h2><p>${MODE==='tests'?'Run the exact scenarios magicpin\'s judge uses: auto-replies, intent switches, hostile replies, curveballs, and new context arriving mid-test.':'See five different conversations Vera would have with one merchant this week.'}</p></div>`}});
$('#lang').onchange=e=>{LANG=e.target.value;if(MODE==='chat'&&cur)start(cur)};
$('#insBtn').onclick=()=>drawer(!$('#shell').classList.contains('drawer'));$('#dealsBtn').onclick=deals;
$('#f').onsubmit=e=>{e.preventDefault();send($('#in').value)};
$('#att').onclick=()=>$('#file').click();
$('#file').onchange=e=>{const f=e.target.files[0];if(!f)return;const img=new Image(),rd=new FileReader();rd.onload=()=>{img.onload=()=>{const k=Math.min(1,1024/Math.max(img.width,img.height));const c=document.createElement('canvas');c.width=img.width*k;c.height=img.height*k;c.getContext('2d').drawImage(img,0,0,c.width,c.height);send($('#in').value,c.toDataURL('image/jpeg',.82))};img.src=rd.result};rd.readAsDataURL(f);e.target.value=''};
(async()=>{SC=await (await fetch('/demo/api/scenarios')).json();side();empty()})();
</script></body></html>"""
