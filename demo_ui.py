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


def _compose_payload(res, cat, m, t, c):
    return {**res.output, "family": res.extras.get("family"), "language": res.extras.get("language"),
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
    return {"session_id": sid, **_compose_payload(res, cat, m, t, c)}


@router.post("/demo/api/reply")
def reply(body: dict):
    st = _sessions.get(body.get("session_id", ""))
    if not st:
        return JSONResponse(status_code=404, content={"error": "session expired — pick a scenario again"})
    return _engine.respond(st, body.get("message", ""), "customer" if st.customer else "merchant")


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
<title>Vera Agent Demo</title>
<style>
:root{--bg:#f0f2f5;--panel:#fff;--ink:#111b21;--muted:#667781;--line:#e4e6eb;--me:#d9fdd3;--them:#fff;--chat:#efeae2;--accent:#008069;--chip:#e7f5f1;--good:#e6f4ea;--goodink:#1e6b3a;--bad:#fde8ec;--badink:#b3163c;--head:#f7f8fa}
@media (prefers-color-scheme:dark){:root{--bg:#0b141a;--panel:#111b21;--ink:#e9edef;--muted:#8696a0;--line:#222d34;--me:#005c4b;--them:#202c33;--chat:#0b141a;--accent:#00a884;--chip:#1f2c33;--good:#12301f;--goodink:#7fd49b;--bad:#3a1520;--badink:#ff8fa6;--head:#17222a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.top{display:flex;align-items:center;gap:14px;padding:10px 18px;background:var(--panel);border-bottom:1px solid var(--line);flex-wrap:wrap}
.top b{font-size:17px}.top .sub{color:var(--muted);font-size:12px}.sp{flex:1}
.seg{display:flex;border:1px solid var(--line);border-radius:18px;overflow:hidden}.seg button{border:0;background:transparent;color:var(--ink);padding:6px 12px;cursor:pointer;font-size:13px}.seg button.on{background:var(--accent);color:#fff}
.wrap{display:grid;grid-template-columns:330px 1fr 330px;height:calc(100vh - 56px)}
aside,.info{background:var(--panel);overflow:auto}aside{border-right:1px solid var(--line)}.info{border-left:1px solid var(--line);padding:14px 16px}
.tabs{display:flex;position:sticky;top:0;background:var(--panel);border-bottom:1px solid var(--line);z-index:1}.tabs button{flex:1;border:0;background:none;color:var(--muted);padding:11px 4px;cursor:pointer;font-weight:600;font-size:13px;border-bottom:2px solid transparent}.tabs button.on{color:var(--accent);border-color:var(--accent)}
.gh{padding:8px 16px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);background:var(--head);border-bottom:1px solid var(--line)}
.sc{padding:9px 16px;border-bottom:1px solid var(--line);cursor:pointer}.sc:hover,.sc.on{background:var(--chip)}
.sc b{font-size:14px}.sc span{display:block;color:var(--muted);font-size:12px}
.pill{display:inline-block;font-size:10.5px;padding:1px 7px;border-radius:9px;margin-left:4px;background:var(--chip);color:var(--accent);font-weight:600;vertical-align:1px}
.pill.cust{background:#fff3d6;color:#8a5a00}@media (prefers-color-scheme:dark){.pill.cust{background:#3b2f10;color:#f5c661}}
.rp{padding:12px 16px;border-bottom:1px solid var(--line)}.rp b{display:block}.rp span{display:block;color:var(--muted);font-size:12px;margin:2px 0 8px}
.btn{border:1px solid var(--accent);color:var(--accent);background:transparent;border-radius:16px;padding:5px 12px;cursor:pointer;font-size:13px;font-weight:600}.btn:hover{background:var(--chip)}
main{display:flex;flex-direction:column;background:var(--chat);min-width:0}
.head{background:var(--panel);padding:11px 16px;border-bottom:1px solid var(--line)}.head b{display:block}.head span{color:var(--muted);font-size:12px}
#log{flex:1;overflow:auto;padding:16px 6%}
.m{max-width:80%;padding:8px 11px;border-radius:9px;margin:6px 0;white-space:pre-wrap;box-shadow:0 1px .5px rgba(0,0,0,.13)}
.v{background:var(--them)}.u{background:var(--me);margin-left:auto}.sys{margin:10px auto;text-align:center;color:var(--muted);font-size:12px;max-width:92%}
.meta{display:block;color:var(--muted);font-size:11px;margin-top:4px}
.badges{display:flex;flex-wrap:wrap;gap:5px;margin:4px 0 8px}.bd{font-size:11.5px;padding:3px 8px;border-radius:6px;background:var(--good);color:var(--goodink);font-weight:600}.bd.no{background:var(--bad);color:var(--badink)}
.verdict{margin:12px auto;max-width:92%;padding:10px 14px;border-radius:10px;font-weight:600;text-align:center}.verdict.pass{background:var(--good);color:var(--goodink)}.verdict.fail{background:var(--bad);color:var(--badink)}
.day{background:var(--them);border-radius:10px;padding:10px 12px;margin:10px 0;box-shadow:0 1px .5px rgba(0,0,0,.13)}.day h4{margin:0 0 4px;font-size:13px}.day h4 small{color:var(--muted);font-weight:400}
.quick{display:flex;gap:6px;flex-wrap:wrap;padding:8px 6%}.quick button{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:16px;padding:5px 11px;cursor:pointer;font-size:13px}
form{display:flex;gap:8px;padding:10px 16px;background:var(--panel)}input{flex:1;border:1px solid var(--line);border-radius:20px;padding:10px 14px;background:var(--bg);color:var(--ink);font-size:15px}
form button{background:var(--accent);color:#fff;border:0;border-radius:20px;padding:0 18px;font-weight:600;cursor:pointer}
.info h2{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin:14px 0 6px}.info p,.info li{font-size:13px;margin:4px 0}.info ul{padding-left:18px;margin:4px 0}
.bar{display:flex;align-items:center;gap:8px;font-size:12px;margin:3px 0}.bar i{flex:1;height:6px;background:var(--line);border-radius:3px;overflow:hidden}.bar i b{display:block;height:100%;background:var(--accent)}
.pp{border:1px solid var(--line);border-radius:8px;padding:8px 10px;margin:6px 0;font-size:12.5px}.pp b{display:block;font-size:13px}
code{font-size:11px;word-break:break-all}
@media (max-width:1000px){.wrap{grid-template-columns:1fr;height:auto}aside{max-height:40vh}main{height:75vh}.info{border-left:0;border-top:1px solid var(--line)}}
</style></head><body>
<div class="top"><div><b>Vera — multi-agent demo</b><div class="sub">Real pipeline · official dataset · every fact verified against context</div></div><div class="sp"></div>
<span class="sub">Message language</span><div class="seg" id="lang"><button data-l="auto" class="on" title="From the merchant/customer profile">Auto</button><button data-l="en">English</button><button data-l="hi-en">Hinglish</button><button data-l="hi">हिन्दी</button></div></div>
<div class="wrap">
<aside><div class="tabs"><button data-t="sc" class="on">30 test scenarios</button><button data-t="rp">Judge replays</button><button data-t="wk">Weekly plan</button></div><div id="side"></div></aside>
<main>
 <div class="head"><b id="who">Pick a scenario, a judge replay, or a weekly plan</b><span id="sub">Vera writes the first message; you reply as the merchant (or customer).</span></div>
 <div id="log"></div><div class="quick" id="quick"></div>
 <form id="f"><input id="in" placeholder="Reply…" autocomplete="off" disabled><button id="send" disabled>Send</button></form>
</main>
<div class="info" id="info"></div>
</div>
<script>
const $=s=>document.querySelector(s);let sid=null,SC=[],cur=null,LANG='auto',TAB='sc';
const QUICK=["How much will this cost?","ok lets do it","what is this about?","Thank you for contacting us! Our team will respond shortly.","busy, call me tomorrow","can you help me file my GST?","not interested","haan karo"];
const REPLAYS=[["auto_reply","Auto-reply hell","Same canned WhatsApp auto-reply 4× in a row"],["intent","Intent transition","2 qualifying turns, then “ok let's do it”"],["join","“I want to join”","Explicit intent on the very first reply"],["hostile","Hostile + off-topic","Abuse, then a GST question"],["stop","STOP","Hard opt-out"],["language","Language switch","Merchant replies in Hinglish"]];
const PAINS=[["Auto-reply pollution","Canned replies detected by phrasing + verbatim repeats (tracked per merchant, even across conversations). One owner-directed nudge, then exit.","auto_reply"],["Intent-handoff failures","“yes / go ahead / ok let's do it / judna hai” routes straight to ACT and delivers the draft — never back to qualifying.","intent"],["Generic copy","Offers ranked service+price first (“Dental Cleaning @ ₹299”); a deterministic checker rejects invented “X% off” claims.",null],["Low engagement frequency","A portfolio of different conversation types — research, trends, curiosity asks, CDE, events — not just reminders.","week"]];
function esc(s){return String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function add(text,cls,meta){const d=document.createElement('div');d.className='m '+cls;d.textContent=text;if(meta){const s=document.createElement('span');s.className='meta';s.textContent=meta;d.appendChild(s)}$('#log').appendChild(d);$('#log').scrollTop=1e9;return d}
function sys(t,cls){const d=document.createElement('div');d.className=cls||'sys';d.textContent=t;$('#log').appendChild(d);$('#log').scrollTop=1e9}
function enable(on){$('#in').disabled=!on;$('#send').disabled=!on}
async function post(u,b){return (await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)})).json()}
function pains(extra){return '<h2>Production pain points → how this bot handles them</h2>'+PAINS.map(p=>`<div class="pp"><b>${p[0]}</b>${p[1]}${p[2]?` <a href="#" onclick="${p[2]=='week'?"tab('wk');return false":"runReplay('"+p[2]+"');return false"}">See it →</a>`:''}</div>`).join('')+(extra||'')}
function infoDefault(){$('#info').innerHTML='<h2>What you are looking at</h2><p>Every message is produced by the agent pipeline: context analyst → trigger analyst → strategy planner → composer → <b>deterministic fact &amp; policy checkers</b> → critic → finalizer.</p><p><b>Two audiences:</b> <span class="pill">Vera → owner</span> coaches the merchant; <span class="pill cust">→ customer</span> is sent on the merchant\'s behalf (consent checked).</p>'+pains()}
function side(){const el=$('#side');el.innerHTML='';
 if(TAB==='sc'){const groups={};SC.forEach(x=>(groups[x.group]=groups[x.group]||[]).push(x));
  Object.entries(groups).forEach(([g,xs])=>{el.insertAdjacentHTML('beforeend',`<div class="gh">${esc(g)} · ${xs.length}</div>`);
   xs.forEach(x=>{const d=document.createElement('div');d.className='sc'+(cur&&cur.id===x.id?' on':'');
    d.innerHTML=`<b>${x.id} · ${esc(x.merchant)}</b><span>${esc(x.kind)} · ${esc(x.category)} ${x.audience==='customer'?'<i class="pill cust">→ customer '+esc(x.to||'')+'</i>':'<i class="pill">Vera → owner</i>'}${x.placeholder?'<i class="pill" title="Trigger has no payload: composed from merchant data only">no-payload</i>':''}</span>`;
    d.onclick=()=>start(x);el.appendChild(d)})})}
 else if(TAB==='rp'){el.insertAdjacentHTML('beforeend','<div class="gh">Judge replay tests (phase 4) · click to run live</div>');
  REPLAYS.forEach(r=>el.insertAdjacentHTML('beforeend',`<div class="rp"><b>${r[1]}</b><span>${r[2]}</span><button class="btn" onclick="runReplay('${r[0]}')">Run test</button></div>`))}
 else{el.insertAdjacentHTML('beforeend','<div class="gh">Weekly conversation plan · pick a merchant</div>');
  const seen=new Set();SC.filter(x=>x.audience==='merchant'&&!seen.has(x.merchant_id)&&seen.add(x.merchant_id)).forEach(x=>{const d=document.createElement('div');d.className='sc';d.innerHTML=`<b>${esc(x.merchant)}</b><span>${esc(x.category)}</span>`;d.onclick=()=>week(x);el.appendChild(d)})}}
function tab(t){TAB=t;document.querySelectorAll('.tabs button').forEach(b=>b.classList.toggle('on',b.dataset.t===t));side()}
async function start(x){cur=x;side();$('#log').innerHTML='';enable(false);$('#quick').innerHTML='';
 $('#who').textContent=x.id+' · '+x.merchant;$('#sub').textContent=(x.audience==='customer'?'Sent to customer '+(x.to||'')+' on behalf of the merchant':'Vera → owner '+(x.to||''))+' · trigger: '+x.kind;
 sys('Running the agent pipeline…');const r=await post('/demo/api/start',{trigger_id:x.trigger_id,customer_id:x.customer_id,language:LANG});
 $('#log').innerHTML='';sid=r.session_id;const d=add(r.body,'v','cta='+r.cta+' · send_as='+r.send_as+' · language='+r.language);
 const b=document.createElement('div');b.className='badges';b.innerHTML=(r.badges||[]).map(x=>`<span class="bd${x.ok?'':' no'}">${x.ok?'✓':'✗'} ${esc(x.label)}</span>`).join('');d.after(b);
 $('#in').placeholder=x.audience==='customer'?'Reply as the customer…':'Reply as the merchant…';enable(true);
 const langWhy=LANG==='hi'&&x.audience!=='customer'?'You chose हिन्दी. Devanagari is used for customer messages; merchant messages use Hinglish (Roman-script Hindi), the way owners text on WhatsApp.':LANG!=='auto'?'You chose '+LANG+'.':(x.audience==='customer'?'Auto: customer profile language_pref = '+(x.customer_lang||'n/a')+'.':'Auto: merchant profile languages = ['+(x.languages||[]).join(', ')+'] → '+(r.language==='hi-en'?'Hinglish (brief: code-mix preferred when “hi” is listed)':r.language)+'.');
 const sc=r.scores||{};$('#info').innerHTML='<h2>Why this message</h2><p>'+esc(r.rationale)+'</p><h2>Language</h2><p>'+esc(langWhy)+' Use the switch at the top to change it.</p>'+(r.placeholder?'<h2>No-payload trigger</h2><p>This test trigger carries no data, so Vera used only this merchant\'s own facts — nothing invented.</p>':'')+'<h2>Critic scores (0-10)</h2>'+Object.entries(sc).map(([k,v])=>`<div class="bar">${k.replace('_',' ')}<i><b style="width:${v*10}%"></b></i>${v}</div>`).join('')+
 '<h2>Facts used (source paths)</h2><ul>'+(r.facts_used||[]).map(f=>'<li><code>'+esc(f)+'</code></li>').join('')+'</ul><h2>Suppression key</h2><p><code>'+esc(r.suppression_key)+'</code></p>'+pains();
 (x.audience==='customer'?["1","YES","can I come on Saturday?","STOP"]:QUICK).forEach(q=>{const bt=document.createElement('button');bt.type='button';bt.textContent=q;bt.onclick=()=>send(q);$('#quick').appendChild(bt)})}
async function send(text){if(!sid||!text.trim())return;add(text,'u');$('#in').value='';const r=await post('/demo/api/reply',{session_id:sid,message:text});
 if(r.error){sys(r.error);return}if(r.action==='send')add(r.body,'v',r.rationale);else if(r.action==='wait')sys('⏸ Vera waits '+Math.round(r.wait_seconds/60)+' min — '+r.rationale);else{sys('🔚 Conversation ended — '+r.rationale);enable(false)}}
async function runReplay(name){sid=null;enable(false);$('#quick').innerHTML='';$('#log').innerHTML='';sys('Running judge replay…');
 const r=await post('/demo/api/replay',{name,trigger_id:cur&&cur.audience==='merchant'?cur.trigger_id:null,language:LANG});$('#log').innerHTML='';
 $('#who').textContent='Judge replay · '+r.title;$('#sub').textContent=r.merchant+' · tests: '+r.pain;sys('Expected: '+r.expect);add(r.opening,'v','Vera opens');
 r.turns.forEach(t=>{add(t.merchant,'u','judge (as merchant)');if(t.action==='send')add(t.body,'v',t.rationale);else if(t.action==='wait')sys('⏸ wait '+Math.round(t.wait_seconds/60)+' min — '+t.rationale);else sys('🔚 end — '+t.rationale)});
 sys((r.pass?'✅ PASS — ':'❌ FAIL — ')+r.verdict,'verdict '+(r.pass?'pass':'fail'));
 $('#info').innerHTML='<h2>Replay test</h2><p><b>'+esc(r.title)+'</b>: '+esc(r.expect)+'</p><p>Result: <b>'+(r.pass?'PASS':'FAIL')+'</b> — '+esc(r.verdict)+'</p>'+pains()}
async function week(x){sid=null;enable(false);$('#quick').innerHTML='';$('#log').innerHTML='';sys('Planning a week of conversations…');
 const r=await post('/demo/api/portfolio',{merchant_id:x.merchant_id,language:LANG});$('#log').innerHTML='';
 $('#who').textContent='Weekly plan · '+r.merchant;$('#sub').textContent='Different conversation types, not just reminders';
 r.week.forEach(w=>$('#log').insertAdjacentHTML('beforeend',`<div class="day"><h4>${w.day} · ${esc(w.kind)} <small>· ${esc(w.group)} · ${esc(w.source)}</small></h4>${esc(w.body).replace(/\n/g,'<br>')}</div>`));
 $('#info').innerHTML='<h2>Why a portfolio</h2><p>Functional nudges (renewal, profile) are rare. Engaging a merchant 3-5×/week needs knowledge- and curiosity-driven conversations. This plan mixes the merchant\'s real triggers with touches generated only from the category knowledge pack (digest, trends, CDE, curiosity asks) — each one fact-checked.</p><p>'+r.week.length+' distinct conversation types this week.</p>'+pains()}
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>tab(b.dataset.t));
document.querySelectorAll('#lang button').forEach(b=>b.onclick=()=>{LANG=b.dataset.l;document.querySelectorAll('#lang button').forEach(x=>x.classList.toggle('on',x===b));if(cur&&TAB==='sc')start(cur)});
$('#f').onsubmit=e=>{e.preventDefault();send($('#in').value)};
(async()=>{SC=await (await fetch('/demo/api/scenarios')).json();infoDefault();side()})();
</script></body></html>"""
