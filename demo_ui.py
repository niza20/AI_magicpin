"""Interactive chat demo served by the bot at /demo (WhatsApp-style UI over the real agent pipeline).

Uses the official dataset in dataset/expanded, with its own session store: it never touches the judge's
/v1 state.
"""
from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path

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


def _dataset():
    global _ds
    if _ds is None and _DS_DIR.exists():
        _ds = load_dataset(str(_DS_DIR))
    return _ds


@router.get("/demo/api/scenarios")
def scenarios():
    ds = _dataset()
    if not ds:
        return JSONResponse(status_code=404, content={"error": "dataset/expanded not found"})
    pairs = load_pairs(str(_DS_DIR / "test_pairs.json"))
    out = []
    for p in pairs:
        t = ds.triggers.get(p["trigger_id"])
        m = ds.merchants.get(p["merchant_id"])
        if not t or not m:
            continue
        cid = p.get("customer_id") or t.get("customer_id")
        c = ds.customers.get(cid) if cid else None
        out.append({"id": p["test_id"], "trigger_id": p["trigger_id"], "customer_id": cid,
                    "merchant": m["identity"]["name"], "category": m.get("category_slug"),
                    "kind": t["kind"].replace("_", " "),
                    "to": (c or {}).get("identity", {}).get("name") if c else m["identity"].get("owner_first_name")})
    return out


@router.post("/demo/api/start")
def start(body: dict):
    ds = _dataset()
    t = ds.triggers[body["trigger_id"]]
    m = ds.merchants[t["merchant_id"]]
    cid = body.get("customer_id") or t.get("customer_id")
    c = ds.customers.get(cid) if cid else None
    cat = ds.category_for(m)
    res = _orch.compose(cat, m, t, c)
    sid = uuid.uuid4().hex[:12]
    st = ConversationState(conversation_id=sid, merchant_id=m["merchant_id"], customer_id=cid, trigger_id=t["id"],
                           category=cat, merchant=m, trigger=t, customer=c, merchant_memory={})
    st.record_bot(res.output["body"], res.output["cta"])
    with _lock:
        _sessions[sid] = st
    return {"session_id": sid, **res.output, "family": res.extras.get("family"), "language": res.extras.get("language"),
            "scores": res.extras.get("scores"), "facts_used": res.extras.get("facts_used", [])[:6],
            "agents": [s.agent for s in res.trace if s.agent != "orchestrator"]}


@router.post("/demo/api/reply")
def reply(body: dict):
    st = _sessions.get(body.get("session_id", ""))
    if not st:
        return JSONResponse(status_code=404, content={"error": "session expired — pick a scenario again"})
    return _engine.respond(st, body.get("message", ""), "customer" if st.customer else "merchant")


@router.get("/demo", response_class=HTMLResponse)
def page():
    return HTMLResponse(_PAGE)


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vera Chat Demo</title>
<style>
:root{--bg:#f0f2f5;--panel:#fff;--ink:#111b21;--muted:#667781;--line:#e4e6eb;--me:#d9fdd3;--them:#fff;--chat:#efeae2;--accent:#008069;--chip:#e7f5f1}
@media (prefers-color-scheme:dark){:root{--bg:#0b141a;--panel:#111b21;--ink:#e9edef;--muted:#8696a0;--line:#222d34;--me:#005c4b;--them:#202c33;--chat:#0b141a;--accent:#00a884;--chip:#1f2c33}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{display:grid;grid-template-columns:320px 1fr 300px;height:100vh;max-width:1400px;margin:0 auto}
aside,.info{background:var(--panel);border-right:1px solid var(--line);overflow:auto}.info{border-left:1px solid var(--line);border-right:0;padding:16px}
h1{font-size:17px;margin:0;padding:16px;border-bottom:1px solid var(--line)}h1 small{display:block;color:var(--muted);font-weight:400;font-size:12px}
.sc{padding:10px 16px;border-bottom:1px solid var(--line);cursor:pointer}.sc:hover,.sc.on{background:var(--chip)}
.sc b{font-size:14px}.sc span{display:block;color:var(--muted);font-size:12px}
main{display:flex;flex-direction:column;background:var(--chat);min-width:0}
.head{background:var(--panel);padding:12px 16px;border-bottom:1px solid var(--line)}.head b{display:block}.head span{color:var(--muted);font-size:12px}
#log{flex:1;overflow:auto;padding:18px 6%}
.m{max-width:78%;padding:8px 11px;border-radius:9px;margin:6px 0;white-space:pre-wrap;box-shadow:0 1px .5px rgba(0,0,0,.13)}
.v{background:var(--them)}.u{background:var(--me);margin-left:auto}.sys{margin:10px auto;text-align:center;color:var(--muted);font-size:12px;max-width:90%}
.meta{display:block;color:var(--muted);font-size:11px;margin-top:4px}
.quick{display:flex;gap:6px;flex-wrap:wrap;padding:8px 6%}.quick button{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:16px;padding:5px 11px;cursor:pointer;font-size:13px}
form{display:flex;gap:8px;padding:10px 16px;background:var(--panel)}input{flex:1;border:1px solid var(--line);border-radius:20px;padding:10px 14px;background:var(--bg);color:var(--ink);font-size:15px}
form button{background:var(--accent);color:#fff;border:0;border-radius:20px;padding:0 18px;font-weight:600;cursor:pointer}
.info h2{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:14px 0 6px}.info p,.info li{font-size:13px;margin:4px 0}
.bar{display:flex;align-items:center;gap:8px;font-size:12px;margin:3px 0}.bar i{flex:1;height:6px;background:var(--line);border-radius:3px;overflow:hidden}.bar i b{display:block;height:100%;background:var(--accent)}
code{font-size:11px;word-break:break-all}
@media (max-width:900px){.wrap{grid-template-columns:1fr;height:auto}aside{max-height:34vh}.info{display:none}main{height:70vh}}
</style></head><body>
<div class="wrap">
<aside><h1>Vera — live agent demo<small>30 official test scenarios · pick one</small></h1><div id="list"></div></aside>
<main>
 <div class="head"><b id="who">Pick a scenario on the left</b><span id="sub">Vera writes the first message; you reply as the merchant (or customer).</span></div>
 <div id="log"></div>
 <div class="quick" id="quick"></div>
 <form id="f"><input id="in" placeholder="Reply as the merchant…" autocomplete="off" disabled><button id="send" disabled>Send</button></form>
</main>
<div class="info" id="info"><h2>How it works</h2><p>Every message goes through the multi-agent pipeline: context → trigger → strategy → composer → deterministic fact &amp; policy checks → critic → finalizer.</p><p>Try the quick replies below the chat: they cover the judge's replay tests.</p></div>
</div>
<script>
const $=s=>document.querySelector(s);let sid=null;
const QUICK=["How much will this cost?","ok lets do it","what is this about?","Thank you for contacting us! Our team will respond shortly.","busy, call me tomorrow","can you help me file my GST?","not interested","haan karo"];
function add(text,cls,meta){const d=document.createElement('div');d.className='m '+cls;d.textContent=text;if(meta){const s=document.createElement('span');s.className='meta';s.textContent=meta;d.appendChild(s)}$('#log').appendChild(d);$('#log').scrollTop=1e9}
function sys(t){const d=document.createElement('div');d.className='sys';d.textContent=t;$('#log').appendChild(d);$('#log').scrollTop=1e9}
function enable(on){$('#in').disabled=!on;$('#send').disabled=!on}
async function load(){const r=await fetch('/demo/api/scenarios');const xs=await r.json();
 if(!Array.isArray(xs)){$('#list').textContent='Dataset not found on server.';return}
 xs.forEach(x=>{const d=document.createElement('div');d.className='sc';d.innerHTML=`<b>${x.id} · ${x.merchant}</b><span>${x.kind} · ${x.category}${x.customer_id?' · to customer '+(x.to||''):''}</span>`;d.onclick=()=>start(x,d);$('#list').appendChild(d)})}
async function start(x,el){document.querySelectorAll('.sc').forEach(e=>e.classList.remove('on'));el.classList.add('on');
 $('#log').innerHTML='';$('#who').textContent=x.merchant;$('#sub').textContent=(x.customer_id?'Vera → customer '+(x.to||'')+' (on behalf of merchant)':'Vera → '+(x.to||'merchant'))+' · trigger: '+x.kind;
 sys('Composing with the agent pipeline…');const r=await(await fetch('/demo/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({trigger_id:x.trigger_id,customer_id:x.customer_id})})).json();
 $('#log').innerHTML='';sid=r.session_id;add(r.body,'v','Vera · cta='+r.cta+' · send_as='+r.send_as);
 $('#in').placeholder=x.customer_id?'Reply as the customer…':'Reply as the merchant…';enable(true);
 const sc=r.scores||{};$('#info').innerHTML='<h2>Why this message</h2><p>'+esc(r.rationale)+'</p><h2>Critic scores</h2>'+Object.entries(sc).map(([k,v])=>`<div class="bar">${k.replace('_',' ')}<i><b style="width:${v*10}%"></b></i>${v}</div>`).join('')+
 '<h2>Facts used (source paths)</h2><ul>'+(r.facts_used||[]).map(f=>'<li><code>'+esc(f)+'</code></li>').join('')+'</ul><h2>Details</h2><p>family: '+r.family+' · language: '+r.language+'<br>suppression key: <code>'+esc(r.suppression_key)+'</code></p>';
 $('#quick').innerHTML='';(x.customer_id?["1","YES","can I come on Saturday?","STOP"]:QUICK).forEach(q=>{const b=document.createElement('button');b.type='button';b.textContent=q;b.onclick=()=>send(q);$('#quick').appendChild(b)})}
function esc(s){return String(s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
async function send(text){if(!sid||!text.trim())return;add(text,'u');$('#in').value='';
 const r=await(await fetch('/demo/api/reply',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid,message:text})})).json();
 if(r.error){sys(r.error);return}
 if(r.action==='send')add(r.body,'v','Vera · '+r.rationale);
 else if(r.action==='wait'){sys('⏸ Vera waits '+Math.round(r.wait_seconds/60)+' min — '+r.rationale)}
 else{sys('🔚 Conversation ended — '+r.rationale);enable(false)}}
$('#f').onsubmit=e=>{e.preventDefault();send($('#in').value)};load();
</script></body></html>"""
