"""Browser view of the judge API: opening /v1/<endpoint> (or /) in a browser shows a styled console with a live
"Send" button for every endpoint, instead of raw JSON. Only requests whose Accept header asks for text/html get
this page — the judge harness (JSON / */*) always gets the normal JSON responses. Add ?format=json to see raw JSON.
"""
from __future__ import annotations

import functools
import json

ENDPOINTS = [
    ("healthz", "GET", "Liveness + how many contexts are loaded per scope.", None),
    ("metadata", "GET", "Team, model and approach.", None),
    ("context", "POST", "Push a category / merchant / customer / trigger context (idempotent by version; stale → 409).",
     {"scope": "category", "context_id": "demo_console", "version": 1, "payload": {"slug": "restaurants"}}),
    ("tick", "POST", "Vera's periodic wake-up: returns the messages it decides to send for the available triggers.",
     {"now": "2026-04-26T10:30:00Z", "available_triggers": []}),
    ("reply", "POST", "A merchant / customer replied: returns send · wait · end with the next message.",
     {"conversation_id": "console_1", "merchant_id": "m_006_southindiancafe_restaurant_bangalore", "from_role": "merchant", "message": "Yes, go ahead", "turn_number": 2}),
    ("teardown", "POST", "Wipe all in-memory state at the end of a test run.", {}),
]


_FLOW_TRIGGER = "trg_012_milestone_mylari"


@functools.lru_cache(maxsize=1)
def sample_flow() -> tuple:
    """A real dataset scenario as the judge would send it: category → merchant → trigger contexts, then tick, then a reply."""
    try:
        import os
        from vera.dataset import load_dataset
        ds = load_dataset(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "expanded"))
        t = ds.triggers[_FLOW_TRIGGER]
        m = ds.merchants[t["merchant_id"]]
        c = ds.category_for(m)
    except Exception:
        return ()
    ctx = lambda scope, cid, payload: {"step": f"push {scope} context", "endpoint": "context",
                                       "body": {"scope": scope, "context_id": cid, "version": 1, "payload": payload}}
    return (ctx("category", c.get("slug"), c), ctx("merchant", m["merchant_id"], m), ctx("trigger", t["id"], t),
            {"step": "tick — Vera decides whether to message", "endpoint": "tick", "body": {"now": "2026-04-26T10:30:00Z", "available_triggers": [t["id"]]}},
            {"step": "merchant replies", "endpoint": "reply", "body": {"merchant_id": m["merchant_id"], "from_role": "merchant", "message": "Yes, go ahead", "turn_number": 2}},
            {"step": "merchant asks for the draft again", "endpoint": "reply", "body": {"merchant_id": m["merchant_id"], "from_role": "merchant", "message": "give me the draft", "turn_number": 3}})


def wants_html(request) -> bool:
    if request.query_params.get("format") == "json":
        return False
    return "text/html" in (request.headers.get("accept") or "")


def page(current: str = "") -> str:
    flow_js = json.dumps(list(sample_flow()), ensure_ascii=False).replace("</", "<\\/")
    cards = []
    for name, method, desc, body in ENDPOINTS:
        sample = json.dumps(body, indent=2, ensure_ascii=False) if body is not None else ""
        box = f'<textarea spellcheck="false" id="b_{name}">{sample}</textarea>' if body is not None else ""
        cards.append(f'''<section class="card{' cur' if name == current else ''}" id="{name}">
  <div class="row"><span class="m {method.lower()}">{method}</span><code>/v1/{name}</code>
  <button onclick="send('{name}','{method}')">Send</button><a class="raw" href="/v1/{name}?format=json">JSON</a></div>
  <p>{desc}</p>{box}<pre id="r_{name}" class="out" hidden></pre>
</section>''')
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vera API Console</title><style>
:root{{--bg:#f0f2f5;--panel:#fff;--ink:#111b21;--muted:#667781;--line:#e9edef;--accent:#008069;--get:#1a73e8;--post:#b06000;--me:#d9fdd3}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#0c1317;--panel:#111b21;--ink:#e9edef;--muted:#8696a0;--line:#222d34;--accent:#00a884;--get:#8ab4f8;--post:#fdd663;--me:#005c4b}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
header{{display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:12px 16px;background:var(--panel);border-bottom:1px solid var(--line)}}
header b{{font-size:17px}}header .sp{{flex:1}}header a{{color:var(--accent);font-weight:600;text-decoration:none;padding:6px 12px;border:1px solid var(--line);border-radius:16px;font-size:13px}}
main{{max-width:900px;margin:0 auto;padding:16px}}.intro{{color:var(--muted);margin:4px 0 16px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:12px}}.card.cur{{border-color:var(--accent);box-shadow:0 0 0 2px var(--accent) inset}}
.row{{display:flex;align-items:center;gap:10px;flex-wrap:wrap}}.row code{{font-size:15px;font-weight:600;flex:1;min-width:0}}
.m{{font:700 12px monospace;padding:3px 8px;border-radius:6px;color:#fff}}.m.get{{background:var(--get);color:var(--bg)}}.m.post{{background:var(--post);color:var(--bg)}}
button{{background:var(--accent);color:#fff;border:0;border-radius:16px;padding:6px 16px;font-weight:600;cursor:pointer}}.raw{{color:var(--muted);font-size:13px}}
p{{margin:8px 0;color:var(--muted)}}textarea,.out{{width:100%;font:13px/1.4 ui-monospace,Menlo,monospace;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:10px;overflow:auto;white-space:pre-wrap;word-break:break-word}}
textarea{{min-height:110px;resize:vertical}}.out{{margin:8px 0 0;max-height:320px}}details{{margin-top:10px}}summary{{cursor:pointer;font-weight:600}}.bubble{{white-space:pre-wrap;background:var(--me);border-radius:10px;padding:10px 12px;margin:8px 0 0;max-width:640px}}.card.flow{{border-color:var(--accent)}}
.ok{{color:var(--accent);font-weight:700}}.err{{color:#d93025;font-weight:700}}
</style></head><body>
<header><b>Vera · API console</b><span class="sp"></span><a href="/demo">Features</a><a href="/demo#chat">Live chat</a></header>
<main><p class="intro">These are the endpoints the judge calls. The judge sends JSON; a browser visit shows this console instead.
Edit a request body and press <b>Send</b> to call the live bot. Base URL: <code id="base"></code></p>
<section class="card flow"><div class="row"><span class="m post">FLOW</span><code>Full judge flow on a real merchant</code>
<button onclick="flow()">Run</button></div><p>Pushes the category, merchant and trigger contexts for Mylari South Indian Cafe, calls tick,
then replies twice as the merchant — exactly the order the judge uses. Every request and response is shown.</p><div id="flowOut"></div></section>
{"".join(cards)}</main>
<script>
document.getElementById('base').textContent=location.origin;
async function send(n,m){{const o=document.getElementById('r_'+n);o.hidden=false;o.textContent='…';const t0=performance.now();
try{{let opt={{method:m,headers:{{'Content-Type':'application/json','Accept':'application/json'}}}};
if(m==='POST'){{const b=document.getElementById('b_'+n).value.trim()||'{{}}';JSON.parse(b);opt.body=b}}
const r=await fetch('/v1/'+n,opt);const j=await r.json();const ms=Math.round(performance.now()-t0);
o.innerHTML='<span class="'+(r.ok?'ok':'err')+'">HTTP '+r.status+'</span> · '+ms+' ms\\n';o.append(JSON.stringify(j,null,2));}}
catch(e){{o.innerHTML='<span class="err">Error</span>\\n';o.append(String(e))}}}}
const FLOW={flow_js};
async function flow(){{const out=document.getElementById('flowOut');out.innerHTML='';let conv=null;
for(const s of FLOW){{const body=Object.assign({{}},s.body);if(s.endpoint==='reply')body.conversation_id=conv||'console_flow';
 const d=document.createElement('details');d.open=s.endpoint!=='context';const sm=document.createElement('summary');sm.textContent=s.step+' → POST /v1/'+s.endpoint+' …';d.append(sm);
 const pre=document.createElement('pre');pre.className='out';d.append(pre);out.append(d);
 try{{const r=await fetch('/v1/'+s.endpoint,{{method:'POST',headers:{{'Content-Type':'application/json','Accept':'application/json'}},body:JSON.stringify(body)}});const j=await r.json();
  sm.textContent=s.step+' → POST /v1/'+s.endpoint+' · HTTP '+r.status;
  if(s.endpoint==='tick'&&j.actions&&j.actions[0])conv=j.actions[0].conversation_id;
  pre.textContent=JSON.stringify(j,null,2);
  if(j.body||(j.actions&&j.actions[0])){{const msg=document.createElement('div');msg.className='bubble';msg.textContent=j.body||j.actions[0].body;d.append(msg)}}
 }}catch(e){{pre.textContent=String(e)}}}}}}
const c=document.querySelector('.card.cur');if(c)c.scrollIntoView({{block:'center'}});
</script></body></html>'''
