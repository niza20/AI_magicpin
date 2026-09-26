"""Live demo of the bot over HTTP, exactly as the judge harness calls it.
    uvicorn bot:app --port 8080   then   python scripts/demo.py [BOT_URL]"""
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vera.dataset import load_dataset  # noqa: E402

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080").rstrip("/")


def call(method, path, body=None):
    req = urllib.request.Request(URL + path, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def line(t=""):
    print(f"\n{'=' * 8} {t} {'=' * (60 - len(t))}")


ds = load_dataset(str(Path(__file__).resolve().parent.parent / "dataset" / "expanded"))
line("1. GET /v1/healthz + /v1/metadata")
print(call("GET", "/v1/healthz"))
print({k: v for k, v in call("GET", "/v1/metadata").items() if k in ("team_name", "version", "model")})

line("2. POST /v1/context  (judge pushes the base dataset)")
n = 0
for scope, store in (("category", ds.categories), ("merchant", ds.merchants), ("customer", ds.customers), ("trigger", ds.triggers)):
    for cid, payload in store.items():
        n += call("POST", "/v1/context", {"scope": scope, "context_id": cid, "version": 1, "payload": payload,
                                          "delivered_at": "2026-04-26T10:00:00Z"})["accepted"]
print(f"accepted {n} contexts →", call("GET", "/v1/healthz")["contexts_loaded"])

line("3. POST /v1/tick  (bot decides what to send)")
picks = ["trg_023_competitor_opened_dentist", "trg_019_chronic_refill_grandfather", "trg_010_ipl_match_delhi"]
acts = call("POST", "/v1/tick", {"now": "2026-04-26T10:30:00Z", "available_triggers": picks})["actions"]
for a in acts:
    print(f"\n→ {a['merchant_id']}  send_as={a['send_as']}  cta={a['cta']}\n  {a['body']}\n  rationale: {a['rationale'][:160]}…")

line("4. POST /v1/reply  (judge plays the merchant)")
conv = next(a for a in acts if a["trigger_id"] == "trg_023_competitor_opened_dentist")
for turn, msg in enumerate(["Hmm, how much will this cost?", "ok lets do it", "Thank you for contacting us! Our team will respond shortly."], 2):
    r = call("POST", "/v1/reply", {"conversation_id": conv["conversation_id"], "merchant_id": conv["merchant_id"],
                                   "from_role": "merchant", "message": msg, "received_at": "2026-04-26T10:40:00Z", "turn_number": turn})
    print(f"\nMERCHANT: {msg}\nVERA [{r['action']}]: {r.get('body', '')}\n  ({r['rationale'][:110]})")

line("5. Auto-reply hell + STOP on another merchant")
acts2 = call("POST", "/v1/tick", {"now": "2026-04-26T10:35:00Z", "available_triggers": ["trg_004_perf_dip_bharat"]})["actions"]
c2 = acts2[0]
print(f"VERA: {c2['body']}")
for i in range(3):
    r = call("POST", "/v1/reply", {"conversation_id": c2["conversation_id"], "merchant_id": c2["merchant_id"], "from_role": "merchant",
                                   "message": "Thank you for contacting us! Our team will respond shortly.", "turn_number": i + 2})
    print(f"  auto-reply #{i + 1} → {r['action']}: {r.get('body', r['rationale'])[:110]}")
    if r["action"] == "end":
        break
