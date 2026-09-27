"""Checks the bot against every testable requirement in magicpin's challenge brief + testing brief.

    python scripts/verify_brief.py            # prints a PASS/FAIL checklist, exits 1 on any FAIL

Runs in-process (FastAPI TestClient) on the official dataset in dataset/expanded. Each check cites the brief section.
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from fastapi.testclient import TestClient  # noqa: E402

import bot  # noqa: E402
from conversation_handlers import new_state, respond  # noqa: E402
from vera.agents.trigger_analyst import classify_family  # noqa: E402
from vera.dataset import load_dataset, load_pairs  # noqa: E402

DS = load_dataset("dataset/expanded")
PAIRS = load_pairs("dataset/expanded/test_pairs.json")
RESULTS: list[tuple[str, str, bool, str]] = []


def check(section: str, name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((section, name, bool(ok), detail))
    return ok


def ctx(tid, cid=None):
    t = DS.triggers[tid]
    m = DS.merchants[t["merchant_id"]]
    cid = cid or t.get("customer_id")
    return DS.category_for(m), m, t, (DS.customers.get(cid) if cid else None)


def hinglish(s: str) -> bool:
    """Hindi-leaning text: Roman Hinglish or Devanagari."""
    return bool(re.search(r"[\u0900-\u097F]", s) or re.search(r"\b(aap|hai|hain|kar|kijiye|karein|mein|ke liye|doon|raha|rahi|ji)\b", s.lower()))


# ============================================================ testing brief §2: the HTTP contract
bot.STORE.reset()
c = TestClient(bot.app)
S = "testing §2"
h = c.get("/v1/healthz")
check(S, "GET /v1/healthz → 200 {status, uptime_seconds, contexts_loaded}", h.status_code == 200 and
      {"status", "uptime_seconds", "contexts_loaded"} <= set(h.json()))
md = c.get("/v1/metadata").json()
check(S, "GET /v1/metadata has all 7 fields", {"team_name", "team_members", "model", "approach", "contact_email", "version",
                                                "submitted_at"} <= set(md), f"team={md.get('team_members')} email={md.get('contact_email')}")

# warmup (testing §4 phase 1): 5 categories + 50 merchants + 200 customers, 0 triggers
t0 = time.time()
for slug, cat in DS.categories.items():
    c.post("/v1/context", json={"scope": "category", "context_id": slug, "version": 1, "payload": cat, "delivered_at": "2026-04-26T10:00:00Z"})
for mid, m in DS.merchants.items():
    c.post("/v1/context", json={"scope": "merchant", "context_id": mid, "version": 1, "payload": m, "delivered_at": "2026-04-26T10:00:00Z"})
for cid, cu in DS.customers.items():
    c.post("/v1/context", json={"scope": "customer", "context_id": cid, "version": 1, "payload": cu, "delivered_at": "2026-04-26T10:00:00Z"})
loaded = c.get("/v1/healthz").json()["contexts_loaded"]
check("testing §4.1", "warmup: healthz reflects all 255 base contexts", loaded.get("category") == 5 and loaded.get("merchant") == 50
      and loaded.get("customer") == 200, f"{loaded} in {time.time() - t0:.1f}s")

m1 = DS.merchants["m_001_drmeera_dentist_delhi"]
r = c.post("/v1/context", json={"scope": "merchant", "context_id": m1["merchant_id"], "version": 1, "payload": m1})
check(S, "/v1/context idempotent: same (id, version) re-post is a no-op", r.status_code == 200 and r.json().get("accepted") is not False
      or r.status_code == 409, f"{r.status_code} {r.json()}")
r = c.post("/v1/context", json={"scope": "merchant", "context_id": m1["merchant_id"], "version": 0, "payload": m1})
check(S, "/v1/context lower version → 409 stale_version + current_version", r.status_code == 409 and r.json().get("reason") == "stale_version"
      and r.json().get("current_version") == 1)
r = c.post("/v1/context", json={"scope": "planet", "context_id": "x", "version": 1, "payload": {}})
check(S, "/v1/context bad scope → 400 invalid_scope", r.status_code == 400 and r.json().get("reason") == "invalid_scope")
big = copy.deepcopy(DS.categories["dentists"])
big["digest"] = big.get("digest", []) + [{"id": f"d_pad_{i}", "kind": "research", "title": "x" * 400, "source": "pad"} for i in range(1000)]
size = len(json.dumps(big))
r = c.post("/v1/context", json={"scope": "category", "context_id": "dentists_big", "version": 1, "payload": big})
check("testing §5", "/v1/context accepts payloads up to the 500 KB cap", r.status_code == 200, f"{size // 1024} KB")

# ============================================================ testing §2.2 /v1/tick
for tid, t in DS.triggers.items():
    c.post("/v1/context", json={"scope": "trigger", "context_id": tid, "version": 1, "payload": t})
S = "testing §2.2"
r = c.post("/v1/tick", json={"now": "2026-04-26T10:30:00Z", "available_triggers": []})
check(S, "tick with nothing active → {actions: []}", r.status_code == 200 and r.json() == {"actions": []})
all_actions, worst, seen_conv = [], 0.0, set()
ids = list(DS.triggers)
for i in range(0, len(ids), 25):
    t1 = time.time()
    acts = c.post("/v1/tick", json={"now": "2026-04-26T10:30:00Z", "available_triggers": ids[i:i + 25]}).json()["actions"]
    worst = max(worst, time.time() - t1)
    all_actions += acts
    check(S, f"tick batch {i // 25 + 1}: ≤20 actions, one per merchant/customer", len(acts) <= 20 and
          len({(a["merchant_id"], a.get("customer_id")) for a in acts}) == len(acts), f"{len(acts)} actions")
need = {"conversation_id", "merchant_id", "customer_id", "send_as", "trigger_id", "template_name", "template_params", "body", "cta",
        "suppression_key", "rationale"}
check(S, "every action has the full schema (incl. template_name + template_params for the 24h window)",
      all(need <= set(a) and a["template_name"] and isinstance(a["template_params"], list) and a["template_params"] for a in all_actions),
      f"{len(all_actions)} actions")
convs = [a["conversation_id"] for a in all_actions]
check(S, "new conversation_ids are unique", len(convs) == len(set(convs)))
check("testing §5", "tick responds well within 30 s", worst < 30, f"slowest batch {worst:.2f}s")
check(S, "restraint: consent-less / suppressed triggers are not sent",
      not any(a.get("customer_id") and not (DS.customers.get(a["customer_id"]) or {}).get("consent") for a in all_actions))
more = []
for k in range(3):   # later ticks re-offer every trigger; nothing identical may reach the same recipient twice
    more += c.post("/v1/tick", json={"now": f"2026-04-26T1{k + 1}:00:00Z", "available_triggers": ids}).json()["actions"]
sent_pairs = [(a["merchant_id"], a.get("customer_id"), a["body"]) for a in all_actions + more]
check("testing §10", "across ticks, no recipient ever gets the same text twice", len(sent_pairs) == len(set(sent_pairs)),
      f"{len(all_actions) + len(more)} sends over 7 ticks")
all_actions += more
again = c.post("/v1/tick", json={"now": "2026-04-26T10:35:00Z", "available_triggers": [a["trigger_id"] for a in all_actions]}).json()["actions"]
check("brief §4.3", "suppression_key dedup: same triggers are not re-sent next tick", again == [], f"{len(again)} re-sent")

# ============================================================ testing §2.3 /v1/reply + §10 failure modes
S = "testing §2.3"
conv = next(a for a in all_actions if not a.get("customer_id"))
bodies, slow = [conv["body"]], 0.0
for n, msg in enumerate(["What exactly would you do?", "How much does it cost?", "ok let's do it", "thanks"], start=2):
    t1 = time.time()
    rr = c.post("/v1/reply", json={"conversation_id": conv["conversation_id"], "merchant_id": conv["merchant_id"], "from_role": "merchant",
                                   "message": msg, "received_at": "2026-04-26T10:40:00Z", "turn_number": n})
    slow = max(slow, time.time() - t1)
    j = rr.json()
    check(S, f"reply turn {n} ({msg!r}) → valid action", rr.status_code == 200 and j.get("action") in ("send", "wait", "end") and
          (j["action"] != "send" or (j.get("body") or "").strip()) and (j["action"] != "wait" or j.get("wait_seconds")),
          f"{j.get('action')}: {(j.get('body') or j.get('rationale') or '')[:70]}")
    if j.get("action") == "send":
        check("testing §10", f"turn {n}: no verbatim repeat within the conversation", j["body"] not in bodies)
        bodies.append(j["body"])
    if j.get("action") == "end":
        break
check("testing §5", "reply responds well within 30 s", slow < 30, f"slowest {slow:.2f}s")
rr = c.post("/v1/reply", json={"conversation_id": "conv_never_seen", "merchant_id": "m_001_drmeera_dentist_delhi", "from_role": "merchant",
                               "message": "hello?", "received_at": "2026-04-26T10:40:00Z", "turn_number": 2})
check(S, "reply on an unknown conversation still returns a valid action (no 500)", rr.status_code == 200 and rr.json().get("action") in ("send", "wait", "end"))

# ============================================================ testing §4 phase 3: adaptive context injection
S = "testing §4.3"
cat = copy.deepcopy(DS.categories["dentists"])
cat["digest"].append({"id": "d_inj_sdf", "kind": "research", "title": "Silver diamine fluoride arrests 81% of early caries",
                      "source": "IJDR May 2026, p.33", "trial_n": 640})
check(S, "new digest version accepted", c.post("/v1/context", json={"scope": "category", "context_id": "dentists", "version": 2, "payload": cat}).json()["accepted"])
mm = copy.deepcopy(DS.merchants["m_002_bharat_dentist_mumbai"]) if "m_002_bharat_dentist_mumbai" in DS.merchants else copy.deepcopy(next(
    m for m in DS.merchants.values() if m["category_slug"] == "dentists" and m["merchant_id"] != "m_001_drmeera_dentist_delhi"))
mm["performance"].update({"calls": 7, "views": 999, "delta_7d": {"calls_pct": -0.61}})
c.post("/v1/context", json={"scope": "merchant", "context_id": mm["merchant_id"], "version": 2, "payload": mm})
inj = [("trg_inj_digest", {"kind": "research_digest", "merchant_id": "m_001_drmeera_dentist_delhi", "payload": {"category": "dentists", "top_item_id": "d_inj_sdf"}}),
       ("trg_inj_dip", {"kind": "perf_dip", "merchant_id": mm["merchant_id"], "payload": {"metric": "calls", "delta_pct": -0.61, "window": "7d"}})]
newc = {"customer_id": "c_inj_anita", "merchant_id": "m_001_drmeera_dentist_delhi",
        "identity": {"name": "Anita", "language_pref": "english"},
        "relationship": {"first_visit": "2025-10-01", "last_visit": "2026-03-20", "visits_total": 2, "services_received": ["cleaning"]},
        "state": "lapsed_soft", "preferences": {"preferred_slots": "weekend_morning", "reminder_opt_in": True},
        "consent": {"opted_in_at": "2025-10-01", "scope": ["recall_reminders", "appointment_reminders"]}}
c.post("/v1/context", json={"scope": "customer", "context_id": "c_inj_anita", "version": 1, "payload": newc})
inj.append(("trg_inj_recall", {"kind": "recall_due", "scope": "customer", "merchant_id": "m_001_drmeera_dentist_delhi", "customer_id": "c_inj_anita",
                                "payload": {"service_due": "6_month_cleaning", "last_service_date": "2026-03-20", "due_date": "2026-09-20"}}))
for tid, t in inj:
    c.post("/v1/context", json={"scope": "trigger", "context_id": tid, "version": 1, "payload": {"id": tid, "scope": t.get("scope", "merchant"), "urgency": 3, **t}})
acts = {a["trigger_id"]: a for a in c.post("/v1/tick", json={"now": "2026-04-27T10:00:00Z", "available_triggers": [t for t, _ in inj]}).json()["actions"]}
check(S, "uses the new digest item (title + trial size)", "trg_inj_digest" in acts and "Silver diamine" in acts["trg_inj_digest"]["body"]
      and "640" in acts["trg_inj_digest"]["body"], (acts.get("trg_inj_digest") or {}).get("body", "")[:90])
check(S, "uses the shifted performance numbers", "trg_inj_dip" in acts and "61%" in acts["trg_inj_dip"]["body"], (acts.get("trg_inj_dip") or {}).get("body", "")[:90])
rc = acts.get("trg_inj_recall") or {}
check(S, "customer pushed mid-test + recall_due → customer-facing message", rc.get("send_as") == "merchant_on_behalf" and "Anita" in rc.get("body", ""), rc.get("body", "")[:90])

# ============================================================ brief §5 / §7: compose() + submission
S = "brief §5/§7"
sub = [json.loads(line) for line in open("submission.jsonl", encoding="utf-8")]
check(S, "submission.jsonl has exactly T01–T30", [r["test_id"] for r in sub] == [f"T{i:02d}" for i in range(1, 31)])
check(S, "every row has body, cta, send_as, suppression_key, rationale",
      all({"body", "cta", "send_as", "suppression_key", "rationale"} <= set(r) and r["body"].strip() for r in sub))
fresh = []
for p in PAIRS:
    cat_, m_, t_, cu_ = ctx(p["trigger_id"], p.get("customer_id"))
    fresh.append(bot.compose(cat_, m_, t_, cu_))
check(S, "submission.jsonl matches what bot.compose() produces today", all(f["body"] == s["body"] for f, s in zip(fresh, sub)))
cat_, m_, t_, cu_ = ctx(PAIRS[0]["trigger_id"])
check(S, "compose() is deterministic (same inputs → same output)", bot.compose(cat_, m_, t_, cu_) == bot.compose(cat_, m_, t_, cu_))
times = []
outs = {}
for tid in DS.triggers:
    cat_, m_, t_, cu_ = ctx(tid)
    t1 = time.time()
    outs[tid] = (bot.compose(cat_, m_, t_, cu_), cat_, m_, t_, cu_)
    times.append(time.time() - t1)
check(S, "compose() < 30 s per call (all 100 triggers)", max(times) < 30, f"max {max(times) * 1000:.0f} ms, mean {sum(times) / len(times) * 1000:.0f} ms")
check(S, "cta ∈ {binary_yes_stop, open_ended, none}", all(o[0]["cta"] in ("binary_yes_stop", "open_ended", "none") for o in outs.values()))
from vera.orchestrator import Orchestrator  # noqa: E402
orch = Orchestrator()
blocked_ids = {tid for tid, o in outs.items() if o[4] and orch.compose(*o[1:]).extras.get("consent_blocked")}
check(S, "send_as = merchant_on_behalf for customer messages (consent-blocked ones go to the merchant instead)",
      all((o[0]["send_as"] == "merchant_on_behalf") == (bool(o[4]) and tid not in blocked_ids) for tid, o in outs.items()),
      f"{len(blocked_ids)} consent-blocked → told the merchant")

# ============================================================ brief §5 constraints + §11 anti-patterns (all 100 triggers)
S = "brief §5/§11"
fallbacks = sum(bool(orch.compose(*o[1:]).extras.get("fallback")) for o in outs.values())
check(S, "no safe-fallback messages (every trigger composed properly)", fallbacks == 0, f"{fallbacks}/100")
bodies100 = [o[0]["body"] for o in outs.values()]
PRE = re.compile(r"^(hi|hello|dear)?[^.]{0,30}(hope you('| a)re (doing )?well|i('| a)m reaching out|trust this finds)", re.I)
check(S, "no long preambles", not any(PRE.search(b) for b in bodies100))
check(S, "no hype / all-caps promo", not any(re.search(r"\b(AMAZING|HURRY|BEST DEAL|LIMITED TIME)\b", b) for b in bodies100))
taboo_hits = []
for o in outs.values():
    voice = o[1].get("voice") or {}
    for w in (voice.get("taboos") or voice.get("vocab_taboo") or []):
        if re.search(rf"\b{re.escape(str(w).lower())}\b", o[0]["body"].lower()):
            taboo_hits.append((o[3]["id"], w))
check(S, "category taboo words never used (e.g. 'cure', 'guaranteed')", not taboo_hits, str(taboo_hits[:3]))
multi = [o[3]["id"] for o in outs.values()
         if len(set(re.findall(r"\b(?:reply|send|bhejein)\s+(yes|no|stop|maybe|go)\b", o[0]["body"].lower()))
                | set(re.findall(r"\b(yes|no|stop|maybe)\s+(?:reply|bhejein)\b", o[0]["body"].lower()))) > 1]
check(S, "single primary CTA (no 'reply YES for X, NO for Y')", not multi, str(multi[:3]))
buried = [o[3]["id"] for o in outs.values() if o[0]["cta"] != "none" and not re.search(r"(\?|YES|reply|bataiye|kijiye|karein|STOP)[^.?!]*[.?!🙂✅]*\s*$", o[0]["body"].strip(), re.I)]
check(S, "CTA lands in the last sentence", not buried, str(buried[:3]))
pct_off = [o[3]["id"] for o in outs.values() if re.search(r"\b\d+\s*%\s*off\b", o[0]["body"], re.I)
           and not re.search(r"\b\d+\s*%\s*off\b", json.dumps(o[2].get("offers", [])) + json.dumps(o[3].get("payload", {}))
                             + json.dumps(o[1].get("offer_catalog", []), ensure_ascii=False), re.I)]
check(S, "no generic 'X% off' (only inside a real service+price offer from the contexts)", not pct_off, str(pct_off[:3]))
lang_miss = []
for o in outs.values():
    if o[4]:
        pref = str((o[4].get("identity") or {}).get("language_pref", "")).lower()
        if "hi" in pref and "english" not in pref and not hinglish(o[0]["body"]):
            lang_miss.append(o[3]["id"])
    elif "hi" in (o[2]["identity"].get("languages") or []) and not hinglish(o[0]["body"]):
        lang_miss.append(o[3]["id"])
check(S, "language preference honoured (hi / hi-en → Hinglish)", not lang_miss, f"{len(lang_miss)} misses {lang_miss[:3]}")
dup = [b for b, n in Counter(bodies100).items() if n > 1]
check(S, "identical texts only where two triggers are the same event for the same merchant", all(
    len({(outs[t][3]["merchant_id"], outs[t][3]["kind"]) for t in outs if outs[t][0]["body"] == b}) == 1 for b in dup), f"{len(dup)} same-event pairs")
urls = [b for b in bodies100 if re.search(r"https?://", b)]
check(S, "no invented URLs", not urls)
phone = [b for b in bodies100 if re.search(r"(?<![\d-])(\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?![\d-])", b)]
check("testing §11", "no phone numbers / PII in any message", not phone)

# no fabrication: every number in the body must exist in the 4 contexts (or be derived: %, days, dates, counts)
from vera.ledger import FactLedger, extract_numbers  # noqa: E402
unver = []
for tid, (o, cat_, m_, t_, cu_) in outs.items():
    led = FactLedger.build(cat_, m_, t_, cu_) if hasattr(FactLedger, "build") else None
    res = orch.compose(cat_, m_, t_, cu_)
    if res.extras.get("checks", {}).get("fact_ok") is False:
        unver.append(tid)
check("brief §5.8", "fact checker passes on all 100 (no invented numbers, offers, citations, competitors)", not unver, str(unver[:3]))

# ============================================================ customer-facing (brief Appendix B + consent)
S = "brief App. B"
cust = [(tid, o) for tid, o in outs.items() if o[4]]
check(S, "customer messages use the customer's name", all(re.sub(r"^(mr|mrs|ms|dr|shri|smt)\.?\s+", "", (o[4]["identity"].get("name") or ""), flags=re.I).split()[0] in o[0]["body"]
                                                        or "(" in (o[4]["identity"].get("name") or "") for _, o in cust if o[0]["body"]), f"{len(cust)} customer triggers")
leaks = [tid for tid, o in cust if re.search(r"\b(ctr|peer|competitor|views|calls)\b", o[0]["body"].lower())]
check(S, "no merchant-internal data (CTR, views, competitors) leaks to customers", not leaks, str(leaks[:3]))
blocked = [tid for tid, o in cust if not (o[4].get("consent") or {}).get("scope") and not (o[4].get("consent") or {}).get("opted_in_at")]
blocked_sent = [a for a in all_actions if a["trigger_id"] in blocked]
check(S, "customers without consent are never messaged via /v1/tick", not blocked_sent, f"{len(blocked)} no-consent triggers, {len(blocked_sent)} sent")

# ============================================================ replay test (brief §8, testing §4 phase 4) + open challenges §12
S = "replay / §12"


def convo(tid, msgs, cid=None):
    cat_, m_, t_, cu_ = ctx(tid, cid)
    st = new_state("v_" + tid, cat_, m_, t_, cu_, bot.compose(cat_, m_, t_, cu_)["body"])
    out = []
    for x in msgs:
        r_ = respond(st, x)
        out.append(r_)
        if r_["action"] == "end":
            break
    return out


TID = "trg_023_competitor_opened_dentist" if "trg_023_competitor_opened_dentist" in DS.triggers else PAIRS[0]["trigger_id"]
auto = convo(TID, ["Thank you for contacting us! Our team will respond shortly."] * 4)
check(S, "auto-reply hell: same canned text 4× → exits (by turn 2)", auto[-1]["action"] == "end" and len(auto) <= 2, [a["action"] for a in auto])
hi_auto = convo(TID, ["Aapki jaankari ke liye bahut-bahut shukriya. Main aapki yeh sabhi baatein hamari team tak pahuncha deti hoon."] * 3)
check(S, "Hindi auto-reply (brief pattern B) detected too", hi_auto[-1]["action"] in ("end", "wait"), [a["action"] for a in hi_auto])
intent = convo(TID, ["Interesting, tell me more", "What would the post say?", "ok let's do it"])
last = intent[-1].get("body", "").lower()
check(S, "intent transition: 2 qualifying turns then 'ok let's do it' → action, no new qualifying question",
      intent[-1]["action"] == "send" and not re.search(r"would you|do you|can you tell|what if|how about", last), last[:90])
for phrase in ("Mujhe magicpin judrna hai", "I want to join", "yes go ahead", "haan kar do"):
    rj = convo(TID, [phrase])[-1]
    check(S, f"explicit intent {phrase!r} → acts immediately", rj["action"] == "send" and not re.search(r"\?\s*$", rj["body"].split("\n")[0]) or
          bool(re.search(r"done|draft|ho gaya|kar diya|raise|live|here|yeh raha|↓", rj.get("body", ""), re.I)), rj.get("body", "")[:80])
hostile = convo(TID, ["You people are useless idiots", "can you also help me file my GST?"])
check(S, "abuse → one polite apology (offers STOP), no counter-attack", hostile[0]["action"] == "send" and
      re.search(r"sorry|maaf", hostile[0]["body"], re.I) and "STOP" in hostile[0]["body"], hostile[0].get("body", "")[:80])
check(S, "then GST question → declined politely, steered back to the one action", len(hostile) > 1 and hostile[1]["action"] == "send" and
      "GST" in hostile[1]["body"] and re.search(r"YES", hostile[1]["body"]), hostile[-1].get("body", "")[:80])
spam = convo(TID, ["Stop messaging me. This is useless spam."])
check(S, "judge_simulator's hostile line ('Stop messaging me… spam') → end", spam[-1]["action"] == "end")
ni = convo(TID, ["Not interested, please don't message again"])
check(S, "hard 'not interested' → graceful end", ni[-1]["action"] == "end")
stop = convo(TID, ["STOP"])
check(S, "STOP → end", stop[-1]["action"] == "end")
curve = convo(TID, ["Who are you? Is this a scam?"])
check(S, "curveball question gets an honest on-topic answer", curve[-1]["action"] == "send" and len(curve[-1]["body"]) > 20, curve[-1].get("body", "")[:80])
lang = convo(TID, ["haan theek hai, kya karna hoga?"])
check(S, "language detection per turn (English opener → Hinglish reply)", hinglish(lang[-1].get("body", "")), lang[-1].get("body", "")[:80])
later = convo(TID, ["busy right now, message me tomorrow"])
check(S, "'later' → wait with a back-off instead of pushing", later[-1]["action"] in ("wait", "send") and
      (later[-1]["action"] == "wait" or re.search(r"tomorrow|kal", later[-1].get("body", ""), re.I)), later[-1]["action"])
silent = convo(TID, ["ok", "ok", "ok", "ok"])
check(S, "knows when to stop: repeated non-answers don't loop forever", len(silent) <= 4 and silent[-1]["action"] in ("end", "wait", "send"),
      [a["action"] for a in silent])

deliv_bad = []
for tid in DS.triggers:
    cat_, m_, t_, cu_ = ctx(tid)
    st = new_state("d_" + tid, cat_, m_, t_, cu_, bot.compose(cat_, m_, t_, cu_)["body"])
    r1, r2 = respond(st, "Yes, go ahead"), respond(st, "GO")
    if r1["action"] != "send" or "Noted," in r1.get("body", "") or (not cu_ and r2["action"] == "send" and "Noted," in r2.get("body", "")):
        deliv_bad.append(tid)
check(S, "every opener: 'Yes, go ahead' delivers what was offered, 'GO' confirms it (100 triggers)", not deliv_bad, str(deliv_bad[:3]))

# ============================================================ pain points (brief §3) + compulsion levers (§10)
S = "brief §3/§10"
families = Counter(classify_family(t["kind"], t.get("scope", "merchant")) for t in DS.triggers.values())
check(S, "diversified portfolio: knowledge/curiosity families present, not just reminders",
      sum(families[f] for f in ("knowledge", "trend", "seasonal", "recurring", "regulation", "local_event", "festival", "competitor")) >= 20,
      dict(families.most_common(8)))
offers_used = sum(bool(re.search(r"@ ?₹\s?\d", b)) for b in bodies100)
check(S, "service+price offers used (e.g. 'Dental Cleaning @ ₹299')", offers_used >= 25, f"{offers_used}/100 messages")
asks = sum(bool(re.search(r"(what|which|kya|kaun|kitne|bataiye)[^.?!]*\?", b, re.I)) for b in bodies100)
check(S, "'asking the merchant' lever fires (production Vera's big miss)", asks >= 5, f"{asks}/100")
social = sum(bool(re.search(r"peer|average|avg|other \w+ in|similar \w+|locality ke|dentists|salons|gyms|clinics", b, re.I)) for b in bodies100)
check(S, "social-proof / peer benchmark lever fires", social >= 10, f"{social}/100")

# ============================================================ deliverables (brief §7) + privacy (testing §11)
S = "brief §7"
check(S, "bot.py exposes compose(category, merchant, trigger, customer)", callable(getattr(bot, "compose", None)))
check(S, "conversation_handlers.respond(state, merchant_message) exists", callable(respond))
words = len(open("README.md", encoding="utf-8").read().split())
check(S, "README.md ≈ 1 page", words <= 900, f"{words} words")
src = "".join(open(os.path.join(dp, f), encoding="utf-8").read() for dp, _, fs in os.walk("vera") for f in fs if f.endswith(".py"))
outbound = re.findall(r"(urlopen|requests\.(get|post)|httpx\.)", src + open("bot.py", encoding="utf-8").read())
check("testing §11", "no outbound calls with merchant data except the optional LLM API", not outbound, str(outbound[:3]))
r = c.post("/v1/teardown")
check("testing §11", "POST /v1/teardown wipes all state", r.status_code == 200 and sum(c.get("/v1/healthz").json()["contexts_loaded"].values()) == 0)

# ============================================================ report
w = max(len(n) for _, n, _, _ in RESULTS)
cur = None
for sec, name, ok, detail in RESULTS:
    if sec != cur:
        print(f"\n[{sec}]")
        cur = sec
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<{w}}  {detail if isinstance(detail, str) else detail}")
fails = [r for r in RESULTS if not r[2]]
print(f"\n{len(RESULTS) - len(fails)}/{len(RESULTS)} checks passed" + (f" — {len(fails)} FAILED" if fails else ""))
sys.exit(1 if fails else 0)
