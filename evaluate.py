"""Internal evaluation simulator.

Approximates the judge's 5 dimensions with the Engagement Critic (deterministic), re-verifies
zero-fabrication with the Fact Checker, and replays the judge's multi-turn scenarios.

    python evaluate.py --dataset dev_fixtures --pairs dev_fixtures/dev_pairs.json [--show]
"""
from __future__ import annotations

import argparse
import statistics
import sys

from conversation_handlers import new_state, respond
from vera.dataset import find_test_pairs, load_dataset, load_pairs
from vera.orchestrator import Orchestrator

DIMS = ["specificity", "category_fit", "merchant_fit", "trigger_relevance", "engagement"]

REPLAYS = {
    "auto_reply_hell": ["Thank you for contacting us! Our team will respond shortly."] * 4,
    "intent_transition": ["Interesting, tell me more", "What would you post?", "ok let's do it"],
    "hostile_off_topic": ["This is useless spam you idiots", "can you also help me file my GST?"],
    "stop": ["STOP"],
    "price": ["how much?"],
    "language_switch": ["haan theek hai, kya karna hoga?", "ok karo"],
}
QUALIFYING = ["would you", "do you", "can you tell", "what if", "how about"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    ds = load_dataset(args.dataset)
    pairs_path = args.pairs or find_test_pairs(args.dataset)
    if pairs_path:
        pairs = load_pairs(pairs_path)
    else:
        pairs = [{"test_id": tid, "trigger_id": tid, "merchant_id": t.get("merchant_id"), "customer_id": t.get("customer_id")}
                 for tid, t in ds.triggers.items()]
    orch = Orchestrator()
    rows, weak, fails = [], [], 0
    for p in pairs:
        t = ds.triggers.get(p["trigger_id"])
        m = ds.merchants.get(p.get("merchant_id") or (t or {}).get("merchant_id"))
        if not t or not m:
            continue
        cid = p.get("customer_id") or t.get("customer_id")
        c = ds.customers.get(cid) if cid else None
        r = orch.compose(ds.category_for(m), m, t, c)
        sc = r.extras.get("scores") or {}
        if r.extras.get("fallback"):
            fails += 1
        rows.append((p["test_id"], sc, r))
        for d in DIMS:
            if sc and sc.get(d, 10) < 7:
                weak.append((p["test_id"], d, sc[d]))
        if args.show:
            print(f"\n[{p['test_id']}] {r.extras.get('family')} {sc}\n{r.output['body']}")
    scored = [sc for _, sc, _ in rows if sc]
    print("\n=== composition (internal critic, 0-10 per dimension) ===")
    for d in DIMS:
        vals = [s[d] for s in scored]
        print(f"  {d:18} avg {statistics.mean(vals):4.1f}   min {min(vals):4.1f}" if vals else f"  {d}: n/a")
    print(f"  total avg {statistics.mean(sum(s.values()) for s in scored):.1f}/50 over {len(scored)} messages; "
          f"safe-fallbacks={fails}; consent-blocked={sum(1 for *_, r in rows if r.extras.get('consent_blocked'))}")
    if weak:
        print("  weak dimensions:", ", ".join(f"{t}:{d}={v}" for t, d, v in weak[:15]))

    print("\n=== multi-turn replays ===")
    ok_all = True
    sample = next(((ds.category_for(ds.merchants[t['merchant_id']]), ds.merchants[t['merchant_id']], t)
                   for t in ds.triggers.values() if not t.get("customer_id") and t.get("merchant_id") in ds.merchants), None)
    if sample:
        for name, msgs in REPLAYS.items():
            st = new_state(f"eval_{name}", *sample)
            actions = [respond(st, msg) for msg in msgs if st.exit_state != "ended"]
            acts = [a["action"] for a in actions]
            ok = {
                "auto_reply_hell": "end" in acts and acts.index("end") <= 1,
                "intent_transition": acts[-1] == "send" and not any(q in actions[-1].get("body", "").lower() for q in QUALIFYING),
                "hostile_off_topic": acts[-1] in ("end", "send"),
                "stop": acts == ["end"],
                "price": acts == ["send"],
                "language_switch": all(a == "send" for a in acts),
            }[name]
            ok_all &= ok
            print(f"  {'PASS' if ok else 'FAIL'} {name}: {acts}")
            if args.show:
                for msg, a in zip(msgs, actions):
                    print(f"      >> {msg}\n      << {a.get('body', a['rationale'])[:160]}")
    return 0 if ok_all and not fails else 1


if __name__ == "__main__":
    sys.exit(main())
