"""Generate submission.jsonl from the OFFICIAL dataset + canonical test pairs.

    python generate_submission.py                      # uses ./dataset and auto-discovers the pairs file
    python generate_submission.py --pairs path.json    # explicit pairs file
    python generate_submission.py --dataset dev_fixtures --pairs dev_fixtures/dev_pairs.json --out dev_outputs/dev_submission.jsonl

Never invents test IDs: if no pairs file is found it exits with an error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from bot import compose
from vera.agents.finalizer import validate_output
from vera.dataset import find_test_pairs, load_dataset, load_pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--out", default="submission.jsonl")
    args = ap.parse_args()

    ds = load_dataset(args.dataset)
    pairs_path = args.pairs or find_test_pairs(args.dataset)
    if not pairs_path or not os.path.exists(pairs_path):
        print(f"ERROR: no canonical test-pair file found under '{args.dataset}'. Put the official dataset (with its "
              f"30 test pairs) there, or pass --pairs. Refusing to invent test IDs.", file=sys.stderr)
        return 2
    pairs = load_pairs(pairs_path)
    print(f"dataset: {ds.counts()}  pairs: {len(pairs)} from {pairs_path}")
    rows, problems = [], []
    t0 = time.time()
    for p in pairs:
        trig = ds.triggers.get(p["trigger_id"])
        mid = p.get("merchant_id") or (trig or {}).get("merchant_id") or ((trig or {}).get("payload") or {}).get("merchant_id")
        merchant = ds.merchants.get(mid)
        if not trig or not merchant:
            problems.append(f"{p['test_id']}: missing trigger/merchant ({p['trigger_id']}, {mid})")
            continue
        cid = p.get("customer_id") or trig.get("customer_id")
        customer = ds.customers.get(cid) if cid else None
        s = time.time()
        out = compose(ds.category_for(merchant), merchant, trig, customer)
        dt = time.time() - s
        errs = validate_output(out)
        if errs or dt > 30:
            problems.append(f"{p['test_id']}: {errs or ''} {dt:.1f}s")
        rows.append({"test_id": p["test_id"], **out})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.out} in {time.time() - t0:.1f}s")
    for pr in problems:
        print("PROBLEM:", pr, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
