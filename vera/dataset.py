"""Dataset loading. Supports both layouts seen in the challenge material:
  (a) per-entity files: <root>/categories/*.json, merchants/*.json, customers/*.json, triggers/*.json
  (b) seed bundles:     <root>/categories/*.json + merchants_seed.json / customers_seed.json / triggers_seed.json
and discovers the canonical test-pair file (never invents one)."""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Dataset:
    root: str
    categories: dict = field(default_factory=dict)
    merchants: dict = field(default_factory=dict)
    customers: dict = field(default_factory=dict)
    triggers: dict = field(default_factory=dict)

    def category_for(self, merchant: dict) -> dict:
        slug = merchant.get("category_slug") or merchant.get("category")
        return self.categories.get(slug, {"slug": slug or "general"})

    def counts(self) -> dict:
        return {"category": len(self.categories), "merchant": len(self.merchants),
                "customer": len(self.customers), "trigger": len(self.triggers)}


def _load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _items(blob, container: str) -> list:
    if isinstance(blob, list):
        return blob
    if isinstance(blob, dict):
        for k in (container, container.rstrip("s"), "items", "data"):
            if isinstance(blob.get(k), list):
                return blob[k]
        return [blob]
    return []


def load_dataset(root: str) -> Dataset:
    ds = Dataset(root=root)
    for f in sorted(glob.glob(os.path.join(root, "categories", "*.json"))):
        c = _load_json(f)
        ds.categories[c.get("slug") or os.path.splitext(os.path.basename(f))[0]] = c
    for container, key, store in (("merchants", "merchant_id", ds.merchants), ("customers", "customer_id", ds.customers),
                                  ("triggers", "id", ds.triggers)):
        for f in sorted(glob.glob(os.path.join(root, container, "*.json"))):
            for item in _items(_load_json(f), container):
                if isinstance(item, dict) and item.get(key):
                    store[item[key]] = item
        seed = os.path.join(root, f"{container}_seed.json")
        if os.path.exists(seed):
            for item in _items(_load_json(seed), container):
                if isinstance(item, dict) and item.get(key):
                    store.setdefault(item[key], item)
    return ds


PAIR_GLOBS = ["test_pairs*.json", "*test*pair*.json", "*submission*pair*.json", "*canonical*test*.json",
              "test_set*.json", "*test_pairs*.jsonl", "*pairs*.json"]


def find_test_pairs(root: str) -> Optional[str]:
    for pat in PAIR_GLOBS:
        for base in (root, os.path.join(root, "*")):
            hits = sorted(glob.glob(os.path.join(base, pat)))
            if hits:
                return hits[0]
    return None


def load_pairs(path: str) -> list[dict]:
    if path.endswith(".jsonl"):
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    else:
        blob = _load_json(path)
        rows = blob if isinstance(blob, list) else next(
            (blob[k] for k in ("pairs", "tests", "test_pairs", "test_set", "items") if isinstance(blob.get(k), list)), [])
    out = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            continue
        tid = r.get("test_id") or r.get("id") or r.get("pair_id")
        trig = r.get("trigger_id") or r.get("trigger")
        if not (tid and trig):
            continue
        out.append({"test_id": tid, "merchant_id": r.get("merchant_id") or r.get("merchant"),
                    "trigger_id": trig, "customer_id": r.get("customer_id") or r.get("customer")})
    return out
