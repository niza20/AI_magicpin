"""AGENT 7 — Personalization. Picks the highest-value facts about THIS merchant.

Output `pz` is a dict name -> {"text", "value", "fid"} so the composer can only reference facts
that exist in the ledger. Priority: specific > verifiable > recent > actionable, weighted by the
trigger family (a perf dip wants CTR-vs-peer; a festival wants the active offer, etc.).
"""
from __future__ import annotations

import re
from typing import Any, Optional

from ..tools import ContextTools
from ..types import CategoryProfile, Personalization, TriggerAnalysis
from ..util import as_float, fmt_int, fmt_money, fmt_pct, humanize, owner_display
from .base import Agent

FAMILY_PRIORITY = {
    "perf_dip": ["ctr_gap", "ctr", "peer_ctr", "calls", "views", "offer", "stale_days", "locality"],
    "perf_spike": ["views", "calls", "offer", "ctr", "locality"],
    "knowledge": ["segment_count", "segment_match", "lapsed", "customers_total", "ctr_gap", "locality"],
    "planning": ["offer", "praise", "customers_total", "locality"],
    "seasonal": ["offer", "calls", "views", "locality"],
    "supply": ["customers_total", "locality"],
    "regulation": ["locality", "customers_total", "segment_match"],
    "competitor": ["rating", "reviews", "peer_rating", "offer", "praise", "ctr_above", "locality", "ctr_gap"],
    "festival": ["offer", "expired_offer", "views", "locality"],
    "weather": ["offer", "locality", "views"],
    "local_event": ["locality", "offer", "calls"],
    "trend": ["offer", "catalog_offer", "views", "locality"],
    "reputation": ["rating", "reviews", "peer_rating", "locality"],
    "milestone": ["rating", "reviews", "peer_rating", "views"],
    "dormant": ["stale_days", "views", "calls", "offer", "ctr_gap"],
    "recurring": ["stale_days", "views", "offer", "segment_match"],
    "account": ["views", "calls", "directions", "leads"],
    "profile": ["stale_days", "views", "ctr_gap", "rating"],
    "offer": ["offer", "expired_offer", "views", "calls"],
    "customer_recall": ["offer", "locality"],
    "customer_appointment": ["locality"],
    "generic": ["views", "calls", "offer", "ctr_gap", "locality"],
}


class PersonalizationAgent(Agent):
    name = "personalization"

    def run(self, tools: ContextTools, ta: TriggerAnalysis, prof: CategoryProfile) -> tuple[Personalization, dict]:
        L = tools.ledger
        pz: dict[str, dict] = {}

        def put(name: str, value: Any, path: Optional[str], text: str) -> None:
            if value in (None, "", []):
                return
            f = L.by_path(path) if path else None
            if f is None:
                f = L.derive(name, value, text, [path] if path else [])
            pz[name] = {"text": text, "value": value, "fid": f.id, "path": path}

        ident = tools.get_merchant_fact("identity") or {}
        sal = owner_display({"identity": ident}, prof.slug)
        sal_path = "merchant.identity.owner_first_name" if ident.get("owner_first_name") else "merchant.identity.name"
        put("salutation", sal, sal_path, sal)
        put("name", ident.get("name"), "merchant.identity.name", str(ident.get("name", "")))
        put("locality", ident.get("locality"), "merchant.identity.locality", str(ident.get("locality", "")))
        put("city", ident.get("city"), "merchant.identity.city", str(ident.get("city", "")))

        window = tools.perf_window_days()
        for m in ("views", "calls", "directions", "leads", "ctr"):
            v, p = tools.perf_metric(m)
            if v is not None:
                put(m, v, p, fmt_pct(v) if m == "ctr" else fmt_int(v))
        if window:
            put("window", window, "merchant.performance.window_days", f"{window} days")
        for m in ("views", "calls", "ctr", "directions"):
            d, p = tools.perf_delta(m)
            if d is not None:
                put(f"{m}_delta", d, p, fmt_pct(d))
                pz[f"{m}_delta"]["signed"] = d

        pctr, pctr_p = tools.get_peer_stat("ctr")
        if pctr is not None:
            put("peer_ctr", pctr, pctr_p, fmt_pct(pctr))
        prat, prat_p = tools.get_peer_stat("rating")
        if prat is not None:
            put("peer_rating", prat, prat_p, f"{prat:g}★")
        prev, prev_p = tools.get_peer_stat("reviews")
        if prev is not None:
            put("peer_reviews", prev, prev_p, fmt_int(prev))
        if "ctr" in pz and "peer_ctr" in pz and pz["ctr"]["value"] < pz["peer_ctr"]["value"]:
            pz["ctr_gap"] = {"text": f"{pz['ctr']['text']} vs {pz['peer_ctr']['text']}", "value": True,
                             "fid": pz["ctr"]["fid"], "path": pz["ctr"]["path"]}

        for label, keys in (("rating", ("rating", "avg_rating", "google_rating", "gbp_rating")),
                            ("reviews", ("reviews", "review_count", "total_reviews", "num_reviews"))):
            for f in L.find(layer="merchant"):
                if f.label in keys and as_float(f.value) is not None and "history" not in f.path:
                    v = as_float(f.value)
                    put(label, v, f.path, f"{v:g}★" if label == "rating" else fmt_int(v))
                    break

        offers = tools.get_active_offers()
        if offers:
            put("offer", offers[0][0], offers[0][1], offers[0][0])
        inactive = tools.get_inactive_offers()
        if inactive:
            put("expired_offer", inactive[0][0], inactive[0][2], inactive[0][0])
            pz["expired_offer"]["status"] = inactive[0][1]
        cat = tools.get_catalog_offers()
        if cat:
            # brief §3: "Haircut @ ₹99" beats "10% off" — rank service+price, then free service, never % discounts first
            def rank(t):
                tl = t[0].lower()
                return 0 if "@ ₹" in t[0] or "@₹" in t[0] else 1 if "free" in tl and "%" not in tl else 3 if "%" in tl else 2
            best = sorted(cat, key=rank)[0]
            put("catalog_offer", best[0], best[1], best[0])

        agg = tools.get_merchant_fact("customer_aggregate") or {}
        for name, keys in (("customers_total", ("total_unique_ytd", "total_unique", "active_count", "total", "unique_customers")),
                           ("lapsed", ("lapsed_180d_plus", "lapsed_count", "lapsed", "lapsed_90d_plus")),
                           ("retention", ("retention_6mo_pct", "retention_pct", "retention"))):
            for k in keys:
                if k in agg and as_float(agg[k]) is not None:
                    v = as_float(agg[k])
                    put(name, v, f"merchant.customer_aggregate.{k}", fmt_pct(v) if name == "retention" else fmt_int(v))
                    break

        for i, s in enumerate(tools.signals()):
            s_low = s.lower()
            m = re.search(r"stale_posts?[:_ ](\d+)", s_low)
            if m:
                put("stale_days", int(m.group(1)), f"merchant.signals[{i}]", m.group(1))
            if re.search(r"ctr_below_peer", s_low):
                put("ctr_below_signal", True, f"merchant.signals[{i}]", "below the peer median")
            seg = ta.anchor.get("segment", {}).get("value")
            if seg and _overlap(str(seg), s_low):
                put("segment_match", s, f"merchant.signals[{i}]", humanize(str(seg)).replace("adults", "adult"))

        if "ctr" in pz and "peer_ctr" in pz and pz["ctr"]["value"] >= pz["peer_ctr"]["value"]:
            pz["ctr_above"] = {"text": f"{pz['ctr']['text']} vs {pz['peer_ctr']['text']}", "value": True,
                               "fid": pz["ctr"]["fid"], "path": pz["ctr"]["path"]}
        for i, rt in enumerate(tools.get_merchant_fact("review_themes") or []):
            if not isinstance(rt, dict) or not rt.get("theme"):
                continue
            sent = str(rt.get("sentiment", "")).lower()
            key = "praise" if sent.startswith("pos") else "complaint" if sent.startswith("neg") else None
            if key and key not in pz:
                put(key, rt["theme"], f"merchant.review_themes[{i}].theme", humanize(rt["theme"]))
                if rt.get("common_quote"):
                    put(f"{key}_quote", rt["common_quote"], f"merchant.review_themes[{i}].common_quote", str(rt["common_quote"]))
        seg = ta.anchor.get("segment", {}).get("value")
        if seg:
            for k, v in (agg or {}).items():
                if k.endswith("_count") and _overlap(str(seg), k) and as_float(v):
                    put("segment_count", v, f"merchant.customer_aggregate.{k}", fmt_int(as_float(v)))
                    break
        hist = tools.get_conversation_history(4)
        if hist:
            last = hist[-1]
            put("last_touch", last.get("body") or last.get("text"), None, str(last.get("engagement", "")))
            pz["last_touch"]["from"] = last.get("from")

        order = FAMILY_PRIORITY.get(ta.family, FAMILY_PRIORITY["generic"])
        chosen = [k for k in order if k in pz][:4]
        out = Personalization(
            personalization_facts=[pz[k]["fid"] for k in chosen],
            reason=f"{ta.family}: prioritised {', '.join(chosen) or 'identity only'} (specific + verifiable + actionable)",
            salutation=sal)
        self.log(out.reason, {k: pz[k]["text"] for k in chosen})
        return out, pz


def _overlap(a: str, b: str) -> bool:
    ta = {w.rstrip("s") for w in re.split(r"[_\W]+", a.lower()) if len(w) > 3}
    tb = {w.rstrip("s") for w in re.split(r"[_\W]+", b.lower()) if len(w) > 3}
    return len(ta & tb) >= 2 or (len(ta) == 1 and ta <= tb)
