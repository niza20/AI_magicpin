"""Retrieval tools. Agents call these instead of receiving the raw dataset, so each agent only
touches the slice it needs, and every lookup is logged (the trace doubles as an audit trail)."""
from __future__ import annotations

import re
from typing import Any, Optional

from .ledger import FactLedger
from .util import as_float, deep_get, first_present, parse_date


class ContextTools:
    def __init__(self, category: dict, merchant: dict, trigger: dict,
                 customer: Optional[dict], ledger: FactLedger) -> None:
        self._c = category or {}
        self._m = merchant or {}
        self._t = trigger or {}
        self._cu = customer
        self.ledger = ledger
        self.calls: list[str] = []

    def _log(self, name: str, arg: Any = "") -> None:
        self.calls.append(f"{name}({arg})")

    # ---------------------------------------------------------- raw getters
    def get_merchant_fact(self, path: str, default: Any = None) -> Any:
        self._log("get_merchant_fact", path)
        return deep_get(self._m, path, default)

    def get_category_fact(self, path: str, default: Any = None) -> Any:
        self._log("get_category_fact", path)
        return deep_get(self._c, path, default)

    def get_trigger_fact(self, path: str, default: Any = None) -> Any:
        self._log("get_trigger_fact", path)
        return deep_get(self._t, path, default)

    def get_customer_fact(self, path: str, default: Any = None) -> Any:
        self._log("get_customer_fact", path)
        return deep_get(self._cu, path, default) if self._cu else default

    @property
    def has_customer(self) -> bool:
        return bool(self._cu)

    @property
    def category_slug(self) -> str:
        return str(self._c.get("slug") or self._m.get("category_slug") or "")

    # ------------------------------------------------------ semantic tools
    def perf_metric(self, name: str) -> tuple[Optional[float], Optional[str]]:
        """Find a performance metric regardless of nesting (performance.ctr / performance.30d.ctr)."""
        self._log("perf_metric", name)
        perf = self._m.get("performance") or {}
        if name in perf and as_float(perf[name]) is not None:
            return as_float(perf[name]), f"merchant.performance.{name}"
        for k, v in perf.items():
            if isinstance(v, dict) and name in v and as_float(v[name]) is not None and not k.startswith("delta"):
                return as_float(v[name]), f"merchant.performance.{k}.{name}"
        return None, None

    def perf_delta(self, name: str) -> tuple[Optional[float], Optional[str]]:
        self._log("perf_delta", name)
        perf = self._m.get("performance") or {}
        for k, v in perf.items():
            if isinstance(v, dict) and k.startswith("delta"):
                for key in (f"{name}_pct", name, f"{name}_delta"):
                    if key in v and as_float(v[key]) is not None:
                        return as_float(v[key]), f"merchant.performance.{k}.{key}"
        return None, None

    def perf_window_days(self) -> Optional[int]:
        v = deep_get(self._m, "performance.window_days")
        return int(v) if isinstance(v, (int, float)) else None

    def get_peer_stat(self, name: str) -> tuple[Optional[float], Optional[str]]:
        self._log("get_peer_stat", name)
        ps = self._c.get("peer_stats") or {}
        for key in (name, f"avg_{name}", f"median_{name}", f"{name}_median", f"typical_{name}"):
            if key in ps and as_float(ps[key]) is not None:
                return as_float(ps[key]), f"category.peer_stats.{key}"
        return None, None

    def peer_scope(self) -> Optional[str]:
        return deep_get(self._c, "peer_stats.scope")

    def get_active_offers(self) -> list[tuple[str, str]]:
        self._log("get_active_offers")
        return [(o.get("title"), f"merchant.offers[{i}].title") for i, o in enumerate(self._m.get("offers") or [])
                if isinstance(o, dict) and o.get("title") and str(o.get("status", "active")).lower() == "active"]

    def get_inactive_offers(self) -> list[tuple[str, str, str]]:
        self._log("get_inactive_offers")
        return [(o.get("title"), str(o.get("status")), f"merchant.offers[{i}].title")
                for i, o in enumerate(self._m.get("offers") or [])
                if isinstance(o, dict) and o.get("title") and str(o.get("status", "active")).lower() != "active"]

    def get_catalog_offers(self) -> list[tuple[str, str]]:
        self._log("get_catalog_offers")
        out = []
        for i, o in enumerate(self._c.get("offer_catalog") or []):
            title = o.get("title") if isinstance(o, dict) else o
            if title:
                out.append((str(title), f"category.offer_catalog[{i}]" + (".title" if isinstance(o, dict) else "")))
        return out

    def get_conversation_history(self, n: int = 6) -> list[dict]:
        self._log("get_conversation_history", n)
        hist = self._m.get("conversation_history") or []
        if isinstance(hist, dict):
            hist = hist.get("turns") or hist.get("messages") or []
        return [h for h in hist if isinstance(h, dict)][-n:]

    def signals(self) -> list[str]:
        sig = self._m.get("signals") or []
        return [s if isinstance(s, str) else str(first_present(s, ["name", "kind", "signal"], s)) for s in sig]

    def search_category_digest(self, item_id: Optional[str] = None, terms: Optional[list[str]] = None) -> list[tuple[dict, str]]:
        """Return digest items (with their ledger path). By id first; otherwise rank by term overlap."""
        self._log("search_category_digest", item_id or terms)
        items = [(d, f"category.digest[{i}]") for i, d in enumerate(self._c.get("digest") or []) if isinstance(d, dict)]
        if item_id:
            hit = [x for x in items if x[0].get("id") == item_id]
            if hit:
                return hit
        if not terms:
            return items
        tl = [t.lower() for t in terms if t]

        def score(d: dict) -> int:
            blob = " ".join(str(v) for v in d.values()).lower()
            return sum(1 for t in tl if t in blob)
        return sorted(items, key=lambda x: -score(x[0]))

    def seasonal_beats(self) -> list[tuple[dict, str]]:
        return [(b, f"category.seasonal_beats[{i}]") for i, b in enumerate(self._c.get("seasonal_beats") or []) if isinstance(b, dict)]

    def trend_signals(self) -> list[tuple[dict, str]]:
        return [(t, f"category.trend_signals[{i}]") for i, t in enumerate(self._c.get("trend_signals") or []) if isinstance(t, dict)]

    def content_library(self) -> list[tuple[dict, str]]:
        return [(c, f"category.patient_content_library[{i}]")
                for i, c in enumerate(self._c.get("patient_content_library") or self._c.get("content_library") or [])
                if isinstance(c, dict)]

    def reference_date(self):
        """'Now' for this composition, taken from context (never the wall clock → deterministic)."""
        for path in ("trigger.detected_at", "trigger.created_at", "trigger.payload.date", "trigger.ts"):
            d = parse_date(deep_get({"trigger": self._t}, path))
            if d:
                return d
        exp = parse_date(self._t.get("expires_at"))
        if exp:
            from datetime import timedelta
            return exp - timedelta(days=7)
        return None
