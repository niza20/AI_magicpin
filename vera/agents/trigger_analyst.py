"""AGENT 2 — Trigger Analyst. Answers: WHY IS VERA MESSAGING NOW?

Classifies the trigger into a family (robust to unseen `kind` strings), resolves the most important
trigger facts into an `anchor` bundle (each entry carries its ledger fact id), and states the
goal/action. Payload keys vary between dataset versions, so every lookup goes through alias lists.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any, Optional

from ..ledger import FactLedger
from ..tools import ContextTools
from ..types import TriggerAnalysis
from ..util import (as_float, fmt_date, fmt_int, fmt_money, fmt_num, fmt_pct, humanize,
                    months_between, parse_date, parse_datetime)
from .base import Agent

# Order matters: first match wins.
FAMILY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("customer_appointment", ("appointment", "booking_reminder", "visit_tomorrow")),
    ("customer_recall", ("recall", "lapsed", "winback", "win_back", "unplanned_slot", "refill", "renewal_customer",
                         "membership_expir", "birthday", "followup", "follow_up", "trial", "wedding", "package")),
    ("supply", ("supply", "batch", "stock_alert", "shortage", "product_recall", "drug_recall", "withdrawal")),
    ("planning", ("planning", "intent", "program_draft", "proposal")),
    ("regulation", ("regulation", "compliance", "circular", "policy", "mandate", "license")),
    ("knowledge", ("research", "digest", "journal", "study", "cde", "webinar", "conference", "clinical_update", "knowledge")),
    ("perf_dip", ("dip", "drop", "decline", "below_peer", "fall", "down")),
    ("perf_spike", ("spike", "surge", "jump", "growth", "peak")),
    ("seasonal", ("category_seasonal", "seasonal_demand", "demand_shift", "seasonal")),
    ("milestone", ("milestone", "crossed", "anniversary", "record")),
    ("competitor", ("competitor", "rival", "new_entrant", "opened_nearby")),
    ("trend", ("trend", "search", "query", "demand")),
    ("festival", ("festival", "holiday", "diwali", "holi", "eid", "christmas", "navratri", "season")),
    ("weather", ("weather", "heat", "rain", "monsoon", "cold", "aqi", "pollution", "flood", "storm")),
    ("local_event", ("news", "event", "local", "match", "ipl", "traffic", "closure", "strike", "concert")),
    ("reputation", ("review", "rating", "feedback", "complaint")),
    ("dormant", ("dormant", "inactive", "no_reply", "silent", "unresponsive")),
    ("recurring", ("scheduled", "recurring", "weekly", "curious", "check_in", "cadence", "ask")),
    ("offer", ("offer", "catalog", "price", "deal")),
    ("account", ("renewal", "subscription", "expir", "payment", "invoice", "winback", "win_back", "trial_ending", "plan_expir")),
    ("profile", ("profile", "gbp", "photo", "hours", "stale", "post", "listing", "verification", "unverified", "verify")),
]

FAMILY_GOALS = {
    "knowledge": ("INFORM", "share a verifiable, practice-relevant update and offer to package it", "send_digest_summary"),
    "regulation": ("INFORM", "flag a compliance change before it bites and offer a checklist", "send_compliance_checklist"),
    "perf_dip": ("RECOMMEND", "name the drop and get a YES to one concrete recovery action", "draft_google_post"),
    "perf_spike": ("RECOMMEND", "convert the spike while traffic is high", "feature_offer_in_post"),
    "milestone": ("RECOMMEND", "celebrate and turn the milestone into social proof", "draft_google_post"),
    "competitor": ("RECOMMEND", "protect share against the new entrant", "feature_offer_in_post"),
    "trend": ("RECOMMEND", "ride rising demand with a matching post/offer", "draft_google_post"),
    "festival": ("RECOMMEND", "get a festival-timed post/offer live before the date", "feature_offer_in_post"),
    "weather": ("RECOMMEND", "adapt this week's messaging to the weather", "draft_google_post"),
    "local_event": ("RECOMMEND", "adapt to a local disruption/opportunity today", "draft_google_post"),
    "reputation": ("RECOMMEND", "respond to an emerging review theme", "reply_to_reviews_draft"),
    "dormant": ("DISCOVER", "re-open the conversation with a low-effort ask", "draft_google_post"),
    "recurring": ("DISCOVER", "curiosity/ask-the-merchant touch that yields content", "draft_google_post"),
    "account": ("ACT", "keep the account active — renewal/plan action", "subscription_renewal"),
    "planning": ("ACT", "merchant already asked — deliver a concrete first draft now", "draft_program"),
    "customer_promo": ("ACT", "invite the customer back with the merchant's real offer", "book_slot"),
    "seasonal": ("RECOMMEND", "adjust stock/offers to the season's demand shift", "draft_google_post"),
    "supply": ("ACT", "protect customers from an affected batch — shelf check now", "shelf_check"),
    "profile": ("RECOMMEND", "fix a concrete profile gap", "draft_google_post"),
    "offer": ("RECOMMEND", "get the right service+price offer live", "reactivate_offer"),
    "customer_recall": ("ACT", "bring the customer back with a concrete slot/price", "book_slot"),
    "customer_appointment": ("ACT", "confirm tomorrow's appointment", "confirm_appointment"),
    "generic": ("INFORM", "surface the event and offer one concrete next step", "draft_google_post"),
}

INFORMATIONAL = {"knowledge", "regulation"}
_HONORIFIC = re.compile(r"^(mr|mrs|ms|miss|dr|shri|smt)\.?\s+", re.I)


def classify_family(kind: str, scope: str) -> str:
    k = (kind or "").lower()
    for fam, keys in FAMILY_RULES:
        if any(key in k for key in keys):
            if scope == "customer" and not fam.startswith("customer_"):
                continue
            if scope != "customer" and fam.startswith("customer_"):
                continue      # e.g. merchant-scope "winback_eligible" is about the merchant's own plan
            return fam
    return "customer_recall" if scope == "customer" else "generic"


class TriggerAnalyst(Agent):
    name = "trigger_analyst"

    def run(self, tools: ContextTools, ledger: FactLedger, now: Optional[str] = None) -> TriggerAnalysis:
        self.tools, self.ledger = tools, ledger
        trig = tools.get_trigger_fact("") or {}
        kind = str(tools.get_trigger_fact("kind") or tools.get_trigger_fact("type") or "unknown")
        scope = str(tools.get_trigger_fact("scope") or ("customer" if tools.has_customer else "merchant"))
        fam = classify_family(kind, scope)
        self.payload = tools.get_trigger_fact("payload") or {}
        anchor: dict[str, dict] = {}
        self.placeholder = bool(isinstance(self.payload, dict) and self.payload.get("placeholder"))
        if self.placeholder:
            anchor["_placeholder"] = {"value": True}
        getattr(self, f"_a_{fam}", self._a_generic)(anchor)
        real = [k for k in anchor if not k.startswith("_")]
        payload_has_data = isinstance(self.payload, dict) and any(
            v not in (None, "", [], {}) for k, v in self.payload.items() if k not in ("merchant_id", "customer_id", "category"))
        if not real and not self.placeholder:
            # the kind name matched a family but the payload doesn't fit it (e.g. "price_benchmark_shift" ≠ an offer):
            # don't force a wrong story — surface the payload as-is
            if payload_has_data and fam not in ("generic",) and not fam.startswith("customer_"):
                fam = "generic"
            self._a_generic(anchor)
        if tools.has_customer and not fam.startswith("customer_"):
            # a customer context attached to a merchant-level event (brief §8 twist): write to the customer,
            # never exposing merchant-internal facts (competitors, CTR, ...)
            anchor["_event_family"] = {"value": fam}
            self._a_customer_recall(anchor)
            fam = "customer_promo"
        urgency = int(as_float(tools.get_trigger_fact("urgency")) or 2)
        mode, goal, action = FAMILY_GOALS.get(fam, FAMILY_GOALS["generic"])
        exp = tools.get_trigger_fact("expires_at")
        expired = False
        if now and exp:
            n, e = parse_datetime(now), parse_datetime(exp)
            expired = bool(n and e and n > e)
        eff = anchor.get("effective")
        if now and eff and fam == "regulation":
            n, d = parse_datetime(now), parse_datetime(str(eff.get("value"))[:10] + "T00:00:00Z")
            if n and d and 0 < (d - n).days <= 365:          # days to the payload deadline, from the judge's own clock
                days = (d - n).days
                self._derive_put(anchor, "days_left", days, f"{days} days", [eff.get("path") or "trigger.payload.deadline_iso"])
        why = self._why_now(fam, kind, anchor)
        ta = TriggerAnalysis(trigger_type=kind, family=fam, urgency=max(1, min(5, urgency)), why_now=why,
                             primary_goal=goal, recommended_action=action, expiration=exp,
                             key_fact_ids=[v["fid"] for k, v in anchor.items() if not k.startswith("_") and v.get("fid")],
                             anchor=anchor, is_expired=expired, is_informational=fam in INFORMATIONAL)
        self.log(f"kind={kind} → family={fam}, urgency={ta.urgency}, anchor={[k for k in anchor if not k.startswith('_')]}",
                 {"why_now": why, "expired": expired})
        return ta

    # ---------------------------------------------------------------- helpers
    def _pv(self, *keys: str, src: Optional[dict] = None, base: str = "trigger.payload") -> tuple[Any, Optional[str]]:
        src = self.payload if src is None else src
        if not isinstance(src, dict):
            return None, None
        for k in keys:
            v = src.get(k)
            if v not in (None, "", [], {}):
                return v, f"{base}.{k}"
        return None, None

    def _put(self, anchor: dict, name: str, value: Any, path: Optional[str], text: Optional[str] = None) -> None:
        if value in (None, "", [], {}):
            return
        fact = self.ledger.by_path(path) if path else None
        if fact is None:
            fact = self.ledger.derive(name, value, text or str(value), [path] if path else [], kind="text")
        anchor[name] = {"text": text if text is not None else str(value), "value": value, "fid": fact.id, "path": path}

    def _derive_put(self, anchor: dict, name: str, value: Any, text: str, sources: list[str]) -> None:
        fact = self.ledger.derive(name, value, text, sources)
        anchor[name] = {"text": text, "value": value, "fid": fact.id, "path": fact.path}

    def _put_pct(self, anchor: dict, name: str, value: Any, path: Optional[str]) -> None:
        f = as_float(value)
        if f is None:
            self._put(anchor, name, value, path)
            return
        is_ratio = -1.5 < f < 1.5 and not (isinstance(value, str) and "%" in value)
        self._put(anchor, name, f, path, fmt_pct(f, ratio=is_ratio))
        anchor[name]["signed"] = f

    def _put_date(self, anchor: dict, name: str, value: Any, path: Optional[str], weekday: bool = False) -> None:
        d = parse_date(value)
        if d:
            self._put(anchor, name, value, path, fmt_date(d, weekday))
            anchor[name]["date"] = d
            dt = parse_datetime(value)
            if dt and isinstance(value, str) and "T" in value and (dt.hour or dt.minute):
                anchor[name]["time"] = dt.strftime("%-I:%M%p").replace(":00", "").lower()
        else:
            self._put(anchor, name, value, path)

    def _days_until(self, anchor: dict, key: str) -> None:
        ref = self.tools.reference_date(strong_only=True)
        d = anchor.get(key, {}).get("date")
        if ref and d and "days_until" not in anchor:
            delta = (d - ref).days
            if 0 <= delta <= 60:
                self._derive_put(anchor, "days_until", delta, str(delta), [anchor[key]["path"] or "", "trigger.expires_at"])

    # ------------------------------------------------------------ family anchors
    def _resolve_digest_item(self, prefer_kinds: tuple[str, ...]) -> tuple[Optional[dict], str]:
        item, path = self._pv("top_item", "item", "digest_item")
        if isinstance(item, dict):
            return item, path
        iid, _ = self._pv("top_item_id", "item_id", "digest_item_id", "digest_id", "id")
        if isinstance(iid, str):
            hits = self.tools.search_category_digest(item_id=iid)
            if hits and hits[0][0].get("id") == iid:
                return hits[0]
        terms = self.tools.signals() + [str(self.payload.get("topic", ""))]
        ranked = self.tools.search_category_digest(terms=terms)
        for d, p in ranked:
            if str(d.get("kind", "")).lower() in prefer_kinds:
                return d, p
        return ranked[0] if ranked else (None, "")

    def _a_knowledge(self, anchor: dict, prefer=("research", "study", "trial", "cde", "tech")) -> None:
        item, base = self._resolve_digest_item(prefer)
        if not item:
            return
        is_event = str(item.get("kind", "")).lower() in ("cde", "event", "webinar", "conference")
        if is_event:
            v, p = self._pv("date", "starts_at", src=item, base=base)
            if v:
                self._put_date(anchor, "event_date", v, p)
            for name, keys in (("credits", ("credits", "cde_credits")), ("fee", ("fee", "price", "cost"))):
                v, p = self._pv(*keys)
                if v is None:
                    v, p = self._pv(*keys, src=item, base=base)
                if v is not None:
                    self._put(anchor, name, v, p, humanize(v) if isinstance(v, str) else fmt_num(as_float(v) or 0))
        for name, keys in (("title", ("title", "headline")), ("source", ("source", "journal", "publisher")),
                           ("trial_n", ("trial_n", "n", "sample_size")), ("segment", ("patient_segment", "segment", "audience")),
                           ("summary", ("summary", "abstract", "takeaway")), ("url", ("url", "link")),
                           ("effective", ("effective_date", "effective_from", "deadline", "date")),
                           ("item_kind", ("kind", "type")), ("actionable", ("actionable", "action", "what_to_do"))):
            v, p = self._pv(*keys, src=item, base=base)
            if v is None:
                continue
            if name == "trial_n" and as_float(v):
                self._put(anchor, name, v, p, fmt_int(as_float(v)))
            elif name == "segment":
                self._put(anchor, name, v, p, humanize(v))
            elif name == "effective" and parse_date(v):
                if not is_event:
                    self._put_date(anchor, name, v, p)
            else:
                self._put(anchor, name, v, p)
        v, p = self._pv("deadline_iso", "deadline", "effective_date")
        if v and parse_date(v):
            self._put_date(anchor, "effective", v, p)
        anchor["_item"] = {"value": item}
        anchor["_is_event"] = {"value": is_event}

    def _a_regulation(self, anchor: dict) -> None:
        self._a_knowledge(anchor, prefer=("regulation", "compliance", "circular", "policy"))
        for name, keys in (("title", ("title", "headline", "rule", "change")), ("source", ("authority", "source", "regulator")),
                           ("summary", ("summary", "details")), ("effective", ("effective_date", "deadline", "effective_from"))):
            if name in anchor:
                continue
            v, p = self._pv(*keys)
            if name == "effective" and parse_date(v):
                self._put_date(anchor, name, v, p)
            elif v is not None:
                self._put(anchor, name, v, p)

    def _metric_name(self) -> tuple[Optional[str], Optional[str]]:
        v, p = self._pv("metric", "kpi", "measure", "signal_metric")
        return (str(v), p) if v else (None, None)

    def _a_perf(self, anchor: dict, direction: int) -> None:
        metric, mpath = self._metric_name()
        if not metric:
            # placeholder trigger: take the merchant's own strongest move in the trigger's direction
            best = None
            for m in ("calls", "views", "ctr", "directions", "leads"):
                d, p = self.tools.perf_delta(m)
                if d is not None and d * direction > 0 and (best is None or abs(d) > abs(best[1])):
                    best = (m, d, p)
            if not best:
                if direction < 0 and str(self.tools.get_merchant_fact("subscription.status", "")).lower() == "expired":
                    d = self.tools.get_merchant_fact("subscription.days_since_expiry")
                    if d:
                        self._put(anchor, "expired_days", d, "merchant.subscription.days_since_expiry", fmt_num(as_float(d) or 0))
                        plan = self.tools.get_merchant_fact("subscription.plan")
                        if plan:
                            self._put(anchor, "plan", plan, "merchant.subscription.plan")
                for m in ("views", "calls"):
                    d, p = self.tools.perf_delta(m)
                    if d is not None:
                        self._put_pct(anchor, f"flat_{m}", d, p)
                anchor["_no_move"] = {"value": True}
                anchor["_direction"] = {"value": direction}
                return
            if best:
                metric, mpath = best[0], None
                self._put(anchor, "metric", best[0], None, best[0].replace("ctr", "CTR"))
                self._put_pct(anchor, "delta", best[1], best[2])
                self._put(anchor, "window", "7d", "merchant.performance.delta_7d", "7d")
                v, p = self.tools.perf_metric(best[0])
                if v is not None:
                    self._put(anchor, "metric_30d", v, p, fmt_pct(v) if best[0] == "ctr" else fmt_int(v))
                anchor["_direction"] = {"value": direction}
                return
        if metric:
            self._put(anchor, "metric", metric, mpath, humanize(metric).replace("ctr", "CTR"))
        dv, dp = self._pv("delta_pct", "change_pct", "pct_change", "delta", "change", "drop_pct", "increase_pct", "wow_pct")
        if dv is not None:
            self._put_pct(anchor, "delta", dv, dp)
        elif metric:
            d, p = self.tools.perf_delta(metric)
            if d is not None:
                self._put_pct(anchor, "delta", d, p)
        for name, keys in (("window", ("window", "period", "vs", "compare_to", "baseline_window", "comparison")),
                           ("current", ("value", "current", "now", "current_value", "yesterday")),
                           ("baseline", ("baseline", "vs_baseline", "avg", "average", "previous", "prior", "baseline_value"))):
            v, p = self._pv(*keys)
            if v is None:
                continue
            f = as_float(v)
            if f is not None and name != "window":
                txt = fmt_pct(f) if metric and "ctr" in metric.lower() and f < 1 else fmt_num(f)
                self._put(anchor, name, f, p, txt)
            else:
                self._put(anchor, name, v, p, humanize(v))
        v, p = self._pv("likely_driver", "driver", "cause")
        if v:
            self._put(anchor, "driver", v, p, humanize(v))
        v, p = self._pv("season_note", "seasonal_note")
        if v and self.payload.get("is_expected_seasonal") is not False:
            self._put(anchor, "season_note", v, p, humanize(v))
        if metric and "current" not in anchor:
            v, p = self.tools.perf_metric(metric)
            if v is not None:
                txt = fmt_pct(v) if "ctr" in metric.lower() else fmt_int(v)
                self._put(anchor, "metric_30d", v, p, txt)
        anchor["_direction"] = {"value": direction}

    def _a_perf_dip(self, anchor: dict) -> None:
        self._a_perf(anchor, -1)

    def _a_perf_spike(self, anchor: dict) -> None:
        self._a_perf(anchor, +1)

    def _a_milestone(self, anchor: dict) -> None:
        metric, mpath = self._metric_name()
        if not metric:
            metric, mpath = self._pv("milestone", "milestone_type", "type")
        if metric:
            self._put(anchor, "metric", metric, mpath, humanize(metric))
        cur, cp = self._pv("value_now", "current", "current_value")
        tgt, tp = self._pv("milestone_value", "target", "threshold")
        if cur is not None and tgt is not None and as_float(cur) is not None and as_float(tgt) is not None:
            self._put(anchor, "current", cur, cp, fmt_int(as_float(cur)))
            self._put(anchor, "target", tgt, tp, fmt_int(as_float(tgt)))
            gap = as_float(tgt) - as_float(cur)
            if gap > 0:
                self._derive_put(anchor, "gap", gap, fmt_int(gap), [cp, tp])
        v, p = self._pv("value", "count", "reached") if cur is None else (None, None)
        if v is None and cur is None and tgt is not None:
            v, p = tgt, tp
        if v is not None:
            f = as_float(v)
            self._put(anchor, "value", v, p, fmt_int(f) if f is not None and f >= 1 else str(v))
        if self.placeholder and "value" not in anchor and "current" not in anchor:
            for m in ("views", "calls", "directions"):
                val, path = self.tools.perf_metric(m)
                if val:
                    self._put(anchor, "metric", m, None, m)
                    self._put(anchor, "value", val, path, fmt_int(val))
                    anchor["_from_merchant"] = {"value": True}
                    break
        v, p = self._pv("date", "reached_on", "reached_at")
        if v:
            self._put_date(anchor, "date", v, p)

    def _a_competitor(self, anchor: dict) -> None:
        for name, keys in (("name", ("competitor_name", "name", "business_name")),
                           ("distance", ("distance_km", "distance", "km")),
                           ("locality", ("locality", "area", "location")),
                           ("rating", ("rating", "competitor_rating")),
                           ("offer", ("offer", "their_offer", "headline_offer"))):
            v, p = self._pv(*keys)
            if v is None:
                continue
            if name == "distance" and as_float(v) is not None:
                self._put(anchor, name, v, p, f"{fmt_num(as_float(v))} km")
            else:
                self._put(anchor, name, v, p)
        v, p = self._pv("opened_on", "opened_date", "opened_at", "date", "detected_at")
        if v:
            self._put_date(anchor, "date", v, p)

    def _a_trend(self, anchor: dict) -> None:
        q, qp = self._pv("query", "term", "keyword", "search_term", "topic")
        src = self.payload
        base = "trigger.payload"
        if not q:
            trends = self.tools.trend_signals()
            if trends:
                src, base = trends[0]
                q, qp = self._pv("query", "term", "keyword", src=src, base=base)
        if q:
            self._put(anchor, "query", q, qp)
        dv, dp = self._pv("delta_yoy", "delta_pct", "delta", "change_pct", "growth", src=src, base=base)
        if dv is not None:
            self._put_pct(anchor, "delta", dv, dp)
        for name, keys in (("segment", ("segment_age", "segment", "audience")), ("window", ("window", "period")),
                           ("city", ("city", "geo", "region"))):
            v, p = self._pv(*keys, src=src, base=base)
            if v is not None:
                self._put(anchor, name, v, p)

    def _a_festival(self, anchor: dict) -> None:
        v, p = self._pv("festival", "name", "event", "occasion", "holiday")
        if v:
            self._put(anchor, "name", v, p, humanize(v).title() if isinstance(v, str) and v.islower() else str(v))
        v, p = self._pv("date", "festival_date", "starts_on", "start_date", "event_date")
        if v:
            self._put_date(anchor, "date", v, p)
        dv, dp = self._pv("days_until", "days_to", "days_left", "in_days")
        if dv is not None:
            self._put(anchor, "days_until", dv, dp, fmt_num(as_float(dv) or 0) if as_float(dv) is not None else str(dv))
        else:
            self._days_until(anchor, "date")
        self._seasonal(anchor)

    _WEATHER_WORDS = {"heat": ("heat", "summer", "ors", "electrolyte", "hydration", "sun"),
                      "rain": ("monsoon", "rain", "frizz", "dengue", "fever", "humid"),
                      "monsoon": ("monsoon", "rain", "frizz", "dengue", "fever", "humid"),
                      "cold": ("winter", "cold", "flu", "cough"), "aqi": ("pollution", "aqi", "asthma", "mask"),
                      "pollution": ("pollution", "aqi", "asthma", "mask")}

    def _seasonal(self, anchor: dict, weather: bool = False) -> None:
        """Attach a seasonal beat only when it is topically relevant (never by month alone for weather)."""
        name = str(anchor.get("name", {}).get("value", "")).lower()
        cond = " ".join(str(anchor.get(k, {}).get("value", "")) for k in ("condition", "temp")).lower()
        if weather and anchor.get("temp") and as_float(anchor["temp"]["value"]) and as_float(anchor["temp"]["value"]) >= 35:
            cond += " heat"
        d = anchor.get("date", {}).get("date") or self.tools.reference_date()
        for beat, p in self.tools.seasonal_beats():
            note = str(beat.get("note", ""))
            rng = str(beat.get("month_range", beat.get("months", "")))
            if weather:
                keys = {w for k, ws in self._WEATHER_WORDS.items() if k in cond for w in ws}
                hit = any(k in note.lower() for k in keys)
            else:
                hit = (name and name.split()[0] in note.lower()) or (d and _month_in_range(d, rng))
            if hit and note:
                self._put(anchor, "beat", note, f"{p}.note")
                break

    def _a_weather(self, anchor: dict) -> None:
        for name, keys in (("temp", ("temp_c", "temperature", "max_temp", "temp", "max_temp_c")),
                           ("condition", ("condition", "alert", "kind", "event", "type", "description")),
                           ("city", ("city", "locality", "area", "region")),
                           ("aqi", ("aqi",)), ("rain_mm", ("rain_mm", "rainfall_mm"))):
            v, p = self._pv(*keys)
            if v is None:
                continue
            if name == "temp" and as_float(v) is not None:
                self._put(anchor, name, v, p, f"{fmt_num(as_float(v))}°C")
            else:
                self._put(anchor, name, v, p, humanize(v) if name == "condition" else str(v))
        v, p = self._pv("date", "forecast_date", "valid_until", "until")
        if v:
            self._put_date(anchor, "date", v, p)
        self._seasonal(anchor, weather=True)

    def _a_local_event(self, anchor: dict) -> None:
        for name, keys in (("headline", ("headline", "title", "event", "name", "match", "description")),
                           ("location", ("location", "area", "locality", "venue", "road")),
                           ("duration", ("duration", "duration_hours", "hours")),
                           ("impact", ("impact", "expected_impact", "note")),
                           ("source", ("source", "publisher"))):
            v, p = self._pv(*keys)
            if v is None:
                continue
            if name == "duration" and as_float(v) is not None:
                self._put(anchor, name, v, p, f"{fmt_num(as_float(v))} hours")
            else:
                self._put(anchor, name, v, p)
        v, p = self._pv("date", "match_time_iso", "starts_at", "start", "event_date", "time")
        if v:
            self._put_date(anchor, "date", v, p)
        if isinstance(self.payload, dict) and "is_weeknight" in self.payload:
            anchor["_weeknight"] = {"value": bool(self.payload["is_weeknight"])}

    def _a_reputation(self, anchor: dict) -> None:
        for name, keys in (("theme", ("theme", "topic", "keyword", "issue")),
                           ("count", ("count", "review_count", "mentions", "n_reviews", "occurrences", "occurrences_30d")),
                           ("window", ("window", "period", "timeframe")),
                           ("sentiment", ("sentiment", "polarity")),
                           ("rating", ("avg_rating", "rating"))):
            v, p = self._pv(*keys)
            if v is not None:
                self._put(anchor, name, v, p, humanize(v) if name in ("window", "sentiment", "theme") else str(v))
        v, p = self._pv("common_quote", "quote")
        if isinstance(v, str):
            self._put(anchor, "quote", v, p, v[:90])
        v, p = self._pv("trend")
        if isinstance(v, str):
            self._put(anchor, "trend", v, p, humanize(v))
        v, p = self._pv("quotes", "samples", "examples", "sample_reviews", "excerpts")
        if isinstance(v, list) and v:
            q = v[0] if isinstance(v[0], str) else (v[0].get("text") if isinstance(v[0], dict) else None)
            if q:
                qp = f"{p}[0]" + ("" if isinstance(v[0], str) else ".text")
                self._put(anchor, "quote", q, qp, q[:90])

    def _a_dormant(self, anchor: dict) -> None:
        v, p = self._pv("days_since_last", "days_since_last_merchant_message", "days_inactive", "days", "days_since_last_reply", "silent_days")
        if v is not None:
            self._put(anchor, "days", v, p, fmt_num(as_float(v) or 0) if as_float(v) is not None else str(v))
        v, p = self._pv("last_message_at", "last_seen", "last_reply_at")
        if v:
            self._put_date(anchor, "last", v, p)
        v, p = self._pv("last_topic")
        if v:
            self._put(anchor, "last_topic", v, p, humanize(v))
        self._a_recurring(anchor)

    def _a_recurring(self, anchor: dict) -> None:
        for name, keys in (("topic", ("topic", "theme", "question", "prompt", "ask", "ask_template")), ("cadence", ("cadence", "day", "schedule"))):
            v, p = self._pv(*keys)
            if v is not None and name not in anchor:
                self._put(anchor, name, v, p, humanize(v) if name == "cadence" else str(v))

    def _a_account(self, anchor: dict) -> None:
        v, p = self._pv("days_remaining", "days_left", "days_to_expiry")
        if v is None:
            v = self.tools.get_merchant_fact("subscription.days_remaining")
            p = "merchant.subscription.days_remaining" if v is not None else None
        if v is not None:
            self._put(anchor, "days_left", v, p, fmt_num(as_float(v) or 0) if as_float(v) is not None else str(v))
        v, p = self._pv("renewal_date", "expires_on", "expiry_date", "due_date")
        if v:
            self._put_date(anchor, "date", v, p)
        v, p = self._pv("plan", "plan_name")
        if v is None:
            v = self.tools.get_merchant_fact("subscription.plan")
            p = "merchant.subscription.plan" if v is not None else None
        if v is not None:
            self._put(anchor, "plan", v, p)
        v, p = self._pv("amount", "price", "renewal_amount")
        if v is not None and as_float(v) is not None:
            self._put(anchor, "amount", v, p, fmt_money(as_float(v)))
        v, p = self._pv("days_since_expiry")
        if v is None:
            v = self.tools.get_merchant_fact("subscription.days_since_expiry")
            p = "merchant.subscription.days_since_expiry" if v is not None else None
        if v is not None and str(self.tools.get_merchant_fact("subscription.status", "")).lower() != "active":
            self._put(anchor, "expired_days", v, p, fmt_num(as_float(v) or 0))
            anchor.pop("days_left", None)
        v, p = self._pv("perf_dip_pct")
        if v is not None:
            self._put_pct(anchor, "dip", v, p)
        v, p = self._pv("lapsed_customers_added_since_expiry", "lapsed_customers")
        if v is not None:
            self._put(anchor, "lapsed_new", v, p, fmt_num(as_float(v) or 0))

    def _a_profile(self, anchor: dict) -> None:
        v, p = self._pv("verified")
        if v is False or (v is None and self.tools.get_merchant_fact("identity.verified") is False):
            self._put(anchor, "unverified", "false", p or "merchant.identity.verified", "not verified")
            v, p = self._pv("verification_path", "path")
            if v:
                self._put(anchor, "verify_path", v, p, humanize(v).replace(" or ", " or a "))
            v, p = self._pv("estimated_uplift_pct", "uplift_pct")
            if v is not None:
                self._put_pct(anchor, "uplift", v, p)
        v, p = self._pv("missing_fields", "missing", "gaps", "fields")
        if isinstance(v, list) and v:
            self._put(anchor, "missing", ", ".join(map(str, v[:3])), p, ", ".join(humanize(x) for x in v[:3]))
        elif v:
            self._put(anchor, "missing", v, p, humanize(v))
        for name, keys in (("completeness", ("completeness", "completeness_pct", "profile_pct")),
                           ("days", ("days_since_last_post", "last_post_days", "days"))):
            v, p = self._pv(*keys)
            if v is not None:
                if name == "completeness":
                    self._put_pct(anchor, name, v, p)
                else:
                    self._put(anchor, name, v, p, fmt_num(as_float(v) or 0) if as_float(v) is not None else str(v))

    def _a_offer(self, anchor: dict) -> None:
        v, p = self._pv("offer_title", "title", "offer", "name")
        if v is None:
            oid, _ = self._pv("offer_id", "id")
            for i, o in enumerate(self.tools.get_merchant_fact("offers") or []):
                if isinstance(o, dict) and o.get("id") == oid:
                    v, p = o.get("title"), f"merchant.offers[{i}].title"
        if v is not None:
            self._put(anchor, "offer", v, p)
        v, p = self._pv("status", "state")
        if v:
            self._put(anchor, "status", v, p, humanize(v))
        v, p = self._pv("expired_on", "expires_on", "date", "ended_at")
        if v:
            self._put_date(anchor, "date", v, p)

    def _a_customer_recall(self, anchor: dict) -> None:
        v, p = self._pv("last_visit", "last_visit_date", "last_service_date", "last_refill", "trial_date", "trial_completed")
        if v is None:
            v = self.tools.get_customer_fact("relationship.last_visit")
            p = "customer.relationship.last_visit" if v else None
        if v:
            self._put_date(anchor, "last_visit", v, p)
        v, p = self._pv("due_date", "recall_date", "due_on")
        if v:
            self._put_date(anchor, "due", v, p)
        v, p = self._pv("service", "service_due", "recall_type", "treatment", "next_step_window_open")
        if v:
            self._put(anchor, "service", v, p, humanize(v).replace("6 month", "6-month").replace("30day", "30-day"))
        v, p = self._pv("molecule_list", "medicines", "items")
        if isinstance(v, list) and v:
            self._put(anchor, "molecules", ", ".join(map(str, v[:4])), p, ", ".join(map(str, v[:4])))
        v, p = self._pv("stock_runs_out_iso", "runs_out_on")
        if v:
            self._put_date(anchor, "runs_out", v, p)
        if isinstance(self.payload, dict) and self.payload.get("delivery_address_saved"):
            anchor["_delivery_saved"] = {"value": True}
        v, p = self._pv("days_since_last_visit", "days_since")
        if v is not None:
            self._put(anchor, "days_since", v, p, fmt_num(as_float(v) or 0))
        v, p = self._pv("previous_focus", "focus", "goal")
        if v:
            self._put(anchor, "focus", v, p, humanize(v))
        v, p = self._pv("previous_membership_months", "membership_months")
        if v is not None:
            self._put(anchor, "member_months", v, p, fmt_num(as_float(v) or 0))
        v, p = self._pv("wedding_date", "event_date")
        if v:
            self._put_date(anchor, "wedding", v, p)
        v, p = self._pv("days_to_wedding", "days_to_event")
        if v is not None:
            self._put(anchor, "days_to_event", v, p, fmt_num(as_float(v) or 0))
        svcs = self.tools.get_customer_fact("relationship.services_received")
        if isinstance(svcs, list) and svcs:
            self._put(anchor, "last_service", svcs[-1], f"customer.relationship.services_received[{len(svcs) - 1}]", humanize(svcs[-1]))
        v, p = self._pv("months_since", "months_since_last_visit")
        if v is not None:
            self._put(anchor, "months", v, p, fmt_num(as_float(v) or 0))
        elif "last_visit" in anchor:
            ref = anchor.get("due", {}).get("date") or self.tools.reference_date(strong_only=True)
            lv = anchor["last_visit"]["date"]
            if ref and ref > lv:
                m = months_between(lv, ref)
                if 1 <= m <= 36:
                    self._derive_put(anchor, "months", m, str(m), [anchor["last_visit"]["path"]])
        self._slots(anchor)

    def _slots(self, anchor: dict) -> None:
        v, p = self._pv("slots", "available_slots", "open_slots", "slot_options", "next_session_options")
        if not v:
            v = self.tools.get_merchant_fact("availability.slots") or self.tools.get_merchant_fact("available_slots")
            p = "merchant.availability.slots" if self.tools.get_merchant_fact("availability.slots") else "merchant.available_slots"
        if isinstance(v, list):
            slots = []
            for i, s in enumerate(v[:2]):
                if isinstance(s, dict) and s.get("label"):
                    self._put(anchor, f"slot{i + 1}", s["label"], f"{p}[{i}].label", str(s["label"]))
                    continue
                raw = s.get("start") or s.get("iso") or s.get("datetime") if isinstance(s, dict) else s
                sp = f"{p}[{i}]" + (".start" if isinstance(s, dict) and s.get("start") else ".datetime" if isinstance(s, dict) and s.get("datetime") else ".label" if isinstance(s, dict) else "")
                dt = parse_datetime(raw) if isinstance(raw, str) and re.match(r"\d{4}-\d{2}-\d{2}", raw) else None
                if dt:
                    t = dt.strftime("%-I:%M%p").replace(":00", "").lower()
                    slots.append((raw, sp, f"{fmt_date(dt.date(), True)}, {t}"))
                elif raw:
                    slots.append((raw, sp, str(raw)))
            for i, (raw, sp, txt) in enumerate(slots):
                if f"slot{i + 1}" not in anchor:
                    self._put(anchor, f"slot{i + 1}", raw, sp, txt)

    def _a_customer_appointment(self, anchor: dict) -> None:
        v, p = self._pv("appointment_at", "datetime", "slot", "time", "scheduled_at", "appointment_time")
        if v:
            self._put_date(anchor, "when", v, p, weekday=True)
            if "when" not in anchor:
                self._put(anchor, "when", v, p)
        v, p = self._pv("service", "treatment", "appointment_type")
        if v:
            self._put(anchor, "service", v, p, humanize(v))
        v, p = self._pv("staff", "doctor", "stylist", "trainer", "with")
        if v:
            self._put(anchor, "staff", v, p)

    def _a_planning(self, anchor: dict) -> None:
        v, p = self._pv("intent_topic", "topic", "program", "idea")
        if v:
            self._put(anchor, "topic", v, p, humanize(v))
        v, p = self._pv("merchant_last_message", "last_message", "request")
        if v:
            self._put(anchor, "ask", v, p, str(v)[:120])

    def _a_seasonal(self, anchor: dict) -> None:
        v, p = self._pv("season", "season_name")
        if v:
            self._put(anchor, "season", v, p, humanize(v).title())
        v, p = self._pv("trends", "demand_shifts", "signals")
        if isinstance(v, list):
            ups, downs = [], []
            for i, t in enumerate(v[:6]):
                m = re.match(r"(.+?)_?(?:demand)?_?([+-]\d+)$", str(t))
                if not m:
                    continue
                item = humanize(m.group(1)).replace(" demand", "").strip()
                item = item.upper() if len(item) <= 3 else item
                (ups if m.group(2).startswith("+") else downs).append((item, m.group(2), f"{p}[{i}]"))
            for j, (item, d, path) in enumerate(ups[:3]):
                self._put(anchor, f"up{j + 1}", str(v), path, f"{item} {d}%")
            for j, (item, d, path) in enumerate(downs[:1]):
                self._put(anchor, f"down{j + 1}", str(v), path, f"{item} {d.replace('-', '−')}%")
        if not any(k.startswith("up") for k in anchor):
            self._seasonal(anchor)

    def _a_supply(self, anchor: dict) -> None:
        for name, keys in (("molecule", ("molecule", "product", "item", "drug")), ("manufacturer", ("manufacturer", "mfr", "brand"))):
            v, p = self._pv(*keys)
            if v:
                self._put(anchor, name, v, p)
        v, p = self._pv("affected_batches", "batches")
        if isinstance(v, list) and v:
            self._put(anchor, "batches", ", ".join(map(str, v[:4])), p, ", ".join(map(str, v[:4])))
        aid, _ = self._pv("alert_id", "digest_item_id", "top_item_id")
        if isinstance(aid, str):
            hits = self.tools.search_category_digest(item_id=aid)
            if hits and hits[0][0].get("id") == aid:
                item, base = hits[0]
                for name, keys in (("title", ("title",)), ("source", ("source",)), ("actionable", ("actionable",))):
                    vv, pp = self._pv(*keys, src=item, base=base)
                    if vv:
                        self._put(anchor, name, vv, pp)

    def _a_generic(self, anchor: dict) -> None:
        """Unknown trigger kind: surface up to 3 scalar payload facts verbatim (no interpretation)."""
        n = 0
        for k, v in (self.payload or {}).items():
            if n >= 3 or k in ("merchant_id", "customer_id", "category", "placeholder") or v is None or isinstance(v, (dict, list)):
                continue
            if isinstance(v, bool):
                continue
            p = f"trigger.payload.{k}"
            f = as_float(v)
            if f is not None and ("pct" in k or "delta" in k or "rate" in k):
                self._put_pct(anchor, f"g_{k}", v, p)
            elif f is not None and any(x in k for x in ("amount", "price", "inr", "revenue", "value_rs", "fee")):
                self._put(anchor, f"g_{k}", v, p, fmt_money(f))
            elif parse_date(v):
                self._put_date(anchor, f"g_{k}", v, p)
            else:
                self._put(anchor, f"g_{k}", v, p, str(v) if f is None else fmt_num(f))
            anchor[f"g_{k}"]["label"] = humanize(k)
            n += 1

    # --------------------------------------------------------------- why now
    @staticmethod
    def _why_now(fam: str, kind: str, a: dict) -> str:
        t = lambda k: a.get(k, {}).get("text")
        parts = {
            "knowledge": f"new digest item: {t('title')} ({t('source')})",
            "regulation": f"compliance change: {t('title')} effective {t('effective')}",
            "perf_dip": f"{t('metric') or 'performance'} fell {t('delta')} {t('window') or ''}",
            "perf_spike": f"{t('metric') or 'performance'} rose {t('delta')} {t('window') or ''}",
            "milestone": f"crossed {t('value')} {t('metric') or ''}",
            "competitor": f"new competitor {t('name') or ''} {t('distance') or ''} away",
            "trend": f"'{t('query')}' demand {t('delta')}",
            "festival": f"{t('name')} in {t('days_until')} days ({t('date')})",
            "weather": f"{t('temp') or t('condition')} in {t('city')}",
            "local_event": f"{t('headline')}",
            "reputation": f"{t('count')} reviews mention '{t('theme')}'",
            "dormant": f"no merchant reply in {t('days')} days",
            "recurring": "scheduled curiosity touch",
            "account": f"{t('plan')} plan: {t('days_left')} days left",
            "customer_recall": f"recall due; last visit {t('last_visit')}",
            "customer_appointment": f"appointment {t('when') or 'tomorrow'}",
            "planning": f"merchant asked about {t('topic')}: '{t('ask')}'",
            "seasonal": f"{t('season')} demand shift: {t('up1')}",
            "supply": f"batch alert: {t('molecule')} {t('batches')}",
        }
        s = parts.get(fam) or f"{humanize(kind)} event"
        return re.sub(r"\s+", " ", s.replace("None", "")).strip()


def _month_in_range(d: date, rng: str) -> bool:
    from ..util import MONTHS
    m = re.findall(r"[A-Za-z]{3}", rng or "")
    if len(m) < 1:
        return False
    try:
        idx = [MONTHS.index(x[:3].title()) for x in m[:2]]
    except ValueError:
        return False
    a, b = idx[0], idx[-1]
    cur = d.month - 1
    return a <= cur <= b if a <= b else (cur >= a or cur <= b)
