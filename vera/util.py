"""Deterministic helpers shared by every agent: safe lookups, number/date formatting."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timezone
from typing import Any, Iterable, Optional

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS_FULL = ["January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WEEKDAYS_FULL = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def deep_get(obj: Any, path: str, default: Any = None) -> Any:
    """Dotted-path lookup that tolerates missing keys and list indices ("a.b[0].c")."""
    cur = obj
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        if part.startswith("["):
            idx = int(part[1:-1])
            if isinstance(cur, list) and -len(cur) <= idx < len(cur):
                cur = cur[idx]
            else:
                return default
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def first_present(obj: dict, keys: Iterable[str], default: Any = None) -> Any:
    """Return the first non-empty value among alias keys (schemas vary between dataset versions)."""
    if not isinstance(obj, dict):
        return default
    for k in keys:
        v = obj.get(k)
        if v not in (None, "", [], {}):
            return v
    return default


def as_float(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.fullmatch(r"\s*[₹$]?\s*(-?[\d,]*\.?\d+)\s*%?\s*", v)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                return None
    return None


def fmt_int(n: float) -> str:
    """Indian digit grouping: 2410 -> 2,410; 123456 -> 1,23,456."""
    n = int(round(n))
    neg = n < 0
    s = str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups + [tail])
    return ("-" if neg else "") + s


def fmt_num(v: float) -> str:
    if float(v).is_integer():
        return fmt_int(v)
    return f"{v:.1f}".rstrip("0").rstrip(".") if abs(v) < 100 else fmt_int(v)


def ratio_to_pct(v: float) -> float:
    """Values in (-1.5, 1.5) are treated as ratios (0.021 -> 2.1); larger ones already are %."""
    return v * 100 if -1.5 < v < 1.5 else v


def fmt_pct(v: float, ratio: bool = True, signed: bool = False) -> str:
    p = ratio_to_pct(v) if ratio else v
    mag = abs(p)
    body = f"{mag:.0f}" if mag >= 10 else f"{mag:.1f}".rstrip("0").rstrip(".")
    sign = ("+" if p > 0 else "-" if p < 0 else "") if signed else ""
    return f"{sign}{body}%"


def fmt_money(v: float) -> str:
    return "₹" + fmt_int(v)


def parse_date(v: Any) -> Optional[date]:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if not isinstance(v, str):
        return None
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", v.strip())
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def parse_datetime(v: Any) -> Optional[datetime]:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if not isinstance(v, str):
        return None
    s = v.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        d = parse_date(s)
        if not d:
            return None
        dt = datetime(d.year, d.month, d.day)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def fmt_date(d: date, with_weekday: bool = False) -> str:
    s = f"{d.day} {MONTHS[d.month - 1]}"
    return f"{WEEKDAYS[d.weekday()]} {s}" if with_weekday else s


def iso_week(d: date) -> str:
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


def months_between(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month) - (1 if b.day < a.day else 0)


def humanize(key: str) -> str:
    return re.sub(r"[_\-]+", " ", str(key)).strip()


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()


def norm_space(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r" +([,.?!:;])", r"\1", s)
    s = re.sub(r"\.\.+", ".", s)
    s = re.sub(r"\?\.", "?", s)
    return s.strip()


def owner_display(merchant: dict, category_slug: str = "") -> str:
    """Best salutation for the merchant, derived only from identity fields."""
    ident = merchant.get("identity", {}) or {}
    owner = first_present(ident, ["owner_first_name", "owner_name", "contact_name"])
    name = ident.get("name", "") or ""
    doctorish = category_slug in ("dentists", "doctors", "clinics") or name.lower().startswith("dr")
    if owner:
        owner = str(owner).strip()
        if doctorish and not owner.lower().startswith("dr"):
            return f"Dr. {owner}"
        return owner
    m = re.match(r"^(Dr\.?\s+[A-Z][a-z]+)", name)
    if m:
        return m.group(1).replace("Dr ", "Dr. ")
    return name or ""
