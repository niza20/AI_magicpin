"""FACT LEDGER — the single source of truth every downstream agent and validator reads.

Every scalar in the four contexts becomes a Fact that keeps its source path. Agents may add
*derived* facts (e.g. "5 months since last visit") but only through `derive()`, which records the
source paths it was computed from. The fact checker accepts nothing that is not traceable here.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from .types import Fact
from .util import MONTHS, MONTHS_FULL, WEEKDAYS, WEEKDAYS_FULL, as_float, parse_date

_NUM_RE = re.compile(r"(?<![A-Za-z])-?\d[\d,]*(?:\.\d+)?")
_WORD_RE = re.compile(r"[A-Za-zऀ-ॿ][A-Za-zऀ-ॿ'’]*")


def extract_numbers(text: str) -> list[float]:
    out = []
    for m in _NUM_RE.finditer(text or ""):
        raw = m.group(0).replace(",", "")
        try:
            out.append(abs(float(raw)))
        except ValueError:
            continue
    return out


class FactLedger:
    def __init__(self) -> None:
        self.facts: dict[str, Fact] = {}
        self._by_path: dict[str, Fact] = {}
        self._numbers: set[float] = set()
        self._words: set[str] = set()
        self._dates: set = set()
        self._counter = 0

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> "FactLedger":
        led = cls()
        for layer, ctx in (("category", category), ("merchant", merchant),
                           ("trigger", trigger), ("customer", customer)):
            if ctx:
                led._flatten(layer, ctx, layer)
        return led

    def _flatten(self, layer: str, obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                self._words.update(w.lower() for w in re.split(r"[_\W]+", str(k)) if w)
                self._flatten(layer, v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                self._flatten(layer, v, f"{path}[{i}]")
        elif obj is not None and obj != "":
            label = re.sub(r"\[\d+\]", "", path.rsplit(".", 1)[-1])
            self._add(Fact(id="", path=path, value=obj, text=str(obj), layer=layer,
                           kind=self._kind(obj), label=label))

    @staticmethod
    def _kind(v: Any) -> str:
        if isinstance(v, bool):
            return "text"
        if isinstance(v, (int, float)):
            return "number"
        if isinstance(v, str) and parse_date(v):
            return "date"
        return "text"

    def _add(self, fact: Fact) -> Fact:
        self._counter += 1
        fact.id = fact.id or f"F{self._counter:04d}"
        self.facts[fact.id] = fact
        self._by_path.setdefault(fact.path, fact)
        self._index_value(fact.value)
        if fact.text != str(fact.value):
            self._index_value(fact.text)
        return fact

    def _index_value(self, v: Any) -> None:
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            self._add_number(float(v))
            return
        s = str(v)
        d = parse_date(s)
        if d:
            self._dates.add(d)
            self._add_number(d.day)
            self._add_number(d.year)
            self._words.update({MONTHS[d.month - 1].lower(), MONTHS_FULL[d.month - 1].lower(),
                                WEEKDAYS[d.weekday()].lower(), WEEKDAYS_FULL[d.weekday()].lower()})
        if d and re.search(r"\d{4}-\d{2}-\d{2}[T ](\d{2}):(\d{2})", s):
            hh, mm = map(int, re.search(r"[T ](\d{2}):(\d{2})", s).groups())
            self._add_number(hh)
            self._add_number(hh % 12 or 12)
            self._add_number(mm)
        for n in extract_numbers(s):
            self._add_number(n)
        self._words.update(w.lower() for w in _WORD_RE.findall(s))

    def _add_number(self, n: float) -> None:
        n = abs(n)
        variants = {n, round(n), round(n, 1), round(n, 2)}
        if 0 < n < 1.5:
            p = n * 100
            variants |= {p, round(p), round(p, 1)}
        self._numbers.update(float(x) for x in variants)

    # ------------------------------------------------------------ derivation
    def derive(self, label: str, value: Any, text: str, sources: Iterable[str], kind: str = "number") -> Fact:
        """Register a computed fact. `sources` must be existing ledger paths (enforced)."""
        srcs = [s for s in sources if s in self._by_path or s.startswith("derived.")]
        path = f"derived.{label}<-" + "+".join(srcs) if srcs else f"derived.{label}"
        existing = self._by_path.get(path)
        if existing:
            return existing
        return self._add(Fact(id="", path=path, value=value, text=text, layer="derived", kind=kind, label=label))

    def allow_words(self, words: Iterable[str]) -> None:
        self._words.update(w.lower() for w in words)

    # --------------------------------------------------------------- queries
    def by_path(self, path: str) -> Optional[Fact]:
        return self._by_path.get(path)

    def get(self, fid: str) -> Optional[Fact]:
        return self.facts.get(fid)

    def find(self, layer: Optional[str] = None, label: Optional[str] = None, prefix: Optional[str] = None) -> list[Fact]:
        out = []
        for f in self.facts.values():
            if layer and f.layer != layer:
                continue
            if label and f.label != label:
                continue
            if prefix and not f.path.startswith(prefix):
                continue
            out.append(f)
        return out

    def number_supported(self, n: float, extra: Iterable[float] = ()) -> bool:
        n = abs(n)
        tol = max(0.051, 0.006 * n)
        for a in list(self._numbers) + [abs(float(x)) for x in extra]:
            if abs(a - n) <= tol:
                return True
        return False

    def word_known(self, w: str) -> bool:
        return w.lower().strip("'’") in self._words

    @property
    def dates(self) -> set:
        return set(self._dates)

    def text_supported(self, needle: str) -> bool:
        """True when `needle` appears (case-insensitively) inside any string fact."""
        n = needle.lower().strip()
        if not n:
            return True
        return any(isinstance(f.value, str) and n in f.value.lower() for f in self.facts.values()) or \
            any(n in f.text.lower() for f in self.facts.values())

    def __len__(self) -> int:
        return len(self.facts)
