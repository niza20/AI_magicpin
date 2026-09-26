"""LLM boundary. Used ONLY for language understanding, strategy phrasing, drafting and critique.

Design rules:
  * Optional. With no credentials (or VERA_LLM=off) every agent runs its deterministic path.
  * Deterministic: current Claude models reject sampling params, so determinism comes from a
    content-addressed response cache (memory + optional JSON file). Same inputs -> same output.
  * Never trusted: callers validate every LLM output with deterministic code.
  * Never blocks: hard per-call timeout and a caller-supplied deadline; failures return None.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Optional

from .util import stable_hash

DEFAULT_MODEL = os.environ.get("VERA_LLM_MODEL", "claude-opus-5")


def _repair_json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    blob = m.group(0)
    for candidate in (blob, re.sub(r",\s*([}\]])", r"\1", blob)):
        try:
            v = json.loads(candidate)
            return v if isinstance(v, dict) else None
        except json.JSONDecodeError:
            continue
    return None


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self._client = None
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = {}
        self._cache_path = os.environ.get("VERA_LLM_CACHE")
        self.stats = {"calls": 0, "cache_hits": 0, "failures": 0}
        self.enabled = self._init_client()
        if self._cache_path and os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, encoding="utf-8") as f:
                    self._cache = json.load(f)
            except (OSError, json.JSONDecodeError):
                self._cache = {}

    def _init_client(self) -> bool:
        if os.environ.get("VERA_LLM", "auto").lower() in ("off", "0", "false", "none"):
            return False
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            return False
        try:
            import anthropic  # official SDK
            self._client = anthropic.Anthropic(timeout=20.0, max_retries=1)
            return True
        except Exception:
            return False

    def _persist(self) -> None:
        if not self._cache_path:
            return
        try:
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=0)
        except OSError:
            pass

    def complete_json(self, agent: str, system: str, user: str, deadline: Optional[float] = None,
                      max_tokens: int = 2000) -> Optional[dict]:
        """Return a parsed JSON object, or None on any failure / missing budget."""
        key = stable_hash([self.model, agent, system, user])
        with self._lock:
            if key in self._cache:
                self.stats["cache_hits"] += 1
                return self._cache[key]
        if not self.enabled:
            return None
        for attempt in range(2):
            remaining = (deadline - time.monotonic()) if deadline else 20.0
            if remaining < 4.0:
                return None
            prompt = user if attempt == 0 else user + "\n\nYour previous reply was not valid JSON. Reply with ONLY the JSON object."
            try:
                self.stats["calls"] += 1
                resp = self._client.with_options(timeout=min(remaining - 1.0, 20.0)).beta.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                    output_config={"effort": "low"},
                    betas=["server-side-fallback-2026-07-01"],
                    extra_body={"fallbacks": "default"},
                )
                if getattr(resp, "stop_reason", None) == "refusal":
                    self.stats["failures"] += 1
                    return None
                text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
                parsed = _repair_json(text)
                if parsed is not None:
                    with self._lock:
                        self._cache[key] = parsed
                        self._persist()
                    return parsed
            except Exception:
                self.stats["failures"] += 1
                return None
        self.stats["failures"] += 1
        return None


_LLM: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    global _LLM
    if _LLM is None:
        _LLM = LLMClient()
    return _LLM


def reset_llm() -> None:
    global _LLM
    _LLM = None
