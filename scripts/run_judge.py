"""Run magicpin's judge_simulator.py against a bot, reading the official dataset from dataset/expanded.

Real LLM scoring (key stays in your shell, never in a file):
    export JUDGE_PROVIDER=gemini          # gemini | groq | xai (Grok) | openai | anthropic | openrouter | ollama
    export JUDGE_API_KEY=your-key         # not needed for ollama
    export JUDGE_MODEL=                   # optional, e.g. gemini-2.0-flash
    python scripts/run_judge.py https://your-bot.onrender.com phase2_short

Without JUDGE_API_KEY it uses a stub scorer (conversation tests still run; scores are placeholders).
Optional: JUDGE_DELAY=3 (seconds between judge calls), JUDGE_MAX=15 (score only the first N messages).
Scenarios: all | warmup | phase2_short | auto_reply_hell | intent_transition | hostile | full_evaluation
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import judge_simulator as js  # noqa: E402
from urllib import request as _urlreq  # noqa: E402

# Some providers (e.g. Groq, behind Cloudflare) return 403 to Python's default "Python-urllib" user agent.
_opener = _urlreq.build_opener()
_opener.addheaders = [("User-Agent", "Mozilla/5.0 (vera-judge-runner)"), ("Accept", "application/json")]
_urlreq.install_opener(_opener)
from vera.dataset import load_dataset  # noqa: E402

bot_url = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("BOT_URL", "http://localhost:8080")
scenario = sys.argv[2] if len(sys.argv) > 2 else "phase2_short"
data_dir = os.environ.get("DATASET_DIR", str(ROOT / "dataset" / "expanded"))


class Loader(js.DatasetLoader):
    def load(self):
        ds = load_dataset(data_dir)
        self.categories, self.merchants, self.customers, self.triggers = ds.categories, ds.merchants, ds.customers, ds.triggers
        return bool(ds.merchants)


class StubLLM(js.LLMProvider):
    def name(self):
        return "stub (no JUDGE_API_KEY) — scores are placeholders"

    def complete(self, prompt, system=None):
        return '{"specificity":5,"category_fit":5,"merchant_fit":5,"decision_quality":5,"engagement_compulsion":5,"hint":"stub"}'


class XAIProvider(js.LLMProvider):
    """xAI Grok (OpenAI-compatible chat completions API) — not built into judge_simulator.py."""

    def __init__(self, api_key: str, model: str = ""):
        self.api_key, self.model = api_key, model or "grok-3-mini"

    def name(self):
        return f"xAI Grok ({self.model})"

    def complete(self, prompt, system=None):
        import json
        from urllib import request as rq
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        req = rq.Request("https://api.x.ai/v1/chat/completions",
                         data=json.dumps({"model": self.model, "messages": msgs, "temperature": 0.2, "max_tokens": 1500}).encode(),
                         headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        return json.loads(rq.urlopen(req, timeout=js.TIMEOUT_LLM).read())["choices"][0]["message"]["content"]


provider = os.environ.get("JUDGE_PROVIDER", "").lower()
if provider == "grok":
    provider = "xai"
key = os.environ.get("JUDGE_API_KEY", "")
if provider and (key or provider == "ollama"):
    js.LLM_PROVIDER, js.LLM_API_KEY, js.LLM_MODEL = provider, key, os.environ.get("JUDGE_MODEL", "")
    llm = XAIProvider(key, js.LLM_MODEL) if provider == "xai" else js.create_provider()
    try:
        llm.complete("Say ready.", "You are a test assistant.")
    except Exception as e:  # show a clear message instead of a stack trace
        detail = ""
        if hasattr(e, "read"):
            try:
                detail = e.read().decode("utf-8", "ignore")[:300]
            except Exception:
                pass
        sys.exit(f"Could not reach {provider}: {e}\n{detail}\nCheck JUDGE_PROVIDER / JUDGE_API_KEY / JUDGE_MODEL.")
else:
    llm = StubLLM()

class Throttled(js.LLMProvider):
    """Wraps the judge LLM: spaces calls out and retries on rate limits (HTTP 429 / 5xx) instead of
    silently falling back to judge_simulator's placeholder scores (5/5/5/5 + digit-counting specificity)."""

    def __init__(self, inner, delay: float, retries: int = 6):
        self.inner, self.delay, self.retries = inner, delay, retries
        self.ok = self.failed = 0

    def name(self):
        return self.inner.name() + f" · {self.delay:g}s between calls, retry on 429"

    def complete(self, prompt, system=None):
        import time
        from urllib.error import HTTPError
        wait = max(self.delay, 2.0)
        for attempt in range(self.retries + 1):
            time.sleep(self.delay)
            try:
                out = self.inner.complete(prompt, system)
                self.ok += 1
                return out
            except HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    retry_after = e.headers.get("retry-after") if e.headers else None
                    pause = float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() else wait
                    print(f"  … judge LLM {e.code}, waiting {pause:.0f}s (retry {attempt + 1}/{self.retries})")
                    time.sleep(pause)
                    wait = min(wait * 2, 60)
                    continue
                self.failed += 1
                raise
            except Exception:
                self.failed += 1
                raise
        self.failed += 1
        raise RuntimeError("judge LLM kept failing")


limit = int(os.environ.get("JUDGE_MAX", "0") or 0)
if limit:
    _orig = js.LLMScorer.score
    _count = {"n": 0}

    def _limited(self, *a, **k):
        _count["n"] += 1
        if _count["n"] > limit:
            return js.ScoreResult(hint=SKIPPED)
        return _orig(self, *a, **k)
    js.LLMScorer.score = _limited

SKIPPED = "skipped (JUDGE_MAX reached)"
_orig_summary = js.JudgeSimulator._final_summary


def _real_summary(self):
    """Average only messages the judge actually scored (JUDGE_MAX-skipped ones are not zeros)."""
    self.all_scores = [s for s in self.all_scores if s.hint != SKIPPED]
    _orig_summary(self)
    if not self.all_scores:
        return
    dims = [("specificity", "specificity_reason"), ("category_fit", "category_fit_reason"),
            ("merchant_fit", "merchant_fit_reason"), ("decision_quality", "decision_quality_reason"),
            ("engagement_compulsion", "engagement_reason")]
    n = len(self.all_scores)
    print(f"\nExact averages over {n} judged messages (the simulator rounds down):")
    for d, _ in dims:
        print(f"  {d:<22} {sum(getattr(s, d) for s in self.all_scores) / n:.1f}")
    print(f"  {'TOTAL':<22} {sum(s.total for s in self.all_scores) / n:.1f}/50")
    rows = [{"body": b, "total": s.total, **{d: getattr(s, d) for d, _ in dims},
             "reasons": {d: getattr(s, r) for d, r in dims}, "penalties": s.penalty_reasons, "hint": s.hint}
            for s, b in zip(self.all_scores, _bodies)]
    with open("judge_scores.json", "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("Per-message scores + judge reasons saved to judge_scores.json")
    low = sorted(rows, key=lambda r: r["total"])[:3]
    for r in low:
        print(f"\n  {r['total']}/50  {r['body'][:90]!r}")
        for d, _ in dims:
            if r[d] < 7 and r["reasons"][d]:
                print(f"    {d}={r[d]}: {r['reasons'][d][:200]}")


_bodies = []
_orig_score_msg = js.LLMScorer.score


def _capture(self, action, *a, **k):
    res = _orig_score_msg(self, action, *a, **k)
    if res.hint != SKIPPED:
        _bodies.append(action.get("body", ""))
    return res


js.LLMScorer.score = _capture
js.JudgeSimulator._final_summary = _real_summary

if not isinstance(llm, StubLLM):
    llm = Throttled(llm, float(os.environ.get("JUDGE_DELAY", "3")))
js.DatasetLoader = Loader
js.BOT_URL = bot_url
print(f"Judge LLM: {llm.name()} · bot: {bot_url} · scenario: {scenario} · data: {data_dir}")
judge = js.JudgeSimulator(llm)
judge.client = js.BotClient(bot_url)
ok = judge.run(scenario)
if isinstance(llm, Throttled):
    real, fb = llm.ok, llm.failed
    print(f"\nJudge calls: {real} real LLM judgements, {fb} failed → fallback placeholder scores.")
    if fb:
        print("⚠ Fallback scores are 5/5/5/5 + digit-count specificity — they are NOT the judge's opinion. "
              "Increase JUDGE_DELAY (e.g. export JUDGE_DELAY=8) or lower JUDGE_MAX and re-run.")
sys.exit(0 if ok else 1)
