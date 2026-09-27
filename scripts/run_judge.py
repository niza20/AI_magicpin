"""Run magicpin's judge_simulator.py against a bot, reading the official dataset from dataset/expanded.

Real LLM scoring (key stays in your shell, never in a file):
    export JUDGE_PROVIDER=gemini          # gemini | groq | xai (Grok) | openai | anthropic | openrouter | ollama
    export JUDGE_API_KEY=your-key         # not needed for ollama
    export JUDGE_MODEL=                   # optional, e.g. gemini-2.0-flash
    python scripts/run_judge.py https://your-bot.onrender.com phase2_short

Without JUDGE_API_KEY it uses a stub scorer (conversation tests still run; scores are placeholders).
Scenarios: all | warmup | phase2_short | auto_reply_hell | intent_transition | hostile | full_evaluation
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import judge_simulator as js  # noqa: E402
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
        sys.exit(f"Could not reach {provider}: {e}\nCheck JUDGE_PROVIDER / JUDGE_API_KEY / JUDGE_MODEL.")
else:
    llm = StubLLM()

js.DatasetLoader = Loader
js.BOT_URL = bot_url
print(f"Judge LLM: {llm.name()} · bot: {bot_url} · scenario: {scenario} · data: {data_dir}")
judge = js.JudgeSimulator(llm)
judge.client = js.BotClient(bot_url)
sys.exit(0 if judge.run(scenario) else 1)
