"""Run magicpin's judge_simulator.py scenarios against a running bot WITHOUT an LLM key.

The LLM-scored parts use a stub provider (neutral scores); the conversation scenarios
(auto_reply_hell, intent_transition, hostile) are rule-checked by the simulator itself.

    uvicorn bot:app --port 8080 &   then   python scripts/run_judge_offline.py [dataset_dir] [scenario]
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import judge_simulator as js  # noqa: E402
from vera.dataset import load_dataset  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else "dataset"
scenario = sys.argv[2] if len(sys.argv) > 2 else "all"


class StubLLM(js.LLMProvider):
    def name(self):
        return "stub (no key) — scores are placeholders"

    def complete(self, prompt, system=None):
        return '{"specificity":5,"category_fit":5,"merchant_fit":5,"decision_quality":5,"engagement_compulsion":5,"hint":"stub"}'


class Loader(js.DatasetLoader):
    def load(self):
        ds = load_dataset(root)
        self.categories, self.merchants, self.customers, self.triggers = ds.categories, ds.merchants, ds.customers, ds.triggers
        return True


js.DatasetLoader = Loader
js.BOT_URL = os.environ.get("BOT_URL", "http://localhost:8080")
judge = js.JudgeSimulator(StubLLM())
judge.client = js.BotClient(js.BOT_URL)
sys.exit(0 if judge.run(scenario) else 1)
