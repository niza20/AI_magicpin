import copy
import os
import sys

import pytest

os.environ["VERA_LLM"] = "off"          # tests exercise the deterministic agent path unless they inject a fake LLM
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vera.dataset import load_dataset  # noqa: E402
from vera.llm import reset_llm  # noqa: E402

DS = load_dataset(os.path.join(ROOT, "dev_fixtures"))


@pytest.fixture(autouse=True)
def _fresh_llm():
    reset_llm()
    yield
    reset_llm()


@pytest.fixture
def ds():
    return DS


def ctx(tid: str, ds=DS):
    t = copy.deepcopy(ds.triggers[tid])
    m = copy.deepcopy(ds.merchants[t["merchant_id"]])
    c = copy.deepcopy(ds.customers[t["customer_id"]]) if t.get("customer_id") else None
    return copy.deepcopy(ds.category_for(m)), m, t, c
