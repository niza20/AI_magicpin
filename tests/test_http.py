import pytest
from fastapi.testclient import TestClient

import bot
from conftest import DS


@pytest.fixture
def client():
    bot.STORE.reset()
    c = TestClient(bot.app)
    for slug, cat in DS.categories.items():
        assert c.post("/v1/context", json={"scope": "category", "context_id": slug, "version": 1, "payload": cat}).json()["accepted"]
    for mid, m in DS.merchants.items():
        c.post("/v1/context", json={"scope": "merchant", "context_id": mid, "version": 1, "payload": m})
    for cid, cu in DS.customers.items():
        c.post("/v1/context", json={"scope": "customer", "context_id": cid, "version": 1, "payload": cu})
    for tid, t in DS.triggers.items():
        c.post("/v1/context", json={"scope": "trigger", "context_id": tid, "version": 1, "payload": t})
    return c


def test_healthz_and_metadata(client):
    h = client.get("/v1/healthz").json()
    assert h["status"] == "ok" and h["contexts_loaded"]["merchant"] == len(DS.merchants)
    assert client.get("/v1/metadata").json()["team_name"]


def test_context_versioning(client):
    m = DS.merchants["m_001_drmeera"]
    same = client.post("/v1/context", json={"scope": "merchant", "context_id": "m_001_drmeera", "version": 1, "payload": m})
    assert same.status_code == 200 and same.json()["accepted"]
    stale = client.post("/v1/context", json={"scope": "merchant", "context_id": "m_001_drmeera", "version": 0, "payload": m})
    assert stale.status_code == 409 and stale.json()["reason"] == "stale_version"
    bad = client.post("/v1/context", json={"scope": "nope", "context_id": "x", "version": 1, "payload": {}})
    assert bad.status_code == 400


def test_tick_actions_suppression_and_restraint(client):
    trig = list(DS.triggers)
    r = client.post("/v1/tick", json={"now": "2026-09-26T10:00:00Z", "available_triggers": trig}).json()
    acts = r["actions"]
    assert 0 < len(acts) <= 20
    for a in acts:
        for k in ("conversation_id", "merchant_id", "send_as", "trigger_id", "template_name", "template_params",
                  "body", "cta", "suppression_key", "rationale"):
            assert k in a
    per_merchant = [(a["merchant_id"], a["customer_id"]) for a in acts]
    assert len(per_merchant) == len(set(per_merchant)), "one action per merchant/customer per tick"
    assert not any(a["trigger_id"] == "trg_017_recall_noconsent" for a in acts), "no-consent customer must not be messaged"
    again = client.post("/v1/tick", json={"now": "2026-09-26T10:05:00Z", "available_triggers": [a["trigger_id"] for a in acts]}).json()
    assert again["actions"] == [], "suppression keys must prevent re-sends"


def test_expired_trigger_skipped(client):
    r = client.post("/v1/tick", json={"now": "2027-01-01T00:00:00Z", "available_triggers": ["trg_001_research_dentists"]}).json()
    assert r["actions"] == []


def test_reply_flow_and_stop_blocks_future_ticks(client):
    acts = client.post("/v1/tick", json={"now": "2026-09-26T10:00:00Z", "available_triggers": ["trg_003_perf_dip_smilecare"]}).json()["actions"]
    conv = acts[0]["conversation_id"]
    r = client.post("/v1/reply", json={"conversation_id": conv, "merchant_id": "m_006_smilecare", "from_role": "merchant",
                                        "message": "yes go ahead", "received_at": "2026-09-26T10:10:00Z", "turn_number": 2}).json()
    assert r["action"] == "send" and r["body"]
    r = client.post("/v1/reply", json={"conversation_id": conv, "merchant_id": "m_006_smilecare", "from_role": "merchant",
                                        "message": "STOP", "received_at": "2026-09-26T10:11:00Z", "turn_number": 3}).json()
    assert r["action"] == "end"
    later = client.post("/v1/tick", json={"now": "2026-09-26T10:20:00Z", "available_triggers": ["trg_011_review_theme_smilecare"]}).json()
    assert later["actions"] == []


def test_judge_simulator_style_auto_reply_on_fresh_conversations(client):
    actions = []
    for i in range(1, 5):
        r = client.post("/v1/reply", json={"conversation_id": f"conv_auto_{i}", "merchant_id": "m_001_drmeera", "customer_id": None,
                                            "from_role": "merchant", "message": "Thank you for contacting us! Our team will respond shortly.",
                                            "received_at": "2026-09-26T10:00:00Z", "turn_number": i + 1}).json()
        actions.append(r["action"])
    assert "end" in actions[:2]


def test_teardown(client):
    assert client.post("/v1/teardown").json()["ok"]
    assert client.get("/v1/healthz").json()["contexts_loaded"]["merchant"] == 0
