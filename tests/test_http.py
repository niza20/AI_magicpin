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


def test_judge_list_is_authoritative_but_fresh_triggers_win(client):
    # judge_simulator stamps wall-clock `now`; dataset expiries are simulated time → still send what the judge lists
    r = client.post("/v1/tick", json={"now": "2027-01-01T00:00:00Z", "available_triggers": ["trg_001_research_dentists"]}).json()
    assert len(r["actions"]) == 1


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


def test_post_submission_context_injection_is_used_without_leaks(client):
    """Brief §8 twist: new digest version, shifted metrics, new trigger, customer attached to a merchant pair."""
    import copy
    cat = copy.deepcopy(DS.categories["dentists"])
    cat["digest"].append({"id": "d_new_sdf", "kind": "research", "title": "Silver diamine fluoride arrests 81% of early caries",
                          "source": "IJDR May 2026, p.33", "trial_n": 640})
    assert client.post("/v1/context", json={"scope": "category", "context_id": "dentists", "version": 2, "payload": cat}).json()["accepted"]
    m = copy.deepcopy(DS.merchants["m_006_smilecare"]); m["performance"].update({"calls": 7, "delta_7d": {"calls_pct": -0.61}})
    client.post("/v1/context", json={"scope": "merchant", "context_id": m["merchant_id"], "version": 2, "payload": m})
    trig = [("trg_inj_1", {"kind": "research_digest", "merchant_id": "m_001_drmeera", "payload": {"top_item_id": "d_new_sdf"}}),
            ("trg_inj_2", {"kind": "perf_dip", "merchant_id": "m_006_smilecare", "payload": {"metric": "calls", "delta_pct": -0.61, "window": "7d"}}),
            ("trg_inj_3", {"kind": "competitor_opened", "merchant_id": "m_002_studio11", "customer_id": "c_002_rohit",
                           "payload": {"competitor_name": "Glow Rivals", "distance_km": 0.8}})]
    for tid, t in trig:
        client.post("/v1/context", json={"scope": "trigger", "context_id": tid, "version": 1,
                                          "payload": {"id": tid, "scope": "merchant", "urgency": 3, **t}})
    acts = {a["trigger_id"]: a for a in client.post("/v1/tick", json={"now": "2026-09-27T10:00:00Z",
                                                                        "available_triggers": [t for t, _ in trig]}).json()["actions"]}
    assert "Silver diamine fluoride" in acts["trg_inj_1"]["body"] and "640" in acts["trg_inj_1"]["body"]
    assert "61%" in acts["trg_inj_2"]["body"] and "41%" not in acts["trg_inj_2"]["body"]
    cust = acts["trg_inj_3"]
    assert cust["send_as"] == "merchant_on_behalf" and "Glow Rivals" not in cust["body"], "merchant intel must not reach customers"


def test_demo_photo_and_deals_flow():
    import base64
    import os
    if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "expanded")):
        pytest.skip("official dataset not expanded")
    c = TestClient(bot.app)
    s = c.post("/demo/api/start", json={"trigger_id": "trg_013_corporate_thali_planning", "language": "en"}).json()
    img = "data:image/png;base64," + base64.b64encode(b"\x89PNG....").decode()
    r = c.post("/demo/api/reply", json={"session_id": s["session_id"], "image": img, "message": "Our new masala dosa"}).json()
    assert r["action"] == "send" and r["draft_image"] == img and "Our new masala dosa" in r["body"] and "22 photos" in r["body"]
    assert c.post("/demo/api/reply", json={"session_id": s["session_id"], "message": "GO"}).json()["body"].startswith("Scheduled")
    d = c.post("/demo/api/deals", json={"session_id": s["session_id"]}).json()
    assert "Weekday Lunch Thali @ ₹149" in d["body"] and "Flat 30%" not in d["body"], "service+price deals first, no % discounts"
    assert "Done" in c.post("/demo/api/reply", json={"session_id": s["session_id"], "message": "Yes, go ahead"}).json()["body"]
    bad = c.post("/demo/api/reply", json={"session_id": s["session_id"], "image": "javascript:alert(1)"})
    assert bad.status_code == 400
    # customers only get deal alerts with promotional consent
    s2 = c.post("/demo/api/start", json={"trigger_id": "trg_019_chronic_refill_grandfather"}).json()
    assert c.post("/demo/api/deals", json={"session_id": s2["session_id"]}).json()["action"] == "end"


def test_demo_category_photos_and_local_override(tmp_path):
    import os
    import demo_ui
    if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "expanded")):
        pytest.skip("official dataset not expanded")
    c = TestClient(bot.app)
    sc = {x["merchant"]: x for x in c.get("/demo/api/scenarios").json()}
    s = c.post("/demo/api/start", json={"trigger_id": sc["Mylari South Indian Cafe"]["trigger_id"]}).json()
    assert s["media"]["label"] == "South Indian thali" and "South_Indian_meal_served_on_a_banana_leaf" in s["media"]["url"]
    assert s["media"]["srcs"][-1] == "/demo/assets/south_indian.svg" and s["media"]["credits"][0].startswith("https://commons.wikimedia.org/wiki/File:")
    assert c.get("/demo/assets/south_indian.svg").headers["content-type"].startswith("image/svg+xml")
    assert sc["Dr. Meera's Dental Clinic"]["thumb"], "every category gets a photo now"
    # a local file in demo_assets/photos/<key>.jpg wins over the online sample and is served safely
    old = demo_ui._ASSETS
    demo_ui._ASSETS = tmp_path
    try:
        (tmp_path / "pizza.jpg").write_bytes(b"\xff\xd8\xff fake jpg")
        s = c.post("/demo/api/start", json={"trigger_id": sc["SK Pizza Junction"]["trigger_id"]}).json()
        assert s["media"]["srcs"][0] == "/demo/assets/pizza.jpg" and s["media"]["credits"][0] == ""
        assert c.get("/demo/assets/pizza.jpg").status_code == 200
        assert c.get("/demo/assets/..%2Fbot.py").status_code == 404
    finally:
        demo_ui._ASSETS = old


def test_brand_new_merchant_and_trigger_outside_the_dataset(client):
    """A judge can push any new merchant/trigger through /v1/context and get a grounded message."""
    m = {"merchant_id": "m_900_biryani_house", "category_slug": "restaurants",
         "identity": {"name": "Hyderabadi Biryani House", "owner_first_name": "Imran", "city": "Hyderabad", "languages": ["en"]},
         "performance": {"window_days": 30, "views": 3100, "calls": 19, "ctr": 0.021},
         "offers": [{"title": "Family Biryani Bucket @ ₹699", "status": "active"}]}
    t = {"id": "trg_900_rain", "kind": "weather_alert", "scope": "merchant", "merchant_id": m["merchant_id"], "urgency": 4,
         "payload": {"condition": "heavy rain", "city": "Hyderabad", "date": "2026-09-28"}}
    assert client.post("/v1/context", json={"scope": "merchant", "context_id": m["merchant_id"], "version": 1, "payload": m}).json()["accepted"]
    assert client.post("/v1/context", json={"scope": "trigger", "context_id": t["id"], "version": 1, "payload": t}).json()["accepted"]
    acts = client.post("/v1/tick", json={"now": "2026-09-27T10:00:00Z", "available_triggers": [t["id"]]}).json()["actions"]
    assert len(acts) == 1
    b = acts[0]["body"]
    assert "Imran" in b and "rain" in b.lower() and "delivery" in b.lower() and "appointments" not in b


def test_try_your_own_scenario_every_template_and_category():
    import demo_ui
    c = TestClient(bot.app)
    meta = c.get("/demo/api/custom/templates").json()
    assert {t["id"] for t in meta["templates"]} >= {"weather", "festival", "competitor", "perf_dip", "reviews", "recall"}
    for cat in meta["categories"]:
        for t in meta["templates"]:
            fields = {f["name"]: f["default"] for f in t["fields"]}
            fields.update((meta["field_defaults"].get(cat) or {}).get(t["id"], {}))
            r = c.post("/demo/api/custom", json={"category": cat, "template": t["id"], "fields": fields, "language": "en",
                                                 **meta["defaults"][cat]}).json()
            assert r["body"] and not r["fallback"], (cat, t["id"], r.get("body"))
            assert meta["defaults"][cat]["owner"].split()[-1] in r["body"] or t["audience"] == "customer"
    # it's a real conversation: the owner can reply
    r = c.post("/demo/api/custom", json={"category": "restaurants", "template": "weather", "name": "Rain Cafe", "owner": "Imran",
                                         "fields": {"condition": "heavy rain"}, "language": "en"}).json()
    assert "Imran" in r["body"] and "delivery" in r["body"].lower() and r["context_sent"]["trigger"]["kind"] == "weather_alert"
    assert c.post("/demo/api/reply", json={"session_id": r["session_id"], "message": "yes go ahead"}).json()["action"] == "send"
    # no consent → Vera does not message the customer
    r = c.post("/demo/api/custom", json={"category": "salons", "template": "recall", "language": "en",
                                         "fields": {"customer_name": "Aarti", "service": "hair spa", "consent": "0"}}).json()
    assert r["consent_blocked"]
    # hostile input is bounded and never breaks composition
    r = c.post("/demo/api/custom", json={"category": "nope", "template": "nope", "name": "<script>x</script>" * 20, "views": "abc"}).json()
    assert r["body"] and len(r["context_sent"]["merchant"]["identity"]["name"]) <= 80


def test_demo_chats_are_saved_listed_restored_and_private(tmp_path):
    import os
    import demo_store
    import demo_ui
    if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "expanded")):
        pytest.skip("official dataset not expanded")
    demo_store.reset_for_tests(str(tmp_path / "chats.db"))
    c = TestClient(bot.app)
    me, other = "owner_aaaaaaaa1", "owner_bbbbbbbb2"
    sc = next(x for x in c.get("/demo/api/scenarios").json() if x["merchant"].startswith("Mylari"))
    s = c.post("/demo/api/start", json={"trigger_id": sc["trigger_id"], "owner": me}).json()
    sid = s["session_id"]
    c.post("/demo/api/reply", json={"session_id": sid, "message": "Yes, go ahead", "owner": me})
    demo_ui._sessions.clear()                                   # server restart: in-memory state gone
    r = c.post("/demo/api/reply", json={"session_id": sid, "message": "GO", "owner": me}).json()
    assert r["body"].startswith(("Sent", "Bhej diya")), "rebuilt by replay and continued from the right state"
    lst = c.get("/demo/api/chats", params={"owner": me}).json()
    assert [x["id"] for x in lst] == [sid] and lst[0]["turns"] == 2
    chat = c.get(f"/demo/api/chats/{sid}", params={"owner": me}).json()
    assert [e["role"] for e in chat["log"]] == ["vera", "me", "vera", "me", "vera"] and all(e.get("ts") for e in chat["log"])
    assert c.get(f"/demo/api/chats/{sid}", params={"owner": other}).status_code == 404, "other browsers can't read it"
    assert c.get("/demo/api/chats", params={"owner": other}).json() == []
    assert c.get("/demo/api/chats", params={"owner": "bad id!"}).json() == []
    assert not c.delete(f"/demo/api/chats/{sid}", params={"owner": other}).json()["deleted"]
    assert c.delete(f"/demo/api/chats/{sid}", params={"owner": me}).json()["deleted"]
    assert c.get("/demo/api/chats", params={"owner": me}).json() == []
    # the judge API never touches the demo database
    assert c.post("/v1/teardown").json()["ok"]
