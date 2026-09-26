import re

import pytest

from bot import compose
from conftest import ctx
from conversation_handlers import new_state, respond

QUALIFYING = ["would you", "do you", "can you tell", "what if", "how about"]
AUTO = "Thank you for contacting us! Our team will respond shortly."


def state(tid="trg_003_perf_dip_smilecare"):
    cat, m, t, c = ctx(tid)
    return new_state("conv_test", cat, m, t, c, compose(cat, m, t, c)["body"])


def test_auto_reply_detected_and_exits_after_one_clarification():
    st = state()
    r1 = respond(st, AUTO)
    assert r1["action"] == "send" and "auto-reply" in r1["body"].lower()
    r2 = respond(st, AUTO)
    assert r2["action"] == "end"
    assert respond(st, AUTO)["action"] == "end"


def test_auto_reply_hindi_canned_text():
    st = state("trg_004_perf_spike_studio11")
    msg = "Aapki jaankari ke liye bahut-bahut shukriya. Main aapki yeh sabhi baatein team tak pahuncha deti hoon."
    assert respond(st, msg)["action"] == "send"
    assert respond(st, msg)["action"] == "end"


@pytest.mark.parametrize("msg", ["YES", "Yes.", "Go ahead", "Start it", "I want to join", "Update my profile",
                                 "Ok lets do it. Whats next?", "haan karo", "theek hai, kar do"])
def test_explicit_action_goes_straight_to_act(msg):
    st = state()
    r = respond(st, msg)
    assert r["action"] == "send"
    low = r["body"].lower()
    assert not any(q in low for q in QUALIFYING), r["body"]
    assert re.search(r"done|draft|sending|on it|ho gaya|here", low)


def test_act_after_qualification_turns_does_not_requalify():
    st = state()
    respond(st, "Interesting, tell me more")
    respond(st, "what would you post?")
    r = respond(st, "ok let's do it")
    assert r["action"] == "send" and not any(q in r["body"].lower() for q in QUALIFYING)


@pytest.mark.parametrize("msg", ["STOP", "Stop messaging me. This is useless spam.", "not interested", "nahi chahiye", "no"])
def test_rejection_ends(msg):
    assert respond(state(), msg)["action"] == "end"


def test_price_question_has_no_invented_numbers():
    st = state()
    r = respond(st, "how much?")
    assert r["action"] == "send"
    nums = set(re.findall(r"\d[\d,]*", r["body"]))
    assert nums <= {"1", "30", "22", "41"}, nums       # only context numbers (none needed here)
    assert "won't guess" in r["body"] or "guess nahi" in r["body"]


def test_language_switch_per_turn():
    st = state()                                         # English merchant
    r = respond(st, "haan theek hai, yeh kya hai aur kaise karna hoga?")
    assert r["action"] == "send"
    assert re.search(r"\b(main|hai|karein|aap|mein)\b", r["body"].lower())


def test_follow_up_question_answered_with_new_facts_no_repeats():
    st = state("trg_012_dormant_glowup")
    bodies = [respond(st, q).get("body") for q in ["why?", "what else?", "and?"]]
    bodies = [b for b in bodies if b]
    assert len(bodies) == len(set(bodies)), "bot must never repeat itself verbatim"


def test_hostile_then_off_topic():
    st = state()
    r1 = respond(st, "you are useless idiots, can you also help me file my GST?")
    assert r1["action"] == "send" and "gst" in r1["body"].lower() and "stop" in r1["body"].lower()
    assert respond(st, "stupid")["action"] == "end"


def test_off_topic_stays_on_mission():
    r = respond(state(), "Can you help me with a bank loan?")
    assert r["action"] == "send" and "outside" in r["body"].lower()


def test_later_waits():
    r = respond(state(), "busy right now, call me tomorrow")
    assert r["action"] == "wait" and r["wait_seconds"] >= 3600


def test_customer_slot_selection_books():
    st = state("trg_014_recall_priya")
    r = respond(st, "1")
    assert r["action"] == "send" and "Booked" in r["body"] and "6pm" in r["body"]


def test_repeated_identical_merchant_messages_become_auto_reply():
    st = state()
    respond(st, "ok noted")
    r = respond(st, "ok noted")
    assert r["action"] in ("send", "end")
    r3 = respond(st, "ok noted")
    assert r3["action"] == "end"


# ---- replay curveballs (judge's merchant-simulator improvises) --------------------------------------
def _official_state():
    from vera.dataset import load_dataset
    import os
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "expanded")
    if not os.path.exists(root):
        pytest.skip("official dataset not expanded")
    ds = load_dataset(root)
    t = ds.triggers["trg_023_competitor_opened_dentist"]; m = ds.merchants[t["merchant_id"]]; cat = ds.category_for(m)
    return new_state("cb", cat, m, t, None, compose(cat, m, t, None)["body"])


@pytest.mark.parametrize("msg,must", [
    ("Who are you? Are you from Google?", "not google"),
    ("Is this a scam?", "never invent"),
    ("What is CTR?", "ctr ="),
    ("Can you reduce my magicpin commission?", "magicpin team"),
    ("I already posted on Instagram yesterday", "google is a separate"),
    ("Which competitor?", "smile studio"),
    ("My nephew handles my Google page", "forward"),
    ("Do it tomorrow morning", "schedule it for tomorrow morning"),
    ("Kitne customers aayenge isse?", "waada nahi"),
    ("Call me", "call you"),
])
def test_curveballs_get_on_topic_answers(msg, must):
    r = respond(_official_state(), msg)
    assert r["action"] == "send" and must in r["body"].lower(), r.get("body")


def test_curveball_never_invents_a_forecast():
    r = respond(_official_state(), "How many customers will I get from this?")
    assert "can't promise" in r["body"].lower()
