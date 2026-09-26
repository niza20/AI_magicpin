"""Generates SYNTHETIC development fixtures that follow the schemas in challenge-brief.md §4 and
challenge-testing-brief.md §3. They are NOT the official magicpin dataset — they exist so the
agent system and its tests can run before the official dataset is dropped into ./dataset.

Run: python dev_fixtures/build_fixtures.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

CATEGORIES = {
    "dentists": {
        "slug": "dentists",
        "offer_catalog": [{"title": "Dental Cleaning @ ₹299", "value": "299", "audience": "new_user"},
                          {"title": "Free Consultation", "value": "0", "audience": "all"},
                          {"title": "Teeth Whitening @ ₹1,499", "value": "1499", "audience": "all"}],
        "voice": {"tone": "peer_clinical", "vocab_allowed": ["fluoride varnish", "caries", "recall", "scaling", "aligners"],
                  "vocab_taboo": ["cure", "guaranteed", "painless", "best dentist"]},
        "peer_stats": {"avg_rating": 4.4, "avg_reviews": 62, "avg_ctr": 0.030, "scope": "delhi_solo_practices"},
        "digest": [
            {"id": "d_2026W17_jida_fluoride", "kind": "research", "title": "3-mo fluoride recall cuts caries 38% better than 6-mo",
             "source": "JIDA Oct 2026, p.14", "trial_n": 2100, "patient_segment": "high_risk_adults",
             "summary": "Multi-centre trial across 2,100 high-risk adults; 3-month fluoride varnish recall reduced caries recurrence 38% vs 6-month recall."},
            {"id": "d_2026W17_dci_xray", "kind": "regulation", "title": "DCI revises radiograph dose limits for dental X-rays",
             "source": "DCI circular 2026/14", "effective_date": "2026-06-01",
             "summary": "Clinics must log per-patient dose for every IOPA and OPG from 1 June."}],
        "patient_content_library": [{"id": "pc_001", "title": "3 things your teeth tell you about your heart", "channel": "whatsapp"}],
        "seasonal_beats": [{"month_range": "Nov-Feb", "note": "exam-stress bruxism spike"},
                           {"month_range": "Oct-Dec", "note": "wedding whitening peak"}],
        "trend_signals": [{"query": "clear aligners delhi", "delta_yoy": 0.62, "segment_age": "28-45"}],
    },
    "salons": {
        "slug": "salons",
        "offer_catalog": [{"title": "Haircut @ ₹99", "value": "99"}, {"title": "Hair Spa @ ₹499", "value": "499"},
                          {"title": "Bridal Makeup Trial @ ₹999", "value": "999"}],
        "voice": {"tone": "warm_friendly_practical", "vocab_allowed": ["hair spa", "keratin", "balayage", "blow-dry"],
                  "vocab_taboo": ["guaranteed results", "permanent"]},
        "peer_stats": {"avg_rating": 4.3, "avg_reviews": 140, "avg_ctr": 0.042, "scope": "hyderabad_unisex_salons"},
        "digest": [{"id": "d_sal_keratin", "kind": "trend", "title": "Keratin bookings up ahead of monsoon frizz season",
                    "source": "magicpin salon bookings index", "summary": "Keratin and smoothening bookings typically rise in June-July."}],
        "patient_content_library": [{"id": "sc_001", "title": "5-minute monsoon hair care routine", "channel": "whatsapp"}],
        "seasonal_beats": [{"month_range": "Oct-Dec", "note": "wedding and festive makeup rush"},
                           {"month_range": "Jun-Aug", "note": "monsoon frizz treatments"}],
        "trend_signals": [{"query": "keratin treatment near me", "delta_yoy": 0.41, "segment_age": "22-35"}],
    },
    "restaurants": {
        "slug": "restaurants",
        "offer_catalog": [{"title": "Weekend Biryani Combo @ ₹249", "value": "249"}, {"title": "Thali @ ₹179", "value": "179"}],
        "voice": {"tone": "operator_to_operator", "vocab_allowed": ["covers", "AOV", "delivery radius", "footfall"],
                  "vocab_taboo": ["world's best", "guaranteed"]},
        "peer_stats": {"avg_rating": 4.1, "avg_reviews": 380, "avg_ctr": 0.051, "scope": "delhi_casual_dining"},
        "digest": [{"id": "d_res_fssai", "kind": "regulation", "title": "FSSAI makes hygiene rating display mandatory on menus",
                    "source": "FSSAI order 2026-09", "effective_date": "2026-11-01",
                    "summary": "Outlets must show their hygiene rating on dine-in and online menus."}],
        "patient_content_library": [],
        "seasonal_beats": [{"month_range": "Oct-Nov", "note": "festive family dining and sweets orders peak"}],
        "trend_signals": [{"query": "biryani delivery late night", "delta_yoy": 0.35, "segment_age": "18-30"}],
    },
    "gyms": {
        "slug": "gyms",
        "offer_catalog": [{"title": "7-Day Trial @ ₹199", "value": "199"}, {"title": "Quarterly Membership @ ₹4,999", "value": "4999"}],
        "voice": {"tone": "coach_motivational", "vocab_allowed": ["PT session", "HIIT", "strength block", "retention"],
                  "vocab_taboo": ["guaranteed weight loss", "miracle"]},
        "peer_stats": {"avg_rating": 4.5, "avg_reviews": 95, "avg_ctr": 0.038, "scope": "pune_independent_gyms"},
        "digest": [{"id": "d_gym_retention", "kind": "research", "title": "Members with 2+ PT sessions in month 1 retain 2x longer",
                    "source": "IHRSA India brief 2026", "summary": "Early personal-training touchpoints strongly predict 6-month retention."}],
        "patient_content_library": [{"id": "gc_001", "title": "Beat-the-heat workout timing guide", "channel": "whatsapp"}],
        "seasonal_beats": [{"month_range": "Jan-Feb", "note": "new-year resolution sign-up spike"}],
        "trend_signals": [{"query": "hiit classes pune", "delta_yoy": 0.28, "segment_age": "24-38"}],
    },
    "pharmacies": {
        "slug": "pharmacies",
        "offer_catalog": [{"title": "Free Home Delivery above ₹499", "value": "499"}, {"title": "BP Check @ ₹49", "value": "49"}],
        "voice": {"tone": "trustworthy_precise", "vocab_allowed": ["refill", "generic substitute", "cold chain"],
                  "vocab_taboo": ["cure", "guaranteed", "miracle", "no side effects"]},
        "peer_stats": {"avg_rating": 4.2, "avg_reviews": 48, "avg_ctr": 0.027, "scope": "mumbai_neighbourhood_chemists"},
        "digest": [{"id": "d_ph_schedule_h1", "kind": "regulation", "title": "Schedule H1 register audits tightened in Maharashtra",
                    "source": "FDA Maharashtra notice 2026/31", "effective_date": "2026-10-15",
                    "summary": "Inspectors will check H1 registers for prescriber details on every sale."}],
        "patient_content_library": [{"id": "ph_001", "title": "How to store insulin in summer", "channel": "whatsapp"}],
        "seasonal_beats": [{"month_range": "Apr-Jun", "note": "heat-related ORS and electrolyte demand"},
                           {"month_range": "Jul-Sep", "note": "monsoon fever and dengue test demand"}],
        "trend_signals": [{"query": "dengue test near me", "delta_yoy": 0.55, "segment_age": "25-50"}],
    },
}


def merchant(mid, slug, name, owner, city, locality, langs, perf, offers, signals, agg=None, history=None, extra=None):
    m = {"merchant_id": mid, "category_slug": slug,
         "identity": {"name": name, "owner_first_name": owner, "city": city, "locality": locality,
                      "place_id": f"ChIJ_{mid}", "verified": True, "languages": langs},
         "subscription": {"status": "active", "plan": "Pro", "days_remaining": 82},
         "performance": perf, "offers": offers, "conversation_history": history or [],
         "customer_aggregate": agg or {}, "signals": signals}
    m.update(extra or {})
    return m


MERCHANTS = [
    merchant("m_001_drmeera", "dentists", "Dr. Meera's Dental Clinic", "Meera", "Delhi", "Lajpat Nagar", ["en", "hi"],
             {"window_days": 30, "views": 2410, "calls": 18, "directions": 45, "ctr": 0.021, "delta_7d": {"views_pct": 0.18, "calls_pct": -0.05}},
             [{"id": "o1", "title": "Dental Cleaning @ ₹299", "status": "active"}, {"id": "o2", "title": "Deep Cleaning @ ₹499", "status": "expired"}],
             ["stale_posts:22d", "ctr_below_peer_median", "high_risk_adult_cohort"],
             {"total_unique_ytd": 540, "lapsed_180d_plus": 78, "retention_6mo_pct": 0.38},
             [{"ts": "2026-04-24T10:00:00Z", "from": "vera", "body": "Your profile got 2,410 views this month.", "engagement": "merchant_replied"},
              {"ts": "2026-04-24T10:05:00Z", "from": "merchant", "body": "Haan dekha, thik hai", "engagement": "replied"}],
             {"rating": 4.6, "review_count": 112}),
    merchant("m_006_smilecare", "dentists", "SmileCare Dental Studio", "Arjun", "Bengaluru", "Indiranagar", ["en"],
             {"window_days": 30, "views": 1320, "calls": 22, "directions": 31, "ctr": 0.034, "delta_7d": {"views_pct": -0.12, "calls_pct": -0.41}},
             [{"id": "o1", "title": "Free Consultation", "status": "active"}], ["reviews_mention_wait_time"],
             {"total_unique_ytd": 310, "lapsed_180d_plus": 41}, [], {"rating": 4.3, "review_count": 58}),
    merchant("m_002_studio11", "salons", "Studio11 Family Salon", "Lakshmi", "Hyderabad", "Kondapur", ["en", "hi"],
             {"window_days": 30, "views": 5120, "calls": 64, "directions": 120, "ctr": 0.047, "delta_7d": {"views_pct": 0.28}},
             [{"id": "o1", "title": "Haircut @ ₹99", "status": "active"}, {"id": "o2", "title": "Hair Spa @ ₹499", "status": "paused"}],
             ["stale_posts:9d"], {"total_unique_ytd": 1900, "lapsed_180d_plus": 260}, [], {"rating": 4.9, "review_count": 298}),
    merchant("m_007_glowup", "salons", "GlowUp Unisex Salon", "Rohan", "Pune", "Baner", ["en"],
             {"window_days": 30, "views": 980, "calls": 9, "ctr": 0.022}, [], ["no_active_offer", "stale_posts:31d"],
             {"total_unique_ytd": 420}, [], {"rating": 4.0, "review_count": 37}),
    merchant("m_003_pizzajunction", "restaurants", "Pizza Junction", "Vikram", "Delhi", "Karol Bagh", ["en", "hi"],
             {"window_days": 30, "views": 8900, "calls": 140, "directions": 410, "ctr": 0.044, "delta_7d": {"views_pct": 0.31, "calls_pct": 0.12}},
             [{"id": "o1", "title": "Weekend Biryani Combo @ ₹249", "status": "active"}], ["delivery_heavy"],
             {"total_unique_ytd": 6100}, [], {"rating": 4.2, "review_count": 1204}),
    merchant("m_008_southspice", "restaurants", "South Spice Kitchen", "Anand", "Chennai", "T Nagar", ["en"],
             {"window_days": 30, "views": 3400, "calls": 51, "directions": 205, "ctr": 0.039, "delta_7d": {"calls_pct": -0.22}},
             [{"id": "o1", "title": "Thali @ ₹179", "status": "active"}], [], {}, [], {"rating": 4.4, "review_count": 610}),
    merchant("m_004_ironcore", "gyms", "IronCore Fitness", "Kabir", "Pune", "Kothrud", ["en", "hi"],
             {"window_days": 30, "views": 2100, "calls": 30, "ctr": 0.029}, [{"id": "o1", "title": "7-Day Trial @ ₹199", "status": "active"}],
             ["stale_posts:14d"], {"total_unique_ytd": 380, "lapsed_180d_plus": 95, "retention_6mo_pct": 0.44}, [],
             {"rating": 4.7, "review_count": 99}),
    merchant("m_009_fitnation", "gyms", "FitNation Studio", "Priyanka", "Mumbai", "Andheri West", ["en"],
             {"window_days": 30, "views": 1500, "calls": 12, "ctr": 0.031}, [{"id": "o1", "title": "Quarterly Membership @ ₹4,999", "status": "active"}],
             [], {"total_unique_ytd": 260}, [], {"rating": 4.5, "review_count": 70}),
    merchant("m_005_apollocare", "pharmacies", "CareWell Pharmacy", "Suresh", "Mumbai", "Dadar", ["en", "hi"],
             {"window_days": 30, "views": 1750, "calls": 88, "ctr": 0.025}, [{"id": "o1", "title": "Free Home Delivery above ₹499", "status": "active"}],
             ["refill_customers_high"], {"total_unique_ytd": 2300, "lapsed_180d_plus": 310}, [], {"rating": 4.1, "review_count": 44}),
    merchant("m_010_medplus", "pharmacies", "MedPoint Chemists", "Farah", "Delhi", "Saket", ["en"],
             {"window_days": 30, "views": 900, "calls": 40, "ctr": 0.031}, [{"id": "o1", "title": "BP Check @ ₹49", "status": "active"}],
             [], {}, [], {"rating": 4.6, "review_count": 52}),
]

CUSTOMERS = [
    {"customer_id": "c_001_priya", "merchant_id": "m_001_drmeera",
     "identity": {"name": "Priya", "phone_redacted": "<phone>", "language_pref": "hi-en mix"},
     "relationship": {"first_visit": "2025-11-04", "last_visit": "2026-05-12", "visits_total": 4,
                      "services_received": ["cleaning", "cleaning", "whitening", "cleaning"]},
     "state": "lapsed_soft", "preferences": {"preferred_slots": "weekday_evening", "channel": "whatsapp"},
     "consent": {"opted_in_at": "2025-11-04", "scope": ["recall_reminders", "appointment_reminders"]}},
    {"customer_id": "c_002_rohit", "merchant_id": "m_002_studio11",
     "identity": {"name": "Rohit", "language_pref": "en"},
     "relationship": {"first_visit": "2025-06-01", "last_visit": "2026-03-20", "visits_total": 6, "services_received": ["haircut"]},
     "state": "lapsed_soft", "preferences": {"preferred_slots": "weekend_morning"},
     "consent": {"opted_in_at": "2025-06-01", "scope": ["promotions", "appointment_reminders"]}},
    {"customer_id": "c_003_anita", "merchant_id": "m_004_ironcore",
     "identity": {"name": "Anita", "language_pref": "hi"},
     "relationship": {"first_visit": "2026-01-10", "last_visit": "2026-08-01", "visits_total": 40},
     "state": "lapsed_soft", "preferences": {"preferred_slots": "early_morning"},
     "consent": {"opted_in_at": "2026-01-10", "scope": ["membership_reminders"]}},
    {"customer_id": "c_004_nokonsent", "merchant_id": "m_001_drmeera",
     "identity": {"name": "Kavya", "language_pref": "en"},
     "relationship": {"first_visit": "2025-01-10", "last_visit": "2025-12-01", "visits_total": 2},
     "state": "lapsed_hard", "preferences": {}, "consent": {}},
    {"customer_id": "c_005_imran", "merchant_id": "m_005_apollocare",
     "identity": {"name": "Imran", "language_pref": "hi-en"},
     "relationship": {"first_visit": "2025-02-02", "last_visit": "2026-08-28", "visits_total": 12, "services_received": ["bp medicine refill"]},
     "state": "active", "preferences": {"preferred_slots": "evening"},
     "consent": {"opted_in_at": "2025-02-02", "scope": ["refill_reminders"]}},
    {"customer_id": "c_006_sana", "merchant_id": "m_002_studio11",
     "identity": {"name": "Sana", "language_pref": "en"},
     "relationship": {"first_visit": "2026-02-02", "last_visit": "2026-09-10", "visits_total": 3, "services_received": ["hair spa"]},
     "state": "active", "preferences": {}, "consent": {"opted_in_at": "2026-02-02", "scope": ["appointment_reminders"]}},
]


def trg(tid, kind, mid, payload, urgency=2, scope="merchant", cid=None, source="internal", key=None, exp="2026-10-10T00:00:00Z", det="2026-09-26T09:00:00Z"):
    return {"id": tid, "scope": scope, "kind": kind, "source": source, "merchant_id": mid, "customer_id": cid,
            "payload": payload, "urgency": urgency, "suppression_key": key, "expires_at": exp, "detected_at": det}


TRIGGERS = [
    trg("trg_001_research_dentists", "research_digest", "m_001_drmeera", {"category": "dentists", "top_item_id": "d_2026W17_jida_fluoride"},
        2, source="external", key="research:dentists:2026-W39"),
    trg("trg_002_regulation_dci", "regulation_change", "m_006_smilecare", {"top_item_id": "d_2026W17_dci_xray"}, 3, source="external"),
    trg("trg_003_perf_dip_smilecare", "perf_dip", "m_006_smilecare", {"metric": "calls", "delta_pct": -0.41, "window": "wow"}, 4),
    trg("trg_004_perf_spike_studio11", "perf_spike", "m_002_studio11", {"metric": "views", "delta_pct": 0.28, "window": "yesterday"}, 3),
    trg("trg_005_milestone_studio11", "milestone_reached", "m_002_studio11", {"metric": "reviews", "value": 300}, 2),
    trg("trg_006_competitor_meera", "competitor_opened", "m_001_drmeera", {"competitor_name": "BrightSmile Dental", "distance_km": 1.3, "opened_on": "2026-09-22"}, 3, source="external"),
    trg("trg_007_trend_aligners", "category_trend_movement", "m_001_drmeera", {"query": "clear aligners delhi", "delta_yoy": 0.62, "segment_age": "28-45"}, 2, source="external"),
    trg("trg_008_diwali_pizza", "festival_upcoming", "m_003_pizzajunction", {"festival": "Diwali", "date": "2026-11-08"}, 3, source="external"),
    trg("trg_009_heatwave_pharmacy", "weather_heatwave", "m_010_medplus", {"temp_c": 42, "city": "Delhi", "condition": "heatwave"}, 3, source="external"),
    trg("trg_010_local_news_southspice", "local_news_event", "m_008_southspice", {"headline": "Metro line work closes Usman Road", "location": "Usman Road, T Nagar", "duration_hours": 3}, 3, source="external"),
    trg("trg_011_review_theme_smilecare", "review_theme_emerged", "m_006_smilecare", {"theme": "wait time", "count": 3, "window": "this_week", "quotes": ["Waited 40 minutes past my slot"]}, 3),
    trg("trg_012_dormant_glowup", "dormant_with_vera", "m_007_glowup", {"days_since_last": 14}, 1),
    trg("trg_013_scheduled_ironcore", "scheduled_recurring", "m_004_ironcore", {"cadence": "weekly_friday"}, 1),
    trg("trg_014_recall_priya", "recall_due", "m_001_drmeera",
        {"last_visit": "2026-05-12", "due_date": "2026-11-12", "service": "cleaning",
         "available_slots": [{"start": "2026-11-04T18:00:00+05:30"}, {"start": "2026-11-05T17:00:00+05:30"}]}, 3, scope="customer", cid="c_001_priya"),
    trg("trg_015_lapsed_rohit", "customer_lapsed_soft", "m_002_studio11", {"last_visit": "2026-03-20"}, 2, scope="customer", cid="c_002_rohit"),
    trg("trg_016_appointment_sana", "appointment_tomorrow", "m_002_studio11", {"appointment_at": "2026-09-27T17:30:00+05:30", "service": "hair_spa"}, 4, scope="customer", cid="c_006_sana"),
    trg("trg_017_recall_noconsent", "recall_due", "m_001_drmeera", {"last_visit": "2025-12-01"}, 2, scope="customer", cid="c_004_nokonsent"),
    trg("trg_018_renewal_fitnation", "renewal_due", "m_009_fitnation", {"days_remaining": 5, "plan": "Pro", "renewal_date": "2026-10-01"}, 4),
    trg("trg_019_perf_dip_southspice", "perf_dip", "m_008_southspice", {"metric": "calls", "delta_pct": -0.22, "window": "7d"}, 3),
    trg("trg_020_research_gym", "research_digest", "m_004_ironcore", {"top_item_id": "d_gym_retention"}, 2, source="external"),
    trg("trg_021_regulation_pharmacy", "regulation_change", "m_005_apollocare", {"top_item_id": "d_ph_schedule_h1"}, 4, source="external"),
    trg("trg_022_profile_glowup", "gbp_profile_incomplete", "m_007_glowup", {"missing_fields": ["business_hours", "description", "photos"], "completeness_pct": 0.625}, 2),
    trg("trg_023_offer_expired_meera", "offer_expired", "m_001_drmeera", {"offer_id": "o2", "status": "expired"}, 2),
    trg("trg_024_unknown_kind", "upi_settlement_delayed", "m_003_pizzajunction", {"amount": 18450, "delay_days": 2, "bank": "HDFC"}, 3),
    trg("trg_025_refill_imran", "refill_due", "m_005_apollocare", {"last_visit": "2026-08-28", "service": "bp_medicine_refill"}, 3, scope="customer", cid="c_005_imran"),
    trg("trg_026_festival_salon", "festival_upcoming", "m_002_studio11", {"festival": "Diwali", "date": "2026-11-08", "days_until": 43}, 2, source="external"),
]

# A small labelled pair list in the same shape the loader expects for the real test set.
PAIRS = [{"test_id": f"DEV{i + 1:02d}", "merchant_id": t["merchant_id"], "trigger_id": t["id"], "customer_id": t["customer_id"]}
         for i, t in enumerate(TRIGGERS)]


def main():
    for sub in ("categories", "merchants", "customers", "triggers"):
        os.makedirs(os.path.join(HERE, sub), exist_ok=True)
    for slug, c in CATEGORIES.items():
        json.dump(c, open(os.path.join(HERE, "categories", f"{slug}.json"), "w"), ensure_ascii=False, indent=2)
    for m in MERCHANTS:
        json.dump(m, open(os.path.join(HERE, "merchants", f"{m['merchant_id']}.json"), "w"), ensure_ascii=False, indent=2)
    for c in CUSTOMERS:
        json.dump(c, open(os.path.join(HERE, "customers", f"{c['customer_id']}.json"), "w"), ensure_ascii=False, indent=2)
    for t in TRIGGERS:
        json.dump(t, open(os.path.join(HERE, "triggers", f"{t['id']}.json"), "w"), ensure_ascii=False, indent=2)
    json.dump({"pairs": PAIRS, "note": "DEV pairs over synthetic fixtures — NOT the official 30 test pairs"},
              open(os.path.join(HERE, "dev_pairs.json"), "w"), indent=2)
    print(f"wrote {len(CATEGORIES)} categories, {len(MERCHANTS)} merchants, {len(CUSTOMERS)} customers, {len(TRIGGERS)} triggers")


if __name__ == "__main__":
    main()
