# Put the OFFICIAL magicpin dataset here

Either layout works:
- `categories/*.json`, `merchants/*.json`, `customers/*.json`, `triggers/*.json`, or
- `categories/*.json` + `merchants_seed.json`, `customers_seed.json`, `triggers_seed.json` (the layout `judge_simulator.py` reads)

plus the canonical **30 test pairs** file (auto-discovered: `test_pairs*.json`, `*test*pair*.json`, ... or pass
`--pairs`). Then run `python generate_submission.py` to write `submission.jsonl`. The generator refuses to invent
test IDs.
