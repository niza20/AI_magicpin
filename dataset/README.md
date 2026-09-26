# Official dataset

- `seed/`: the files as uploaded (5 category packs, `merchants_seed.json`, `customers_seed.json`, `triggers_seed.json`,
  and magicpin's `generate_dataset.py`).
- `expanded/`: output of `python seed/generate_dataset.py --out expanded` (run from this folder). The generator is
  deterministic with a fixed seed. It contains 50 merchants, 200 customers, 100 triggers, and `test_pairs.json`
  with the 30 canonical pairs T01-T30.

`python generate_submission.py --dataset dataset/expanded` writes `submission.jsonl`.
