# Internal real-chain cases

INTERNAL A/B mechanism cases only; official comparisons require unmodified dev cohort via run_dev.py.

Each `<name>.json` is an ordered list of chain prefixes with ordered `req_ids`. Manifests give chain reasons, source hashes, composition and availability limits. Prediction JSONL contains approximate stock/role_conservative uncached tokens per request.

The supplied data cannot reach the formal 91% intra mix: see the formal_like manifest. Band cutoffs other than population p95 are weighted-dev proxies. Predictions use a fixed 340-token tail guess, not observed role boundaries; rerun replay_chains with the tokenizer for exact F24-model predictions. These files are not official scores or new sampling weights.

Regenerate: `python3 -B scripts/make_case_sets.py`. Functional replay: `python3 -B scripts/replay_chains.py --dev-root s1-dev --case-file cases/smoke.json --output /path/new-run --plan-only`. Long cases may require `--max-prompt-tokens 262144`; prompts are never truncated.
