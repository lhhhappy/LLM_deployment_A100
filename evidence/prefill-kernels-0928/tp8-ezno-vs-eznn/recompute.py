"""Recompute the closed diagnostic pair with the repository's original harness.

Usage: python3 recompute.py REPO EZNN_N30_DIR EZNO_N30_DIR OUT_DIR
The two *.tgz capsules contain the corresponding N30 directories.
This script does not issue a full-cohort or official verdict.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys

repo, baseline_dir, candidate_dir, out = map(Path, sys.argv[1:])
sys.path.insert(0, str(repo / "scripts/analysis"))
import compare_window as compare

scorer = compare.gates.score_formal.load_harness()
loaded = []
receipts = []
for directory in (baseline_dir, candidate_dir):
    summary = json.loads((directory / "summary.json").read_text())
    raw = directory / Path(summary["raw"]).name
    rows = compare.gates.load_raw(raw)
    run = json.loads((directory / Path(summary["run"]).name).read_text())
    flush = json.loads((directory / "flush_evidence.json").read_text())
    assert run["n_attempted"] == run["dispatched"] == len(rows)
    assert flush["flush_success"] and flush["runner_rc"] == 0
    assert summary["scope"] == "fixed_duration_diagnostic"
    assert not summary["full_cohort_complete"]
    assert all(not r.get("error") and not r.get("error_class") for r in rows)
    assert all(r["output_tokens"] == r["max_output_i"] for r in rows)
    assert all(r["prompt_tokens"] == r["glm_tokens"] for r in rows)
    t0 = min(r["client_dispatch_at_s"] for r in rows)
    t1 = max(r["client_finish_at_s"] for r in rows)
    metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    metrics = [r for r in metrics if t0 <= r["t"] <= t1]
    peaks = {k: max(r[k] for r in metrics if isinstance(r.get(k), (int, float)))
             for k in ("token_usage", "mamba_usage", "kv_used_tokens",
                       "mamba_used_tokens", "num_retracted_reqs")}
    receipts.append(dict(raw=raw.name, raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                         rows=len(rows), dispatched=run["dispatched"],
                         cohort_sha256=run["config"]["cohort_sha256"],
                         workload_hash=run["workload_hash"], measured_pool_peaks=peaks))
    loaded.append(rows)
assert receipts[0]["cohort_sha256"] == receipts[1]["cohort_sha256"]
assert receipts[0]["workload_hash"] == receipts[1]["workload_hash"]
base, candidate = loaded
ids = {r["req_id"] for r in base} & {r["req_id"] for r in candidate}
summary, rows = compare.compare(base, [r for r in candidate if r["req_id"] in ids], scorer)
summary["scope"] = "DRAINED fixed-duration diagnostic; common IDs; not a full-cohort verdict"
summary["runs"] = receipts
summary["candidate_unpaired"] = len(candidate) - len(ids)
summary["all_candidate"] = compare.gates.stats(candidate, scorer)
tails = [r for r in rows if "chain_start" in r["baseline_failed"] or "chain_start" in r["candidate_failed"]]
summary["chain_tail_relative_reductions"] = [(r["base_ttft_s"] - r["candidate_ttft_s"]) / r["base_ttft_s"] for r in tails]
out.mkdir(parents=True, exist_ok=True)
(out / "paired-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
with (out / "paired.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(dict(common=len(ids), chain=summary["gate_changes"]["chain_start"],
                      tpot=[summary[k]["tpot"] for k in ("same_request_baseline", "candidate")]), indent=2))
