#!/usr/bin/env python3
"""Read-only replay analysis; use the original harness selectors, never rescore.

Run from repository root: python3 -B evidence/slo-levers-20260924/analyze.py
Outputs remain beside this script. All timing partitions are descriptive:
recv->exec includes admission/preparation; exec->first includes interleaving.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts/analysis"))
sys.path.insert(0, str(ROOT / "s1-dev/harness"))
from compare_runs import GATES, load, pct
from s1_common import in_ttft_gate

cohort = json.loads((ROOT / "data/s1-dev-longchain/cohort.json").read_text())
ids = [rid for ch in cohort["chains"] for rid in ch["req_ids"]]
assert len(ids) == len(set(ids)) == cohort["n_requests"]
summary = {
    "scope": "068/069 complete local N30, same frozen cohort; descriptive, no causal attribution",
    "limits": [
        "queue-heavy is not proof of a scheduler defect or idle GPU",
        "exec-to-first is not pure kernel time",
        "final cached_tokens does not identify device vs host residency at arrival",
        "later arrivals executing earlier do not identify the admission rejection reason",
        "bucket quantiles are separate requests and must not be summed",
        "local N30 does not predict official N34/N38",
    ],
    "runs": {},
}
details = []
identities = []
for name in ("068-official_b_pace_off_full_n30_shortwarm",
             "069-official_b_pace_off_host64_full_n30_shortwarm"):
    directory = ROOT / "evidence" / ("L" + name) / "N30"
    by, verdict, batches, pace, boundary, identity = load(directory, set(ids), cohort)
    identities.append(identity)
    rows = list(by.values())
    t0 = min(r["client_dispatch_at_s"] for r in rows)
    raw = next(directory.glob("raw_*.jsonl"))
    run = {
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "rows": len(rows),
        "uncached_prompt": sum(r["prompt_tokens"] - r["cached_tokens"] for r in rows),
        "tpot_mean": verdict["tpot_mean"], "tpot_p95": verdict["tpot_p95"],
        "buckets": {},
    }
    for gate, limit in GATES:
        selected = [r for r in rows if in_ttft_gate(r, gate)]
        bad = [r for r in selected if r["ttft_s"] > limit]
        saved = verdict["ttft_gates"][gate]
        assert len(selected) == saved["n"] and len(bad) == saved["over_limit"]
        wait = lambda r: r["t_exec_start_s"] - r["t_recv_s"]
        execute = lambda r: r["t_first_token_s"] - r["t_exec_start_s"]
        run["buckets"][gate] = {
            "n": len(selected), "bad": len(bad), "allowed": saved["allowed_over"],
            "ttft_p95_s": pct([r["ttft_s"] for r in selected], .95),
            "recv_to_exec_p95_s": pct([wait(r) for r in selected], .95),
            "exec_to_first_p95_s": pct([execute(r) for r in selected], .95),
            "bad_recv_to_exec_ge_80pct": sum(wait(r) >= .8 * r["ttft_s"] for r in bad),
            "bad_queue_timer_ge_80pct": sum(r["queue_time_s"] >= .8 * r["ttft_s"] for r in bad),
            "bad_exec_to_first_gt_limit": sum(execute(r) > limit for r in bad),
            "bad_actual_uncached_le4096": sum(r["prompt_tokens"] - r["cached_tokens"] <= 4096 for r in bad),
            "bad_dispatch_min_range": [min((r["client_dispatch_at_s"] - t0) / 60 for r in bad),
                                        max((r["client_dispatch_at_s"] - t0) / 60 for r in bad)],
            "uncached_prompt": sum(r["prompt_tokens"] - r["cached_tokens"] for r in selected),
        }
        for r in bad:
            details.append({
                "run": name[:3], "gate": gate, "req_id": r["req_id"],
                "session_id": r["session_id"], "idx_in_chain": r["idx_in_chain"],
                "phase": r["phase"], "dispatch_min": (r["client_dispatch_at_s"] - t0) / 60,
                "ttft_s": r["ttft_s"], "recv_to_exec_s": wait(r),
                "queue_time_s": r["queue_time_s"], "exec_to_first_s": execute(r),
                "prompt_tokens": r["prompt_tokens"], "cached_tokens": r["cached_tokens"],
                "actual_uncached": r["prompt_tokens"] - r["cached_tokens"],
                "later_arrivals_exec_before_this_exec": sum(
                    r["t_recv_s"] < other["t_recv_s"] < other["t_exec_start_s"] < r["t_exec_start_s"]
                    for other in rows),
            })
    summary["runs"][name[:3]] = run
assert identities[0] == identities[1]
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
with (OUT / "badcases.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=details[0].keys())
    writer.writeheader()
    writer.writerows(details)
print(json.dumps(summary, ensure_ascii=False, indent=2))
