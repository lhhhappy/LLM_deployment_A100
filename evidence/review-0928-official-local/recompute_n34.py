"""Reproduce the archived N30/N34 diagnostic; no service/GPU writes."""
import csv
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts/analysis"))
import compare_window as compare
import window_gates as gates

scorer = gates.score_formal.load_harness()
summary = json.loads((OUT / "n34-paired-summary.json").read_text())
raws, metrics = {}, {}
for n, archived in summary["runs"].items():
    directory = ROOT / archived["source"]
    raw_path, = directory.glob("raw_*.jsonl")
    rows = gates.load_raw(raw_path)
    compare.compare(rows, rows, scorer)  # Original frozen metadata and token checks.
    assert all(not r.get("error") for r in rows)
    assert gates.stats(rows, scorer) == archived["all_completed"]
    first = min(r["client_dispatch_at_s"] for r in rows)
    last = max(r["client_finish_at_s"] for r in rows)
    for lo, hi in [(0, 300), (300, 600), (600, 99999)]:
        subset = [r for r in rows if lo <= r["client_dispatch_at_s"] - first < hi]
        assert gates.stats(subset, scorer) == archived["time_buckets"][f"{lo}-{hi}s_by_dispatch"]
    samples = [json.loads(x) for x in (directory / "metrics.jsonl").read_text().splitlines() if x.strip()]
    samples = [r for r in samples if first <= r["t"] <= last]
    assert len(samples) == archived["metric_samples"]
    assert sum(r.get("token_usage", 0) >= .93 for r in samples) / len(samples) == archived["kv_usage_ge_0_93_fraction"]
    for key, expected in archived["peaks"].items():
        assert max(r.get(key, 0) for r in samples) == expected
    gpu = {}
    for row in csv.reader((directory / "gpu_util.csv").read_text().splitlines()):
        if len(row) != 4:
            continue
        t = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
        if first <= t <= last:
            rank, used = row[1].strip(), int(row[3].strip().split()[0])
            gpu[rank] = max(gpu.get(rank, 0), used)
    assert gpu == archived["gpu_used_peak_mib_by_rank"]
    if n == "N34":
        receipt = json.loads((directory / "timed_verdict.json").read_text())
        assert receipt["status"] == "DRAINED" and not receipt["full_cohort_complete"]
        assert receipt["raw_sha256"] == hashlib.sha256(raw_path.read_bytes()).hexdigest()
        assert len(rows) == receipt["n_completed"] == len(set(receipt["dispatched_req_ids"]))
        assert {r["req_id"] for r in rows} == set(receipt["dispatched_req_ids"])
    raws[n], metrics[n] = rows, samples

base_ids = {r["req_id"] for r in raws["N30"]}
candidate = [r for r in raws["N34"] if r["req_id"] in base_ids]
paired_summary, details = compare.compare(raws["N30"], candidate, scorer)
for key, value in paired_summary.items():
    if key != "scope":  # Original helper's OPEN label does not describe drained inputs.
        assert value == summary[key], key
new_fast = [r for r in details if "fast_intra" in r["candidate_failed"] and "fast_intra" not in r["baseline_failed"]]
assert len(new_fast) == 49
for field, both in summary["new_fast_timing"].items():
    for prefix, expected in both.items():
        values = [r[prefix + "_" + field] for r in new_fast]
        assert statistics.median(values) == expected["median"]
        assert abs(statistics.mean(values) - expected["mean"]) < 1e-12
saved_memory = json.loads((OUT / "n34-new-fast-memory.json").read_text())
assert {r["req_id"] for r in saved_memory} == {r["req_id"] for r in new_fast}
for row in saved_memory:
    detail, = [r for r in new_fast if r["req_id"] == row["req_id"]]
    assert detail["delta_cached_tokens"] == row["cached_token_delta"]
    assert detail["candidate_recv_to_exec_s"] == row["recv_to_exec_s"]
print(json.dumps({"status": "PASS", "common_ids": len(candidate), "new_fast": len(new_fast),
                  "scope": "DRAINED diagnostic; not an official or full-cohort verdict"}))
