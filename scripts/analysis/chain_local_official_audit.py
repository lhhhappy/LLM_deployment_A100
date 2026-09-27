#!/usr/bin/env python3
"""Recompute chain diagnostics from preserved raw records; never assign a full score.

Uses the original harness gate selector and quantile, not a copied implementation.
Inputs remain in place. Outputs contain numeric metadata only, no request bodies.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

GATES = {"chain_start": 30, "turn_start": 15, "overall_intra": 5, "fast_intra": 3}
PAIRS = [
    ("130ee", "130ed"), ("130eez", "130ee5"), ("130eezy", "130eezz"),
    ("130ez4", "130ez5"), ("130ez4", "130ez6"),
    ("130ez6zz", "130ez6zzz"), ("130ez6zz", "130ez6zzzz"),
    ("130ez1", "130ez7"), ("130ez1", "130ez8"), ("130ez1", "130ez9"),
    ("130ez1", "130eze2"), ("130ez1", "130ezf"),
    ("130ezf", "130ezh"), ("130ezh", "130ezi"), ("130ezh", "130ezj"),
    ("130ezh", "130ezk"), ("130ezh", "130ezm"),
    ("130ezl", "130ezm5"), ("130ezm5", "130ezm6"), ("130ezl", "130ezm6"),
]


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def numeric(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root, out = args.repo.resolve(), args.out.resolve()
    sys.path.insert(0, str(root / "s1-dev/harness"))
    from s1_common import in_ttft_gate, load_index, q
    # Execute the unchanged organizer function without importing its engine adapter.
    loadgen_path = root / "s1-dev/harness/s1_loadgen.py"
    gap_function = next(node for node in ast.parse(loadgen_path.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == "build_gap_plan")
    gap_ns = {"GAP_ROUNDING": "last-nonzero-remainder"}
    exec(compile(ast.Module(body=[gap_function], type_ignores=[]),
                 str(loadgen_path) + " [original build_gap_plan]", "exec"), gap_ns)

    def distribution(values, limit=None):
        values = [v for v in values if numeric(v)]
        result = dict(n=len(values), p50=q(values, .5), p95=q(values, .95),
                      p99=q(values, .99), max=max(values) if values else None)
        if limit is not None:
            result["over"] = sum(v > limit for v in values)
        return result

    def metrics(rows):
        successful = [r for r in rows if not r.get("error") and not r.get("error_class")]
        result = {gate: distribution([r.get("ttft_s") for r in successful
                                     if in_ttft_gate(r, gate)], limit)
                  for gate, limit in GATES.items()}
        tpot = [r["tpot_s"] for r in successful if numeric(r.get("tpot_s"))]
        result["tpot"] = distribution(tpot, .10)
        result["tpot"]["mean"] = sum(tpot) / len(tpot) if tpot else None
        result["rows"] = len(rows)
        return result

    inputs, runs, raw_by_run, chain_rows, time_rows = [], {}, {}, [], []

    def receipt(path):
        data = path.read_bytes()
        inputs.append(dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        return data

    for directory in sorted((root / "evidence").glob("L130e*")):
        code = directory.name.split("-", 1)[0][1:]
        candidates = list(directory.glob("N*/raw*.jsonl"))
        if len(candidates) > 1:
            raise ValueError(f"ambiguous measured raw: {directory}")
        if not candidates:
            candidates = list(directory.glob("window/raw.jsonl"))
        if not candidates:
            continue
        path = candidates[0]
        rows = [json.loads(line) for line in receipt(path).splitlines() if line.strip()]
        ids = [r["req_id"] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate request IDs: {path}")
        raw_by_run[code] = {r["req_id"]: r for r in rows}
        levels = list(directory.glob("N[0-9]*"))
        level = path.parent if path.parent.name.startswith("N") else (levels[0] if levels else None)
        run_paths = list(level.glob("run_*.json")) if level else []
        run = read_json(run_paths[0]) if len(run_paths) == 1 else {}
        for p in run_paths:
            receipt(p)
        summary_path = level / "summary.json" if level else None
        summary = read_json(summary_path) if summary_path else {}
        if summary_path and summary_path.exists():
            receipt(summary_path)
        log_path = level / "job.log" if level else None
        log = log_path.read_text(errors="replace") if log_path and log_path.exists() else ""
        if log:
            receipt(log_path)
        declared = re.findall(r"TIMED_DIAGNOSTIC DRAINED (\d+) requests", log)
        status = "DRAINED_DIAGNOSTIC" if declared and int(declared[-1]) == len(rows) else "INCOMPLETE_SNAPSHOT"
        if code in {"130ezl", "130ezd"}:
            status = "ABORTED_SNAPSHOT"
        t0 = min(r["t_recv_s"] for r in rows if numeric(r.get("t_recv_s")))
        d = dict(code=code, directory=str(directory), raw=str(path), scope=status,
                 full_cohort_score=False, rows=len(rows), duplicate_ids=0,
                 errors=sum(bool(r.get("error") or r.get("error_class")) for r in rows),
                 config=run.get("config", {}), n=summary.get("n") or (int(level.name[1:]) if level else None),
                 completed_snapshot_last_arrival_s=max(r["t_recv_s"] for r in rows)-t0,
                 drained_receipt_count=int(declared[-1]) if declared else None,
                 dispatch_count=run.get("dispatched"),
                 dispatch_ledger_independently_checked=False,
                 engine_started=next((s for s in log.splitlines() if s.startswith("ENGINE_STARTED:")), None),
                 mechanism_receipt=next((s for s in log.splitlines() if s.startswith("MECHANISMS ")), None),
                 all=metrics(rows))
        timestamp_issues = Counter()
        for r in rows:
            ts = [r.get(k) for k in ("t_recv_s", "t_exec_start_s", "t_first_token_s")]
            if not all(numeric(v) for v in ts):
                timestamp_issues["missing_server_timestamps"] += 1
            elif not ts[0] <= ts[1] <= ts[2]:
                timestamp_issues["nonmonotonic_server_timestamps"] += 1
            elif numeric(r.get("ttft_s")) and abs(ts[2]-ts[0]-r["ttft_s"]) > 1e-5:
                timestamp_issues["server_ttft_mismatch"] += 1
        d["timestamp_issues"] = dict(timestamp_issues)
        d["ttft_sources"] = dict(Counter(r.get("ttft_source") for r in rows))
        for boundary in (30, 60, 300, 600):
            before = [r for r in rows if r["t_recv_s"]-t0 < boundary]
            after = [r for r in rows if r["t_recv_s"]-t0 >= boundary]
            d[f"before_{boundary}s"] = metrics(before)
            d[f"after_{boundary}s"] = metrics(after)
        for label, subset in (
            ("heads_only", [r for r in rows if r.get("idx_in_chain") == 0]),
            ("later_heads", [r for r in rows if r.get("idx_in_chain") == 0 and r["t_recv_s"]-t0 >= 60]),
            ("internal_resets", [r for r in rows if r.get("idx_in_chain", 0) > 0 and in_ttft_gate(r, "chain_start")]),
        ):
            d[label] = metrics(subset)
            d[label]["uncached_tokens"] = distribution([
                r["prompt_tokens"]-r["cached_tokens"] for r in subset
                if numeric(r.get("prompt_tokens")) and numeric(r.get("cached_tokens"))])
        # Arrival cohorts at fixed cutoffs: no comparison of unequal drain periods.
        for cutoff in (600, 1800, 2400, 3600):
            subset = [r for r in rows if r["t_recv_s"]-t0 < cutoff]
            z = metrics(subset)["chain_start"]
            time_rows.append(dict(run=code, cutoff_s=cutoff, observed_rows=len(subset),
                                  chain_n=z["n"], chain_over=z["over"], chain_p95=z["p95"],
                                  fully_observed_arrival_window=(status == "DRAINED_DIAGNOSTIC"
                                      and d["completed_snapshot_last_arrival_s"] >= cutoff-2)))
        selected = [r for r in rows if in_ttft_gate(r, "chain_start")]
        for r in selected:
            valid = all(numeric(r.get(k)) for k in ("t_recv_s", "t_exec_start_s", "t_first_token_s"))
            wait = r["t_exec_start_s"]-r["t_recv_s"] if valid else None
            after = r["t_first_token_s"]-r["t_exec_start_s"] if valid else None
            chain_rows.append(dict(run=code, scope=status, req_id=r["req_id"],
                phase=r.get("phase"), idx_in_chain=r.get("idx_in_chain"),
                arrival_s=r["t_recv_s"]-t0, source_edge=r.get("edge_type"),
                ttft_s=r.get("ttft_s"), wait_to_first_batch_s=wait,
                first_batch_to_first_token_s=after, queue_time_s=r.get("queue_time_s"),
                prompt_tokens=r.get("prompt_tokens"), cached_tokens=r.get("cached_tokens"),
                uncached_tokens=(r["prompt_tokens"]-r["cached_tokens"]
                                 if numeric(r.get("prompt_tokens")) and numeric(r.get("cached_tokens")) else None),
                output_tokens=r.get("output_tokens"), tpot_s=r.get("tpot_s")))
        misses = [r for r in chain_rows if r["run"] == code and numeric(r["ttft_s"]) and r["ttft_s"] > 30]
        d["miss_wait_over_30"] = sum(numeric(r["wait_to_first_batch_s"]) and r["wait_to_first_batch_s"] > 30 for r in misses)
        d["miss_after_entry_over_30"] = sum(numeric(r["first_batch_to_first_token_s"]) and r["first_batch_to_first_token_s"] > 30 for r in misses)
        d["summary_quantiles_match"] = all(
            next(d["all"][g]["p95"] for g in GATES if label.startswith(g)) == value
            for label, value in summary.get("ttft_p95_by_gate", {}).items()) if summary else None
        if summary and d["summary_quantiles_match"] is not True:
            raise ValueError(f"original summary does not match harness recomputation: {code}")
        runs[code] = d

    pairs, paired_rows = [], []
    for a, b in PAIRS:
        if a not in raw_by_run or b not in raw_by_run:
            continue
        aa, bb = raw_by_run[a], raw_by_run[b]
        ids = sorted(aa.keys() & bb.keys())
        left, right = [aa[k] for k in ids], [bb[k] for k in ids]
        change_keys = ("phase", "idx_in_chain", "glm_tokens", "uncached_expected", "max_output_i", "replay_gap_ms")
        p = dict(a=a, b=b, common=len(ids), left=metrics(left), right=metrics(right),
                 frozen_metadata_differences={k:sum(aa[r].get(k) != bb[r].get(k) for r in ids) for k in change_keys},
                 note="Matched completed IDs only. Configs, arrivals, coverage and bodies may differ; no causal estimate.")
        for gate, limit in GATES.items():
            fixed = [k for k in ids if in_ttft_gate(aa[k], gate) and in_ttft_gate(bb[k], gate)
                     and numeric(aa[k].get("ttft_s")) and numeric(bb[k].get("ttft_s"))]
            p[gate + "_repaired"] = sum(aa[k]["ttft_s"] > limit >= bb[k]["ttft_s"] for k in fixed)
            p[gate + "_new"] = sum(bb[k]["ttft_s"] > limit >= aa[k]["ttft_s"] for k in fixed)
            if gate == "chain_start":
                for k in fixed:
                    paired_rows.append(dict(pair=f"{a}__{b}", req_id=k, a_ttft_s=aa[k]["ttft_s"],
                        b_ttft_s=bb[k]["ttft_s"], a_cached=aa[k].get("cached_tokens"),
                        b_cached=bb[k].get("cached_tokens"), a_over=aa[k]["ttft_s"] > 30,
                        b_over=bb[k]["ttft_s"] > 30))
        pairs.append(p)

    official = []
    for ident in (46676, 46677, 46757, 46758):
        path = root / f"evidence/official/attempt-{ident}-20260927.json"
        j = json.loads(receipt(path))
        official.append(dict(attempt=ident, source=str(path), exec_status=j.get("execStatus"),
            final=j.get("scoringState", {}).get("scoreIsFinal"), updated_at=j.get("updatedAt"),
            stress=j.get("scorecard", {}).get("scorewheel_stress"),
            failed_level_details_available=False, per_request_raw_available=False))
    public_root = root / "s1-dev/data/dev-combined-v1"
    public, _, _ = load_index(str(public_root))
    receipt(public_root / "requests.jsonl")
    datasets, cohorts = {}, {}
    for version in ("v3", "v3g", "v4", "v5"):
        dataset = root / "cache" / ("s1-dev-longchain-" + version)
        rows, _, _ = load_index(str(dataset))
        receipt(dataset / "requests.jsonl")
        cohort = json.loads(receipt(dataset / "cohort.json"))
        cohorts[version] = cohort
        heads = [r for r in rows.values() if r["_idx_in_chain"] == 0]
        resets = [r for r in rows.values() if r["_idx_in_chain"] > 0 and in_ttft_gate(r, "chain_start")]
        datasets[version] = dict(requests=len(rows), chains=len(heads),
            chain_gate_requests=sum(in_ttft_gate(r, "chain_start") for r in rows.values()),
            internal_resets=len(resets), internal_resets_public=sum(r["_req_id"] in public for r in resets),
            internal_resets_synthetic=sum(r["_req_id"] not in public for r in resets),
            internal_reset_edges=dict(Counter(r.get("edge_type") for r in resets)),
            internal_reset_expected_uncached=distribution([r["uncached_expected"] for r in resets]),
            output_budgets=distribution([r["max_output_i"] for r in rows.values()]),
            cohort_sha256=cohort["cohort_sha256"])
        gap_plan, gap_stats = gap_ns["build_gap_plan"](cohort["chains"], rows, 3600000)
        nonhead_gaps = [g/1000 for ch in cohort["chains"] for g in gap_plan[ch["chain_id"]][1:]]
        datasets[version]["gap_nonhead_s"] = distribution(nonhead_gaps)
        datasets[version]["gap_nonhead_s"].update(
            sum_s=sum(nonhead_gaps), mean_s=sum(nonhead_gaps)/len(nonhead_gaps),
            over_60s=sum(g > 60 for g in nonhead_gaps))
        datasets[version]["gap_all_s"] = gap_stats["effective_gap_sum_ms"]/1000
        datasets[version]["gap_capped_chains"] = gap_stats["n_chains_compressed"]
    a, b = (cohorts[v]["chains"] for v in ("v3", "v4"))
    cohort_comparison = dict(
        same_chain_set={c["chain_id"] for c in a} == {c["chain_id"] for c in b},
        identical_positions=sum(x["chain_id"] == y["chain_id"] for x, y in zip(a, b)),
        common_first_n={n:len({c["chain_id"] for c in a[:n]} & {c["chain_id"] for c in b[:n]}) for n in (26,34,38)},
        giant_positions={v:{needle:[i+1 for i,c in enumerate(cohorts[v]["chains"]) if needle in c["req_ids"][0]]
                           for needle in ("QSdTYVbowNG_k_R8lOG7T", "YM1Suj4E5fA4Bx-ZGQj4c")} for v in ("v3", "v4")})
    receipt(root / "s1-dev/harness/s1_common.py")
    receipt(loadgen_path)
    receipt(Path(__file__).resolve())
    result = dict(scope="diagnostic_reanalysis_not_formal_score", official=official, runs=runs, pairs=pairs,
        datasets=datasets, cohort_v3_v4=cohort_comparison,
        definitions={"opening": "server receive < first receive + 60 s; also report 30/300/600 cutoffs",
                     "after_entry": "first batch admission to first token, includes all chunks, decode and stalls; not pure GPU time",
                     "p95": "unmodified s1_common.q", "gate": "unmodified s1_common.in_ttft_gate",
                     "censoring": "Aborted snapshots only contain completed rows; all window runs omit undispatched requests."}, inputs=inputs)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    write_csv(out / "chain_requests.csv", chain_rows)
    write_csv(out / "chain_pairs.csv", paired_rows)
    write_csv(out / "arrival_windows.csv", time_rows)
    print(json.dumps(dict(runs=len(runs), official=len(official), pairs=len(pairs),
                          chain_rows=len(chain_rows), out=str(out))))


if __name__ == "__main__":
    main()
