#!/usr/bin/env python3
"""Recheck 118 numerical evidence and pair the drained TP8 opening probes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="shared evidence and original harness")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    own = Path(__file__).resolve().parents[2]
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.repo / "s1-dev/harness"))
    from s1_common import in_ttft_gate, q
    inputs, runs, totals = [], [], []
    for name in ("L130ezm7-v3_open_S5b_118off_n34", "L130ezm8-v3_open_S5b_118prefill_n34"):
        directory = args.repo / "evidence" / name / "N34"
        paths = list(directory.glob("raw*.jsonl"))
        if len(paths) != 1:
            raise ValueError(f"ambiguous raw: {directory}")
        path = paths[0]
        blob = path.read_bytes()
        rows = [json.loads(line) for line in blob.splitlines() if line.strip()]
        if len({r["req_id"] for r in rows}) != len(rows) or any(r.get("error") or r.get("error_class") for r in rows):
            raise ValueError(f"duplicates or errors: {path}")
        inputs.append(dict(path=str(path), bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest()))
        runs.append({r["req_id"]: r for r in rows})
        flush = json.loads((directory / "flush_evidence.json").read_text())
        if not flush["flush_success"] or flush["runner_rc"] != 0:
            raise ValueError(f"flush/runner failed: {directory}")
        totals.append(len(rows))
    off, on = runs
    common = sorted(off.keys() & on.keys())
    frozen = ("idx_in_chain", "phase", "uncached_expected", "max_output_i", "glm_tokens")
    for rid in common:
        if any(off[rid].get(k) != on[rid].get(k) for k in frozen):
            raise ValueError(f"frozen request fields changed: {rid}")
    summary = {"scope": "DRAINED_DIAGNOSTIC, paired IDs; not a full-cohort score",
               "inputs": inputs, "total_rows_off_on": totals, "common_ids": len(common), "gates": {}}
    for gate, limit in (("chain_start", 30), ("turn_start", 15), ("overall_intra", 5), ("fast_intra", 3)):
        ids = [rid for rid in common if in_ttft_gate(off[rid], gate)]
        before = {rid for rid in ids if off[rid]["ttft_s"] > limit}
        after = {rid for rid in ids if on[rid]["ttft_s"] > limit}
        summary["gates"][gate] = dict(count=len(ids), misses_off_on=[len(before), len(after)],
            p95_off_on=[q([run[rid]["ttft_s"] for rid in ids], .95) for run in runs],
            fixed=sorted(before-after), newly_late=sorted(after-before))
    summary["tpot_over_0_10_off_on"] = [sum((run[rid].get("tpot_s") or 0) > .10 for rid in common) for run in runs]
    chains = []
    for rid in common:
        if not in_ttft_gate(off[rid], "chain_start"):
            continue
        item = {"req_id": rid, "idx_in_chain": off[rid]["idx_in_chain"], "phase": off[rid]["phase"]}
        for label, run in zip(("off", "on"), runs):
            r = run[rid]
            recv, forward, first = (r[k] for k in ("t_recv_s", "t_exec_start_s", "t_first_token_s"))
            if not recv <= forward <= first or abs(first-recv-r["ttft_s"]) > 1e-5:
                raise ValueError(f"invalid server timing: {rid}, {label}")
            item.update({label + "_" + k: v for k, v in dict(
                prompt_tokens=r["prompt_tokens"], cached_tokens=r["cached_tokens"], ttft_s=r["ttft_s"],
                wait_to_forward_s=forward-recv, forward_to_first_s=first-forward).items()})
        chains.append(item)
    summary["late_before_first_forward_off_on"] = [
        sum(r[label+"_ttft_s"] > 30 and r[label+"_wait_to_forward_s"] > 30 for r in chains)
        for label in ("off", "on")]
    evidence = own / "evidence/prefill-sm80-0927"
    numeric = []
    for line in (evidence / "118-prefill-tests.log").read_text().splitlines():
        if "{" not in line:
            continue
        try:
            record = json.loads(line[line.index("{"):])
        except json.JSONDecodeError:
            continue
        if record.get("kind") == "numerics":
            numeric.append(record)
    costs = [json.loads(line) for line in (evidence / "118-prefill-cost.jsonl").read_text().splitlines()]
    costs = [{k: r[k] for k in ("case", "query_rows", "reference_rows", "error")}
             for r in costs if r.get("kind") == "case"]
    summary["numerical_evidence"] = dict(scope="Single GPU synthetic operator, not full-model logits/sequence equality",
        bitwise_equal=False, numeric_cases=numeric, sampled_full_kv_cases=costs,
        worst_bound_ratio=max(r["triton_bound"] for r in numeric),
        max_triton_tilelang_difference=max(r.get("triton_vs_served_max_abs", 0) for r in numeric),
        on_mean_error_below_off_in_all_cost_cases=all(r["error"]["on"]["mean_abs"] <= r["error"]["off"]["mean_abs"] for r in costs))
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with (args.out / "paired-chain.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(chains[0]))
        writer.writeheader()
        writer.writerows(chains)
    print(json.dumps({k:v for k,v in summary.items() if k != "numerical_evidence"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
