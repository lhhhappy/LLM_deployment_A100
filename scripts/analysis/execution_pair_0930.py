#!/usr/bin/env python3
"""Pair two drained execution diagnostics without changing their scoring.

Each directory needs timed_verdict.json and its SHA-bound measured raw file.
Optional timed_score.json retains all original harness gate reports. This tool
reports common-ID diagnostics separately from each run's complete observed set;
it never declares a full-cohort PASS or promotes a candidate.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
FROZEN_RUNTIME = ROOT / "build/queue/queue-0929-chain-n42/verify_kit"
META = ("phase", "idx_in_chain", "chain_id", "session_id", "edge_type",
        "logical_call_id", "glm_tokens", "uncached_expected", "max_output_i",
        "replay_gap_ms", "pack", "view")


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load(directory):
    verdict = json.loads((directory / "timed_verdict.json").read_text())
    if verdict.get("status") != "DRAINED" or verdict.get("errors") != 0:
        raise ValueError(f"{directory}: requires DRAINED zero-error receipt")
    raw = directory / Path(verdict["raw"]).name
    contents = raw.read_bytes()
    if hashlib.sha256(contents).hexdigest() != verdict["raw_sha256"]:
        raise ValueError(f"{directory}: measured raw SHA mismatch")
    rows = [json.loads(line) for line in contents.splitlines() if line.strip()]
    by = {r["req_id"]: r for r in rows}
    if len(rows) != len(by) or len(rows) != verdict["n_completed"]:
        raise ValueError(f"{directory}: duplicate IDs or receipt census mismatch")
    if set(by) != set(verdict["dispatched_req_ids"]):
        raise ValueError(f"{directory}: dispatched/raw ID mismatch")
    start, end = verdict["first_dispatch_at_s"], verdict["admission_deadline_s"]
    if end != start + verdict["admission_seconds"]:
        raise ValueError(f"{directory}: admission interval mismatch")
    for rid, row in by.items():
        if row.get("error") or row.get("error_class") in ("HARNESS_DATA", "HARNESS_RENDER"):
            raise ValueError(f"{directory}: failed request {rid}")
        if not number(row.get("ttft_s")) or row["ttft_s"] < 0:
            raise ValueError(f"{directory}: invalid TTFT {rid}")
        dispatch, finish = row.get("client_dispatch_at_s"), row.get("client_finish_at_s")
        if not number(dispatch) or not number(finish) or not start <= dispatch < end or finish < dispatch:
            raise ValueError(f"{directory}: invalid client interval {rid}")
        if not all(key in row for key in META):
            raise ValueError(f"{directory}: missing frozen metadata {rid}")
        output = row.get("output_tokens")
        if not number(output) or output < 1 or int(output) != output:
            raise ValueError(f"{directory}: invalid output count {rid}")
        if output > 1 and (not number(row.get("tpot_s")) or row["tpot_s"] < 0):
            raise ValueError(f"{directory}: invalid TPOT {rid}")
    score_path = directory / "timed_score.json"
    score = json.loads(score_path.read_text()) if score_path.exists() else None
    if score is not None and (score["tpot"] != verdict["tpot"] or score["ttft_estimated"] != verdict["ttft"]):
        raise ValueError(f"{directory}: original score/verdict disagreement")
    return by, verdict, score


def stats(values, scorer):
    return dict(n=len(values), mean=math.fsum(values) / len(values) if values else None,
                p50=scorer.q(values, .5) if values else None,
                p95=scorer.q(values, .95) if values else None)


def tpot_pair(ids, a, b, scorer):
    valid = [rid for rid in ids if a[rid]["output_tokens"] > 1 and b[rid]["output_tokens"] > 1]
    equal = [rid for rid in valid if a[rid]["output_tokens"] == b[rid]["output_tokens"]]
    def describe(selected):
        return dict(n=len(selected), baseline=stats([a[r]["tpot_s"] for r in selected], scorer),
                    candidate=stats([b[r]["tpot_s"] for r in selected], scorer),
                    candidate_minus_baseline=stats([b[r]["tpot_s"]-a[r]["tpot_s"] for r in selected], scorer))
    return dict(all_common_multitoken=describe(valid), equal_output_counts=describe(equal),
                changed_output_count_ids=[rid for rid in ids if a[rid]["output_tokens"] != b[rid]["output_tokens"]],
                single_token_either_ids=[rid for rid in ids if min(a[rid]["output_tokens"], b[rid]["output_tokens"]) == 1],
                note="Equal-output comparison is a diagnostic subset; retain every observed request in original gates. Raw client SSE TPOT is never token-weighted or recomputed from HTTP finish time.")


def timing(row, key):
    return row.get(key) if number(row.get(key)) else None


def elapsed(row, last, first):
    x, y = timing(row, last), timing(row, first)
    return x-y if x is not None and y is not None else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, default=FROZEN_RUNTIME)
    parser.add_argument("--harness-dir", type=Path, default=ROOT / "s1-dev/harness")
    args = parser.parse_args(argv)
    spec = importlib.util.spec_from_file_location("_execution_frozen_score", args.runtime_dir / "score_formal.py")
    formal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(formal)
    scorer = formal.load_harness(args.harness_dir)
    a, va, sa = load(args.baseline)
    b, vb, sb = load(args.candidate)
    if va["n"] != vb["n"] or va["admission_seconds"] != vb["admission_seconds"]:
        raise ValueError("paired execution comparison requires identical N and admission duration")
    ids = sorted(set(a) & set(b))
    if not ids:
        raise ValueError("no common requests")
    for rid in ids:
        changed = [key for key in META if a[rid].get(key) != b[rid].get(key)]
        if changed:
            raise ValueError(f"frozen replay metadata differs for {rid}: {changed}")
        if any(scorer.in_ttft_gate(a[rid], selector) != scorer.in_ttft_gate(b[rid], selector)
               for _, selector, _ in scorer.TTFT_GATE_SPECS):
            raise ValueError(f"gate membership differs for {rid}")
    # Compare each source receipt to its measured rows, including unpaired IDs.
    for rows, verdict in ((a, va), (b, vb)):
        values = [r["tpot_s"] for r in rows.values() if r["output_tokens"] > 1]
        actual = stats(values, scorer)
        for key, value in (("tpot_mean", actual["mean"]), ("tpot_p95", actual["p95"])):
            if not math.isclose(verdict["tpot"][key], value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"source TPOT receipt mismatch: {key}")
        for name, selector, limit in scorer.TTFT_GATE_SPECS:
            selected = [r for r in rows.values() if scorer.in_ttft_gate(r, selector)]
            gate = verdict["ttft"][name]
            over = sum(r["ttft_s"] > limit for r in selected)
            if gate["n"] != len(selected) or gate["over_limit"] != over or gate["allowed_over"] != formal.allowed_over(len(selected)):
                raise ValueError(f"source TTFT receipt mismatch: {name}")
    paired, csv_rows = {}, []
    for name, selector, limit in scorer.TTFT_GATE_SPECS:
        selected = [rid for rid in ids if scorer.in_ttft_gate(a[rid], selector)]
        repaired = [r for r in selected if a[r]["ttft_s"] > limit >= b[r]["ttft_s"]]
        new = [r for r in selected if b[r]["ttft_s"] > limit >= a[r]["ttft_s"]]
        groups = collections.defaultdict(list)
        for rid in selected:
            aa, bb = a[rid], b[rid]
            opening_a = aa["client_dispatch_at_s"] - va["first_dispatch_at_s"] < 600
            opening_b = bb["client_dispatch_at_s"] - vb["first_dispatch_at_s"] < 600
            period = "opening_both" if opening_a and opening_b else "steady_both" if not opening_a and not opening_b else "opening_boundary_mixed"
            transition = "repaired" if rid in repaired else "new_bad" if rid in new else "bad_both" if aa["ttft_s"] > limit else "good_both"
            groups[(period, aa["edge_type"], aa["phase"])].append((rid, transition))
            csv_rows.append(dict(gate=name, req_id=rid, period=period, edge_type=aa["edge_type"], phase=aa["phase"],
                                 transition=transition, baseline_ttft_s=aa["ttft_s"], candidate_ttft_s=bb["ttft_s"],
                                 delta_ttft_s=bb["ttft_s"]-aa["ttft_s"],
                                 baseline_dispatch_offset_s=aa["client_dispatch_at_s"]-va["first_dispatch_at_s"],
                                 candidate_dispatch_offset_s=bb["client_dispatch_at_s"]-vb["first_dispatch_at_s"],
                                 baseline_wait_s=elapsed(aa,"t_exec_start_s","t_recv_s"), candidate_wait_s=elapsed(bb,"t_exec_start_s","t_recv_s"),
                                 baseline_exec_to_first_s=elapsed(aa,"t_first_token_s","t_exec_start_s"), candidate_exec_to_first_s=elapsed(bb,"t_first_token_s","t_exec_start_s"),
                                 baseline_cached_tokens=aa.get("cached_tokens"), candidate_cached_tokens=bb.get("cached_tokens"),
                                 baseline_prompt_tokens=aa.get("prompt_tokens"), candidate_prompt_tokens=bb.get("prompt_tokens"),
                                 baseline_output_tokens=aa["output_tokens"], candidate_output_tokens=bb["output_tokens"],
                                 baseline_tpot_s=aa.get("tpot_s"), candidate_tpot_s=bb.get("tpot_s")))
        paired[name] = dict(n=len(selected), limit_s=limit, repaired_ids=repaired, new_bad_ids=new,
                            baseline=stats([a[r]["ttft_s"] for r in selected],scorer),
                            candidate=stats([b[r]["ttft_s"] for r in selected],scorer),
                            delta=stats([b[r]["ttft_s"]-a[r]["ttft_s"] for r in selected],scorer),
                            strata=[dict(period=k[0],edge_type=k[1],phase=k[2],n=len(v),transitions=dict(collections.Counter(t for _,t in v)))
                                    for k,v in sorted(groups.items(),key=lambda item:str(item[0]))])
    def original(directory, verdict, score):
        return dict(directory=str(directory), n_completed=verdict["n_completed"],raw_sha256=verdict["raw_sha256"],
                    original_ttft=verdict["ttft"], original_tpot=verdict["tpot"],
                    original_harness_gates=score["dev"]["gates"] if score else None,
                    original_harness_gates_note=None if score else "timed_score.json absent; additional original harness gates unknown")
    result = dict(scope="paired drained diagnostics; no complete-cohort PASS or automatic promotion",
                  baseline=original(args.baseline,va,sa), candidate=original(args.candidate,vb,sb),
                  census=dict(common=len(ids),baseline_only_ids=sorted(set(a)-set(b)),candidate_only_ids=sorted(set(b)-set(a))),
                  paired_ttft=paired, paired_tpot=tpot_pair(ids,a,b,scorer),
                  caveats=["Source DRAINED receipts are SHA/census verified; this tool does not independently rerun the dispatch ledger or verify frozen prompt bodies.",
                           "Closed-loop arrival times change with predecessor completion; common-ID effects are descriptive, not isolated kernel causality.",
                           "Server exec_start-recv is not a pure queue timer. Missing timing remains unknown.",
                           "Opening is the first600s in each run; mixed-boundary requests are reported separately. No failing segment is removed from original scoring."])
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out / "paired.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    with (args.out / "paired.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    chain = next(g for name,g in paired.items() if name.startswith("chain_start"))
    print(json.dumps(dict(common=len(ids),chain_repaired=len(chain["repaired_ids"]),chain_new_bad=len(chain["new_bad_ids"]),
                          equal_output_tpot=result["paired_tpot"]["equal_output_counts"]),ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode=True
    try:
        sys.exit(main())
    except (ValueError,KeyError,OSError,TypeError) as error:
        print(f"INVALID: {error}",file=sys.stderr)
        sys.exit(2)
