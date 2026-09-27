#!/usr/bin/env python3
"""Read-only audit of public chain heads, v3 provenance, and opening raw.

This produces diagnostic evidence, never an N@SLO verdict. The source-body
check compares public bodies with frozen v3 provenance; it does not read the
deployed v3 body shard. Run with --repo-root pointing to the evidence checkout.
"""
import argparse
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys


def records(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "s1-dev/harness"))
    from s1_common import in_ttft_gate, load_index

    public = root / "s1-dev/data/dev-combined-v1"
    v3 = root / "cache/s1-dev-longchain-v3"
    public_rows, _, public_groups = load_index(str(public))
    v3_rows, _, v3_groups = load_index(str(v3))
    heads = {cid: rows[0] for cid, rows in public_groups.items()}
    v3_heads = {cid: rows[0] for cid, rows in v3_groups.items()}
    assert heads.keys() == v3_heads.keys()
    differences = collections.Counter()
    for cid, a in heads.items():
        b = v3_heads[cid]
        differences.update(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
    assert set(differences) <= {"session_id", "body_ref"}, differences

    provenance = {r["req_id"]: r for r in records(v3 / "provenance.jsonl")
                  if r["kind"] == "original"}
    body_mismatches, seen_bodies = [], set()
    for shard in sorted((public / "bodies").rglob("*.jsonl.gz")):
        for body in records(shard):
            rid = body["req_id"]
            assert rid not in seen_bodies, rid
            seen_bodies.add(rid)
            digest = hashlib.sha256(json.dumps(
                body, ensure_ascii=False, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest()
            p = provenance.get(rid, {})
            if p.get("source_body_sha256") != digest or p.get("body_sha256") != digest:
                body_mismatches.append(rid)
    assert seen_bodies == public_rows.keys() == provenance.keys()
    assert not body_mismatches, body_mismatches

    public_cohort = json.loads((root / "s1-dev/harness/g0a/samples_v3/cohort_dev-combined-v1.json").read_text())
    v3_cohort = json.loads((v3 / "cohort.json").read_text())
    initial = {ch["req_ids"][0] for ch in v3_cohort["chains"][:34]}
    public_initial = {ch["req_ids"][0] for ch in public_cohort["chains"][:34]}
    tokens = sorted(r["glm_tokens"] for r in heads.values())
    summary = {
        "scope": "public-source and same-ID opening diagnostics; no formal capacity prediction",
        "input_root": str(root),
        "public": {
            "requests": len(public_rows), "chains": len(heads),
            "head_prompt_quantiles": {str(q): tokens[min(len(tokens)-1, int(q*len(tokens)))]
                                      for q in (0, .25, .5, .75, .9, .95, 1)},
            "head_prompt_at_least": {str(n): sum(t >= n for t in tokens)
                                     for n in (64000, 65536, 100000, 200000)},
            "head_phase": dict(collections.Counter(r["phase"] for r in heads.values())),
            "head_edge": dict(collections.Counter(r["edge_type"] for r in heads.values())),
            "head_split": dict(collections.Counter(r["split"] for r in heads.values())),
            "chain_gate_n": sum(in_ttft_gate(r, "chain_start") for r in public_rows.values()),
        },
        "v3": {"requests": len(v3_rows), "chains": len(v3_heads),
               "chain_gate_n": sum(in_ttft_gate(r, "chain_start") for r in v3_rows.values()),
               "head_metadata_difference_fields": dict(differences)},
        "body_provenance": {"public_bodies_checked": len(seen_bodies),
                            "mismatches": body_mismatches,
                            "deployed_v3_body_shard_rechecked": False},
        "cohort": {"public": public_cohort["cohort_sha256"],
                   "v3": v3_cohort["cohort_sha256"],
                   "first34_head_overlap": len(initial & public_initial),
                   "v3_first34_gap_ms": sorted(v3_rows[i].get("replay_gap_ms") or 0 for i in initial)},
        "runs": {}, "comparisons": {},
    }
    run_ids = ("130ez1", "130ez4", "130ez5", "130ez6", "130ez6z",
               "130ez6zz", "130ez6zzz", "130ez6zzzz")
    raw = {}
    opening_sets = []
    for run_id in run_ids:
        directories = list((root / "evidence").glob(f"L{run_id}-*"))
        assert len(directories) == 1, directories
        directory = directories[0] / "N34"
        paths = list(directory.glob("raw*.jsonl"))
        assert len(paths) == 1, paths
        rows = list(records(paths[0]))
        by_id = {r["req_id"]: r for r in rows}
        assert len(by_id) == len(rows) and initial <= by_id.keys()
        assert all(not r.get("error") and not r.get("error_class") for r in rows)
        for r in rows:
            assert r["t_recv_s"] <= r["t_exec_start_s"] <= r["t_first_token_s"]
            assert abs(r["ttft_s"] - (r["t_first_token_s"] - r["t_recv_s"])) < 1e-4
        origin = min(r["client_dispatch_at_s"] for r in rows)
        opening = {r["req_id"] for r in rows if in_ttft_gate(r, "chain_start")
                   and r["client_dispatch_at_s"] - origin < 30}
        assert opening <= initial
        opening_sets.append(opening)
        bad = [by_id[i] for i in opening if by_id[i]["ttft_s"] > 30]
        receipt = json.loads((directory / "flush_evidence.json").read_text())
        assert receipt["flush_success"] and receipt["runner_rc"] == 0
        run_doc = json.loads(next(directory.glob("run_s1-dev*.json")).read_text())
        assert run_doc["n_attempted"] == len(rows)
        config = run_doc["config"]
        assert config["N"] == 34
        assert config["cohort_sha256"] == v3_cohort["cohort_sha256"]
        assert all(by_id[i]["prompt_tokens"] == v3_rows[i]["glm_tokens"] for i in initial)
        summary["runs"][run_id] = {
            "raw": str(paths[0].relative_to(root)), "rows": len(rows),
            "scope": receipt["scope"], "initial34_present": len(initial & by_id.keys()),
            "opening_heads": len(opening), "opening_bad": len(bad),
            "initial34_bad": sum(by_id[i]["ttft_s"] > 30 for i in initial),
            "bad_first_batch_to_first_token_max_s": max(r["t_first_token_s"] - r["t_exec_start_s"] for r in bad),
            "bad_before_first_batch_fraction_of_summed_ttft":
                sum(r["t_exec_start_s"] - r["t_recv_s"] for r in bad) / sum(r["ttft_s"] for r in bad),
        }
        raw[run_id] = by_id
    assert all(x == opening_sets[0] for x in opening_sets)
    base = raw["130ez6zz"]
    for run_id in ("130ez5", "130ez6zzz", "130ez6zzzz"):
        candidate = raw[run_id]
        common = base.keys() & candidate.keys()
        chain = {i for i in common if in_ttft_gate(base[i], "chain_start")}
        summary["comparisons"][run_id] = {
            "base": "130ez6zz", "common_requests": len(common),
            "common_chain_gate": len(chain),
            "fixed": sorted(i for i in chain if base[i]["ttft_s"] > 30 >= candidate[i]["ttft_s"]),
            "regressed": sorted(i for i in chain if base[i]["ttft_s"] <= 30 < candidate[i]["ttft_s"]),
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with (args.out / "initial34.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["run", "req_id", "prompt_tokens", "cached_tokens", "ttft_s",
                         "recv_to_first_batch_s", "first_batch_to_first_token_s"])
        for run_id in run_ids:
            for rid in sorted(initial):
                r = raw[run_id][rid]
                writer.writerow([run_id, rid, r["prompt_tokens"], r["cached_tokens"], r["ttft_s"],
                                 r["t_exec_start_s"]-r["t_recv_s"], r["t_first_token_s"]-r["t_exec_start_s"]])
    print(json.dumps({"head_metadata_differences": dict(differences),
                      "public_bodies_provenance_checked": len(seen_bodies),
                      "first34_overlap": len(initial & public_initial),
                      "opening_bad": {k: v["opening_bad"] for k, v in summary["runs"].items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
