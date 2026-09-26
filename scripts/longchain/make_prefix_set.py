#!/usr/bin/env python3
"""Derive a chain-prefix dataset root: every chain keeps only its first K requests.

Why: an official stress level replays ~1788 requests and ~1.05e8 prompt tokens (inferred from the
k/1788 slo_attainment values and tpm_all), while data/s1-dev-longchain replays 5601 requests and 3.94e8.
Keeping the first 9 requests of each of its 311 chains gives 1865 requests and 1.050e8 tokens, with
chain starts at 18.8% of requests instead of 7.7%. This root is for calibrating local runs against
official results; it is not the official data.

Order within a chain is the harness's (s1_common.load_index: dispatch_offset_ms, then logical_call_id),
and the cohort's req_ids must list the kept requests in that same order (checked). Request bodies,
samples, event plans and provenance are symlinked to the source (selection is by req_id / chain_id).
Usage: python3 scripts/longchain/make_prefix_set.py --src data/s1-dev-longchain --dst data/s1-dev-longchain-k9 --k 9
       [--harness-dir <s1-dev/harness>]   (the harness whose load_index defines chain order)
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--harness-dir", type=Path, default=ROOT / "s1-dev" / "harness")
    args = ap.parse_args()
    sys.path.insert(0, str(args.harness_dir))
    global load_index
    from s1_common import load_index
    src, dst, k = args.src.resolve(), args.dst, args.k
    if k < 1:
        raise SystemExit("--k must be >= 1")
    if dst.exists():
        raise SystemExit(f"{dst} exists; remove it first")

    rows, _chains, by_chain = load_index(str(src))
    keep = {r["_req_id"] for rs in by_chain.values() for r in rs if r["_idx_in_chain"] < k}

    dst.mkdir(parents=True)
    kept_rows = []
    with open(src / "requests.jsonl") as fin, open(dst / "requests.jsonl", "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["view"] == "canon" and "%s:%s:%s" % (r["pack"], r["view"], r["logical_call_id"]) in keep:
                fout.write(line if line.endswith("\n") else line + "\n")
                kept_rows.append(r)

    per_chain = defaultdict(list)
    for r in kept_rows:
        per_chain[r["chain_id"]].append(r)
    with open(src / "chains.jsonl") as fin, open(dst / "chains.jsonl", "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            c = json.loads(line)
            rs = per_chain.get(c["chain_id"])
            if c["view"] != "canon" or not rs:
                continue
            c.update(n_requests=len(rs), sum_glm_tokens=sum(r["glm_tokens"] for r in rs),
                     sum_uncached_expected=sum(r["uncached_expected"] for r in rs),
                     max_output_i_sum=sum(r.get("max_output_i") or 0 for r in rs),
                     phases=dict(Counter(r["phase"] for r in rs)), prefix_of=str(src.name), prefix_k=k)
            fout.write(json.dumps(c, ensure_ascii=False) + "\n")

    cohort = json.loads((src / "cohort.json").read_text())
    order = {rid: rows[rid]["_idx_in_chain"] for rid in keep}
    chains = []
    for ch in cohort["chains"]:
        ids = [rid for rid in ch["req_ids"] if rid in keep]
        if [order[rid] for rid in ids] != list(range(len(ids))):
            raise SystemExit(f"cohort order differs from the harness order in {ch['chain_id']}")
        if ids:
            chains.append(dict(ch, req_ids=ids))
    if sum(len(c["req_ids"]) for c in chains) != len(keep):
        raise SystemExit("cohort does not cover every kept request exactly once")
    cohort.update(n_chains=len(chains), n_requests=len(keep), chains=chains, prefix_of=str(src.name), prefix_k=k,
                  cohort_sha256=hashlib.sha256(json.dumps(chains, ensure_ascii=False, sort_keys=True)
                                               .encode()).hexdigest()[:16])
    (dst / "cohort.json").write_text(json.dumps(cohort, ensure_ascii=False, indent=2))

    for name in ("bodies", "samples", "event-plans.jsonl", "provenance.jsonl"):
        if (src / name).exists():
            os.symlink(os.path.relpath(src / name, dst), dst / name)
    manifest = json.loads((src / "manifest.json").read_text())
    manifest.update(n_chains=len(chains), n_requests=len(keep),
                    actual_prompt_sum=sum(r["glm_tokens"] for r in kept_rows),
                    prefix_of={"root": src.name, "k": k, "requests_sha256": sha256(src / "requests.jsonl"),
                               "cohort_sha256": json.loads((src / "cohort.json").read_text()).get("cohort_sha256")})
    (dst / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))

    # The harness must see exactly the kept requests with unchanged in-chain positions.
    rows2, _, _ = load_index(str(dst))
    assert set(rows2) == keep and all(rows2[rid]["_idx_in_chain"] == order[rid] for rid in keep)
    print(json.dumps(dict(dst=str(dst), k=k, chains=len(chains), requests=len(keep),
                          prompt_tokens=manifest["actual_prompt_sum"], cohort_sha256=cohort["cohort_sha256"],
                          requests_sha256=sha256(dst / "requests.jsonl"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
