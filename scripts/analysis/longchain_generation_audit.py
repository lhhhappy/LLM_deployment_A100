#!/usr/bin/env python3
"""Read-only workload accounting plus isolated reproductions of generator defects.

Never reads request bodies or modifies a frozen input. The CLI overwrite probes
run the actual producer scripts against disposable, two-request fixtures only.
This is metadata diagnostics, not token re-rendering or formal score acceptance.
"""
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def jsonl(path, values):
    path.write_text("".join(json.dumps(v) + "\n" for v in values))


def fixture(directory):
    directory.mkdir()
    requests = [dict(pack="fixture", view="canon", logical_call_id=f"r{i}",
                     chain_id="c", split="synthetic", max_output_i=budget,
                     glm_tokens=100, gap_imputed=True, replay_gap_ms=1000)
                for i, budget in enumerate((1, 19))]
    jsonl(directory / "requests.jsonl", requests)
    jsonl(directory / "chains.jsonl", [dict(chain_id="c", max_output_i_sum=20,
          first_dispatch_offset_ms=0, last_end_offset_ms=100000)])
    jsonl(directory / "provenance.jsonl", [])
    dump(directory / "cohort.json", dict(set="fixture-parent", cohort_sha256="fixture",
         chains=[dict(chain_id="c", req_ids=["fixture:canon:r0", "fixture:canon:r1"])]))
    dump(directory / "manifest.json", dict(generator="fixture", set="fixture-parent"))


def cli_probe(repo, tool):
    with tempfile.TemporaryDirectory(prefix="longchain-audit-") as temporary:
        parent = Path(temporary) / "parent"
        fixture(parent)
        before = sha(parent / "requests.jsonl")
        if tool == "rebudget":
            public = Path(temporary) / "public"
            public.mkdir()
            jsonl(public / "requests.jsonl", [dict(max_output_i=v) for v in (2, 100)])
            args = ["--parent", str(parent), "--public", str(public), "--out", str(parent)]
        else:
            args = ["--src", str(parent), "--organizer", str(parent / "chains.jsonl"),
                    "--out", str(parent), "--set", "fixture-derived"]
        result = subprocess.run([sys.executable, "-B", str(repo / "scripts/longchain" / (tool + ".py")),
                                 *args], capture_output=True, text=True, check=False, timeout=15)
        after = sha(parent / "requests.jsonl")
        manifest = json.loads((parent / "manifest.json").read_text())
        return dict(tool=tool, scope="disposable fixture only; out equals parent",
                    exit_code=result.returncode, parent_requests_changed=before != after,
                    failure_class="SameFileError" if "SameFileError" in result.stderr else None,
                    parent_receipt_matches_modified_input=(
                        manifest.get("parent_requests_sha256") == after if tool == "regap" else None),
                    original_set="fixture-parent", recorded_parent_set=manifest.get("parent_set"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    protected = [repo / p for p in ("cache", "s1-dev", "llm-challenge-arena-v1", "build/base_exact", "refs")]
    if any(out == p or p in out.parents for p in protected):
        raise ValueError("audit output must be outside input trees")
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "s1-dev/harness"))
    from s1_common import q

    result = dict(scope="metadata accounting; no body re-render or formal representativeness claim",
                  datasets={}, inputs=[], probes=[])
    exported, cohorts = [], {}
    for version in ("v3", "v3g", "v4", "v5"):
        root = repo / "cache" / ("s1-dev-longchain-" + version)
        requests, chains = rows(root / "requests.jsonl"), rows(root / "chains.jsonl")
        manifest = json.loads((root / "manifest.json").read_text())
        cohorts[version] = json.loads((root / "cohort.json").read_text())["chains"]
        for name in ("requests.jsonl", "chains.jsonl", "manifest.json", "cohort.json"):
            p = root / name
            result["inputs"].append(dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size))
        by_chain = collections.defaultdict(list)
        for row in requests:
            by_chain[row["chain_id"]].append(row)
        accounts = []
        summary_by_id = {c["chain_id"]: c for c in manifest.get("chain_summaries", [])}
        summary_mismatches = []
        for chain in chains:
            source = chain["source_chain_targets"]
            rs = by_chain[chain["chain_id"]]
            original = [r for r in rs if r.get("split") != "synthetic"]
            synthetic = [r for r in rs if r.get("split") == "synthetic"]
            target = source["sum_uncached_expected"] - sum(r["uncached_expected"] for r in original)
            actual = sum(r["uncached_expected"] for r in synthetic)
            account = dict(version=version, chain_id=chain["chain_id"], requests=len(rs),
                           original_requests=len(original), synthetic_requests=len(synthetic),
                           target_synthetic_new_tokens=target, actual_synthetic_new_tokens=actual,
                           synthetic_new_relative_error=actual / target - 1 if target > 0 else None,
                           source_prompt_sum=source["sum_glm_tokens"],
                           prompt_sum=sum(r["glm_tokens"] for r in rs),
                           source_output_sum=source["max_output_i_sum"],
                           output_sum=sum(r["max_output_i"] for r in rs))
            account["prompt_relative_error"] = account["prompt_sum"] / account["source_prompt_sum"] - 1
            accounts.append(account)
            recorded = summary_by_id.get(chain["chain_id"])
            if recorded:
                for field, observed in (("target_new_tokens", target), ("synthesized_new_tokens", actual),
                                        ("prompt_sum", account["prompt_sum"]), ("output_sum", account["output_sum"])):
                    if recorded.get(field) != observed:
                        summary_mismatches.append(dict(chain_id=chain["chain_id"], field=field))
        relevant = [a for a in accounts if a["synthetic_requests"] and a["synthetic_new_relative_error"] is not None]
        errors = [abs(a["synthetic_new_relative_error"]) for a in relevant]
        total = lambda key: sum(a[key] for a in accounts)
        internal_reset = [r for rs in by_chain.values()
                          for r in sorted(rs, key=lambda r: (r.get("dispatch_offset_ms") or 0, r["logical_call_id"]))[1:]
                          if r["phase"] == "context_reset"]
        check_path = root / "check.json"
        check = json.loads(check_path.read_text()) if check_path.exists() else {}
        result["datasets"][version] = dict(
            requests=len(requests), chains=len(chains),
            original_requests=sum(a["original_requests"] for a in accounts),
            synthetic_requests=sum(a["synthetic_requests"] for a in accounts),
            global_prompt_relative_error=total("prompt_sum") / total("source_prompt_sum") - 1,
            global_synthetic_new_relative_error=total("actual_synthetic_new_tokens") / total("target_synthetic_new_tokens") - 1,
            chains_with_synthetic_requests=len(relevant),
            per_chain_abs_synthetic_error_median=statistics.median(errors),
            per_chain_abs_synthetic_error_over={str(p): sum(e > p for e in errors) for p in (.1, .2, .5)},
            output_total_delta=total("output_sum") - total("source_output_sum"),
            output_conflict_chains=sum(a["output_sum"] != a["source_output_sum"] for a in accounts),
            stored_summary_mismatches=summary_mismatches,
            most_underfilled_chains=sorted(relevant, key=lambda a: a["synthetic_new_relative_error"])[:5],
            internal_reset=dict(count=len(internal_reset), synthetic=sum(r.get("split") == "synthetic" for r in internal_reset),
                edges=dict(collections.Counter(r["edge_type"] for r in internal_reset)),
                expected_new_p50=q([r["uncached_expected"] for r in internal_reset], .5),
                expected_new_p95=q([r["uncached_expected"] for r in internal_reset], .95)),
            guard_required_manifest_fields_missing=[f for f in ("set", "artifacts", "max_context_tokens") if f not in manifest],
            stored_check=dict(status=check.get("status"), complete_pass=check.get("complete_pass"),
                              rendered_prompts=check.get("counts", {}).get("rendered_prompts")))
        exported.extend(accounts)

    before, after = ([c["chain_id"] for c in cohorts[v]] for v in ("v3", "v4"))
    result["v3_to_v4_cohort"] = dict(same_chain_set=set(before) == set(after),
        equal_positions=sum(a == b for a, b in zip(before, after)),
        common_prefix_members={str(n): len(set(before[:n]) & set(after[:n])) for n in (26, 34, 38)})
    result["probes"] = [cli_probe(repo, tool) for tool in ("rebudget", "regap")]
    spec = importlib.util.spec_from_file_location("audited_longchain_replay", repo / "scripts/longchain/longchain_replay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module.validate(repo / "cache/s1-dev-longchain-v5", repo / "s1-dev/glm_tok")
    except ValueError as error:
        result["v5_replay_guard_probe"] = dict(accepted=False, exception=type(error).__name__, message=str(error))
    else:
        result["v5_replay_guard_probe"] = dict(note="producer changed; inspect returned validation before claiming acceptance")
    dump(out / "analysis.json", result)
    with (out / "per_chain.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    print(json.dumps(dict(datasets=len(result["datasets"]), chain_accounts=len(exported),
                          probes=result["probes"], replay_guard=result["v5_replay_guard_probe"])))


if __name__ == "__main__":
    main()
