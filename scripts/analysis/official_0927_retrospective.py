#!/usr/bin/env python3
"""Reproduce the September 27 official comparison and local diagnostic checks.

Reads existing evidence only; emits no full-cohort or formal performance verdict.
Uses the original harness for bucket membership and quantiles.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import sys


def read_json(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def single(paths):
    found = list(paths)
    assert len(found) == 1, found
    return found[0]


def command_flags(command):
    words = shlex.split(command)
    flags = {}
    for i, word in enumerate(words):
        if word.startswith("--"):
            flags[word] = words[i + 1] if i + 1 < len(words) and not words[i + 1].startswith("--") else True
    return flags


def delta(a, b):
    return {key: [a.get(key), b.get(key)] for key in sorted(a.keys() | b.keys()) if a.get(key) != b.get(key)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "s1-dev/harness"))
    from s1_common import TTFT_GATE_SPECS, in_ttft_gate, qs

    result = {"scope": "official returned levels plus local DRAINED diagnostics; no inferred failing-level verdict",
              "input_root": str(root), "official": {}, "configuration_deltas": {}, "runs": {}, "paired": {}}
    configs = {}
    for attempt, arm in ((46677, "s2"), (46757, "s3"), (46758, "s4")):
        path = root / f"evidence/official/attempt-{attempt}-20260927.json"
        doc = read_json(path)
        assert doc["id"] == attempt and doc["execStatus"] == "completed"
        assert doc["scoringState"]["scoreIsFinal"]
        card = doc["scorecard"]
        assert card["scorewheel_gate_passed"]
        stress = card["scorewheel_stress"]
        assert stress == doc["resultsJson"]["scorewheel_stress"]
        assert stress["passed"] and stress["n"] == stress["n_at_slo"]
        result["official"][attempt] = {"source": str(path.relative_to(root)),
            "updated_at": doc["updatedAt"], "capability": card["scorewheel_datasets"], "stress": stress}
        configs[attempt] = read_json(root / f"evidence/submission-0926-{arm}/submission.json")
    for a, b in ((46677, 46757), (46757, 46758)):
        result["configuration_deltas"][f"{a}->{b}"] = {
            "image": [configs[a]["image"], configs[b]["image"]],
            "flags": delta(command_flags(configs[a]["command"]), command_flags(configs[b]["command"])),
            "env": delta(configs[a]["env"], configs[b]["env"])}

    def stats(records):
        records = list(records)
        return {"requests": len(records),
            "gates": {selector: {"samples": sum(in_ttft_gate(r, selector) for r in records),
                                  "over": sum(in_ttft_gate(r, selector) and r["ttft_s"] > limit for r in records)}
                      for _, selector, limit in TTFT_GATE_SPECS},
            "tpot": qs([r["tpot_s"] for r in records]),
            "tpot_over_100ms": sum(r["tpot_s"] is not None and r["tpot_s"] > .1 for r in records)}

    data, origins = {}, {}
    for name in ("130ez1", "130ez5", "130ez6zzzz", "130ez9", "130ezf", "130ezh"):
        directory = single((root / "evidence").glob(f"L{name}-*")) / "N34"
        path = single(directory.glob("raw_*.jsonl"))
        raw = rows(path)
        by_id = {r["req_id"]: r for r in raw}
        assert len(by_id) == len(raw)
        assert all(not r.get("error") and not r.get("error_class") for r in raw)
        run = read_json(single(directory.glob("run_s1-dev*.json")))
        flush = read_json(directory / "flush_evidence.json")
        assert flush["flush_success"] and flush["runner_rc"] == 0
        assert run["n_attempted"] == run["dispatched"] == len(raw)
        assert run["config"]["N"] == 34
        assert flush["scope"] == "fixed_duration_diagnostic"
        origin = min(r["client_dispatch_at_s"] for r in raw)
        finish = max(r["client_finish_at_s"] for r in raw)
        transitions, counters = [], []
        for line_number, line in enumerate((directory / "server.log").open(), 1):
            match = re.match(r"\[([^]]{19}) TP0\]", line)
            if not match or "[ax-" not in line:
                continue
            epoch = datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
            if not origin <= epoch <= finish:
                continue
            record = {"elapsed_s": epoch - origin, "line": line_number, "text": line.rstrip()}
            if "[ax-125]" in line:
                transitions.append(record)
            count = re.search(r"parks=(\d+) relief_rounds=(\d+)", line)
            if count:
                counters.append(dict(record, parks=int(count[1]), relief_rounds=int(count[2])))
        assert counters and transitions
        final_count = counters[-1]["relief_rounds"]
        plateau = next(c for c in counters if c["relief_rounds"] == final_count)
        opening = [r for r in raw if r["client_dispatch_at_s"] - origin < 30 and in_ttft_gate(r, "chain_start")]
        result["runs"][name] = {"raw": str(path.relative_to(root)), "scope": flush["scope"],
            "cohort": run["config"]["cohort_sha256"], "workload_hash": run["config"]["workload_hash"],
            "origin_s": origin, "finish_s": finish, "stats": stats(raw),
            "opening_chain_n": len(opening), "opening_chain_over": sum(r["ttft_s"] > 30 for r in opening),
            "guard": {"transitions": transitions, "relief_counter_plateau_first_observed": plateau,
                      "last_counter": counters[-1], "trip_timestamp_observable": False}}
        data[name], origins[name] = by_id, origin
    assert len({r["cohort"] for r in result["runs"].values()}) == 1
    assert len({r["workload_hash"] for r in result["runs"].values()}) == 1

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "paired-tail.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["baseline", "candidate", "req_id", "gate", "change", "baseline_arrival_s", "candidate_arrival_s",
                         "baseline_ttft_s", "candidate_ttft_s", "baseline_tpot_s", "candidate_tpot_s",
                         "baseline_prompt", "candidate_prompt", "baseline_cached", "candidate_cached"])
        for base_name, candidate_name in (("130ez5", "130ez6zzzz"), ("130ez1", "130ez9"),
                                          ("130ez1", "130ezf"), ("130ezf", "130ezh")):
            base, candidate = data[base_name], data[candidate_name]
            common = sorted(base.keys() & candidate.keys())
            for rid in common:
                assert all(base[rid][key] == candidate[rid][key]
                           for key in ("phase", "idx_in_chain", "uncached_expected", "prompt_tokens", "output_tokens"))
            comparison = {"common": len(common), "baseline": stats(base[i] for i in common),
                          "candidate": stats(candidate[i] for i in common), "gates": {}}
            for _, selector, limit in TTFT_GATE_SPECS:
                fixes, regressions = [], []
                for rid in common:
                    a, b = base[rid], candidate[rid]
                    if not in_ttft_gate(a, selector) or (a["ttft_s"] > limit) == (b["ttft_s"] > limit):
                        continue
                    change = "fixed" if a["ttft_s"] > limit else "regressed"
                    (fixes if change == "fixed" else regressions).append(rid)
                    writer.writerow([base_name, candidate_name, rid, selector, change,
                        a["client_dispatch_at_s"]-origins[base_name], b["client_dispatch_at_s"]-origins[candidate_name],
                        a["ttft_s"], b["ttft_s"], a["tpot_s"], b["tpot_s"], a["prompt_tokens"], b["prompt_tokens"],
                        a["cached_tokens"], b["cached_tokens"]])
                comparison["gates"][selector] = {"fixed": fixes, "regressed": regressions}
            result["paired"][f"{base_name}->{candidate_name}"] = comparison

    manifest = read_json(root / "cache/s1-dev-longchain-v3/manifest.json")
    source_budget = sum(r["sum_uncached_expected"] for r in rows(root / "s1-dev/data/dev-combined-v1/chains.jsonl"))
    v3_budget = sum(r["uncached_expected"] for r in rows(root / "cache/s1-dev-longchain-v3/requests.jsonl"))
    result["workload_limits"] = {"scope": "metadata budgets, not measured GPU work or formal workload",
        "quality_status": manifest["quality_status"], "public_source_chain_budget": source_budget,
        "v3_annotation_budget": v3_budget, "budget_ratio": v3_budget/source_budget,
        "synthesized_new_tokens": manifest["synthesized_new_tokens"], "target_new_tokens": manifest["target_new_tokens"]}
    (args.out / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"official_N": {k: v["stress"]["n_at_slo"] for k, v in result["official"].items()},
                     "local_requests": {k: v["stats"]["requests"] for k, v in result["runs"].items()},
                     "comparison_common": {k: v["common"] for k, v in result["paired"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
