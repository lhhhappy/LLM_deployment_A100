#!/usr/bin/env python3
"""Build INTERNAL real-prefix A/B cases, never an official cohort replacement.

CPU/stdlib only: python3 -B scripts/make_case_sets.py --out-dir cases
Official comparisons MUST use the unmodified dev cohort through run_dev.py.
The shipped requests are already prefixes: chains.jsonl describes longer source
chains whose missing bodies must not be invented. Whole = whole AVAILABLE prefix.
Population band cutoffs are not published except p95. Other cuts are explicitly
sampling-weighted DEV proxies; target band shares are 50/30/10/5/4/1 percent.

Prediction approximation ports F24's aligned chunk/branch/end/conservative rules.
It uses frozen adjacent LCPs and a CURRENT-prompt role guess (length - 340 tokens,
F3 median reminder tail), NEVER the next prompt's LCP to place a checkpoint.
No tokenization, old-branch resurrection, decoding, eviction or timing model.
Role locations are NOT frozen in requests.jsonl: these are approximate tokens,
not a reproduction of F24's tokenizer results. replay_chains.py recomputes exact
token/path predictions when rendering these cases in the ana environment.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
COHORT = REPO / "s1-dev/harness/g0a/samples_v3/cohort_dev-combined-v1.json"
NOTICE = "INTERNAL A/B mechanism cases only; official comparisons require unmodified dev cohort via run_dev.py."
GATES = ("chain_start", "intra", "turn_start")


def load_data(dev_root=REPO / "s1-dev"):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(dev_root / "harness"))
    from s1_common import load_index
    return load_index(str(dev_root / "data/dev-combined-v1"))


def gate(row):
    from s1_common import phase_gate, P95_CHAIN_START_S, P95_INTRA_S, P95_TURN_START_S
    return {P95_CHAIN_START_S: "chain_start", P95_INTRA_S: "intra",
            P95_TURN_START_S: "turn_start"}[phase_gate(row)]


def quantile(values, p=.95):
    values = sorted(values)
    return values[min(len(values)-1, int(len(values)*p))] if values else None


def band_cuts(rows, cohort):
    cuts = {}
    for g in GATES:
        pairs = sorted((r["uncached_expected"], r.get("sampling_weight") or 0)
                       for r in rows.values() if gate(r) == g)
        total = sum(w for _, w in pairs)
        def wq(p):
            cumulative = 0
            for value, weight in pairs:
                cumulative += weight
                if cumulative > p * total:
                    return value
            return pairs[-1][0]
        # p95 is the one exact population cutoff published in the cohort.
        full = cohort["gate_p95_alignment"][g]["full"]
        cuts[g] = [min(wq(p), full) for p in (.5, .8, .9)] + [full, max(full, wq(.99))]
    return cuts


def composition(selected, cuts):
    counts = Counter(gate(r) for r in selected)
    result = {"n_requests": len(selected), "gates": {},
              "pack": dict(Counter(r["pack"] for r in selected)),
              "split": dict(Counter(r["split"] for r in selected)),
              "edge_subtype": dict(Counter(r.get("edge_subtype") or "none" for r in selected)),
              "sum_output_budget": sum(r["max_output_i"] for r in selected),
              "sum_frozen_uncached": sum(r["uncached_expected"] for r in selected)}
    for g in GATES:
        values = [r["uncached_expected"] for r in selected if gate(r) == g]
        bands = Counter(bisect_left(cuts[g], v) for v in values)
        result["gates"][g] = {"n": counts[g], "share": counts[g]/len(selected) if selected else 0,
            "uncached_p95": quantile(values), "band_counts": [bands[i] for i in range(6)],
            "band_shares": [bands[i]/len(values) if values else 0 for i in range(6)]}
    intra = [r for r in selected if gate(r) == "intra"]
    result["fast_intra_share"] = (sum(r["uncached_expected"] <= 4096 for r in intra)/len(intra)
                                  if intra else None)
    return result


def predict_frozen(rows, policy, tail=340, page=64, chunk=8192):
    """Adjacent-path approximation; current-only fixed tail role heuristic."""
    checkpoints, result = {0}, []
    floor = lambda x: x // page * page
    for i, row in enumerate(rows):
        length = row["glm_tokens"]
        common = min(row.get("glm_lcp_with_prev") or 0, length, rows[i-1]["glm_tokens"]) if i else 0
        checkpoints = {d for d in checkpoints if d <= common}
        hit = max(checkpoints, default=0)
        end = floor(length)
        branch = floor(common) if floor(common) > hit else None
        role = floor(max(0, length - tail))
        role = role if hit < role < end else None
        added = list(range(hit + chunk, length, chunk))
        if policy == "role_conservative" and role is not None and branch is None:
            added += [role, end]
        else:
            added += [branch if branch is not None else end]
        checkpoints.update(added)
        result.append({"cached_tokens": hit, "uncached_tokens": length - hit,
                       "branch_conflict": branch is not None, "role_guess": role})
    return result


def formal_select(groups, metadata, cuts, cohort, target_size):
    """Deterministic greedy prefix search; never repeats or stitches chains."""
    reference = cohort["composition_alignment_measured"]["formal_set_reference"]
    total = sum(reference[g]["n"] for g in GATES)
    target = {g: reference[g]["n"]/total for g in GATES}
    candidates = [(cid, rows[:k]) for cid, rows in groups.items()
                  for k in range(1, len(rows)+1)]
    chosen, used, selected = [], set(), []
    def loss(rows):
        c = composition(rows, cuts)
        phase = sum(abs(c["gates"][g]["share"]-target[g]) for g in GATES)
        bands = sum(sum(abs(a-b) for a, b in zip(c["gates"][g]["band_shares"],
                                (.5, .3, .1, .05, .04, .01))) for g in GATES)/3
        fast = abs((c["fast_intra_share"] or 0) - .852)
        tails = sum(abs(math.log(max(1, c["gates"][g]["uncached_p95"] or 1)/
                                cohort["gate_p95_alignment"][g]["full"])) for g in GATES)/3
        hidden = sum(r["pack"] == "biomaster" and r["split"] == "hidden" for r in rows)/len(rows)
        return 12*phase + 3*bands + .4*fast + .8*tails + .05*(1-hidden)
    # Keep the rare third gate represented (about one request in a 120-row set).
    turn_candidates = [(cid, rs) for cid, rs in candidates
                       if sum(gate(r) == "turn_start" for r in rs) == 1]
    if turn_candidates:
        cid, rs = min(turn_candidates, key=lambda x: (loss(x[1]), -len(x[1]), x[0]))
        chosen.append((cid, rs)); selected.extend(rs); used.add(cid)
    while len(selected) < target_size:
        available = [(cid, rs) for cid, rs in candidates if cid not in used and
                     sum(gate(r) == "turn_start" for r in selected+rs) <=
                     max(1, round(target_size*target["turn_start"]))]
        if not available:
            break
        cid, rs = min(available, key=lambda x: (
            loss(selected+x[1]),
            -(x[1][0]["pack"] == "biomaster" and x[1][0]["split"] == "hidden"),
            -len(x[1]), -metadata[x[0]].get("n_requests", 0), x[0]))
        chosen.append((cid, rs)); selected.extend(rs); used.add(cid)
    return chosen, target


def validate_case(case, groups):
    if not isinstance(case, list) or not case:
        raise ValueError("case file must be a nonempty ordered chain list")
    seen = set()
    for chain in case:
        cid, ids = chain["chain_id"], chain["req_ids"]
        if cid in seen or cid not in groups:
            raise ValueError("duplicate or unknown case chain")
        seen.add(cid)
        available = [r["_req_id"] for r in groups[cid]]
        if not ids or ids != available[:len(ids)] or chain.get("prefix_len", len(ids)) != len(ids):
            raise ValueError("case must be a contiguous ordered prefix from the available chain head")
    return case


def build(dev_root, cohort_file, out_dir, formal_size=120, heavy_chains=12):
    rows, metadata, groups = load_data(dev_root)
    cohort = json.loads(cohort_file.read_text())
    cuts = band_cuts(rows, cohort)
    formal, target = formal_select(groups, metadata, cuts, cohort, formal_size)
    def edge_count(rs, subtype):
        return sum(gate(r) == "intra" and r.get("edge_type") == "append-only"
                   and r.get("edge_subtype") == subtype for r in rs[1:])
    def heavy(subtype):
        candidates = [(cid, rs) for cid, rs in groups.items() if edge_count(rs, subtype)]
        return sorted(candidates, key=lambda x: (-edge_count(x[1], subtype)/(len(x[1])-1),
                      -edge_count(x[1], subtype), -len(x[1]), x[0]))[:heavy_chains]
    cold = sorted(groups.items(), key=lambda x: (
        -sum(gate(r) == "chain_start" for r in x[1])/len(x[1]),
        -sum(r["uncached_expected"] for r in x[1]), x[0]))[:heavy_chains]
    # Include reset-containing prefixes as well as large cold heads.
    resets = sorted(((cid, rs[:i+1]) for cid, rs in groups.items() for i, r in enumerate(rs)
                     if i > 0 and r["phase"] == "context_reset"),
                    key=lambda x: (-x[1][-1]["uncached_expected"], x[0]))
    cold_ids = {cid for cid, _ in cold}
    for cid, rs in resets:
        if cid not in cold_ids:
            cold.append((cid, rs)); cold_ids.add(cid)
    smoke = sorted(((cid, rs) for cid, rs in groups.items() if 2 <= len(rs) <= 3),
                   key=lambda x: (sum(r["glm_tokens"] for r in x[1]), x[0]))[:3]
    sets = dict(formal_like=formal, reminder_heavy=heavy("reminder-replaced"),
                strict_append=heavy("strict"), cold_heavy=cold, smoke=smoke)
    purposes = {"formal_like": "closest available gate/band mix; prefer long hidden biomaster prefixes (F31)",
                "reminder_heavy": "high append-only/reminder-replaced intra density and count: D1 target (F3)",
                "strict_append": "high append-only/strict intra density: preserve prompt-end checkpoints (F3)",
                "cold_heavy": "large frozen-uncached cold heads plus all available internal context resets",
                "smoke": "three complete available short chains with smallest total prompt tokens"}
    max_intra = max(sum(gate(r) == "intra" for r in rs[:k])/k
                    for rs in groups.values() for k in range(1, len(rs)+1))
    min_start = min(sum(gate(r) == "chain_start" for r in rs[:k])/k
                    for rs in groups.values() for k in range(1, len(rs)+1))
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name, chains in sets.items():
        case = [{"chain_id": cid, "prefix_len": len(rs), "req_ids": [r["_req_id"] for r in rs]}
                for cid, rs in chains]
        validate_case(case, groups)
        selected = [r for _, rs in chains for r in rs]
        stats = composition(selected, cuts)
        predictions = []
        for cid, rs in chains:
            policies = {p: predict_frozen(rs, p) for p in ("stock", "role_conservative")}
            for i, row in enumerate(rs):
                predictions.append({"req_id": row["_req_id"], "chain_id": cid, "idx_in_chain": i,
                    "gate": gate(row), "uncached_expected": row["uncached_expected"],
                    "glm_tokens": row["glm_tokens"], "prediction_kind": "frozen_fields_approximation",
                    "stock": policies["stock"][i]["uncached_tokens"],
                    "role_conservative": policies["role_conservative"][i]["uncached_tokens"],
                    "policy_details": {p: policies[p][i] for p in policies}})
        manifest = {"schema_version": 1, "purpose": NOTICE, "selection_reason": purposes[name],
            "source": {"requests_sha256": hashlib.sha256((dev_root/"data/dev-combined-v1/requests.jsonl").read_bytes()).hexdigest(),
                       "cohort_sha256": cohort["cohort_sha256"], "n_available_requests": len(rows)},
            "composition": stats, "formal_target_gate_shares": target,
            "formal_target_difference_pp": {g: 100*(stats["gates"][g]["share"]-target[g]) for g in GATES},
            "band_cutoffs": cuts, "band_target_shares": [.5, .3, .1, .05, .04, .01],
            "band_alignment_proxy": {g: {
                "n": stats["gates"][g]["n"], "underpowered": stats["gates"][g]["n"] < 20,
                "max_share_difference_pp": 100*max(abs(a-b) for a, b in
                    zip(stats["gates"][g]["band_shares"], (.5, .3, .1, .05, .04, .01))),
                "tvd_pct": 50*sum(abs(a-b) for a, b in
                    zip(stats["gates"][g]["band_shares"], (.5, .3, .1, .05, .04, .01)))} for g in GATES},
            "band_basis": "p95 = exact cohort gate_p95_alignment.full; p50/p80/p90/p99 = weighted dev proxies (clamped around p95); population cutoffs unavailable",
            "alignment_source": {"gate_p95_alignment": cohort["gate_p95_alignment"],
                "composition_alignment_measured": cohort["composition_alignment_measured"]},
            "availability_limits": {"max_intra_share_any_available_prefix": max_intra,
                "min_chain_start_share_any_available_prefix": min_start,
                "formal_mix_attainable": max_intra >= target["intra"] and min_start <= target["chain_start"],
                "note": "722 shipped requests are source-chain prefixes; longer chains.jsonl counts do not supply additional requests. No duplicate/stitch/synthetic requests."},
            "prediction": {"kind": "frozen_fields_approximation", "role_tail_guess_tokens": 340,
                "page_size": 64, "chunk_size": 8192,
                "logic_source": "scripts/sim_role_boundary.py F24; approximate role/path inputs",
                "limitations": "current length-340 role guess; adjacent LCP ancestry only; no token path equality, old-branch resurrection, decode, eviction, timing; not F24 reproduction",
                "exact_alternative": "scripts/replay_chains.py --case-file (Renderer/tokenizer required)"},
            "chains": [{"chain_id": cid, "selected_requests": len(rs), "available_requests": len(groups[cid]),
                "source_metadata_requests": metadata[cid]["n_requests"],
                "selection": "whole_available_prefix" if len(rs) == len(groups[cid]) else "prefix_of_available_prefix",
                "reason": purposes[name], "pack": rs[0]["pack"], "split": rs[0]["split"],
                "reminder_intra_edges": edge_count(rs, "reminder-replaced"),
                "strict_intra_edges": edge_count(rs, "strict"),
                "composition": composition(rs, cuts)} for cid, rs in chains]}
        for suffix, value in ((".json", case), (".manifest.json", manifest)):
            (out_dir/(name+suffix)).write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n")
        (out_dir/(name+".predictions.jsonl")).write_text("".join(json.dumps(r)+"\n" for r in predictions))
        summaries[name] = {"chains": len(chains), **stats}
    (out_dir/"README.md").write_text(
        "# Internal real-chain cases\n\n"+NOTICE+"\n\n"
        "Each `<name>.json` is an ordered list of chain prefixes with ordered `req_ids`. "
        "Manifests give chain reasons, source hashes, composition and availability limits. "
        "Prediction JSONL contains approximate stock/role_conservative uncached tokens per request.\n\n"
        "The supplied data cannot reach the formal 91% intra mix: see the formal_like manifest. "
        "Band cutoffs other than population p95 are weighted-dev proxies. Predictions use a fixed "
        "340-token tail guess, not observed role boundaries; rerun replay_chains with the tokenizer "
        "for exact F24-model predictions. These files are not official scores or new sampling weights.\n\n"
        "Regenerate: `python3 -B scripts/make_case_sets.py`. "
        "Functional replay: `python3 -B scripts/replay_chains.py --dev-root s1-dev "
        "--case-file cases/smoke.json --output /path/new-run --plan-only`. "
        "Long cases may require `--max-prompt-tokens 262144`; prompts are never truncated.\n")
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dev-root", type=Path, default=REPO/"s1-dev")
    p.add_argument("--cohort", type=Path, default=COHORT)
    p.add_argument("--out-dir", type=Path, default=REPO/"cases")
    p.add_argument("--formal-requests", type=int, default=120)
    p.add_argument("--heavy-chains", type=int, default=12)
    a = p.parse_args()
    if min(a.formal_requests, a.heavy_chains) < 1:
        p.error("sizes must be positive")
    if (a.dev_root.resolve() == a.out_dir.resolve() or a.dev_root.resolve() in a.out_dir.resolve().parents):
        p.error("outputs must be outside read-only s1-dev")
    summaries = build(a.dev_root, a.cohort, a.out_dir, a.formal_requests, a.heavy_chains)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
