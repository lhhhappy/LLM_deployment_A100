#!/usr/bin/env python3
"""Compare replay datasets by the load the engine sees, from frozen metadata only.

For each root: requests, chains, prompt/new/output token totals, and per harness TTFT gate
(membership from s1_common.in_ttft_gate, i.e. the scorer's own rule) the request share and the
distribution of frozen new tokens (uncached_expected). Generated roots whose chains.jsonl carries
source_chain_targets also report per-chain new-token and prompt totals against the source.
No engine, rendering or GPU. Usage:
  python3 scripts/longchain/workload_compare.py --root dev=s1-dev/data/dev-combined-v1 \
      --root v3=data/s1-dev-longchain-v3 [--out-json report.json]
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GATES = ("chain_start", "turn_start", "fast_intra", "overall_intra")


def quantiles(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    pick = lambda q: values[min(len(values) - 1, int(q * (len(values) - 1)))]
    return {"n": len(values), "mean": round(sum(values) / len(values)),
            "p50": pick(.5), "p90": pick(.9), "p99": pick(.99), "max": values[-1]}


def ratio_quantiles(pairs):
    """Distribution of generated/source over chains with a positive source value."""
    ratios = sorted(g / s for g, s in pairs if s > 0)
    if not ratios:
        return {"n": 0}
    pick = lambda q: round(ratios[min(len(ratios) - 1, int(q * (len(ratios) - 1)))], 3)
    return {"n": len(ratios), "p10": pick(.1), "p50": pick(.5), "p90": pick(.9),
            "total_ratio": round(sum(g for g, _ in pairs) / max(1, sum(s for _, s in pairs)), 3)}


def describe(root, common):
    rows, chains, by_chain = common.load_index(str(root))
    rewrites = {cid: sum(r.get("edge_type") != "append-only" for r in rs[1:]) for cid, rs in by_chain.items()}
    out = {"requests": len(rows), "chains": len(by_chain), "rewrite_edges": sum(rewrites.values()),
           "requests_per_chain": round(len(rows) / max(1, len(by_chain)), 2),
           "prompt_tokens": sum(r["glm_tokens"] for r in rows.values()),
           "new_tokens": sum(r["uncached_expected"] for r in rows.values()),
           "output_tokens": sum(r.get("max_output_i") or 0 for r in rows.values()), "gates": {}}
    for gate in GATES:
        members = [r for r in rows.values() if common.in_ttft_gate(r, gate)]
        out["gates"][gate] = {"share": round(len(members) / max(1, len(rows)), 3),
                              "new_tokens": quantiles([r["uncached_expected"] for r in members])}
    # Chain heads versus later requests of each phase: heads carry the whole visible history.
    out["later_by_phase"] = {}
    for phase in ("intra", "turn_start", "context_reset"):
        out["later_by_phase"][phase] = quantiles([r["uncached_expected"] for r in rows.values()
                                                  if r["_idx_in_chain"] > 0 and r["phase"] == phase])
    targets = [(c, c.get("source_chain_targets")) for c in chains.values()
               if c["chain_id"] in by_chain and isinstance(c.get("source_chain_targets"), dict)]
    def source_rewrites(chain, target):
        # Chain metadata counts the head's incoming edge when the chain starts mid-session.
        head = by_chain[chain["chain_id"]][0]
        return (target["total_edges"] - target["append_only_edges"]
                - (target["total_edges"] == target["n_requests"] and head.get("edge_type") != "append-only"))
    if targets:
        out["vs_source_chains"] = {
            "rewrite_edges": ratio_quantiles([(rewrites[c["chain_id"]], source_rewrites(c, t)) for c, t in targets]),
            "new_tokens": ratio_quantiles([(c["sum_uncached_expected"], t["sum_uncached_expected"]) for c, t in targets]),
            "prompt_tokens": ratio_quantiles([(c["sum_glm_tokens"], t["sum_glm_tokens"]) for c, t in targets])}
    return out


def markdown(report):
    names = list(report)
    lines = ["| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    def row(label, fn):
        lines.append(f"| {label} | " + " | ".join(fn(report[n]) for n in names) + " |")
    row("requests / chains", lambda r: f"{r['requests']} / {r['chains']}")
    row("prompt tokens (M)", lambda r: f"{r['prompt_tokens'] / 1e6:.1f}")
    row("new tokens (M)", lambda r: f"{r['new_tokens'] / 1e6:.2f}")
    row("history rewrites after the head", lambda r: str(r["rewrite_edges"]))
    for gate in GATES:
        row(f"{gate} share; new p50/p90/mean", lambda r, g=gate: "{:.1%}; {}/{}/{}".format(
            r["gates"][g]["share"], *(r["gates"][g]["new_tokens"].get(k, "-") for k in ("p50", "p90", "mean"))))
    for phase in ("intra", "turn_start", "context_reset"):
        row(f"later {phase} n; p50/p90/mean", lambda r, p=phase: "{}; {}/{}/{}".format(
            *(r["later_by_phase"][p].get(k, "-") for k in ("n", "p50", "p90", "mean"))))
    for key in ("new_tokens", "prompt_tokens", "rewrite_edges"):
        row(f"chain {key} vs source (p10/p50/p90; total)", lambda r, k=key: "-" if "vs_source_chains" not in r else
            "{}/{}/{}; {}".format(*(r["vs_source_chains"][k].get(x, "-") for x in ("p10", "p50", "p90", "total_ratio"))))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", required=True, help="label=path of a dataset root")
    ap.add_argument("--harness-dir", type=Path, default=REPO / "s1-dev/harness")
    ap.add_argument("--out-json", type=Path)
    args = ap.parse_args(argv)
    sys.path.insert(0, str(args.harness_dir))
    import s1_common as common
    report = {}
    for spec in args.root:
        label, _, path = spec.partition("=")
        if not path:
            ap.error(f"--root {spec!r} must be label=path")
        report[label] = describe(Path(path), common)
    if args.out_json:
        args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
