#!/usr/bin/env python3
"""Read closed diagnostic runs and demonstrate deadline changes with real policy code.

Uses the original harness selectors/quantiles through dcp_run_audit. This is
analysis, not a full-cohort verdict or a simulation of a different scheduler.
"""
import argparse
import datetime as dt
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import dcp_run_audit as audit


def review_run(directory, harness):
    raw_paths = list(directory.glob("raw_*.jsonl"))
    run_paths = list(directory.glob("run_*.json"))
    if len(raw_paths) != 1 or len(run_paths) != 1:
        raise ValueError(f"Expected one raw and run receipt in {directory}")
    rows = audit.records(raw_paths[0])
    run = json.loads(run_paths[0].read_text())
    assert len(rows) == len({r["req_id"] for r in rows})
    assert len(rows) == run["dispatched"] == run["n_attempted"]
    assert f"TIMED_DIAGNOSTIC DRAINED {len(rows)} requests" in (directory / "job.log").read_text()
    t0 = min(r["client_dispatch_at_s"] for r in rows)
    stats = audit.gates(rows, harness)
    summary = json.loads((directory / "summary.json").read_text())
    for _, selector, _ in harness.TTFT_GATE_SPECS:
        expected = [v for k, v in summary["ttft_p95_by_gate"].items() if k.startswith(selector)]
        assert len(expected) == 1 and abs(expected[0] - stats["ttft"][selector]["p95_s"]) < 1e-8
    details = [audit.components(r, t0, harness) for r in rows if not r.get("error")]
    chain = [r for r in details if "chain_start" in r["gates"]]
    bad = [r for r in chain if r["ttft_s"] > 30]
    stamp = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0\] (.*)")
    lines = (directory / "server.log").read_text().splitlines()
    opening_policy = []
    for number, line in enumerate(lines, 1):
        m = stamp.match(line)
        if m is None or "[ax-125]" not in line:
            continue
        timestamp = dt.datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()
        if 0 <= timestamp - t0 <= 150:
            opening_policy.append(dict(line=number, relative_s=timestamp - t0, text=line))
    return dict(
        source=str(directory), scope="DRAINED diagnostic; not full cohort",
        integrity="Unique raw IDs; raw/run counts and drain receipt agree. No independent dispatch-ID ledger checked.",
        stats=stats, chain_misses=dict(
            total=len(bad), opening_arrivals_0_to_10s=sum(r["arrival_s"] <= 10 for r in bad),
            later_arrivals=sum(r["arrival_s"] > 10 for r in bad),
            first_selection_wait_over30=sum(r["dispatch_to_first_selection_s"] > 30 for r in bad),
            selection_to_first_token_over30=sum(r["selection_to_first_token_s"] > 30 for r in bad),
            window_allowed_over=audit.score_formal.allowed_over(len(chain)),
        ), server_lines_read=len(lines), opening_policy=opening_policy,
    ), chain


def counterexamples(source):
    spec = importlib.util.spec_from_file_location("_chain_review_policy", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cfg = module.DeadlineConfig(warm_budget_s=15)
    req = SimpleNamespace(rid="family_waiter", origin_input_ids=range(36000), output_ids=[], num_matched_prefix_tokens=0)
    other = SimpleNamespace(rid="other_cold", origin_input_ids=range(22000), output_ids=[], num_matched_prefix_tokens=0)
    changes = []
    for matched in (16000, 32000):
        req.num_matched_prefix_tokens = matched
        changes.append(dict(
            matched=matched, remaining=module.remaining_tokens(req), budget_s=module.budget_s(req, cfg),
            slack_s=module.slack_s(req, module.remaining_tokens(req), 20, 8192, cfg),
            order=[r.rid for r in module.tier_order([req, other], lambda _: 20, set(), 8192, cfg)],
        ))
    assert changes[0]["order"] == ["family_waiter", "other_cold"]
    assert changes[1]["order"] == ["other_cold", "family_waiter"]
    req.num_matched_prefix_tokens = 0
    held_order = [r.rid for r in module.tier_order(
        [req, other], lambda _: 1, {req.rid}, 8192, cfg, {req.rid: 7200})]
    assert held_order == ["other_cold", "family_waiter"]
    return dict(source=str(source), scope="Synthetic states through real functions; not observed run decisions",
                cache_growth=changes, family_leader_with_native_holdback=held_order)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    harness = audit.score_formal.load_harness(args.source_root / "s1-dev/harness")
    report = dict(runs={})
    chain_rows = []
    for name, relative in (
        ("130ee5", "evidence/L130ee5-dcp_stack2_w2_n34_60m/N34"),
        ("130eezy", "evidence/L130eezy-dcp_stack2_w2_n38_60m/N38"),
    ):
        result, chain = review_run(args.source_root / relative, harness)
        report["runs"][name] = result
        chain_rows.extend(dict(run=name, **r) for r in chain)
    own_root = Path(__file__).resolve().parents[2]
    report["counterexamples"] = counterexamples(own_root / "engine/sglang/srt/managers/ax_deadline.py")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    audit.csv_out(args.output / "chain.csv", chain_rows)
    print(json.dumps({k: v["chain_misses"] for k, v in report["runs"].items()}))


if __name__ == "__main__":
    main()
