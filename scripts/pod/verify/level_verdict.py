#!/usr/bin/env python3
"""Fail-closed verdict for ONE dev level (T54, Claude 2026-09-24). CPU only; never talks to the engine.

  level_verdict.py OUT_DIR N --harness-dir H --data-root D [--requests R] [--rundev-rc RC]

1. Measured files come from OUT_DIR/summary.json written by run_dev.py (never "the newest raw").
2. The raw must contain every request of the dev index (s1_common.load_index, the harness's own loader) exactly once,
   with the same idx_in_chain, and nothing else. score_formal.py alone does not check this: a raw missing one request,
   or with a duplicate in its place, still scores as PASS (evidence/T54/level_verdict_tests.log).
3. Gates come only from scripts/score_formal.py: the harness's s1_score.evaluate, the task.md statistical allowance on
   the four TTFT gates, and tpot_p95 <= 0.10 (task.md:519, 565-567, 577).
Writes OUT_DIR/level_verdict.json and prints one LEVEL line.
Exit: 0 = VALID and all eleven gates pass; 1 = VALID but some gate fails; 2 = INVALID (the measurement is unusable).
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def fail_invalid(out_dir, n, reasons):
    verdict = {"status": "INVALID", "n": n, "reasons": reasons}
    (Path(out_dir) / "level_verdict.json").write_text(json.dumps(verdict, indent=1) + "\n")
    print(f"LEVEL N={n} status=INVALID reasons={'; '.join(reasons)}")
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("n", type=int)
    ap.add_argument("--harness-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--requests", type=Path, default=None, help="requests.jsonl (default: DATA_ROOT/requests.jsonl)")
    ap.add_argument("--rundev-rc", type=int, default=0)
    a = ap.parse_args()
    reasons = []
    if a.rundev_rc != 0:
        reasons.append(f"run_dev exit code {a.rundev_rc}")
    summary_path = a.out_dir / "summary.json"
    if not summary_path.is_file():
        fail_invalid(a.out_dir, a.n, reasons + ["no summary.json"])
    summary = json.loads(summary_path.read_text())
    raw, run = (a.out_dir / Path(summary.get(k) or "").name for k in ("raw", "run"))
    if not raw.is_file() or not run.is_file() or raw.name == "" or run.name == "":
        fail_invalid(a.out_dir, a.n, reasons + [f"measured files missing: raw={raw.name} run={run.name}"])
    if summary.get("n") not in (None, a.n):
        reasons.append(f"summary n={summary.get('n')} != {a.n}")

    # 2. completeness against the harness's own index
    sys.path.insert(0, str(a.harness_dir))
    from s1_common import load_index  # noqa: E402  (harness is read-only; import only)
    index, _, _ = load_index(str(a.data_root))
    expected = {rid: r["_idx_in_chain"] for rid, r in index.items()}
    rows = []
    for i, line in enumerate(raw.read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                reasons.append(f"raw line {i} is not JSON")
    counts = Counter(r.get("req_id") for r in rows)
    dup = [rid for rid, c in counts.items() if c > 1]
    missing = sorted(set(expected) - set(counts))
    extra = sorted(set(counts) - set(expected))
    wrong_idx = [r["req_id"] for r in rows if r.get("req_id") in expected and r.get("idx_in_chain") != expected[r["req_id"]]]
    for label, items in (("duplicate req_id", dup), ("missing req_id", missing), ("unknown req_id", extra),
                         ("idx_in_chain mismatch", wrong_idx)):
        if items:
            reasons.append(f"{label} x{len(items)} (e.g. {items[0]})")
    if reasons:
        fail_invalid(a.out_dir, a.n, reasons)

    # 3. score with the harness-based formal estimate
    sys.path.insert(0, str(HERE))
    import score_formal  # noqa: E402
    report = score_formal.score_files(raw, run, a.harness_dir, a.requests or (a.data_root / "requests.jsonl"))
    est, tpot, ttft = report["estimated"], report["tpot"], report["ttft_estimated"]
    gates = {}
    for name, g in ttft.items():
        if isinstance(g, dict) and "over_limit" in g:
            gates[name.split("(")[0]] = {k: g[k] for k in ("n", "p95", "over_limit", "allowed_over", "rate_ci_lower",
                                                           "pass_estimated", "pass_point")}
    failed = [k for k, ok in est["gates"].items() if not ok]
    verdict = {"status": "VALID", "n": a.n, "passed": bool(est["passed"]), "failed_gates": failed,
               "tpot_mean": tpot["tpot_mean"], "tpot_p95": tpot["tpot_p95"], "ttft_gates": gates,
               "rows": len(rows), "raw": raw.name, "run": run.name,
               "wall_s": report["dev"].get("wall_s"), "harness_point_all_pass": summary.get("ALL_PASS")}
    (a.out_dir / "level_verdict.json").write_text(json.dumps(verdict, indent=1) + "\n")
    (a.out_dir / "score_formal.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    g = " ".join(f"{k}:{v['p95']:.2f}({v['over_limit']}/{v['allowed_over']})" for k, v in gates.items())
    print(f"LEVEL N={a.n} status=VALID formal_est={'PASS' if est['passed'] else 'FAIL'} "
          f"tpot_mean={tpot['tpot_mean']:.4f} tpot_p95={tpot['tpot_p95']:.4f} | {g} | failed={failed}")
    sys.exit(0 if est["passed"] else 1)


if __name__ == "__main__":
    main()
