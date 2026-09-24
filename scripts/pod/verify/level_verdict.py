#!/usr/bin/env python3
"""Fail-closed verdict for ONE complete dev level; CPU only, no engine requests.

Select measurement files by summary.json, validate cohort/N/runner and a fresh
pre-measurement flush, then call the original harness via score_formal.
Exit 0=VALID/PASS, 1=VALID/FAIL, 2=INVALID. Alternative CI methods are diagnostics.
New runs use run_dev_checked.py receipts. Legacy runs need run_dev.log plus a
timestamped server flush after their runner invocation and before measurement.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent


def score_module():
    for source in (HERE / "score_formal.py", HERE.parents[1] / "score_formal.py"):
        if source.is_file():
            spec = importlib.util.spec_from_file_location("level_score_formal", source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise ValueError("score_formal.py missing from kit/repository")


def selected_files(out_dir, n):
    summary = json.loads((out_dir / "summary.json").read_text())
    if not isinstance(summary, dict) or summary.get("n") != n:
        raise ValueError("summary N mismatch/missing")
    paths = []
    for key in ("raw", "run"):
        value = summary.get(key)
        if not isinstance(value, str) or not value or value.endswith("/"):
            raise ValueError(f"summary {key} missing")
        name = Path(value).name
        if name in (".", "..", ""):
            raise ValueError(f"invalid summary {key} filename")
        path = out_dir / name
        if not path.is_file():
            raise ValueError(f"measured file missing: {name}")
        paths.append(path)
    metadata = json.loads(paths[1].read_text())
    if not isinstance(metadata, dict):
        raise ValueError("run metadata is not an object")
    if metadata.get("config", {}).get("N") != n:
        raise ValueError("run metadata N mismatch/missing")
    if not str(metadata.get("config", {}).get("instance_id", "")).endswith("-measure"):
        raise ValueError("selected run is not a measurement replay")
    if metadata.get("raw_file") not in (None, paths[0].name):
        raise ValueError("run metadata points at a different raw")
    return paths[0], paths[1], summary, metadata


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_flush(out_dir, raw, run, metadata, rows, n, server_log=None):
    log = out_dir / "run_dev.log"
    text = log.read_text()
    if "flushed KV via" not in text or re.search(r"flush failed:|FLUSH_FAILED", text, re.I):
        raise ValueError("flush missing/failed in run_dev.log")
    starts = [r.get("client_dispatch_at_s") for r in rows]
    if not starts or not all(finite(x) for x in starts):
        raise ValueError("cannot bound flush: missing measurement dispatch timestamps")
    first = min(starts)
    receipt_path = out_dir / "flush_evidence.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("flush_success") is not True or receipt.get("runner_rc") != 0:
            raise ValueError("flush/runner receipt not successful")
        if receipt.get("n") != n or receipt.get("raw") != raw.name or receipt.get("run") != run.name:
            raise ValueError("flush receipt is for another measurement")
        start, finish, invocation = (receipt.get(k) for k in
                                      ("flush_started_s", "flush_finished_s", "runner_started_s"))
        if not all(finite(t) for t in (start, finish, invocation)) or not invocation <= start <= finish <= first:
            raise ValueError("flush receipt timestamps stale/out of order")
        mode = "checked_runner_receipt_and_server_log"
    else:
        # Original run_dev.py tag records invocation time before preflight/warmup.
        # It is not inferred from an old server startup log or newest-file mtime.
        instance = metadata.get("config", {}).get("instance_id", "")
        match = re.fullmatch(r".*-(\d{10})-measure", instance)
        if not match:
            raise ValueError("no flush receipt or bounded legacy invocation timestamp")
        start, finish = float(match[1]), first
        if start > finish:
            raise ValueError("legacy invocation follows measurement")
        mode = "legacy_run_log_and_bounded_server_log"
    source = Path(server_log) if server_log else out_dir / "server.log"
    if not source.is_file() and server_log is None:
        source = out_dir.parent / "server.log"
    if not source.is_file() or source.name == "engine_current.log":
        raise ValueError("live/full server.log missing; startup snapshot cannot verify flush")
    matches = []
    for line in source.read_text(errors="replace").splitlines():
        if "Cache flushed successfully!" not in line:
            continue
        stamp = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", line)
        if stamp:
            ts = datetime.strptime(stamp[1], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
            # Server text timestamps have one-second precision.
            if math.floor(start) <= ts <= math.floor(finish):
                matches.append(ts)
    if not matches:
        raise ValueError("no successful server flush in this measurement's pre-flush window")
    return {"verified": True, "mode": mode, "server_log": str(source),
            "window_start_s": start, "window_end_s": finish,
            "matched_server_flush_s": max(matches), "server_log_clock": "UTC, second precision"}


def fail_invalid(out_dir, n, reasons):
    out_dir.mkdir(parents=True, exist_ok=True)
    verdict = {"status": "INVALID", "n": n, "passed": False, "reasons": reasons}
    (out_dir / "level_verdict.json").write_text(json.dumps(verdict, indent=1) + "\n")
    # A failed recheck must not leave a stale valid-looking derived score.
    (out_dir / "score_formal.json").unlink(missing_ok=True)
    print(f"LEVEL N={n} status=INVALID reasons={'; '.join(reasons)}")
    return 2


def evaluate(out_dir, n, harness_dir, data_root, requests=None, rundev_rc=None, server_log=None):
    reasons = []
    try:
        saved_rc = out_dir / "rundev_exit_code"
        if saved_rc.is_file():
            recorded_rc = int(saved_rc.read_text().strip())
            if rundev_rc is not None and recorded_rc != rundev_rc:
                reasons.append("runner exit-code evidence disagrees")
            rundev_rc = recorded_rc
        if rundev_rc not in (None, 0):
            reasons.append(f"run_dev exit code {rundev_rc}")
        raw, run, summary, metadata = selected_files(out_dir, n)
        if summary.get("score_rc") != 0:
            reasons.append("harness score execution failed/missing")
        sys.path.insert(0, str(harness_dir))
        from s1_common import load_index
        index, _, _ = load_index(str(data_root))
        rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
        if any(not isinstance(row, dict) or not isinstance(row.get("req_id"), str) for row in rows):
            raise ValueError("raw row missing req_id/object")
        counts = Counter(r["req_id"] for r in rows)
        missing, extra = set(index) - set(counts), set(counts) - set(index)
        duplicate = [rid for rid, count in counts.items() if count > 1]
        wrong = [r["req_id"] for r in rows if r["req_id"] in index and
                 any(r.get(key) != index[r["req_id"]].get(source) for key, source in
                     (("idx_in_chain", "_idx_in_chain"), ("phase", "phase"),
                      ("edge_type", "edge_type"), ("uncached_expected", "uncached_expected")))]
        for label, values in (("missing", missing), ("extra", extra), ("duplicate", duplicate), ("frozen metadata mismatch", wrong)):
            if values:
                reasons.append(f"{label} req_id x{len(values)}")
        if reasons:
            return fail_invalid(out_dir, n, reasons)
        flush = check_flush(out_dir, raw, run, metadata, rows, n, server_log)
        scorer = score_module()
        report = scorer.score_files(raw, run, harness_dir, requests or data_root / "requests.jsonl")
        est, tpot, ttft = report["estimated"], report["tpot"], report["ttft_estimated"]
        gates = {}
        for name, detail in ttft.items():
            if isinstance(detail, dict) and "over_limit" in detail:
                gates[name.split("(")[0]] = {k: detail[k] for k in
                    ("n", "p95", "over_limit", "allowed_over", "rate_ci_lower",
                     "pass_estimated", "pass_point", "interval_sensitivity")}
        failed = [name for name, passed in est["gates"].items() if not passed]
        verdict = {
            "status": "VALID", "n": n, "passed": bool(est["passed"]), "failed_gates": failed,
            "tpot_mean": tpot["tpot_mean"], "tpot_p95": tpot["tpot_p95"], "ttft_gates": gates,
            "rows": len(rows), "raw": raw.name, "run": run.name, "wall_s": report["dev"].get("wall_s"),
            "harness_point_all_pass": summary.get("ALL_PASS"), "flush": flush,
            "interval_sensitivity": report["interval_sensitivity"],
        }
        (out_dir / "level_verdict.json").write_text(json.dumps(verdict, indent=1, allow_nan=False) + "\n")
        (out_dir / "score_formal.json").write_text(json.dumps(report, indent=1, default=str, allow_nan=False) + "\n")
        fmt = lambda value: "n/a" if value is None else f"{value:.4f}"
        detail = " ".join(f"{key}:{fmt(value['p95'])}({value['over_limit']}/{value['allowed_over']})"
                          for key, value in gates.items())
        print(f"LEVEL N={n} status=VALID formal_est={'PASS' if est['passed'] else 'FAIL'} "
              f"tpot_mean={fmt(tpot['tpot_mean'])} tpot_p95={fmt(tpot['tpot_p95'])} | {detail} | failed={failed}")
        print("CI_DIAGNOSTIC primary=clopper_pearson organizer_method=unknown " +
              json.dumps(report["interval_sensitivity"], ensure_ascii=False))
        return 0 if est["passed"] else 1
    except (ValueError, OSError, KeyError, TypeError) as error:
        return fail_invalid(out_dir, n, reasons + [str(error)])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("n", type=int)
    parser.add_argument("--harness-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--rundev-rc", type=int)
    parser.add_argument("--server-log", type=Path, help="full serving log; UTC timestamps")
    args = parser.parse_args()
    return evaluate(args.out_dir, args.n, args.harness_dir.resolve(), args.data_root.resolve(),
                    args.requests, args.rundev_rc, args.server_log)


if __name__ == "__main__":
    sys.exit(main())
