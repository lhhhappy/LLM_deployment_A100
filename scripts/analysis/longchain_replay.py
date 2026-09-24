#!/usr/bin/env python3
"""Optional guard around the unchanged S1 runner; no new replay/scoring policy.

CPU acceptance (no service calls):
  python -B scripts/analysis/longchain_replay.py --root DATA --out NEW_OUT --self-check
Replay after the same full token acceptance:
  python -B scripts/analysis/longchain_replay.py --root DATA --out NEW_OUT --n 6 --base-url URL

Every invocation needs a fresh output directory outside DATA. This prevents the
original runner's unbound body cache from mixing different frozen artifacts.
The original root/set/cohort schema and GPU queue remain unchanged. Exit 0 means
the tools completed, not that an estimated SLO or official ranking passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO))
from scripts.analysis.longchain_check import check_dataset

HARNESS = REPO / "s1-dev/harness"
RUNNER = REPO / "s1-dev/run_dev.py"
CHECKED_RUNNER = REPO / "scripts/pod/verify/run_dev_checked.py"
SCORER = REPO / "scripts/score_formal.py"


def validate(root, tok_dir):
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    name = manifest.get("set")
    if not manifest.get("generator") or not isinstance(name, str) or not name or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("requires a generated artifact with a simple manifest set name")
    report = check_dataset(str(root), str(HARNESS), str(tok_dir), str(root / "cohort.json"))
    counts = report.get("counts", {})
    expected = counts.get("expected_ids")
    accepted = (report.get("status") == "VALID" and not report.get("errors")
                and isinstance(expected, int) and not isinstance(expected, bool) and expected > 0
                and all(counts.get(k) == expected for k in
                        ("serving_request_ids", "matched_bodies", "rendered_prompts")))
    report["replay_guard"] = {
        "accepted": accepted, "root": str(root), "set": name,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "scope": "CPU data acceptance only; no performance or representativeness claim"}
    if (root / "manifest.json").read_bytes() != manifest_bytes:
        report["replay_guard"]["accepted"] = False
        report.setdefault("errors", []).append("manifest changed during validation")
    return name, report


def execute(args):
    root, out, tok_dir = args.root.resolve(), args.out.resolve(), args.tok_dir.resolve()
    if out == root or root in out.parents:
        raise ValueError("output must be outside the frozen dataset")
    if any(out == p or p in out.parents for p in
           (REPO / name for name in ("s1-dev", "llm-challenge-arena-v1", "build/base_exact", "refs"))):
        raise ValueError("output must not write to a protected source tree")
    if out.exists():
        raise ValueError("output already exists; choose a fresh directory to avoid stale body caches")
    if args.n <= 0:
        raise ValueError("n must be positive")
    if not args.self_check and not args.base_url.strip():
        raise ValueError("--base-url is required for replay")
    # Reserve a fresh directory; failed validation remains reviewable here.
    out.mkdir(parents=True, exist_ok=False)
    name, report = validate(root, tok_dir)
    (out / "dataset-validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if not report["replay_guard"]["accepted"]:
        print("REJECTED: dataset validation incomplete; see dataset-validation.json", file=sys.stderr)
        return 2
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", S1_HARNESS_DIR=str(HARNESS),
               S1_SKIP_PREFLIGHT="0", S1_SKIP_WARMUP="0")
    env.pop("S1_FLUSH_URL", None)  # flush the same engine root as the measured service
    if args.api_key:
        env["API_KEY"] = env["S1_API_KEY"] = args.api_key
    if args.self_check:
        command = [sys.executable, "-B", str(HARNESS / "s1_loadgen.py"),
                   "--root", str(root), "--set", name, "--cohort-file", str(root / "cohort.json"),
                   "--tok-dir", str(tok_dir), "--out-dir", str(out), "--self-check", "--no-body-cache"]
        log_name = "self-check.log"
    else:
        command = [sys.executable, "-B", str(CHECKED_RUNNER), "--runner", str(RUNNER), "--",
                   "--root", str(root), "--set", name, "--cohort", str(root / "cohort.json"),
                   "--tok-dir", str(tok_dir), "--out", str(out), "--n", str(args.n),
                   "--base-url", args.base_url, "--model", args.model]
        if args.skip_warmup:
            command.append("--skip-warmup")
        log_name = "run_dev.log"
    # Never print command/environment: credentials only travel via environment.
    with (out / log_name).open("w") as handle:
        result = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode or args.self_check:
        return result.returncode
    # Same scorer and equations; explicitly use this artifact's full request index.
    with (out / "score_formal.log").open("w") as handle:
        result = subprocess.run(
            [sys.executable, "-B", str(SCORER), "--run-dir", str(out),
             "--requests", str(root / "requests.jsonl"), "--harness-dir", str(HARNESS),
             "--out", str(out / "score_formal.json")],
            env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tok-dir", type=Path, default=REPO / "s1-dev/glm_tok")
    parser.add_argument("--self-check", action="store_true", help="full CPU acceptance only; no engine requests")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="", help="optional; passed only via child environment")
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--skip-warmup", action="store_true", help="reuse JIT warmup; KV flush still required")
    try:
        return execute(parser.parse_args(argv))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"longchain_replay: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
