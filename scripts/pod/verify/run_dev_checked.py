#!/usr/bin/env python3
"""Run the read-only organizer runner, failing BEFORE measurement on flush errors.

run_dev_checked.py --runner /path/to/run_dev.py -- [original run_dev arguments]
Only the flush hook is replaced. Preflight, warmup, loadgen and scoring stay original.
Writes flush_evidence.json with the invocation/flush times and final runner status.
"""
import argparse
import importlib.util
import json
import sys
import time
import urllib.request
from pathlib import Path


def strict_flush(url, evidence, path):
    evidence["flush_started_s"] = time.time()
    evidence["flush_success"] = False
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="POST"), timeout=30) as response:
            evidence["flush_http_status"] = response.status
            if not 200 <= response.status < 300:
                raise ValueError("non-2xx flush response")
            body = response.read()
            payload = json.loads(body)
            evidence["flush_response"] = payload
            # task.md requires positive JSON acknowledgement, not merely 2xx.
            if not isinstance(payload, dict) or payload.get("success") is not True:
                raise ValueError("flush lacks success=true acknowledgement")
        evidence["flush_success"] = True
    except (OSError, ValueError):
        # Do not print credentials or the full URL/error returned by an endpoint.
        evidence["flush_finished_s"] = time.time()
        path.write_text(json.dumps(evidence, indent=2) + "\n")
        print("FLUSH_FAILED: measurement aborted", file=sys.stderr, flush=True)
        raise SystemExit(2)
    evidence["flush_finished_s"] = time.time()
    path.write_text(json.dumps(evidence, indent=2) + "\n")
    print("flushed KV via checked runner", flush=True)
    return True


def run(runner, argv):
    parsed = argparse.ArgumentParser(add_help=False)
    parsed.add_argument("--out", required=True)
    parsed.add_argument("--n", type=int, required=True)
    opts, _ = parsed.parse_known_args(argv)
    out = Path(opts.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    receipt = out / "flush_evidence.json"
    evidence = {"schema_version": 1, "runner_started_s": time.time(), "n": opts.n,
                "flush_success": False, "runner_rc": None}
    receipt.write_text(json.dumps(evidence, indent=2) + "\n")
    spec = importlib.util.spec_from_file_location("arena_original_run_dev", runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.flush_kv = lambda url: strict_flush(url, evidence, receipt)
    rc = 2
    try:
        rc = module.main(argv)
        rc = int(rc or 0)
        return rc
    except SystemExit as error:
        rc = error.code if isinstance(error.code, int) else 2
        raise
    finally:
        evidence["runner_rc"] = rc
        evidence["runner_finished_s"] = time.time()
        summary = out / "summary.json"
        if rc == 0 and summary.is_file():
            data = json.loads(summary.read_text())
            evidence.update({key: Path(data.get(key) or "").name for key in ("raw", "run")})
        receipt.write_text(json.dumps(evidence, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    args, rest = parser.parse_known_args()
    if rest[:1] == ["--"]:
        rest = rest[1:]
    return run(args.runner.resolve(), rest)


if __name__ == "__main__":
    sys.exit(main())
