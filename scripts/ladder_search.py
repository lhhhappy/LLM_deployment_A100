#!/usr/bin/env python3
"""README — capacity search over N=2,6,10,14,... using unmodified run_dev.py.

Usage (the engine must already be running; this tool never launches one):
  python3 scripts/ladder_search.py --base-url http://HOST:8000 --out /path/session
  python3 scripts/ladder_search.py --mode fast --hint 18 --base-url http://HOST:8000 \
      --out /path/session --gate-policy estimated --max-levels 6
  python3 scripts/ladder_search.py --dry-run --base-url http://HOST:8000 \
      --out /path/session --dry-run-outcomes PASS,FAIL

Python >=3.10, Linux/POSIX, standard library only (run_dev has its own deps).
Default mode official-climb reproduces task.md's traversal: start 10, pass +4,
fail -4, stop once a measured failing rung is above a measured passing rung;
an earlier failure is reused on descent. Failing N=2 gives null capacity.
There is no default upper bound. --max-n / --max-levels are optional session
budgets; reaching one reports an incomplete bound, never a critical N.
Fast mode starts at --hint, exponentially brackets in ladder indices, then
bisects until adjacent pass/fail rungs. This assumes monotone SLO behavior;
noise and configuration changes invalidate that inference. No rung repeats.

Gate policies: dev (default) uses the unchanged 10-gate ALL_PASS;
dev+tpot adds the 0.10 TPOT gate; estimated substitutes explicitly estimated
Clopper-Pearson TTFT checks and adds TPOT, retaining all other dev gates.
"Official" names the search order only: all capacities here are dev estimates.

Before EVERY level, POST engine-root/flush_cache and require 2xx plus JSON
{"success": true} (strict boolean). After preflight/warmup, run_dev's unchecked
flush is also routed through a loopback validation guard via S1_FLUSH_URL.
The guard kills the whole runner process group before replying on failure,
preventing run_dev from continuing into measurement. Harness files are never
edited. First level warms up; all later commands include --skip-warmup.
Inherited skip-warmup/preflight/flush overrides cannot silently weaken this.

Each level has its own directory, run_dev.log, unchanged harness artifacts,
score_estimated.json; ledger.json is atomically updated before/after each step.
Flush/execution/scoring errors abort the search, never count as an SLO FAIL.
Use a fresh --out directory; no resume or cross-configuration result reuse.
Bearer auth is read from API_KEY or S1_API_KEY, passed only via environment /
HTTP headers, never command arguments or the ledger. URLs cannot contain auth,
queries or fragments. Defaults target this repo's frozen dev-combined-v1 set.

Dry-run only prints planned shell commands, with no network/process/file writes.
Adaptive commands need assumed results: --dry-run-outcomes defaults PASS,FAIL;
if this trace ends early, the plan ends there (it is not a measured result).
Printed python commands go through this driver so the flush guard stays active;
run_dev/scorer argv are also shown with `#` comments for inspection.
Exit 0 = completed search, 3 = budget-limited, 2 = aborted/invalid, 130 = interrupted.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import time
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request

try:
    from . import score_formal as scoring
except ImportError:
    import score_formal as scoring

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "s1-dev" / "run_dev.py"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rung(n: int) -> int:
    if n < 2 or (n - 2) % 4:
        raise ValueError("N must be on the ladder 2,6,10,14,...")
    return (n - 2) // 4


class Search:
    def __init__(self, mode="official-climb", hint=10, max_n=None):
        self.mode = mode
        self.current = 2 if mode == "official-climb" else rung(hint)
        self.ceiling = rung(max_n) if max_n is not None else None
        if self.ceiling is not None and self.current > self.ceiling:
            raise ValueError("starting N exceeds --max-n")
        self.observations = {}
        self.step = 1
        self.reason = None

    @property
    def next_n(self):
        return None if self.reason else 2 + 4 * self.current

    def observe(self, passed: bool):
        if self.reason or self.current in self.observations:
            raise ValueError("search already stopped or rung already measured")
        if not isinstance(passed, bool):
            raise ValueError("observation must be a boolean")
        self.observations[self.current] = passed
        passing = [i for i, p in self.observations.items() if p]
        failing = [i for i, p in self.observations.items() if not p]
        low, high = max(passing, default=-1), min(failing, default=None)
        if high is not None and high <= low:
            self.reason = "non_monotone_observations"
        elif high == 0:
            self.reason = "no_passing_rung"
        elif low >= 0 and high is not None and high == low + 1:
            self.reason = "critical_bracket"
        elif self.ceiling is not None and low == self.ceiling:
            self.reason = "max_n_reached"
        elif self.mode == "official-climb":
            self.current += 1 if passed else -1
        elif low >= 0 and high is not None:
            self.current = (low + high) // 2
        else:
            self.current = max(0, self.current + (self.step if passed else -self.step))
            if self.ceiling is not None:
                self.current = min(self.current, self.ceiling)
            self.step *= 2

    def result(self):
        passing = [2 + 4 * i for i, p in self.observations.items() if p]
        failing = [2 + 4 * i for i, p in self.observations.items() if not p]
        complete = self.reason in ("critical_bracket", "no_passing_rung")
        return {"label": "estimated", "complete": complete, "reason": self.reason,
                "critical_n": max(passing, default=None) if complete else None,
                "largest_observed_pass": max(passing, default=None),
                "smallest_observed_fail": min(failing, default=None),
                "next_n": self.next_n}


def engine_url(raw: str) -> str:
    base = raw.strip().rstrip("/").removesuffix("/v1")
    parsed = urllib.parse.urlsplit(base)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("--base-url must be an http(s) engine URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("--base-url must not contain credentials, query or fragment")
    return base


class NoRedirect(urllib.request.HTTPRedirectHandler):
    # Avoid forwarding bearer credentials elsewhere or turning a flush POST into GET.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


FLUSH_SERVER_WAIT_S = float(os.environ.get("ARENA_FLUSH_SERVER_WAIT_S", "120"))
FLUSH_ATTEMPTS = int(os.environ.get("ARENA_FLUSH_ATTEMPTS", "3"))


def verified_flush(base: str, api_key: str, timeout: float) -> dict:
    """T14: SGLang flush is honestly refused (400) while requests are in flight.
    Ask the server to wait up to ARENA_FLUSH_SERVER_WAIT_S (``?timeout=``), and
    retry a bounded number of times; the level aborts if all attempts fail."""
    attempts = []
    for attempt in range(max(1, FLUSH_ATTEMPTS)):
        result = _verified_flush_once(base, api_key, timeout)
        attempts.append({k: result.get(k) for k in ("http_status", "error", "success")})
        if result["success"]:
            break
        if attempt + 1 < FLUSH_ATTEMPTS:
            time.sleep(min(30.0, 5.0 * (attempt + 1)))
    result["attempts"] = attempts
    return result


def _verified_flush_once(base: str, api_key: str, timeout: float) -> dict:
    result = {"at": now(), "http_status": None, "success": False}
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(
        base + "/flush_cache?timeout=%g" % FLUSH_SERVER_WAIT_S, method="POST", headers=headers)
    try:
        with urllib.request.build_opener(NoRedirect()).open(
                request, timeout=timeout + FLUSH_SERVER_WAIT_S) as response:
            result["http_status"] = response.status
            if not 200 <= response.status < 300:
                result["error"] = "non_2xx_status"
                return result
            body = response.read(65537)
            if len(body) > 65536:
                result["error"] = "oversized_json_body"
                return result
            payload = json.loads(body)
            result["success"] = isinstance(payload, dict) and payload.get("success") is True
            if not result["success"]:
                result["error"] = "json_success_is_not_true"
    except urllib.error.HTTPError as exc:
        result.update(http_status=exc.code, error="non_2xx_status")
        exc.close()
    except (ValueError, UnicodeError):
        result["error"] = "invalid_json_body"
    except (OSError, urllib.error.URLError, HTTPException) as exc:
        # Do not record response bodies / exception strings which may contain secrets.
        result["error"] = type(exc).__name__
    return result


def kill_group(process):
    if process is not None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class FlushGuard:
    """Intercept only run_dev's cleanup POST, leaving all load traffic untouched."""
    def __init__(self, base, api_key, timeout, on_result):
        self.process = None
        self.failed = False
        self.calls = 0
        self.lock = threading.Lock()
        guard = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                if self.path != "/flush_cache":
                    self.send_error(404)
                    return
                completed = False

                def finish(result):
                    nonlocal completed
                    with guard.lock:
                        if completed:
                            return
                        completed = True
                        guard.calls += 1
                        if not result["success"]:
                            guard.failed = True
                            kill_group(guard.process)
                        try:
                            on_result(result)
                        except Exception:
                            guard.failed = True
                            kill_group(guard.process)

                # run_dev ignores a failure after its 30s urlopen timeout. A total
                # deadline (including slow response bodies) must kill it sooner.
                timer = threading.Timer(min(timeout, 20), lambda: finish({
                    "at": now(), "http_status": None, "success": False,
                    "error": "guard_total_deadline"}))
                timer.daemon = True
                timer.start()
                try:
                    finish(verified_flush(base, api_key, timeout))
                except Exception as exc:
                    # Even an unexpected guard error must not let the unchecked
                    # runner continue after a closed HTTP connection.
                    finish({"at": now(), "http_status": None, "success": False,
                            "error": type(exc).__name__})
                finally:
                    timer.cancel()
                body = json.dumps({"success": not guard.failed}).encode()
                try:
                    self.send_response(200 if not guard.failed else 502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/flush_cache"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def runner_command(args, n, directory, first):
    cmd = [args.python, str(RUNNER), "--base-url", args.base_url, "--n", str(n),
           "--model", args.model, "--root", str(args.root), "--cohort", str(args.cohort),
           "--tok-dir", str(args.tok_dir), "--set", args.set, "--out", str(directory),
           "--warmup-min-chains", str(args.warmup_min_chains)]
    if not first:
        cmd.append("--skip-warmup")
    return cmd


def policy_pass(report, policy):
    if policy == "dev":
        return report["dev"]["ALL_PASS"]
    if policy == "dev+tpot":
        return report["dev"]["ALL_PASS"] and report["tpot"]["passed"]
    return report["estimated"]["passed"]


def dry_run(args):
    search = Search(args.mode, args.hint, args.max_n)
    # Executable command retains validation guard. Comments show adaptive inner argv.
    outer = [args.python, str(Path(__file__).resolve()), "--base-url", args.base_url,
             "--out", str(args.out), "--mode", args.mode, "--hint", str(args.hint),
             "--gate-policy", args.gate_policy, "--harness-dir", str(args.harness_dir),
             "--root", str(args.root), "--cohort", str(args.cohort), "--tok-dir", str(args.tok_dir),
             "--model", args.model, "--set", args.set, "--python", args.python,
             "--warmup-min-chains", str(args.warmup_min_chains), "--flush-timeout", str(args.flush_timeout)]
    for name in ("max_n", "max_levels"):
        if getattr(args, name) is not None:
            outer += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    print(shlex.join(outer))
    for index, passed in enumerate(args.dry_run_outcomes):
        if search.next_n is None or (args.max_levels is not None and index >= args.max_levels):
            break
        n = search.next_n
        directory = args.out / f"level_{index + 1:03d}_N{n}"
        # This is a plan, not executable curl: the real command requires JSON validation.
        print("# " + shlex.join(["POST", args.base_url + "/flush_cache", "require=2xx,JSON-success:true"]))
        print("# " + shlex.join(runner_command(args, n, directory, index == 0)))
        print("# " + shlex.join([args.python, str(Path(scoring.__file__).resolve()),
                                "--run-dir", str(directory), "--harness-dir", str(args.harness_dir),
                                "--out", str(directory / "score_estimated.json")]))
        search.observe(passed)
    return 0


def execute(args):
    search = Search(args.mode, args.hint, args.max_n)
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("--out must be new or empty (no stale result reuse)")
    for path in (RUNNER, args.harness_dir / "s1_loadgen.py", args.harness_dir / "s1_score.py",
                 args.root / "requests.jsonl", args.cohort, args.tok_dir):
        if not path.exists():
            raise ValueError(f"missing input: {path}")
    args.out.mkdir(parents=True, exist_ok=True)
    ledger_path = args.out / "ledger.json"
    ledger = {"schema_version": 1, "label": "estimated", "started_at": now(),
              "status": "running", "mode": args.mode, "gate_policy": args.gate_policy,
              "base_url": args.base_url, "hint": args.hint,
              "max_n": args.max_n, "max_levels": args.max_levels,
              "assumption": "fixed configuration and monotone pass/fail over N; no official score",
              "levels": []}

    def save():
        ledger["updated_at"] = now()
        ledger["search"] = search.result()
        scoring.write_json(ledger_path, ledger)

    api_key = os.environ.get("API_KEY") or os.environ.get("S1_API_KEY") or ""
    process = None
    level = None
    save()
    try:
        while search.next_n is not None:
            if args.max_levels is not None and len(ledger["levels"]) >= args.max_levels:
                search.reason = "max_levels_reached"
                break
            index, n = len(ledger["levels"]) + 1, search.next_n
            process = None
            directory = args.out / f"level_{index:03d}_N{n}"
            directory.mkdir()
            cmd = runner_command(args, n, directory, index == 1)
            level = {"N": n, "status": "flushing", "started_at": now(),
                     "command": cmd, "passed": None, "dev_gates": None,
                     "estimated_gates": None, "flushes": [],
                     "paths": {"directory": str(directory), "runner_log": str(directory / "run_dev.log")}}
            ledger["levels"].append(level)
            save()
            flush = verified_flush(args.base_url, api_key, args.flush_timeout)
            level["flushes"].append(dict(flush, stage="before_level"))
            save()
            if not flush["success"]:
                raise RuntimeError("before_level_flush_failed")

            def after_flush(result):
                level["flushes"].append(dict(result, stage="after_preflight_warmup"))
                save()

            env = os.environ.copy()
            env.update(S1_HARNESS_DIR=str(args.harness_dir), S1_SKIP_WARMUP="0",
                       S1_SKIP_PREFLIGHT="0", PYTHONDONTWRITEBYTECODE="1")
            for name in ("NO_PROXY", "no_proxy"):
                env[name] = env.get(name, "") + ",127.0.0.1,localhost"
            level["status"] = "running"
            save()
            print(f"N={n}: running; ledger {ledger_path}", flush=True)
            with FlushGuard(args.base_url, api_key, args.flush_timeout, after_flush) as guard:
                env["S1_FLUSH_URL"] = guard.url
                with (directory / "run_dev.log").open("w", encoding="utf-8") as log:
                    with guard.lock:
                        process = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log,
                                                   stderr=subprocess.STDOUT, start_new_session=True)
                        guard.process = process
                    try:
                        level["returncode"] = process.wait()
                    except BaseException:
                        kill_group(process)
                        process.wait()
                        raise
                if guard.failed or guard.calls != 1:
                    raise RuntimeError("measurement_flush_failed_or_missing")
            if level["returncode"] != 0:
                raise RuntimeError("run_dev_execution_failed")
            process = None
            raw, run = scoring.resolve_inputs(None, None, directory)
            metadata = scoring.read_json(run)
            if metadata.get("config", {}).get("N") != n:
                raise RuntimeError("run_metadata_N_mismatch")
            summary = scoring.read_json(directory / "summary.json")
            dev_report_path = directory / Path(summary["report"]).name
            original = scoring.read_json(dev_report_path)
            report = scoring.score_files(raw, run, args.harness_dir)
            if report["dev"] != original:
                raise RuntimeError("dev_report_recomputation_mismatch")
            scored = directory / "score_estimated.json"
            scoring.write_json(scored, report)
            passed = policy_pass(report, args.gate_policy)
            level.update(status="completed", finished_at=now(), passed=passed,
                         dev_passed=report["dev"]["ALL_PASS"], dev_gates=report["dev"]["gates"],
                         estimated_gates=report["estimated"]["gates"], tpot=report["tpot"])
            level["paths"].update(raw=str(raw), run=str(run), dev_report=str(dev_report_path),
                                  summary=str(directory / "summary.json"), score_estimated=str(scored))
            search.observe(passed)
            save()
            print(f"N={n}: {'PASS' if passed else 'FAIL'} ({args.gate_policy})", flush=True)
        ledger["status"] = "completed" if search.result()["complete"] else "incomplete"
        save()
        return 0 if ledger["status"] == "completed" else 3
    except (Exception, KeyboardInterrupt) as exc:
        kill_group(process)
        if process is not None:
            process.wait()
        ledger["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "aborted"
        # Record controlled errors or exception class only, not arbitrary secret-bearing text.
        ledger["error"] = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
        if level is not None:
            level.update(status=ledger["status"], error=ledger["error"], finished_at=now())
        save()
        print(f"search {ledger['status']}: {ledger['error']}; see {ledger_path}", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 2


def outcome_trace(value):
    outcomes = value.upper().split(",")
    if not outcomes or any(v not in ("PASS", "FAIL") for v in outcomes):
        raise argparse.ArgumentTypeError("expected a comma-separated PASS,FAIL trace")
    return [v == "PASS" for v in outcomes]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("official-climb", "fast"), default="official-climb")
    parser.add_argument("--hint", type=int, default=18, help="first N in fast mode")
    parser.add_argument("--max-n", type=int)
    parser.add_argument("--max-levels", type=int)
    parser.add_argument("--gate-policy", choices=("dev", "dev+tpot", "estimated"), default="dev")
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL") or os.environ.get("S1_ENGINE_URL") or "")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--harness-dir", type=Path, default=scoring.DEFAULT_HARNESS)
    parser.add_argument("--root", type=Path, default=REPO / "s1-dev/data/dev-combined-v1")
    parser.add_argument("--cohort", type=Path, default=scoring.DEFAULT_HARNESS / "g0a/samples_v3/cohort_dev-combined-v1.json")
    parser.add_argument("--tok-dir", type=Path, default=REPO / "s1-dev/glm_tok")
    parser.add_argument("--set", default="dev-combined-v1")
    parser.add_argument("--model", default=os.environ.get("S1_MODEL") or "glm-5.3-flash")
    parser.add_argument("--warmup-min-chains", type=int, default=16)
    parser.add_argument("--flush-timeout", type=float, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-outcomes", type=outcome_trace, default=outcome_trace("PASS,FAIL"))
    args = parser.parse_args(argv)
    try:
        args.base_url = engine_url(args.base_url)
        rung(args.hint)
        if args.max_levels is not None and args.max_levels < 1:
            raise ValueError("--max-levels must be positive")
        if not scoring.finite_number(args.flush_timeout) or args.flush_timeout <= 0 or args.warmup_min_chains < 1:
            raise ValueError("timeouts and warmup chain count must be positive")
        for name in ("out", "root", "cohort", "tok_dir", "harness_dir"):
            setattr(args, name, getattr(args, name).resolve())
        return dry_run(args) if args.dry_run else execute(args)
    except (OSError, ValueError) as exc:
        print(f"ladder_search: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
