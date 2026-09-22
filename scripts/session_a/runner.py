#!/usr/bin/env python3
"""Unattended full session A controller. --dry-run never invokes Trisol/SSH.

No service creation/deletion/image build/submission. Live use is an explicit
coordinator launch against an already-running, separately authorized hang pod.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tarfile
import tempfile
import time

from common import (Budget, FRAME, POLICIES, candidate_command, digest, framed,
                    next_baseline, safe_extract, select_profiles, write_json)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
GPU_ROOT = "/sjtu/linhang/arena/runs/session_a"


def next_level(args, observations, baseline=False):
    """Choose a new rung without re-running calibration or exceeding caps."""
    n, reason = _next_level(args, observations, baseline)
    if n is not None and args.max_levels is not None and len(observations) >= args.max_levels:
        return None, "max_levels_reached"
    return n, reason


def _next_level(args, observations, baseline):
    if baseline:
        for n in (6, 10):
            if n not in observations:
                return n, None
    if args.ladder_mode == "levels":
        return next(((n, None) for n in args.levels if n not in observations), (None, "levels_completed"))
    passing = [n for n, ok in observations.items() if ok]
    failing = [n for n, ok in observations.items() if not ok]
    if passing and failing and max(passing) >= min(failing):
        return None, "non_monotone"
    if critical_p(observations) is not None:
        return None, "adjacent_bracket"
    if 2 in failing:
        return None, "no_passing_rung"
    if baseline and args.ladder_mode == "official-climb":
        return next_baseline(observations, args.max_n)
    from ladder_search import Search
    search = Search(args.ladder_mode, args.hint, args.max_n)
    while search.next_n in observations:
        search.observe(observations[search.next_n])
    return search.next_n, {"critical_bracket": "adjacent_bracket"}.get(search.reason, search.reason)


def critical_p(observations):
    passing = [n for n, ok in observations.items() if ok]
    failing = [n for n, ok in observations.items() if not ok]
    p, f = max(passing, default=None), min(failing, default=None)
    return p if p is not None and f == p + 4 else None

# Includes every SGLang file cited by R4/R7 plus every touched patch file.
REFERENCE = [
    "srt/managers/schedule_policy.py", "srt/managers/scheduler.py", "srt/managers/schedule_batch.py",
    "srt/layers/attention/hybrid_linear_attn_backend.py", "srt/layers/attention/linear/kda_backend.py",
    "srt/layers/attention/linear/kernels/kda_triton.py", "srt/layers/attention/linear/kernels/kda_flashkda.py",
    "srt/layers/attention/linear/kernels/kda_nvidia.py", "srt/layers/attention/linear/kernels/kda_flashinfer.py",
    "kernels/ops/attention/fla/chunk_delta_h.py", "kernels/ops/attention/fla/kda.py",
    "srt/model_executor/forward_batch_info.py", "srt/runtime_context.py",
    "srt/managers/scheduler_components/batch_result_processor.py", "srt/mem_cache/unified_radix_cache.py",
    "srt/mem_cache/unified_cache/components/mamba.py",
    "srt/layers/attention/dsa/dsa_indexer.py", "srt/layers/attention/dsa_backend.py",
    "srt/observability/metrics_collector.py",
]
SCRIPTS = ["ladder_search.py", "score_formal.py", "preflight_8gpu.sh", "preflight_8gpu.py",
           "logits_check.py", "cap_spot_check.py", "replay_chains.py", "serving_probe.py",
           "plan_8gpu_session.py", "make_case_sets.py"]


def inputs():
    files = {}
    for directory in (REPO / "s1-dev", REPO / "cases", REPO / "scripts/session_a"):
        for path in directory.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files[str(path.relative_to(REPO))] = path
    for path in [REPO / "scripts" / s for s in SCRIPTS] + list((REPO / "scripts").glob("if_checks*")) + list((REPO / "patches").glob("*.patch")) + [REPO / "submission/candidate-01.json"]:
        if path.is_file():
            files[str(path.relative_to(REPO))] = path
        else:
            raise ValueError("missing transfer input " + str(path))
    references = set(REFERENCE)
    for patch in (REPO / "patches").glob("*.patch"):
        for line in patch.read_text().splitlines():
            if line.startswith("+++ b/python/sglang/"):
                references.add(line.split()[1].removeprefix("b/python/sglang/"))
    for name in references:
        path = REPO / "src/sglang/python/sglang" / name
        if not path.is_file():
            if name in REFERENCE:
                raise ValueError("missing reference input " + name)
            continue  # a patch can introduce a file absent from v0.5.20
        files["reference/" + name] = path
    return files


def estimates(args):
    from plan_8gpu_session import make_plan
    a = argparse.Namespace(dev_root=REPO / "s1-dev", cohort=REPO / "s1-dev/harness/g0a/samples_v3/cohort_dev-combined-v1.json",
        budget_minutes=args.minutes, runs="stock:" + ",".join(str(n) for n in range(2, args.max_n + 1, 4)),
        startup_minutes=21., warmup_minutes=5., preflight_minutes=.5, flush_minutes=.5,
        self_test_minutes=2., reserve_fraction=.15, slowdown_exponent=.5, d1_service_factor=1.,
        uncached_per_output=32., out=None)
    return make_plan(a)


def prepare_cap(args):
    if args.cap_file:
        data = json.loads(args.cap_file.read_text())
    else:
        from cap_spot_check import load_questions
        data = {}
        for suite in ("aime", "gpqa"):
            questions, source = load_questions(suite, args.cap_count, 19, args.cap_cache)
            data[suite] = {"questions": questions, "source": source}
    if set(data) != {"aime", "gpqa"}:
        raise ValueError("CAP input needs AIME and GPQA")
    for suite, bundle in data.items():
        questions = bundle["questions"]
        if len(questions) != args.cap_count or not bundle.get("source"):
            raise ValueError("CAP sample count/provenance mismatch")
        if any(q.get("kind") != suite or not all(k in q for k in ("id", "question", "answer")) for q in questions):
            raise ValueError("invalid CAP questions")
    return data


def bundle(path, files, cap_questions):
    manifest = {}
    with tarfile.open(path, "w:gz") as tar:
        for name, source in sorted(files.items()):
            data = source.read_bytes()
            manifest[name] = hashlib.sha256(data).hexdigest()
            item = tarfile.TarInfo(name)
            item.size, item.mode = len(data), 0o755 if name.endswith(".sh") else 0o644
            tar.addfile(item, io.BytesIO(data))
        reference_index = {name.removeprefix("reference/"): manifest[name] for name in manifest if name.startswith("reference/")}
        for patch in (REPO / "patches").glob("*.patch"):
            for line in patch.read_text().splitlines():
                if line.startswith("+++ b/python/sglang/"):
                    reference_index.setdefault(line.split()[1].removeprefix("b/python/sglang/"), None)
        for name, value in (("cap_questions.json", cap_questions), ("reference-index.json", reference_index), ("TRANSFER_SHA256.json", manifest)):
            data = json.dumps(value).encode()
            item = tarfile.TarInfo(name)
            item.size, item.mode = len(data), 0o644
            tar.addfile(item, io.BytesIO(data))
            if name != "TRANSFER_SHA256.json":
                manifest[name] = hashlib.sha256(data).hexdigest()
    return manifest


# Bootstrap is self-contained because common.py has not been transferred yet.
BOOTSTRAP = """import base64,hashlib,io,json,pathlib,sys,tarfile
root=pathlib.Path(sys.argv[1]); expected=sys.argv[2]
parts=sorted((root/'upload').glob('*.part'))
data=b''.join(p.read_bytes() for p in parts)
assert hashlib.sha256(data).hexdigest()==expected,'archive sha256 mismatch'
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as t:
 names=set()
 for m in t.getmembers():
  p=pathlib.Path(m.name)
  assert m.isfile() and not p.is_absolute() and '..' not in p.parts and m.name not in names,'unsafe archive'
  names.add(m.name)
  assert root.resolve() in (root/p).resolve().parents,'archive escape'
 t.extractall(root)
manifest=json.loads((root/'TRANSFER_SHA256.json').read_text())
for name,sha in manifest.items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,'file sha256 mismatch: '+name
print('SESSION_A_JSON:'+json.dumps({'verified_files':len(manifest),'archive_sha256':expected}))
"""
GPU_RECEIVE = """import hashlib,io,json,pathlib,sys,tarfile
root=pathlib.Path(sys.argv[1]); data=sys.stdin.buffer.read()
assert hashlib.sha256(data).hexdigest()==sys.argv[2],'GPU archive sha256 mismatch'
assert root.is_absolute() and '..' not in root.parts and pathlib.Path('/sjtu/linhang/arena/runs') in root.parents,'GPU path outside arena'
root.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as t:
 for m in t.getmembers():
  p=pathlib.Path(m.name)
  assert m.isfile() and not p.is_absolute() and '..' not in p.parts,'unsafe archive'
  assert root.resolve() in (root/p).resolve().parents,'archive escape'
 t.extractall(root)
print('SESSION_A_JSON:'+json.dumps({'copied':True,'sha256':sys.argv[2]}))
"""
UPLOAD_CHUNKS = """import base64,hashlib,pathlib,sys,json
root=pathlib.Path(sys.argv[1]); count=0
for i in range(2,len(sys.argv),3):
 name,encoded,sha=sys.argv[i:i+3]
 assert name.isdigit(),'invalid chunk name'
 data=base64.b64decode(encoded,validate=True)
 assert hashlib.sha256(data).hexdigest()==sha,'chunk sha256 mismatch'
 (root/'upload'/(name+'.part')).write_bytes(data); count+=1
print('SESSION_A_JSON:'+json.dumps({'verified_chunks':count}))
"""


def upload_batches(stream, count):
    """64KiB maximum per argv value; batch to avoid hundreds of websocket setups."""
    index = 0
    while True:
        arguments = []
        for _ in range(count):
            data = stream.read(48 * 1024)
            if not data:
                break
            arguments.extend((f"{index:06d}", base64.b64encode(data).decode(), hashlib.sha256(data).hexdigest()))
            index += 1
        if not arguments:
            return
        yield arguments


class Controller:
    def __init__(self, args):
        self.args = args
        self.stamp = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.remote_root = f"/tmp/arena-session-a-{self.stamp}"
        self.out = args.out.resolve() if args.out else REPO / "runs/session_a" / self.stamp
        self.gpu_out = ("/sjtu/linhang/arena/runs" if args.out else GPU_ROOT) + "/" + self.stamp
        # Recovery has its own bounded pull window, even after work budget=0.
        seconds = args.collect_seconds if args.collect_only else args.minutes * 60
        self.budget = Budget(seconds, min(1, seconds / 2) if args.collect_only else args.final_minutes * 60)
        self.deadline = time.time() + args.minutes * 60 - args.final_minutes * 60
        self.plan = {} if args.collect_only else estimates(args)
        self.level_seconds = {r["N"]: (r["measure_minutes"] * args.time_factor + 1) * 60
                              for r in self.plan.get("ladder_levels", [])}
        self.limits = {"inspect": 600, "start": 21 * 60 * args.time_factor,
                       "warmup": 5 * 60 * args.time_factor + 60,
                       "preflight": args.preflight_minutes * 60, "cap": args.cap_minutes * 60}
        self.known = {}
        self.counter = 0
        self.uploaded = False
        self.current = None
        self.last_sync = 0.
        self.phase_end = None
        self.final_action_end = None
        self.summary = {"label": "dev session; no official scores", "run_id": self.stamp,
                        "status": "running", "levels": {}, "skips": [], "errors": []}
        self.matrix = args.matrix.split(",") if args.matrix else ["baseline", "spf", "spf_d1", "d1"]
        if args.hrrn and "hrrn" not in self.matrix:
            self.matrix.append("hrrn")
        # Put baseline first without reordering the remaining requested configs.
        if "baseline" in self.matrix:
            self.matrix = ["baseline"] + [c for c in self.matrix if c != "baseline"]
        self.profiles, self.selected = select_profiles(
            args.profiles or [REPO / "submission/candidate-01.json"], self.matrix)
        self.selected["stock"] = self.selected[self.matrix[0]]

    def event(self, message):
        line = datetime.now(timezone.utc).isoformat() + " " + self.stamp + " " + message.replace("\n", " ")
        print(line, flush=True)
        for path in (REPO / "logs/session_a.events", self.out / "session_a.events"):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write(line + "\n")

    def bohr(self, command, timeout=60, final=False):
        argv = ["bohr", "trisol", "inference", "exec", self.args.service, "--team", self.args.team, "--", *command]
        if self.phase_end is not None and not final:
            timeout = min(timeout, self.phase_end - time.monotonic())
            if timeout <= 0:
                raise TimeoutError("transfer step time box")
        if final and self.final_action_end is not None:
            timeout = min(timeout, self.final_action_end - time.monotonic())
            if timeout <= 0:
                raise TimeoutError("final action time box")
        p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                           timeout=self.budget.limit(timeout, final))
        if p.returncode:
            # Never dump CLI diagnostics which may contain authentication details.
            raise RuntimeError(f"Trisol exec returned {p.returncode}; command={command[0]}")
        return framed(p.stdout)

    def remote(self, operation, *args, final=False):
        return self.bohr(["python3", "-B", self.remote_root + "/scripts/session_a/remote.py", operation, *args], final=final)

    def transfer(self):
        self.event("TRANSFER begin")
        self.phase_end = time.monotonic() + self.args.transfer_minutes * 60
        cap = prepare_cap(self.args)  # on controller; pod never needs dataset Internet
        with tempfile.TemporaryDirectory(prefix="session-a-") as directory:
            archive = Path(directory) / "payload.tar.gz"
            files = inputs()
            files.update({p["remote"]: Path(p["path"]) for p in self.profiles})
            manifest = bundle(archive, files, cap)
            if any(manifest[p["remote"]] != p["sha256"] for p in self.profiles):
                raise ValueError("profile changed while preparing transfer")
            write_json(self.out / "transfer-manifest.json", manifest)
            archive_sha = digest(archive)
            self.bohr(["python3", "-c", "import pathlib,sys,json; p=pathlib.Path(sys.argv[1]); p.mkdir(exist_ok=False); (p/'upload').mkdir(); print('SESSION_A_JSON:'+json.dumps({'created':True}))", self.remote_root])
            with archive.open("rb") as f:
                index = 0
                for batch in upload_batches(f, self.args.transfer_batch_chunks):
                    receipt = self.bohr(["python3", "-c", UPLOAD_CHUNKS, self.remote_root, *batch])
                    if receipt.get("verified_chunks") != len(batch)//3:
                        raise RuntimeError("upload chunk receipt mismatch")
                    index += len(batch)//3
                    if index % 32 == 0:
                        self.event(f"TRANSFER progress chunks={index}")
            result = self.bohr(["python3", "-c", BOOTSTRAP, self.remote_root, archive_sha], timeout=180)
            self.uploaded = True
            write_json(self.out / "transfer-receipt.json", result)
            self.event(f"TRANSFER done files={result['verified_files']} chunks={index} sha256={archive_sha}")
        self.phase_end = None

    def gpu_sync(self, final=False):
        with tempfile.TemporaryDirectory(prefix="session-a-sync-") as tmp:
            archive = Path(tmp) / "gpu.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                for path in sorted(self.out.rglob("*")):
                    if path.is_file() and not path.is_symlink():
                        tar.add(path, arcname=str(path.relative_to(self.out)), recursive=False)
            command = shlex.join(["python3", "-c", GPU_RECEIVE, self.gpu_out, digest(archive)])
            # Proxy helper is repository-owned; do not depend on /root/.ssh/httpconnect.py.
            argv = ["ssh", "-F", str(REPO / "scripts/ssh_config.GPU"), "-o", "BatchMode=yes",
                    "-o", "ConnectTimeout=15", "-o", "ProxyCommand=" + shlex.join(["python3", str(REPO / "scripts/ssh_httpconnect.py"), "%h", "%p"]), "GPU", command]
            with archive.open("rb") as data:
                timeout = self.budget.limit(120, final)
                if final and self.final_action_end is not None:
                    timeout = min(timeout, self.final_action_end - time.monotonic())
                    if timeout <= 0:
                        raise TimeoutError("final action time box")
                p = subprocess.run(argv, stdin=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   timeout=timeout)
            if p.returncode or not framed(p.stdout.decode()).get("copied"):
                raise RuntimeError("GPU artifact copy/checksum verification failed")

    def sync(self, final=False):
        if self.uploaded:
            # Keep the control argument below ARG_MAX as the run accumulates files.
            # Only hashes still matching local files can be acknowledged remotely.
            known = {name: sha for name, sha in self.known.items() if (self.out / name).is_file()}
            if len(json.dumps(known)) > 60000:
                known = {}  # safe full snapshot; never omit changed evidence
            receipt = self.remote("export", json.dumps({"known": known}), final=final)
            with tempfile.TemporaryDirectory(prefix="session-a-pull-") as tmp:
                archive = Path(tmp) / "snapshot.tar.gz"
                with archive.open("wb") as f:
                    for offset in range(0, receipt["size"], 512 * 1024):
                        chunk = self.remote("read-export", str(offset), str(512 * 1024), final=final)
                        f.write(base64.b64decode(chunk["data"], validate=True))
                if digest(archive) != receipt["sha256"]:
                    raise RuntimeError("pulled artifact archive checksum mismatch")
                with tarfile.open(archive, "r:gz") as tar:
                    safe_extract(tar, self.out)
                for name, sha in receipt["manifest"].items():
                    if digest(self.out / name) != sha:
                        raise RuntimeError("pulled artifact file checksum mismatch")
                self.known.update(receipt["manifest"])
            write_json(self.out / "artifact-manifest.json", self.known)
        self.write_results()
        write_json(self.out / "summary.json", self.summary)
        self.gpu_sync(final=final)
        self.last_sync = time.monotonic()
        self.event("SYNC done local=verified gpu=verified")

    def write_results(self):
        """Rebuild the daemon manifest from durable pod receipts, including recovery."""
        results = []
        for path in sorted(self.out.glob("*/N*/result.json")):
            config = path.parts[-3]
            if config not in self.matrix:
                continue
            result = json.loads(path.read_text())
            n = result["N"]
            receipt = json.loads((self.out / config / "engine-command.json").read_text())
            profile = self.selected[config]
            if receipt.get("profile_sha256") != profile["sha256"]:
                raise ValueError("result profile receipt mismatch")
            paths = {}
            for key in ("raw", "run"):
                relative = Path(result["paths"][key]).relative_to(Path(self.remote_root) / "artifacts")
                if ".." in relative.parts or not (self.out / relative).is_file():
                    raise ValueError("result artifact missing/outside run")
                paths[key] = str(relative)
            self.summary["levels"].setdefault(config, {})[str(n)] = result
            results.append({"profile_sha256": profile["sha256"], "matrix": config, "N": n,
                            **paths, "candidate_exact": False, "p0": {}})
        write_json(self.out / "session_result.json", {"run_id": self.stamp, "results": results})

    def job(self, action, config="baseline", n=None):
        self.counter += 1
        name = f"{self.counter:03d}-{action}-{config}" + (f"-n{n}" if n is not None else "")
        limit = self.level_seconds[n] if action == "measure" else self.limits[action]
        limit = self.budget.limit(limit)
        spec = {"id": name, "action": action, "config": config, "n": n, "port": self.args.port,
                "timeout": limit, "deadline": self.deadline}
        if config in self.selected:
            spec.update(profile=self.selected[config]["remote"], profile_sha256=self.selected[config]["sha256"])
        self.event(f"{action.upper()} begin config={config} N={n} timebox_s={limit:.0f}")
        self.remote("launch", json.dumps(spec))
        end = time.monotonic() + limit + 10
        while True:
            status = self.remote("status", name)
            if status["status"] not in ("pending", "running"):
                break
            if time.monotonic() > end:
                raise TimeoutError("remote job did not complete within its time box")
            if time.monotonic() - self.last_sync >= self.args.sync_seconds:
                self.sync()
            time.sleep(min(5, self.budget.limit(5)))
        self.event(f"{action.upper()} {status['status']} config={config} N={n}")
        # Failure evidence is copied too, before any decision or new rung.
        self.sync()
        if status["status"] != "done":
            raise RuntimeError(f"{name}: {status['status']} {status.get('error', '')}")
        return status["result"]

    def startup(self, config, checks=False):
        self.job("start", config)
        self.current = config
        if checks:
            self.job("preflight", config)
            self.job("cap", config)
        self.job("warmup", config)

    def measure(self, config, n):
        result = self.job("measure", config, n)
        self.summary["levels"].setdefault(config, {})[str(n)] = result
        self.event(f"LEVEL done config={config} N={n} passed={result['passed']} gates=10+tpot")
        self.sync()  # includes the aggregate result and the per-level completion event
        return result["passed"]

    def config_cost(self, p):
        return self.limits["start"] + self.limits["warmup"] + sum(self.level_seconds[n] for n in (p, p+4)) + 120

    def run_ladder(self, config, reserve_matrix=()):
        observations = {}
        while True:
            n, reason = next_level(self.args, observations, config == "baseline")
            if n is None:
                break
            reserve = 0
            if reserve_matrix:
                p_hint = min(n, self.args.max_n - 4)
                reserve = sum(self.config_cost(p_hint) for _ in reserve_matrix[:2])
                reserve += self.limits["start"] + self.limits["cap"]
            if not (config == "baseline" and n in (6, 10)) and not self.budget.fits(self.level_seconds[n], reserve):
                reason = "budget_bound_before_bracket"
                break
            observations[n] = self.measure(config, n)
        result = {"observations": observations, "reason": reason, "critical_p": critical_p(observations)}
        self.summary.setdefault("ladders", {})[config] = result
        self.event(f"LADDER done config={config} reason={reason} observations={observations}")
        return result

    def execute(self):
        # The daemon has already frozen inputs in out/profiles/. Refuse stale
        # execution artifacts, but accept that prepared directory.
        self.out.mkdir(parents=True, exist_ok=True)
        if any(p.name != "profiles" for p in self.out.iterdir()):
            raise ValueError("output contains a previous run; use --collect-only or a fresh --run-id/--out")
        write_json(self.out / "profile-plan.json", {"profiles": self.profiles, "selected": self.selected})
        write_json(self.out / "planning-estimate.json", self.plan)
        write_json(self.out / "timeboxes.json", {"minutes": self.args.minutes, "final_minutes": self.args.final_minutes,
                   "actions_s": self.limits, "rungs_s": self.level_seconds, "factor": self.args.time_factor})
        try:
            self.transfer()
            inspection = self.job("inspect")
            self.event("STEP1 done sglang=" + inspection.get("sglang_version", "unavailable") + " patches=" + " ".join(k + ":" + v["status"] for k, v in inspection["patches"].items()))
            if not inspection["baseline_ready"]:
                raise RuntimeError("D0/model inspection failed; no strict-flush ladder is eligible")
            matrix = list(self.matrix)
            for config in list(matrix):
                missing = [p for p in POLICIES[config][0] if inspection["patches"].get(p, {}).get("status") != "ok"]
                if missing:
                    if self.args.matrix is not None:
                        raise RuntimeError(f"requested {config}: patch {','.join(missing)} absent/incompatible")
                    matrix.remove(config)  # legacy automatic matrix compatibility
                    self.summary["skips"].append(f"{config}: patch {','.join(missing)} absent/incompatible")
            self.job("start", "stock")
            self.current = "stock"
            self.job("cap", "stock")
            complete = True
            if "baseline" in matrix:
                self.startup("baseline", checks=True)
                self.event("STEP2 done baseline=D0 D1=UNSET policy=fcfs IF+TL03+CAP=smoke")
                others = [c for c in matrix if c != "baseline"]
                baseline = self.run_ladder("baseline", others)
                self.summary["baseline"] = baseline
                self.event(f"STEP3 done reason={baseline['reason']} observations={baseline['observations']}")
                p = baseline["critical_p"]
                if others and p is None:
                    self.summary["status"] = "incomplete"
                    self.summary["skips"].append("matrix: no adjacent baseline pass/fail bracket; never invent P")
                    return
                complete = baseline["reason"] in ("adjacent_bracket", "no_passing_rung", "levels_completed")
                for config in others:
                    # Account for config-specific IF/CAP gates as well as the pair.
                    reserve = self.limits["start"] + 2 * self.limits["cap"] + self.limits["preflight"]
                    if not self.budget.fits(self.config_cost(p), reserve):
                        complete = False
                        self.summary["skips"].append(config + ": time box cannot fit a restart and BOTH rungs")
                        self.event(f"MATRIX skip config={config} reason=full_pair_does_not_fit")
                        continue
                    self.startup(config, checks=True)
                    for n in (p, p+4):
                        self.measure(config, n)
            else:
                for config in matrix:
                    self.startup(config, checks=True)
                    result = self.run_ladder(config)
                    complete &= result["reason"] in ("adjacent_bracket", "no_passing_rung", "levels_completed")
            def quality(item):
                name, rows = item
                passed = [int(n) for n, r in rows.items() if r["passed"]]
                high = max(passed, default=0)
                tpot = rows.get(str(high), {}).get("tpot", {}).get("tpot_mean")
                return high, -(tpot if isinstance(tpot, (int, float)) else float("inf")), name == "baseline"
            if self.summary["levels"]:
                best = max(self.summary["levels"].items(), key=quality)[0]
                self.summary["best_measured"] = best
                if self.current != best:
                    self.job("start", best)
                    self.current = best
                self.job("cap", best)
            self.summary["status"] = "completed" if complete else "incomplete"
            self.event(f"STEP4 done matrix={matrix} status={self.summary['status']} CAP=smoke")
        except (Exception, KeyboardInterrupt) as e:
            self.summary["status"] = "aborted"
            self.summary["errors"].append(str(e) if isinstance(e, (RuntimeError, TimeoutError)) else type(e).__name__)
            self.event("SESSION aborted reason=" + self.summary["errors"][-1])
        finally:
            self.finish()

    def collect(self):
        """Pull existing pod artifacts only. No transfer, jobs, engine start/stop."""
        self.out.mkdir(parents=True, exist_ok=True)
        path = self.out / "summary.json"
        if path.exists():
            self.summary = json.loads(path.read_text())
            if self.summary.get("run_id") != self.stamp:
                raise ValueError("recovery run-id does not match summary")
        else:
            self.summary["status"] = "recovered"
        self.uploaded = True  # deterministic remote_root, never create/re-upload it
        manifest = self.out / "artifact-manifest.json"
        if manifest.exists():
            # Recovery must not acknowledge stale/missing local bytes.
            self.known = {name: sha for name, sha in json.loads(manifest.read_text()).items()
                          if (self.out / name).is_file() and digest(self.out / name) == sha}
        self.event("COLLECT begin")
        try:
            self.sync(final=True)
            self.event("COLLECT done")
            return 0
        except Exception as exc:
            self.event("COLLECT failed " + type(exc).__name__)
            return 2

    def finish(self):
        # Each final action is independent: failed copy must not prevent stopping.
        self.event("FINAL begin")
        for label, action in (("pull_before_stop", lambda: self.sync(final=True)),
                              ("stop_owned_engines", lambda: self.remote("stop", final=True) if self.uploaded else None),
                              ("pull_after_stop", lambda: self.sync(final=True))):
            try:
                if label == "pull_before_stop":
                    self.final_action_end = time.monotonic() + min(120, self.budget.remaining(True) / 3)
                else:
                    self.final_action_end = None
                action()
                self.event("FINAL " + label + " done")
            except Exception as e:
                self.summary["errors"].append(label + ": " + type(e).__name__)
                self.event("FINAL " + label + " failed " + type(e).__name__)
        self.summary["delete_command"] = shlex.join(["bohr", "trisol", "inference", "delete", self.args.service, "--team", self.args.team, "--yes"])
        self.summary["local_artifacts"] = str(self.out)
        self.summary["gpu_artifacts"] = self.gpu_out
        self.summary["warning"] = "Delete only after reviewing final sync receipts; no service deletion is performed."
        if self.summary["errors"] and self.summary["status"] == "completed":
            self.summary["status"] = "completed_with_cleanup_errors"
        write_json(self.out / "summary.json", self.summary)
        self.event(f"FINAL done status={self.summary['status']} errors={len(self.summary['errors'])}")
        # Copy final aggregate/events without requiring the pod to remain reachable.
        try:
            self.gpu_sync(final=True)
        except Exception as e:
            self.summary["errors"].append("final_summary_gpu_copy: " + type(e).__name__)
            if self.summary["status"] == "completed":
                self.summary["status"] = "completed_with_cleanup_errors"
            write_json(self.out / "summary.json", self.summary)
        print(json.dumps({k: self.summary[k] for k in ("status", "errors", "local_artifacts", "gpu_artifacts", "delete_command")}, indent=2))


def dry_run(args):
    ctl = Controller(args)
    prefix = ["bohr", "trisol", "inference", "exec", args.service, "--team", args.team, "--"]
    remote = ["python3", "-B", ctl.remote_root + "/scripts/session_a/remote.py"]
    print("DRY RUN: no Trisol, SSH, downloads, subprocesses, file writes or engine starts.")
    print(f"# local={ctl.out} gpu={ctl.gpu_out}; matrix={','.join(ctl.matrix)} mode={args.ladder_mode}")
    if args.collect_only:
        print("# COLLECT ONLY: bounded existing export/read-export + verified local/GPU copy; no jobs or engine operations")
        print(shlex.join(prefix + remote + ["export", '{"known":{}}']))
        print(shlex.join(prefix + remote + ["read-export", "<offset>", "524288"]))
        return 0
    print(f"# global={args.minutes}min final-reserve={args.final_minutes}min; private engine port={args.port}")
    print(f"# transfer {len(inputs())} shared files plus {len(ctl.profiles)} frozen profiles, CAP questions/provenance and SHA256 manifest")
    print(shlex.join(prefix + ["python3", "-c", UPLOAD_CHUNKS, ctl.remote_root, "000000", "<chunk-base64>", "<chunk-sha256>"]))
    print(shlex.join(prefix + ["python3", "-c", BOOTSTRAP, ctl.remote_root, "<archive-sha256>"]))
    def job(action, config="baseline", n=None):
        limit = ctl.level_seconds[n] if n is not None else ctl.limits[action]
        spec = {"action": action, "config": config, "n": n, "timeout": limit,
                "deadline": "<global-work-deadline>", "id": "<unique-job-id>", "port": args.port}
        if config in ctl.selected:
            spec.update(profile=ctl.selected[config]["remote"], profile_sha256=ctl.selected[config]["sha256"])
        print(shlex.join(prefix + remote + ["launch", json.dumps(spec)]))
        if action == "start":
            candidate = json.loads(Path(ctl.selected[config]["path"]).read_text())
            cmd, env, patches = candidate_command(candidate, config, args.port)
            print("# engine patches=" + str(patches) + " env=" + json.dumps(env) + " parent D1 unset before overrides")
            print("# " + shlex.join(cmd))
        print("# Poll status; after each step/rung and periodically export, read-export, verify SHA256 and sync both artifact roots.")
    job("inspect")
    job("start", "stock")
    job("cap", "stock")
    def startup(config):
        for action in ("start", "preflight", "cap", "warmup"):
            job(action, config)
    def ladder(config):
        observed = {}
        for outcome in args.dry_run_outcomes.split(","):
            n, reason = next_level(args, observed, config == "baseline")
            if n is None:
                break
            job("measure", config, n)
            observed[n] = outcome == "PASS"
        _, reason = next_level(args, observed, config == "baseline")
        print(f"# DRY trace config={config} observations={observed} reason={reason or 'needs_more_outcomes'}; live decisions/time boxes use measurements")
        return observed
    if "baseline" in ctl.matrix:
        startup("baseline")
        p = critical_p(ladder("baseline"))
        for config in ctl.matrix[1:]:
            if p is None:
                print(f"# {config}: no adjacent bracket in this trace; no invented P")
                continue
            startup(config)
            for n in (p, p+4):
                job("measure", config, n)
    else:
        for config in ctl.matrix:
            startup(config)
            ladder(config)
    print("# Final paired CAP on best measured config when budget permits.")
    print("# FINALLY, independent of failures: pull, stop owned jobs/engines, pull again, copy final summary to GPU.")
    print(shlex.join(prefix + remote + ["stop"]))
    print("# Engine watchdog stops owned engine at global work deadline even after controller death.")
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--service-id", "--service", dest="service", default="lh-arena-sess-a",
                   help="Trisol service ID or name (exec resolves either)")
    p.add_argument("--team", default="arena")
    p.add_argument("--budget-minutes", "--minutes", dest="minutes", type=float,
                   help="required for live launch: global wall budget; dry-run default 480")
    p.add_argument("--profiles", type=Path, nargs="+", help="candidate JSON paths; matched by policy and D1 switch")
    p.add_argument("--matrix", help="comma-separated configs: " + ",".join(c for c in POLICIES if c != "stock"))
    p.add_argument("--ladder-mode", choices=("official-climb", "fast", "levels"), default="official-climb")
    p.add_argument("--levels", type=lambda s: [int(n) for n in s.split(",")], default=[])
    p.add_argument("--hint", type=int, default=18)
    p.add_argument("--max-levels", type=int, help="maximum unique ladder measurements per config (includes baseline N6/N10)")
    p.add_argument("--out", type=Path, help="artifact root; an existing profiles/ snapshot directory is allowed")
    p.add_argument("--collect-only", action="store_true", help="only pull existing artifacts; accepts budget-minutes=0")
    p.add_argument("--collect-seconds", type=float, default=120, help="independent recovery pull time box")
    p.add_argument("--final-minutes", type=float, default=10)
    p.add_argument("--time-factor", type=float, default=1.25, help="multiplier on planner's 21m startup / 5m warmup / rung estimate")
    p.add_argument("--preflight-minutes", type=float, default=20)
    p.add_argument("--transfer-minutes", type=float, default=20)
    p.add_argument("--transfer-batch-chunks", type=int, choices=range(1, 9), default=8)
    p.add_argument("--cap-minutes", type=float, default=20)
    p.add_argument("--sync-seconds", type=float, default=120)
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--max-n", type=int, default=30)
    p.add_argument("--hrrn", action="store_true")
    p.add_argument("--cap-count", type=int, choices=(2, 3), default=2)
    p.add_argument("--cap-file", type=Path, help="prepared questions/provenance JSON; avoids controller downloads")
    p.add_argument("--cap-cache", type=Path, default=Path("/sjtu/linhang/arena/cache/session_a"))
    p.add_argument("--run-id")
    p.add_argument("--dry-run-outcomes", default="PASS,PASS,PASS,FAIL", help="N6,N10,... assumed outcomes, not measurements")
    return p


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    import re
    if a.minutes is None:
        if not a.dry_run and not a.collect_only:
            p.error("live launch requires --minutes with the coordinator's approved wall budget")
        a.minutes = 0 if a.collect_only else 480
    if not math.isfinite(a.minutes) or a.minutes < 0 or (not a.collect_only and a.minutes == 0):
        p.error("budget-minutes must be finite and positive (zero allowed for collect-only)")
    for key in ("final_minutes", "time_factor", "preflight_minutes", "cap_minutes", "sync_seconds", "transfer_minutes", "collect_seconds"):
        v = getattr(a, key)
        if not math.isfinite(v) or v <= 0:
            p.error(key + " must be finite and positive")
    if (not a.collect_only and a.final_minutes >= a.minutes) or not 1024 <= a.port <= 65535 or a.port == 8000:
        p.error("invalid final reserve or engine port (8000 belongs to hang service)")
    if a.matrix is not None:
        configs = a.matrix.split(",")
        if not configs or len(set(configs)) != len(configs) or any(c not in POLICIES or c == "stock" for c in configs):
            p.error("matrix must contain unique supported configurations")
        if a.hrrn and "hrrn" not in configs:
            p.error("--hrrn conflicts with an explicit matrix without hrrn")
    else:
        configs = ["baseline", "spf", "spf_d1", "d1"]
    if any(n < 2 or (n - 2) % 4 for n in [a.max_n, a.hint, *a.levels]):
        p.error("ladder values must be rungs 2,6,10,14,...")
    if a.max_levels is not None and a.max_levels < 1:
        p.error("max-levels must be positive")
    if not a.collect_only:
        if "baseline" in configs and a.max_n < 10:
            p.error("baseline requires max-n >=10 for N6/N10 calibration")
        if a.ladder_mode == "levels" and (not a.levels or len(set(a.levels)) != len(a.levels)):
            p.error("levels mode requires unique --levels")
        if any(n > a.max_n for n in a.levels) or (a.ladder_mode == "fast" and a.hint > a.max_n):
            p.error("levels/hint exceeds max-n")
        if a.ladder_mode == "official-climb" and a.max_n < 10:
            p.error("official-climb starts at N10; max-n must be >=10")
    if a.run_id and not re.fullmatch(r"[A-Za-z0-9_-]+", a.run_id):
        p.error("run-id must contain only letters/numbers/underscore/hyphen")
    if a.collect_only and not a.run_id:
        p.error("collect-only requires the original --run-id")
    if any(x not in ("PASS", "FAIL") for x in a.dry_run_outcomes.split(",")):
        p.error("dry-run-outcomes needs comma-separated PASS/FAIL")
    if a.dry_run:
        return dry_run(a)
    ctl = Controller(a)
    if a.collect_only:
        return ctl.collect()
    # Convert cooperative termination to the same finalization path as errors.
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    ctl.execute()
    return 0 if ctl.summary["status"] == "completed" and not ctl.summary["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
