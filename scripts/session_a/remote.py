#!/usr/bin/env python3
"""Pod-side actions, invoked only by the authorized controller; stdlib bootstrap.

All engine processes are owned by this run. Port 8000's hang service is untouched.
Supervisors enforce action deadlines even if the local controller disappears.
"""
from __future__ import annotations

import base64
import difflib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request

from common import FRAME, ROLE_ENV, candidate_command, digest, metrics_summary, write_json

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
WORK = ROOT / "work"
sys.path.insert(0, str(ROOT / "scripts"))
for proxy_name in ("NO_PROXY", "no_proxy"):
    os.environ[proxy_name] = os.environ.get(proxy_name, "") + ",127.0.0.1,localhost"


def emit(obj):
    print(FRAME + json.dumps(obj), flush=True)


def run(cmd, timeout=120, cwd=None, env=None):
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=timeout)
        return {"command": cmd, "returncode": p.returncode, "output": p.stdout}
    except subprocess.TimeoutExpired:
        return {"command": cmd, "returncode": 124, "output": "command time box expired"}
    except OSError as e:
        return {"command": cmd, "returncode": 127, "output": type(e).__name__}


def checked(cmd, log, env=None):
    with Path(log).open("w") as f:
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        code = p.wait()
    if code:
        raise RuntimeError(f"command failed rc={code}; see {Path(log).name}")


def process_info(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return {"ppid": int(fields[1]), "start": fields[19]}
    except (OSError, ValueError, IndexError):
        return None


def kill_tree(pid):
    # run_dev deliberately uses a new process group; kill descendants as well.
    children = []
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            info = process_info(int(p.name))
            if info and info["ppid"] == pid:
                children.append(int(p.name))
    for child in children:
        kill_tree(child)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def descendants(pid):
    result = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            child = int(path.name)
            info = process_info(child)
            if info and info["ppid"] == pid:
                result.extend(descendants(child))
                result.append((child, info["start"]))
    return result


def stop_engine(state=None):
    state_path = WORK / "engine.json"
    if state is None:
        if not state_path.exists():
            return {"stopped": False, "reason": "no_owned_engine"}
        state = json.loads(state_path.read_text())
    info = process_info(state["pid"])
    if info and info["start"] != state["start"]:
        raise RuntimeError("engine PID identity changed; refusing signal")
    owned = descendants(state["pid"]) if info else []
    try:
        os.killpg(state["pid"], signal.SIGTERM)
        time.sleep(2)
        os.killpg(state["pid"], signal.SIGKILL)
    except ProcessLookupError:
        pass
    for pid, start in owned:
        info = process_info(pid)
        if info and info["start"] == start:
            kill_tree(pid)  # workers may use their own process groups
    if state_path.exists() and json.loads(state_path.read_text()).get("pid") == state["pid"]:
        state_path.unlink()
    return {"stopped": True, "pid": state["pid"], "config": state["config"]}


def inspect(_spec):
    out = ART / "inspection"
    out.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        raise RuntimeError("sglang package not installed")
    package = Path(spec.origin).parent
    report = {"package_dir": str(package), "reference": "v0.5.20 / 94602c9", "commands": {},
              "files": {}, "patches": {}, "internet": {}}
    # Persist partial inspection after every probe, including a failed dry-run.
    def save():
        write_json(out / "report.json", report)
        (out / "report.txt").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    commands = {
        "pip_show": [sys.executable, "-m", "pip", "show", "sglang"],
        "version_commit": [sys.executable, "-c", "import sglang; print('version=',getattr(sglang,'__version__',None)); print('commit=',getattr(sglang,'__commit__',getattr(sglang,'__git_version__',None))); print('file=',sglang.__file__)"],
        "git_commit": ["git", "-C", str(package), "rev-parse", "HEAD"],
        "dependencies": [sys.executable, "-m", "pip", "list", "--format=json", "--disable-pip-version-check"],
        "nvidia_smi": ["nvidia-smi"], "memory": ["free", "-g"], "disk": ["df", "-h"],
        "models_size": ["du", "-sh", "/mnt/models"],
        # T33: which launch flags the organizer base actually accepts (L2 configs add flags).
        "server_help": [sys.executable, "-m", "sglang.launch_server", "--help"],
    }
    report["models_exists"] = Path("/mnt/models").is_dir()
    for name, cmd in commands.items():
        report["commands"][name] = run(cmd, timeout=60)
        save()
    for name, url in (("pypi", "https://pypi.org/simple/"), ("huggingface", "https://huggingface.co")):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=5) as response:
                report["internet"][name] = {"reachable": True, "http_status": response.status}
        except Exception as e:
            report["internet"][name] = {"reachable": False, "error": type(e).__name__}
        save()
    for ref in sorted((ROOT / "reference").rglob("*")):
        if not ref.is_file():
            continue
        relative = ref.relative_to(ROOT / "reference")
        actual = package / relative
        entry = {"reference_sha256": digest(ref), "exists": actual.is_file()}
        if actual.is_file():
            entry["installed_sha256"] = digest(actual)
            entry["equal"] = entry["reference_sha256"] == entry["installed_sha256"]
            if not entry["equal"]:
                difference = "".join(difflib.unified_diff(ref.read_text(errors="replace").splitlines(True),
                    actual.read_text(errors="replace").splitlines(True), fromfile="v0.5.20/"+str(relative),
                    tofile="installed/"+str(relative)))
                target = out / "diffs" / (str(relative) + ".diff")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(difference)
        report["files"][str(relative)] = entry
    for name, sha in json.loads((ROOT / "reference-index.json").read_text()).items():
        if sha is None:
            actual = package / name
            report["files"][name] = {"reference_sha256": None, "reference_note": "new patch file, absent in v0.5.20",
                "exists": actual.is_file(), "installed_sha256": digest(actual) if actual.is_file() else None}
    evidence = []
    for path in package.rglob("*"):
        if path.suffix not in (".py", ".cu", ".cuh", ".cpp") or not any(x in str(path).lower() for x in ("dsa", "indexer", "deepseek", "glm")):
            continue
        hits = [f"{i}: {line.strip()}" for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1)
                if re.search(r"sm_?80|ampere|capability|triton|tilelang|flashinfer|deep_gemm|topk", line, re.I)]
        if hits:
            evidence.append({"path": str(path), "sha256": digest(path), "lines": hits})
    report["dsa_indexer_sources"] = evidence
    report["dsa_note"] = "Source/backend dispatch evidence only; executed sm80 path must be corroborated by engine logs."
    WORK.mkdir(exist_ok=True)
    pristine = WORK / "stock" / "sglang"
    shutil.copytree(package, pristine, ignore=shutil.ignore_patterns("__pycache__", ".git"))
    # T33: bring the organizer base's own sources home (patch development against the base:
    # DP controller, KDA/mamba cache, scheduler, models). Python only, a few MB compressed.
    import tarfile
    with tarfile.open(out / "base_src.tgz", "w:gz") as tar:
        for path in sorted(pristine.rglob("*.py")):
            rel = path.relative_to(pristine)
            if rel.parts[0] == "srt" and (len(rel.parts) < 2 or rel.parts[1] not in ("test", "tests")):
                tar.add(path, arcname=str(rel))
    report["base_src"] = {"path": "inspection/base_src.tgz", "sha256": digest(out / "base_src.tgz")}
    save()
    chain = WORK / "dry_run" / "sglang"
    shutil.copytree(pristine, chain)
    predecessor_ok = True
    for prefix in ("000", "001", "002", "004"):
        patches = list((ROOT / "patches").glob(prefix + "-*.patch"))
        if not patches:
            report["patches"][prefix] = {"status": "skip", "reason": "absent"}
            if prefix not in ("002", "004"):
                predecessor_ok = False
        elif len(patches) != 1:
            report["patches"][prefix] = {"status": "fail", "reason": "ambiguous patch"}
            predecessor_ok = False
        elif not predecessor_ok:
            report["patches"][prefix] = {"status": "skip", "reason": "dependency failed"}
        else:
            patch = patches[0]
            command = ["patch", "--batch", "--forward", "--dry-run", "-p3", "--fuzz=0", "-i", str(patch)]
            result = run(command, cwd=chain)
            result.update(status="ok" if result["returncode"] == 0 else "fail", sha256=digest(patch))
            report["patches"][prefix] = result
            predecessor_ok = result["status"] == "ok"
            if predecessor_ok:
                applied = run([x for x in command if x != "--dry-run"], cwd=chain)
                predecessor_ok = applied["returncode"] == 0
                result["scratch_apply"] = applied
                if not predecessor_ok:
                    result["status"] = "fail"
        save()
    # Organizer-base line (T33): each 1xx patch is checked on the base + 000 (not on the
    # v0.5.20 001/002/004 chain). Stacks of several 1xx patches are checked at engine start.
    base_line = sorted(p for p in (ROOT / "patches").glob("1[0-9][0-9]-*.patch"))
    first = list((ROOT / "patches").glob("000-*.patch"))
    for patch in base_line:
        prefix = patch.name[:3]
        if report["patches"].get("000", {}).get("status") != "ok" or len(first) != 1:
            report["patches"][prefix] = {"status": "skip", "reason": "000 failed"}
            continue
        chain = WORK / ("dry_run_" + prefix) / "sglang"
        shutil.copytree(pristine, chain)
        run(["patch", "--batch", "--forward", "-p3", "--fuzz=0", "-i", str(first[0])], cwd=chain)
        result = run(["patch", "--batch", "--forward", "--dry-run", "-p3", "--fuzz=0", "-i", str(patch)], cwd=chain)
        result.update(status="ok" if result["returncode"] == 0 else "fail", sha256=digest(patch))
        report["patches"][prefix] = result
    save()
    report["baseline_ready"] = report["patches"]["000"]["status"] == "ok" and report["models_exists"]
    save()
    version_text = report["commands"]["version_commit"]["output"]
    match = re.search(r"version=\s*([^\s]+)", version_text)
    return {"baseline_ready": report["baseline_ready"], "patches": report["patches"],
            "sglang_version": match.group(1) if match else "unavailable",
            "package_dir": str(package), "internet": report["internet"]}


def start(spec):
    stop_engine()
    # A pre-existing listener must never be mistaken for our newly owned engine.
    with socket.socket() as check:
        check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        check.bind(("0.0.0.0", spec["port"]))
    config = spec["config"]
    profile = ROOT / spec.get("profile", "submission/candidate-01.json")
    if ROOT.resolve() not in profile.resolve().parents:
        raise ValueError("profile outside run")
    if spec.get("profile_sha256") and digest(profile) != spec["profile_sha256"]:
        raise ValueError("profile checksum mismatch")
    candidate = json.loads(profile.read_text())
    command, overrides, prefixes = candidate_command(candidate, config, spec["port"], sys.executable)
    source = WORK / "stock" / "sglang"
    target = WORK / config / "sglang"
    if config != "stock":
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        for prefix in prefixes:
            patches = list((ROOT / "patches").glob(prefix + "-*.patch"))
            if len(patches) != 1:
                raise RuntimeError("missing/ambiguous patch " + prefix)
            cmd = ["patch", "--batch", "--forward", "-p3", "--fuzz=0", "-i", str(patches[0])]
            receipt = run(cmd, cwd=target)
            write_json(ART / config / (prefix + "-apply.json"), receipt)
            if receipt["returncode"]:
                raise RuntimeError("patch apply failed: " + prefix)
    env = dict(os.environ)
    env.pop(ROLE_ENV, None)  # remove inherited candidate/parent switch as well
    env.update(overrides, PYTHONPATH=str(target.parent), PYTHONDONTWRITEBYTECODE="1",
               NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
    folder = ART / config
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "engine.log").open("a") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, stdin=subprocess.DEVNULL)
    info = process_info(process.pid)
    state = {"pid": process.pid, "start": info["start"], "config": config,
             "log": str(folder / "engine.log"), "deadline": spec["deadline"]}
    write_json(WORK / "engine.json", state)
    write_json(folder / "engine-command.json", {"argv": command, "env_overrides": overrides,
        "profile": str(profile.relative_to(ROOT)), "profile_sha256": digest(profile),
        "unset": [] if ROLE_ENV in overrides else [ROLE_ENV], "patches": prefixes,
        "pythonpath": str(target.parent), "port_note": "8000 hang process retained; private engine port"})
    subprocess.Popen([sys.executable, "-B", __file__, "watch-engine", json.dumps(state)],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    url = f"http://127.0.0.1:{spec['port']}/v1/models"
    while time.time() < spec["deadline"]:
        if process.poll() is not None:
            raise RuntimeError("engine exited during startup; see engine.log")
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                models = json.load(response)
            if any(m.get("id") == "default" for m in models.get("data", [])):
                return {"ready": True, "config": config}
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError("engine startup deadline")


def base(spec):
    return f"http://127.0.0.1:{spec['port']}"


def preflight(spec):
    folder = ART / spec["config"]
    # Detect missing dependencies first: no implicit Internet-dependent pip install.
    deps = run([sys.executable, "-c", "import transformers; from importlib.metadata import version; v=tuple(map(int,version('transformers').split('.')[:2])); assert (4,51)<=v<(6,0)"])
    write_json(folder / "requirements.json", deps)
    if deps["returncode"]:
        raise RuntimeError("harness dependency missing/incompatible; provide an offline wheelhouse before relaunch")
    checked([sys.executable, "-B", str(ROOT / "scripts/preflight_8gpu.py"), "--base-url", base(spec),
             "--extended-if", "--same-host", "--out", str(folder / "preflight.json")], folder / "preflight.log")
    return {"passed": True, "cases": "IF-01..08,10..12; TL-03; IF-09 not applicable TP8/DP1"}


def cap(spec):
    from cap_spot_check import compare
    from serving_probe import client_for
    questions = json.loads((ROOT / "cap_questions.json").read_text())
    result = {}
    for suite, bundle in questions.items():
        report = compare([client_for(base(spec), 1800)], bundle["questions"], suite, bundle["source"], 19)
        # T19 deliberately returns false for a single-server, small sample.
        # Inspect the actual interface/score evidence, do not relabel CAP-01/02 pass.
        report["session_config"] = spec["config"]
        report["smoke_interface_ok"] = all(r["results"]["stock"]["interface_ok"] for r in report["rows"])
        report["smoke_correct"] = report["scores"]["stock"]
        stock_path = ART / "stock" / ("cap_" + suite + ".json")
        if spec["config"] != "stock" and stock_path.exists():
            stock = json.loads(stock_path.read_text())
            if [r["id"] for r in stock["rows"]] != [r["id"] for r in report["rows"]] or stock["source"] != report["source"]:
                raise RuntimeError("CAP pairing/provenance mismatch")
            report["paired_stock_correct"] = stock["smoke_correct"]
            report["smoke_no_regression"] = report["smoke_correct"] >= stock["smoke_correct"] - 1
        write_json(ART / spec["config"] / ("cap_" + suite + ".json"), report)
        write_json(ART / spec["config"] / "cap_history" / spec["id"] / (suite + ".json"), report)
        result[suite] = {k: report.get(k) for k in ("smoke_interface_ok", "smoke_correct", "smoke_no_regression")}
    if not all(r["smoke_interface_ok"] and r["smoke_correct"] > 0 and r["smoke_no_regression"] is not False for r in result.values()):
        raise RuntimeError("CAP smoke interface/truncation, zero correct in a suite, or paired correctness regression")
    return {"smoke": result, "registry_sample_complete": False, "official_ability_gate": "not established"}


def warmup(spec):
    folder = ART / spec["config"]
    cmd = [sys.executable, "-B", str(ROOT / "s1-dev/harness/s1_loadgen.py"),
           "--root", str(ROOT / "s1-dev/data/dev-combined-v1"),
           "--cohort-file", str(ROOT / "s1-dev/harness/g0a/samples_v3/cohort_dev-combined-v1.json"),
           "--set", "dev-combined-v1", "--n", "6", "--model", "default",
           "--tok-dir", str(ROOT / "s1-dev/glm_tok"), "--out-dir", str(folder / "warmup"),
           "--instance-id", spec["config"] + "-warmup", "--cache-namespace", spec["config"] + "-jit",
           "--warmup", "--warmup-min-chains", "16"]
    checked(cmd, folder / "warmup.log", {**os.environ, "S1_ENGINE_URL": base(spec), "PYTHONDONTWRITEBYTECODE": "1"})
    import ladder_search as ladder
    flush = ladder.verified_flush(base(spec), "", 30)
    write_json(folder / "warmup_flush.json", flush)
    if not flush["success"]:
        raise RuntimeError("post-JIT strict flush failed")
    return {"warmed": True, "flush": flush}


def measure(spec):
    import argparse
    import ladder_search as ladder
    from common import level_result
    folder = ART / spec["config"] / f"N{spec['n']}"
    folder.mkdir(parents=True, exist_ok=False)
    engine_log = Path(json.loads((WORK / "engine.json").read_text())["log"])
    log_start = engine_log.stat().st_size
    stopping = threading.Event()
    summaries = []
    def sample():
        with (folder / "metrics.prom.jsonl").open("a") as f:
            while True:
                try:
                    with urllib.request.urlopen(base(spec) + "/metrics", timeout=4) as response:
                        body = response.read().decode()
                    row = {"time": time.time(), "prometheus": body}
                    summaries.append(metrics_summary(body))
                except Exception as e:
                    row = {"time": time.time(), "error": type(e).__name__}
                f.write(json.dumps(row) + "\n")
                f.flush()
                if stopping.wait(10):
                    break
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    # Reuse the existing strict guard and unmodified harness/scorer for ONE rung.
    # Every engine has already had its own original-harness JIT warmup.
    args = argparse.Namespace(mode="fast", hint=spec["n"], max_n=spec["n"], max_levels=1,
        out=folder / "ladder", base_url=base(spec), python=sys.executable, model="default",
        root=ROOT / "s1-dev/data/dev-combined-v1", cohort=ROOT / "s1-dev/harness/g0a/samples_v3/cohort_dev-combined-v1.json",
        tok_dir=ROOT / "s1-dev/glm_tok", harness_dir=ROOT / "s1-dev/harness", set="dev-combined-v1",
        warmup_min_chains=16, flush_timeout=30, gate_policy="dev+tpot")
    original = ladder.runner_command
    ladder.runner_command = lambda a, n, d, first: original(a, n, d, False)
    try:
        rc = ladder.execute(args)
        if rc not in (0, 3):
            raise RuntimeError("rung execution/flush/scoring failed; not an SLO FAIL")
        row = level_result(json.loads((args.out / "ledger.json").read_text()), spec["n"])
        write_json(folder / "result.json", row)
        return row
    finally:
        stopping.set()
        thread.join(timeout=6)
        with engine_log.open("rb") as f:
            f.seek(log_start)
            text = f.read().decode(errors="replace")
        (folder / "engine-level.log").write_text(text)
        (folder / "timing-cache-lines.log").write_text("\n".join(line for line in text.splitlines()
            if re.search(r"queue|forward|cache|pool|usage|evict|prefill|decode", line, re.I)) + "\n")
        write_json(folder / "diagnostics.json", {"engine_log_offset": log_start,
            "metrics_first": summaries[0] if summaries else {}, "metrics_last": summaries[-1] if summaries else {},
            "samples": len(summaries), "note": "Unavailable metrics remain unavailable; no inferred per-request forward time."})


ACTIONS = {"inspect": inspect, "start": start, "preflight": preflight, "cap": cap,
           "warmup": warmup, "measure": measure, "stop": lambda spec: stop_engine()}


def supervise(job):
    spec = json.loads(Path(job).read_text())
    status_path = ART / "jobs" / (spec["id"] + ".status.json")
    write_json(status_path, {"status": "running", "action": spec["action"], "started": time.time()})
    with (ART / "jobs" / (spec["id"] + ".log")).open("w") as f:
        child = subprocess.Popen([sys.executable, "-B", __file__, "action", job], stdout=f,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            code = child.wait(timeout=max(0.01, min(spec["timeout"], spec["deadline"] - time.time())))
            if not status_path.exists() or json.loads(status_path.read_text()).get("status") == "running":
                write_json(status_path, {"status": "failed", "returncode": code, "error": "action exited without result"})
        except subprocess.TimeoutExpired:
            kill_tree(child.pid)
            child.wait()
            stop_engine()  # timed out generation/startup cannot leak work into another rung
            write_json(status_path, {"status": "timeout", "action": spec["action"]})


def export(spec):
    """Consistent bytes+hash snapshot of changed artifacts, never model/input data."""
    known = spec.get("known", {})
    snapshot = WORK / "export.tar.gz"
    manifest = {}
    with tarfile.open(snapshot, "w:gz") as tar:
        for path in sorted(ART.rglob("*")):
            if not path.is_file() or path.is_symlink() or path.suffix == ".tmp":
                continue
            name = str(path.relative_to(ART))
            data = path.read_bytes()
            import hashlib
            sha = hashlib.sha256(data).hexdigest()
            if known.get(name) == sha:
                continue
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(data), 0o644
            tar.addfile(member, io.BytesIO(data))
            manifest[name] = sha
    return {"manifest": manifest, "sha256": digest(snapshot), "size": snapshot.stat().st_size}


def main():
    operation = sys.argv[1]
    ART.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if operation == "launch":
        spec = json.loads(sys.argv[2])
        if spec["action"] not in ACTIONS or not re.fullmatch(r"[a-z0-9_-]+", spec["id"]):
            raise ValueError("invalid action/id")
        path = WORK / (spec["id"] + ".json")
        if path.exists():
            raise ValueError("job id already used")
        write_json(path, spec)
        (ART / "jobs").mkdir(exist_ok=True)
        write_json(ART / "jobs" / (spec["id"] + ".spec.json"), spec)
        p = subprocess.Popen([sys.executable, "-B", __file__, "supervise", str(path)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        write_json(WORK / "supervisors" / (spec["id"] + ".json"), {"pid": p.pid, "start": process_info(p.pid)["start"]})
        emit({"launched": spec["id"], "pid": p.pid})
    elif operation == "supervise":
        supervise(sys.argv[2])
    elif operation == "action":
        spec = json.loads(Path(sys.argv[2]).read_text())
        try:
            result = ACTIONS[spec["action"]](spec)
            status = {"status": "done", "result": result}
        except Exception as e:
            status = {"status": "failed", "error": str(e) if isinstance(e, RuntimeError) else type(e).__name__}
            import traceback
            traceback.print_exc()
        write_json(ART / "jobs" / (spec["id"] + ".status.json"), status)
    elif operation == "status":
        path = ART / "jobs" / (sys.argv[2] + ".status.json")
        emit(json.loads(path.read_text()) if path.exists() else {"status": "pending"})
    elif operation == "watch-engine":
        state = json.loads(sys.argv[2])
        while time.time() < state["deadline"]:
            info = process_info(state["pid"])
            if not info:
                stop_engine(state)  # clean any surviving process group workers
                return
            if info["start"] != state["start"]:
                return
            time.sleep(min(5, max(0, state["deadline"] - time.time())))
        stop_engine(state)
    elif operation == "stop":
        # Stop all unfinished jobs first so one cannot launch an engine after cleanup.
        for path in (WORK / "supervisors").glob("*.json"):
            data = json.loads(path.read_text())
            info = process_info(data["pid"])
            if info and info["start"] == data["start"]:
                kill_tree(data["pid"])
        result = stop_engine()
        write_json(ART / "final/engine-stop.json", result)
        write_json(ART / "final/nvidia-smi.json", run(["nvidia-smi"], timeout=10))
        emit(result)
    elif operation == "export":
        emit(export(json.loads(sys.argv[2])))
    elif operation == "read-export":
        with (WORK / "export.tar.gz").open("rb") as f:
            f.seek(int(sys.argv[2]))
            emit({"data": base64.b64encode(f.read(int(sys.argv[3]))).decode()})
    else:
        raise ValueError("unknown operation")


if __name__ == "__main__":
    main()
