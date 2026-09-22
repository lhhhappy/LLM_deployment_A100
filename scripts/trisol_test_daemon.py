#!/usr/bin/env python3
"""T24: approval-gated, serialized 8-A100 self-tests. Stdlib, Linux/POSIX.

--dry-run prints plans only: no Trisol, runner, SSH, locks or state writes.
See tests/TRISOL_TEST_DAEMON.md for schema, budget units and recovery.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
ITEM_RE = re.compile(r"[0-9]{2,}-[a-z0-9][a-z0-9-]{0,60}\Z")
NAME_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,44}[a-z0-9])?\Z")
DEFAULTS = {
    # 0 means unlimited; item deadlines are still enforced.
    "daily_gpu_hours": 0, "poll_seconds": 30, "command_timeout_seconds": 60,
    "cleanup_reserve_seconds": 300, "delete_reserve_seconds": 120,
    "keep_alive": True, "idle_hold_seconds": 10800, "target_queue_depth": 5,
    "adopt_service_name": "lh-arena-sess-a",
    "max_admission_wait_minutes": 1440, "team": "arena", "cluster": "w1",
    "model": "glm-5-3-flash:2", "gpu_model": "A100-SXM4-80GB",
    "gpu_product_id": None, "bohr": ["bohr"],
    "runner": ["bash", "scripts/session_a/run_session_a.sh"],
    "mirror_host": "GPU", "mirror_root": "/sjtu/linhang/arena/runs",
    "daily_usage_adjustments": {},
}
TERMINAL = {"done", "failed", "cancelled", "invalid"}


class GuardError(Exception):
    """Static diagnostics only: do not copy CLI output/credentials to events."""


class Deadline(GuardError):
    pass


def read_json(path):
    def invalid(_):
        raise GuardError("non-finite-json")
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".test-daemon-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        dfd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat(timespec="seconds")


def numeric(x, name, minimum=0):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= minimum:
        raise GuardError("invalid-" + name)
    return float(x)


def below(root, relative):
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts:
        raise GuardError("unsafe-relative-path")
    target = (root / p).resolve()
    if not target.is_relative_to(root.resolve()):
        raise GuardError("path-outside-root")
    return target


def registered_configs(root):
    """L2 configs registered in scripts/session_a/configs.json (T33; edited via scripts/l2.py)."""
    p = root / "scripts/session_a/configs.json"
    return set(read_json(p)) if p.is_file() else set()


def approval(item):
    """APPROVED marks an item runnable. scripts/l2.py writes it on enqueue (approval is
    delegated to Claude by the user, 2026-09-22); the daemon only reads it."""
    p = item / "APPROVED"
    return p.is_file() and not p.is_symlink()


def config(root, path=None):
    result = dict(DEFAULTS)
    p = path or root / "tests/trisol_test_config.json"
    if p.exists():
        value = read_json(p)
        if not isinstance(value, dict) or set(value) - set(DEFAULTS):
            raise GuardError("unknown-config-fields")
        result.update(value)
    for k in ("poll_seconds", "command_timeout_seconds", "cleanup_reserve_seconds",
              "delete_reserve_seconds", "max_admission_wait_minutes"):
        numeric(result[k], k)
    if (isinstance(result["daily_gpu_hours"], bool) or
            not isinstance(result["daily_gpu_hours"], (int, float)) or
            not math.isfinite(result["daily_gpu_hours"]) or result["daily_gpu_hours"] < 0):
        raise GuardError("invalid-daily-gpu-hours")
    if (not isinstance(result["keep_alive"], bool) or
            not isinstance(result["idle_hold_seconds"], (int, float)) or
            result["idle_hold_seconds"] <= 0 or
            type(result["target_queue_depth"]) is not int or result["target_queue_depth"] < 1):
        raise GuardError("invalid-keep-alive-config")
    if result["cleanup_reserve_seconds"] <= result["delete_reserve_seconds"]:
        raise GuardError("cleanup-reserve-must-exceed-delete-reserve")
    if result["team"] != "arena" or result["cluster"] != "w1" or result["gpu_model"] != "A100-SXM4-80GB":
        raise GuardError("unsupported-arena-placement")
    if result["mirror_root"] != "/sjtu/linhang/arena/runs":
        raise GuardError("mirror-must-stay-in-arena-runs")
    for k in ("bohr", "runner"):
        if not isinstance(result[k], list) or not result[k] or not all(isinstance(v, str) and v for v in result[k]):
            raise GuardError("invalid-command-config")
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+", result["mirror_host"]):
        raise GuardError("invalid-mirror-host")
    for day, hours in result["daily_usage_adjustments"].items():
        dt.date.fromisoformat(day)
        if not isinstance(hours, (int, float)) or isinstance(hours, bool) or not math.isfinite(hours) or hours < 0:
            raise GuardError("invalid-budget-adjustment")
    return result


def load_spec(root, item, validate_name=True):
    if item.is_symlink() or (validate_name and not ITEM_RE.fullmatch(item.name)):
        raise GuardError("invalid-queue-directory")
    spec = read_json(item / "spec.json")
    allowed = {"image_ref", "profiles", "ladder", "matrix", "time_budget_minutes", "notes",
               "service_name", "promote", "max_admission_wait_minutes", "simulator"}
    if not isinstance(spec, dict) or set(spec) - allowed:
        raise GuardError("invalid-spec-fields")
    if not isinstance(spec.get("image_ref"), str) or not spec["image_ref"] or any(c.isspace() for c in spec["image_ref"]):
        raise GuardError("invalid-image-ref")
    profiles = spec.get("profiles")
    if not isinstance(profiles, list) or not profiles or not all(isinstance(name, str) for name in profiles) or len(profiles) != len(set(profiles)):
        raise GuardError("invalid-profiles")
    if len({Path(name).name for name in profiles}) != len(profiles):
        raise GuardError("duplicate-profile-basenames")
    for name in profiles:
        p = below(root, name)
        if not re.fullmatch(r"candidate-b[^/]*\.json", p.name) or not p.is_file():
            raise GuardError("candidate-b-profile-required")
        candidate = read_json(p)
        if not isinstance(candidate, dict) or set(candidate) != {"image", "command", "env", "model_name"}:
            raise GuardError("invalid-candidate")
    if not isinstance(spec.get("matrix"), list) or not spec["matrix"] or any(
            v not in {"baseline", "spf", "d1", "spf_d1", "hrrn", "d1v12", "spf_d1v12"} | registered_configs(root)
            for v in spec["matrix"]):
        raise GuardError("invalid-matrix")
    ladder = spec.get("ladder", {})
    if set(ladder) - {"mode", "levels", "hint", "max_n", "max_levels"}:
        raise GuardError("invalid-ladder-fields")
    if ladder.get("mode") not in {"levels", "official-climb", "fast"}:
        raise GuardError("invalid-ladder-mode")
    values = ladder.get("levels", [])
    if not isinstance(values, list) or (ladder["mode"] == "levels" and not values):
        raise GuardError("explicit-levels-required")
    if ladder["mode"] == "fast" and "hint" not in ladder:
        raise GuardError("fast-hint-required")
    for n in values + [ladder[k] for k in ("hint", "max_n") if k in ladder]:
        if type(n) is not int or n < 2 or (n - 2) % 4:
            raise GuardError("invalid-ladder-rung")
    if "max_levels" in ladder and (type(ladder["max_levels"]) is not int or ladder["max_levels"] < 1):
        raise GuardError("invalid-max-levels")
    numeric(spec.get("time_budget_minutes"), "time-budget")
    if "max_admission_wait_minutes" in spec:
        numeric(spec["max_admission_wait_minutes"], "admission-budget")
    if "notes" in spec and not isinstance(spec["notes"], str):
        raise GuardError("invalid-notes")
    if "simulator" in spec and (not isinstance(spec["simulator"], dict) or
                                 any(not isinstance(k, str) for k in spec["simulator"])):
        raise GuardError("invalid-simulator-config")
    name = spec.setdefault("service_name", "lh-t24-" + hashlib.sha256(item.name.encode()).hexdigest()[:12])
    if not NAME_RE.fullmatch(name):
        raise GuardError("invalid-service-name")
    if "promote" in spec:
        rule = spec["promote"]
        if not isinstance(rule, dict) or set(rule) != {"min_n"} or type(rule["min_n"]) is not int or rule["min_n"] < 2:
            raise GuardError("invalid-promotion-rule")
    hashes = {name: sha(below(root, name)) for name in profiles}
    pinned = item / "profiles.sha256.json"
    if pinned.exists() and read_json(pinned) != hashes:
        raise GuardError("profile-changed-since-enqueue")
    return spec, hashes


def enqueue(root, source, slug):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,60}", slug):
        raise GuardError("invalid-slug")
    queue = root / "tests/queue"
    queue.mkdir(parents=True, exist_ok=True)
    with exclusive(queue / ".enqueue.lock"):
        number = max([int(p.name.split("-", 1)[0]) for p in queue.iterdir() if ITEM_RE.fullmatch(p.name)], default=0) + 1
        dest = queue / f"{number:02d}-{slug}"
        stage = Path(tempfile.mkdtemp(prefix=".enqueue-", dir=queue))
        try:
            shutil.copyfile(source, stage / "spec.json")
            spec, hashes = load_spec(root, stage, validate_name=False)
            # Default service identity must be derived from final, not validation name.
            if "service_name" not in read_json(source):
                spec["service_name"] = "lh-t24-" + hashlib.sha256(dest.name.encode()).hexdigest()[:12]
            atomic_json(stage / "spec.json", spec)
            atomic_json(stage / "profiles.sha256.json", hashes)
            (stage / "notes.md").write_text("Enqueued. scripts/l2.py add writes APPROVED right after (self-test approval delegated to Claude).\n")
            os.rename(stage, dest)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return dest


@contextlib.contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise GuardError("daemon-already-running") from None
        yield


class Commands:
    def __init__(self, root):
        self.root = root
        self.on_start = None

    def run(self, argv, timeout, interrupted=lambda: False):
        if timeout <= 0:
            raise Deadline("deadline")
        # No shell and no raw child output is persisted; payload files are the protocol.
        with tempfile.TemporaryFile() as output:
            proc = subprocess.Popen(argv, cwd=self.root, stdout=output, stderr=subprocess.DEVNULL,
                                    start_new_session=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            end = time.monotonic() + timeout
            try:
                if self.on_start:
                    self.on_start(proc.pid)
                while proc.poll() is None:
                    if interrupted() or time.monotonic() >= end:
                        raise Deadline("command-interrupted-or-timeout")
                    time.sleep(min(.1, max(0, end - time.monotonic())))
                output.seek(0)
                data = output.read(4 * 1024 * 1024 + 1)
                if len(data) > 4 * 1024 * 1024:
                    raise GuardError("command-output-too-large")
                return proc.returncode, data.decode(errors="replace")
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


def services(payload):
    """List only our own services. Reject truncated/ambiguous envelopes."""
    if not isinstance(payload, dict) or payload.get("scope") != "mine":
        raise GuardError("list-must-prove-mine-scope")
    rows = payload.get("items", payload.get("services", payload.get("data")))
    if isinstance(rows, dict):
        rows = rows.get("items", rows.get("services"))
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise GuardError("invalid-service-list")
    total = payload.get("total", len(rows))
    if payload.get("has_more") or payload.get("next_page") or int(total) > len(rows):
        raise GuardError("incomplete-service-list")
    if any(not service_id(r) or not r.get("name") for r in rows):
        raise GuardError("service-identity-missing")
    return rows


def service_id(row):
    value = row.get("id", row.get("service_id"))
    return str(value) if value is not None and re.fullmatch(r"[A-Za-z0-9_-]+", str(value)) else None


def phase(row):
    # Real list rows (2026-09-22): status=deploying, stage=launching, substage=queueing,
    # reason=WaitingForAdmission while queued; status=deploying + ReadinessPending once admitted.
    if (str(row.get("reason") or "").lower() in {"waitingforadmission", "pendingadmission"}
            or row.get("substage") == "queueing"):
        return "waitingforadmission"
    # Prefer observed phase over operator-intent status=running.
    value = row.get("phase") or row.get("runtime_status") or row.get("status")
    if isinstance(value, dict):
        value = value.get("phase") or value.get("status")
    value = str(value or "unknown").lower().replace("_", "").replace("-", "")
    value = {"deploying": "starting", "stopping": "deleting"}.get(value, value)
    return value if value in {"waitingforadmission", "queued", "pendingadmission", "pending", "provisioning",
                              "starting", "running", "ready", "failed", "error", "stopped", "deleting", "deleted"} else "unknown"


def allocated(row):
    # Unknown / image-pull / provisioning may already hold GPUs; charge conservatively.
    return phase(row) not in {"waitingforadmission", "queued", "pendingadmission"}


def gpu_hours(start, end, day):
    midnight = dt.datetime.combine(dt.date.fromisoformat(day), dt.time(), dt.timezone.utc).timestamp()
    return max(0, min(end, midnight + 86400) - max(start, midnight)) * 8 / 3600


class Daemon:
    def __init__(self, root, cfg, commands=None, now=time.time, sleep=time.sleep):
        self.root, self.cfg = root, cfg
        self.commands = commands or Commands(root)
        self.now, self.sleep = now, sleep
        self.queue = root / "tests/queue"
        self.state_path = root / "data/trisol_tests.json"
        self.events = root / "logs/trisol_test.events"
        self.state = read_json(self.state_path) if self.state_path.exists() else {"schema_version": 1, "items": {}}
        if self.state.get("schema_version") != 1 or not isinstance(self.state.get("items"), dict):
            raise GuardError("invalid-state")
        self.stopping = False

    def save(self):
        atomic_json(self.state_path, self.state)

    def event(self, kind, item=None, **fields):
        self.events.parent.mkdir(parents=True, exist_ok=True)
        row = {"at": stamp(self.now()), "event": kind, "item": item, **fields}
        with self.events.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def stopped(self, item=None):
        return self.stopping or (self.queue / "STOP").exists() or bool(item and (item / "STOP").exists())

    def bohr(self, args, timeout=None):
        argv = self.cfg["bohr"] + ["trisol", "inference"] + args + ["--team", self.cfg["team"], "--no-input", "--output", "json"]
        rc, output = self.commands.run(argv, timeout or self.cfg["command_timeout_seconds"])
        if rc:
            raise GuardError("trisol-command-failed")
        try:
            return json.loads(output)
        except (ValueError, TypeError):
            raise GuardError("invalid-trisol-json") from None

    def inventory(self):
        return services(self.bohr(["list", "--scope", "mine", "--all"]))

    def runner_preflight(self, spec):
        rc, help_text = self.commands.run(self.cfg["runner"] + ["--help"], self.cfg["command_timeout_seconds"])
        required = ["--service-id", "--profiles", "--matrix", "--ladder-mode", "--levels", "--hint",
                    "--budget-minutes", "--run-id", "--out", "--collect-only"]
        required += ["--" + k.replace("_", "-") for k in ("max_n", "max_levels") if k in spec["ladder"]]
        if rc or any(flag not in help_text for flag in required):
            raise GuardError("runner-cli-contract-not-ready-see-T24-T23-dispatch")

    def create_command(self, spec, key):
        if not self.cfg["gpu_product_id"]:
            raise GuardError("gpu-product-id-required-for-create")
        return ["create", "--name", spec["service_name"], "--cluster", self.cfg["cluster"],
                "--gpu-product-id", str(self.cfg["gpu_product_id"]), "--gpu-model", self.cfg["gpu_model"],
                "--gpu-count", "8", "--replicas", "1", "--model", self.cfg["model"],
                "--image-ref", spec["image_ref"], "--command", "python3,-m,http.server,8000",
                "--startup-timeout-seconds", "3600", "--idempotency-key", key]

    def used(self, day, now=None):
        now = self.now() if now is None else now
        return self.cfg["daily_usage_adjustments"].get(day, 0) + sum(
            gpu_hours(r["charged_from"], r.get("deleted_at", now), day)
            for r in self.state["items"].values() if r.get("charged_from") is not None)

    def available_seconds(self):
        if self.cfg["daily_gpu_hours"] == 0:
            return math.inf
        t = self.now()
        day = dt.datetime.fromtimestamp(t, dt.timezone.utc).date().isoformat()
        return max(0, self.cfg["daily_gpu_hours"] - self.used(day, t)) * 3600 / 8

    def runner_command(self, spec, record, collect=False):
        argv = self.cfg["runner"] + ["--service-id", record["service_id"], "--run-id", record["slug"],
                "--out", str(self.root / "runs" / record["slug"]), "--profiles", *record["snapshot_profiles"],
                "--matrix", ",".join(spec["matrix"]), "--ladder-mode", spec["ladder"]["mode"],
                "--budget-minutes", str(max(0, record["work_deadline"] - self.now()) / 60)]
        for key in ("levels", "hint", "max_n", "max_levels"):
            if key in spec["ladder"]:
                value = spec["ladder"][key]
                argv += ["--" + key.replace("_", "-"), ",".join(map(str, value)) if isinstance(value, list) else str(value)]
        if collect:
            argv += ["--collect-only"]
        return argv

    def begin_charge(self, record, start):
        if record.get("charged_from") is not None:
            return
        record["charged_from"] = start
        # Include past allocated time for adoption, never grant a fresh budget on restart.
        remaining = self.available_seconds()
        end = start + record["spec"]["time_budget_minutes"] * 60
        if math.isfinite(remaining):
            end = min(end, self.now() + remaining)
        record.update(hard_deadline=end, work_deadline=end - self.cfg["cleanup_reserve_seconds"])
        self.save()

    def adoption_start(self, row):
        for key in ("allocated_at", "running_at", "started_at", "created_at", "createdAt"):
            value = row.get(key)
            if value:
                try:
                    result = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                    if result.tzinfo is None or result.timestamp() > self.now():
                        raise ValueError()
                    return result.timestamp()
                except ValueError:
                    raise GuardError("invalid-service-start-time") from None
        # Missing allocation history is unsafe for adopting an already allocated pod.
        raise GuardError("adoption-needs-allocation-or-creation-time")

    def run_item(self, item, spec, hashes, rows):
        # In keep-alive mode the one retained service is the owner of the whole
        # approved queue; an item's requested name is only used for first create.
        retained_id = self.state.get("keep_alive_service_id")
        retained = ([r for r in rows if service_id(r) == retained_id] if retained_id else [])
        if retained_id and not retained and self.state.get("keep_alive_service_name"):
            # Platform preemption/death: recreate the same neutral public name,
            # then let this still-approved item resume the FIFO queue.
            spec = dict(spec, service_name=self.state["keep_alive_service_name"])
            self.event("SERVICE_RECREATE", item.name, service_name=spec["service_name"])
        if retained:
            matching = retained
        else:
            matching = [r for r in rows if r["name"] in {spec["service_name"], self.cfg["adopt_service_name"]}]
        if len(matching) > 1 or (not retained and any(r["name"] != spec["service_name"] for r in rows)):
            self.event("BLOCKED", item.name, reason="another-owned-service-exists")
            return False
        if not matching and math.isfinite(self.available_seconds()) and self.available_seconds() <= self.cfg["cleanup_reserve_seconds"]:
            self.event("BLOCKED", item.name, reason="daily-gpu-budget")
            return False
        if spec["time_budget_minutes"] * 60 <= self.cfg["cleanup_reserve_seconds"]:
            raise GuardError("session-budget-too-small-for-cleanup")
        if matching:
            row = matching[0]
            # Validate the actual RequestSpec before taking ownership of an existing pod.
            deployed = self.bohr(["get", service_id(row), "--spec"])
            validate_adoption(spec, deployed, self.cfg)
            # A retained service was already accounted for by its first item;
            # each subsequent item gets its own deadline without requiring a
            # platform allocation timestamp on every status response.
            charge_from = (self.now() if retained and allocated(row)
                           else self.adoption_start(row) if allocated(row) else None)
        else:
            self.create_command(spec, str(uuid.uuid4()))  # prevalidate config before committing intent
            charge_from = None
        out = self.root / "runs" / item.name
        if out.exists():
            raise GuardError("run-output-already-exists")
        out.mkdir(parents=True)
        snapshot = []
        for name in spec["profiles"]:
            dest = out / "profiles" / Path(name).name
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(below(self.root, name), dest)
            if sha(dest) != hashes[name]:
                raise GuardError("profile-changed-during-snapshot")
            snapshot.append(str(dest))
        r = {"status": "creating", "slug": item.name, "spec": spec, "profile_hashes": hashes,
             "spec_sha256": sha(item / "spec.json"), "snapshot_profiles": snapshot,
             "service_name": matching[0]["name"] if matching else spec["service_name"],
             "service_id": service_id(matching[0]) if matching else None,
             "idempotency_key": str(uuid.uuid4()), "queued_at": self.now(), "queue_wait_s": 0,
             "charged_from": None, "create_attempted": False}
        self.state["items"][item.name] = r
        self.save()  # durable intent before any create, so ambiguous creates never duplicate
        try:
            if matching:
                self.state["keep_alive_service_id"] = r["service_id"]
                self.state["keep_alive_service_name"] = r["service_name"]
                self.event("ADOPT", item.name, service_id=r["service_id"])
                if charge_from is not None:
                    self.begin_charge(r, charge_from)
                if math.isfinite(self.available_seconds()) and self.available_seconds() <= self.cfg["cleanup_reserve_seconds"]:
                    raise Deadline("adopted-service-daily-budget-exhausted")
            else:
                # Recheck approval, STOP and content immediately before the mutating call.
                self.validate_approval(item, r)
                self.event("CREATE_INTENT", item.name)
                r["create_attempted"] = True
                self.save()
                response = self.bohr(self.create_command(spec, r["idempotency_key"]))
                if isinstance(response, dict):
                    created = response.get("service", response.get("data", response))
                    if isinstance(created, dict):
                        r["service_id"] = service_id(created)
                r["create_acknowledged"] = True
                self.save()
            r["status"] = "waiting"
            self.save()
            admission_end = r["queued_at"] + spec.get("max_admission_wait_minutes", self.cfg["max_admission_wait_minutes"]) * 60
            previous_poll = r["queued_at"]
            while True:
                if self.stopped(item) or not approval(item):
                    raise Deadline("stopped-or-approval-removed")
                rows = self.inventory()
                matches = [x for x in rows if x["name"] == r["service_name"]]
                if len(matches) != 1:
                    raise GuardError("service-missing-or-ambiguous")
                row = matches[0]
                if r["service_id"] and service_id(row) != r["service_id"]:
                    raise GuardError("service-identity-changed")
                r["service_id"] = service_id(row)
                observed = phase(row)
                if allocated(row):
                    self.begin_charge(r, previous_poll)
                else:
                    r["queue_wait_s"] = self.now() - r["queued_at"]
                if observed != r.get("last_phase"):
                    self.event("PHASE", item.name, phase=observed, queue_wait_s=r["queue_wait_s"])
                    r["last_phase"] = observed
                self.save()
                if self.now() >= admission_end or (r.get("work_deadline") is not None and self.now() >= r["work_deadline"]):
                    raise Deadline("session-or-admission-deadline")
                if observed in {"failed", "error", "stopped", "deleting", "deleted"}:
                    raise GuardError("service-failed")
                if observed in {"running", "ready"}:
                    break
                previous_poll = self.now()
                self.sleep(min(self.cfg["poll_seconds"], max(0, admission_end - self.now()),
                               max(0, r.get("work_deadline", admission_end) - self.now())))
            self.validate_approval(item, r)
            r["status"] = "running"
            self.save()
            self.event("RUNNER_START", item.name, deadline=stamp(r["hard_deadline"]))
            def record_child(pid):
                r["runner_process"] = process_token(pid)
                self.save()
            self.commands.on_start = record_child
            try:
                rc, _ = self.commands.run(self.runner_command(spec, r), r["work_deadline"] - self.now(),
                                          lambda: self.stopped(item) or not approval(item))
            finally:
                self.commands.on_start = None
                r.pop("runner_process", None)
            if rc:
                raise GuardError("runner-failed")
            r["outcome"] = "done"
        except (GuardError, OSError, ValueError, TypeError, KeyError) as exc:
            r["outcome"] = "cancelled" if self.stopped(item) else "failed"
            r["reason"] = str(exc) if isinstance(exc, GuardError) else type(exc).__name__
            self.event("SESSION_ABORT", item.name, reason=r["reason"])
        finally:
            try:
                self.finalize(item, r)
            except (GuardError, OSError, ValueError, TypeError, KeyError):
                # In particular, full local disks must not prevent deleting the pod.
                self.delete_owned(item, r)
                raise
        return True

    def validate_approval(self, item, r):
        if self.stopped(item) or not approval(item):
            raise GuardError("approval-required-or-stopped")
        if sha(item / "spec.json") != r["spec_sha256"] or any(sha(below(self.root, p)) != h for p, h in r["profile_hashes"].items()):
            raise GuardError("approved-content-changed")

    def cleanup_timeout(self, r, deletion=False):
        reserve = 0 if deletion else self.cfg["delete_reserve_seconds"]
        end = r.get("hard_deadline", self.now() + self.cfg["cleanup_reserve_seconds"])
        return max(.05, min(self.cfg["command_timeout_seconds"], end - self.now() - reserve))

    def finalize(self, item, r):
        r["status"] = "cleanup"
        self.save()
        out = self.root / "runs" / item.name
        out.mkdir(parents=True, exist_ok=True)
        try:
            if not r.get("service_id"):
                matches = [x for x in self.inventory() if x["name"] == r["service_name"]]
                if len(matches) == 1:
                    r["service_id"] = service_id(matches[0])
                    if allocated(matches[0]):
                        self.begin_charge(r, r["queued_at"])
                elif len(matches) > 1:
                    raise GuardError("ambiguous-created-service")
            if r.get("service_id") and not r.get("collected"):
                r.setdefault("work_deadline", self.now())
                rc, _ = self.commands.run(self.runner_command(r["spec"], r, collect=True), self.cleanup_timeout(r))
                r["collected"] = rc == 0
                self.event("COLLECT", item.name, ok=r["collected"])
            if not r.get("scored"):
                r["scores"] = self.score(r, out)
                r["scored"] = True
            if not r.get("sim_checked"):
                self.sim_check(item, r, out)
                r["sim_checked"] = True
            r["mirrored"] = self.mirror(out, r)
        except (GuardError, OSError, ValueError, TypeError, KeyError):
            self.event("ARTIFACT_ERROR", item.name)
            r["artifact_error"] = True
        finally:
            # Local disk/reporting errors must never bypass cloud cleanup.
            try:
                if r.get("outcome") == "done" and (not r.get("collected") or not r.get("mirrored") or not r.get("scored")):
                    r["outcome"] = "failed"
                atomic_json(out / "daemon_result.json", self.result(r))
                self.publish(item, r)
                r["status"] = "cleanup"
                self.save()
            finally:
                if self.cfg.get("keep_alive", True):
                    self.event("SERVICE_HELD", item.name, service_id=r.get("service_id"))
                    self.state["keep_alive_service_id"] = r.get("service_id")
                    self.state["keep_alive_service_name"] = r.get("service_name")
                    self.state["keep_alive_idle_since"] = self.now()
                    self.save()
                else:
                    self.delete_owned(item, r)
        if r.get("deleted_at") is not None or self.cfg.get("keep_alive", True):
            r["status"] = r.get("outcome", "failed")
            self.save()
            if r["status"] == "done":
                try:
                    self.promote(item, r)
                except Exception:
                    self.event("PROMOTION_ERROR", item.name)
            # Include the final deletion receipt in both artifact destinations.
            atomic_json(out / "daemon_result.json", self.result(r))
            try:
                self.mirror(out, r, after_delete=True)
            except (GuardError, OSError):
                self.event("FINAL_MIRROR_FAILED", item.name)

    def release_idle_service(self):
        """Release a retained service only after the configured idle hold."""
        sid = self.state.get("keep_alive_service_id")
        since = self.state.get("keep_alive_idle_since")
        if not sid or since is None or self.now() - since < self.cfg["idle_hold_seconds"]:
            return False
        rows = self.inventory()
        matches = [r for r in rows if service_id(r) == sid]
        if not matches:
            self.state.pop("keep_alive_service_id", None)
            self.state.pop("keep_alive_service_name", None)
            self.state.pop("keep_alive_idle_since", None)
            self.save()
            return False
        # Reuse the last terminal record as the identity-checked deletion receipt.
        records = [r for r in self.state["items"].values() if r.get("service_id") == sid]
        if not records:
            raise GuardError("retained-service-record-missing")
        record = records[-1]
        self.event("IDLE_RELEASE", record["slug"], service_id=sid)
        self.delete_owned(self.queue / record["slug"], record)
        if record.get("deleted_at") is not None:
            self.state.pop("keep_alive_service_id", None)
            self.state.pop("keep_alive_service_name", None)
            self.state.pop("keep_alive_idle_since", None)
            self.save()
            return True
        return False

    def delete_owned(self, item, r):
        try:
            rows = self.inventory()
            matches = [x for x in rows if x["name"] == r["service_name"]]
            if not matches and r.get("create_attempted") and not r.get("service_id") and not r.get("create_acknowledged"):
                raise GuardError("unresolved-create-intent-needs-reconciliation")
            if matches:
                if len(matches) != 1 or (r.get("service_id") and service_id(matches[0]) != r["service_id"]):
                    raise GuardError("cleanup-identity-changed")
                r["service_id"] = service_id(matches[0])
                if allocated(matches[0]):
                    self.begin_charge(r, r["queued_at"])
                if phase(matches[0]) != "deleting":
                    self.event("DELETE_INTENT", item.name, service_id=r["service_id"])
                    self.bohr(["delete", r["service_id"], "--yes"], self.cleanup_timeout(r, True))
                end = self.now() + self.cfg["delete_reserve_seconds"]
                while self.now() < end:
                    rows = self.inventory()
                    if not any(service_id(x) == r["service_id"] or x["name"] == r["service_name"] for x in rows):
                        break
                    self.sleep(min(self.cfg["poll_seconds"], max(0, end - self.now())))
                else:
                    raise GuardError("deletion-not-confirmed")
            r["deleted_at"] = self.now()
            self.event("DELETED", item.name, service_id=r.get("service_id"))
        except (GuardError, OSError, ValueError, TypeError):
            # Never mark final or permit another service while cleanup is uncertain.
            self.event("CLEANUP_PENDING", item.name)
        self.save()

    def mirror(self, out, r, after_delete=False):
        remote = self.cfg["mirror_root"] + "/" + r["slug"]
        limit = self.cfg["command_timeout_seconds"] if after_delete else self.cleanup_timeout(r)
        ssh = ["ssh", "-F", str(self.root / "scripts/ssh_config.GPU"), "-o", "BatchMode=yes",
               "-o", "ConnectTimeout=15", "-o", "ProxyCommand=" + shlex.join([
                   "python3", str(self.root / "scripts/ssh_httpconnect.py"), "%h", "%p"])]
        rc, _ = self.commands.run(ssh + [self.cfg["mirror_host"], "mkdir -p -- " + shlex.quote(remote)], limit)
        if rc:
            raise GuardError("mirror-mkdir-failed")
        rc, _ = self.commands.run(["rsync", "-a", "-e", shlex.join(ssh), "--", str(out) + "/", self.cfg["mirror_host"] + ":" + remote + "/"], limit)
        if rc:
            raise GuardError("mirror-transfer-failed")
        return True

    def score(self, r, out):
        manifest = out / "session_result.json"
        if not manifest.exists():
            raise GuardError("runner-result-manifest-missing")
        data = read_json(manifest)
        if data.get("run_id") != r["slug"] or not isinstance(data.get("results"), list):
            raise GuardError("invalid-runner-result-manifest")
        scores = []
        for i, level in enumerate(data["results"]):
            digest = level.get("profile_sha256")
            if digest not in r["profile_hashes"].values() or level.get("matrix") not in r["spec"]["matrix"]:
                raise GuardError("result-profile-or-matrix-mismatch")
            n = level["N"]
            if type(n) is not int or n < 2 or (n - 2) % 4:
                raise GuardError("invalid-result-rung")
            raw, run = below(out, level["raw"]), below(out, level["run"])
            if read_json(run).get("config", {}).get("N") != n:
                raise GuardError("result-rung-metadata-mismatch")
            target = out / "scores" / f"{i:03d}-N{n}.json"
            target.parent.mkdir(exist_ok=True)
            rc, _ = self.commands.run([sys.executable, "-B", str(self.root / "scripts/score_formal.py"),
                                      "--raw", str(raw), "--run", str(run), "--out", str(target)], self.cleanup_timeout(r))
            if rc:
                raise GuardError("scoring-failed")
            report = read_json(target)
            scores.append({"profile_sha256": digest, "matrix": level["matrix"], "N": n,
                           "passed_estimated": report["estimated"]["passed"] is True,
                           "tpot_mean": report["tpot"]["tpot_mean"], "report": str(target.relative_to(out)),
                           "gates": report.get("ttft_estimated", {}),
                           "candidate_exact": level.get("candidate_exact") is True,
                           "p0": level.get("p0", {})})
        return scores

    def sim_check(self, item, r, out):
        """Run the repository simulator and persist a measured-vs-predicted receipt."""
        sim = r["spec"].get("simulator", {})
        evidence = self.root / "evidence" / item.name
        sim_out = evidence / "simulator"
        sim_out.mkdir(parents=True, exist_ok=True)
        levels = sim.get("levels", "2,6,10,14,18,22")
        argv = [sys.executable, "-B", str(self.root / "scripts/sim_closed_loop.py"),
                "--out-dir", str(sim_out), "--summary-only", "--levels", str(levels)]
        for key in ("root", "cohort_file", "prefill_rates", "decode_ms", "policies", "schedulers",
                    "chunk_tokens", "page_tokens", "decode_batch_slope", "forward_overhead_ms",
                    "frontend_ms", "mix", "seed"):
            if key in sim:
                argv += ["--" + key.replace("_", "-"), str(sim[key])]
        rc, _ = self.commands.run(argv, self.cfg["command_timeout_seconds"])
        if rc:
            raise GuardError("simulator-failed")
        sweep = sim_out / "sweep.json"
        predicted = read_json(sweep) if sweep.exists() else {"ladders": []}
        measured = []
        for score in r.get("scores", []):
            measured.append({"N": score["N"], "matrix": score["matrix"],
                             "passed": score["passed_estimated"], "tpot_mean": score["tpot_mean"],
                             "gates": score.get("gates", {})})
        ladders = predicted.get("ladders", [])
        predicted_n = max((ladder.get("max_passing_tested_N") for ladder in ladders
                           if isinstance(ladder.get("max_passing_tested_N"), int)), default=None)
        measured_n = max((x["N"] for x in measured if x["passed"]), default=None)
        predicted_p95 = {}
        for ladder in ladders:
            for run in ladder.get("runs", []):
                for gate, detail in run.get("ttft_gate_detail", {}).items():
                    if isinstance(detail, dict) and isinstance(detail.get("p95"), (int, float)):
                        predicted_p95[gate] = detail["p95"]
        measured_p95 = {}
        for x in measured:
            for gate, detail in x["gates"].items():
                if isinstance(detail, dict) and isinstance(detail.get("p95"), (int, float)):
                    measured_p95[gate] = detail["p95"]
        p95_delta = {gate: measured_p95[gate] - predicted_p95[gate]
                     for gate in measured_p95.keys() & predicted_p95.keys()}
        receipt = {"item": item.name, "config": sim, "predicted": predicted,
                   "measured": measured, "delta": {"critical_N": None,
                   "p95": p95_delta,
                   "predicted_critical_N": predicted_n, "measured_critical_N": measured_n,
                   "critical_N_delta": (measured_n - predicted_n if measured_n is not None and predicted_n is not None else None),
                   "predicted_p95": predicted_p95, "measured_p95": measured_p95},
                   "label": "simulator cross-check; measured values are estimated scorer output"}
        atomic_json(evidence / "sim_vs_real.json", receipt)
        self.event("SIMCHECK", item.name, critical_N_delta=receipt["delta"]["critical_N_delta"],
                    p95_delta=receipt["delta"]["p95"],
                    receipt=str((evidence / "sim_vs_real.json").relative_to(self.root)))

    def result(self, r):
        passed = [x for x in r.get("scores", []) if x["passed_estimated"]]
        best = max(passed, key=lambda x: (x["N"], -(x["tpot_mean"] if x["tpot_mean"] is not None else math.inf)), default=None)
        return {"label": "estimated; dev cohort, not official submission", "item": r["slug"],
                "outcome": r.get("outcome", "failed"), "best": best, "queue_wait_s": r["queue_wait_s"],
                "charged_from": r.get("charged_from"), "deleted_at": r.get("deleted_at"),
                "collected": r.get("collected", False), "mirrored": r.get("mirrored", False),
                "reason": r.get("reason"), "tests": ["PF-01", "PF-02"]}

    def publish(self, item, r):
        result = self.result(r)
        # Deterministic result ID makes the append idempotent across crash recovery.
        key = "trisol-test:" + item.name
        self.events.parent.mkdir(parents=True, exist_ok=True)
        if not self.events.exists() or not any(json.loads(line).get("result_id") == key for line in self.events.read_text().splitlines() if line):
            self.event("RESULT", item.name, result_id=key, **{k: v for k, v in result.items() if k != "item"})
        ledger = self.root / "notes/experiments.md"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        marker = "<!-- " + key + " -->"
        with ledger.open("a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            text = f.read()
            if marker not in text:
                if "### Trisol automated tests (T27)" not in text:
                    f.write("\n### Trisol automated tests (T27)\n\n| Item | UTC | Status | Best N (estimated) | Queue wait s | Tests | Artifacts |\n| --- | --- | --- | --- | --- | --- | --- |\n")
                best_n = result["best"]["N"] if result["best"] else "—"
                f.write(f"| {item.name} | {stamp(self.now())} | {result['outcome']} | {best_n} | {r['queue_wait_s']:.1f} | PF-01/PF-02 | runs/{item.name}/daemon_result.json | {marker}\n")
                f.flush()
                os.fsync(f.fileno())

    def promote(self, item, r):
        if "promote" not in r["spec"] or r.get("promoted"):
            return
        best = self.result(r)["best"]
        required = set(re.findall(r"^\|\s*([A-Z0-9]+-\d+)\s*\|\s*P0\s*\|", (self.root / "tests/TEST_PLAN.md").read_text(), re.M))
        if not best or best["N"] < r["spec"]["promote"]["min_n"] or not required or not best["candidate_exact"] or any(
                best["p0"].get(case) != "pass" for case in required):
            self.event("PROMOTION_BLOCKED", item.name, reason="threshold-exact-candidate-or-p0-evidence")
            return
        candidate = next(Path(p) for p in r["snapshot_profiles"] if sha(p) == best["profile_sha256"])
        value = read_json(candidate)
        if value["image"] != r["spec"]["image_ref"] or not re.search(r"@sha256:[0-9a-f]{64}$", value["image"]):
            self.event("PROMOTION_BLOCKED", item.name, reason="tested-image-is-not-pinned-submission-image")
            return
        # Use W9's locked queue writer and its exact stub-trace filename/schema.
        import submit_daemon
        slug = "test-" + item.name
        queue = self.root / "submission/queue"
        existing = list(queue.glob("*-" + slug)) if queue.exists() else []
        dest = existing[0] if len(existing) == 1 else submit_daemon.create_queue_item(
            self.root, candidate, slug, self.root / "submission/stub-trace.jsonl", None)
        r["promoted"] = str(dest.relative_to(self.root))
        self.save()
        self.event("SUBMISSION_QUEUED_UNAPPROVED", item.name, queue_item=r["promoted"])

    def tick(self):
        # Outstanding cleanup always takes precedence over STOP and missing approval.
        pending = [(name, r) for name, r in self.state["items"].items() if r["status"] not in TERMINAL]
        if pending:
            for name, r in pending:
                r.setdefault("outcome", "failed")
                r.setdefault("reason", "recovered-interrupted-session")
                self.finalize(self.queue / name, r)
            return True
        if self.stopped():
            if self.cfg.get("keep_alive", True):
                self.state["keep_alive_idle_since"] = self.now() - self.cfg["idle_hold_seconds"]
                self.release_idle_service()
            return False
        items = [p for p in self.queue.glob("[0-9]*-*") if ITEM_RE.fullmatch(p.name)]
        approved_items = [p for p in items if p.is_dir() and approval(p) and not (p / "STOP").exists()
                          and p.name not in self.state["items"]]
        if len(approved_items) < self.cfg["target_queue_depth"]:
            if not self.state.get("queue_low_emitted"):
                self.event("QUEUE_LOW", remaining=len(approved_items))
                self.state["queue_low_emitted"] = True
                self.save()
        else:
            self.state.pop("queue_low_emitted", None)
            self.save()
        for item in sorted(items, key=lambda p: (int(p.name.split("-", 1)[0]), p.name)):
            if not item.is_dir() or not ITEM_RE.fullmatch(item.name) or item.name in self.state["items"]:
                continue
            if not approval(item) or (item / "STOP").exists():
                continue
            try:
                spec, hashes = load_spec(self.root, item)
                self.runner_preflight(spec)
                return self.run_item(item, spec, hashes, self.inventory())
            except (GuardError, OSError, ValueError, TypeError, KeyError) as exc:
                self.event("BLOCKED", item.name, reason=str(exc) if isinstance(exc, GuardError) else type(exc).__name__)
                return False
        if self.cfg.get("keep_alive", True):
            self.release_idle_service()
        return False


def validate_adoption(spec, deployed, cfg):
    if not isinstance(deployed, dict):
        raise GuardError("invalid-adoption-spec")
    if "spec" in deployed:
        deployed = deployed["spec"]
    if isinstance(deployed.get("runtime"), dict) or isinstance(deployed.get("resources"), dict):
        # Real `get --spec` output (checked 2026-09-22) nests fields under runtime/resources.
        deployed = {**deployed.get("resources", {}), **deployed.get("runtime", {})}
    gpu = deployed.get("gpu_count")
    command = deployed.get("command")
    args = deployed.get("args") or []
    if isinstance(command, str):
        command = shlex.split(command)
    if command is None and deployed.get("command_line"):
        command = shlex.split(deployed["command_line"])
    if (gpu != 8 or deployed.get("replicas", 1) != 1 or deployed.get("image_ref", deployed.get("image")) != spec["image_ref"]
            or (command or []) + args not in (["python3", "-m", "http.server", "8000"],
                                              ["python3", "-m", "http.server", "8000", "--bind", "0.0.0.0"])
            or deployed.get("gpu_model", cfg["gpu_model"]) != cfg["gpu_model"]):
        raise GuardError("adoption-spec-mismatch")


def dry_run(root, cfg, spec_path=None):
    items = [spec_path.parent] if spec_path else sorted((root / "tests/queue").glob("[0-9]*-*"))
    plans = []
    daemon = Daemon(root, cfg)
    for item in items:
        spec, hashes = load_spec(root, item)
        record = {"slug": item.name, "service_id": "SERVICE_ID", "work_deadline": daemon.now() + spec["time_budget_minutes"] * 60,
                  "snapshot_profiles": [str(root / "runs" / item.name / "profiles" / Path(p).name) for p in spec["profiles"]]}
        try:
            create = daemon.create_command(spec, "UUID_AT_EXECUTION")
        except GuardError as exc:
            create = str(exc)
        plans.append({"item": item.name, "approved": approval(item), "spec": spec, "profile_sha256": hashes,
                      "create_or_adopt": create, "runner": daemon.runner_command(spec, record),
                      "collect": daemon.runner_command(spec, record, True),
                      "daily_gpu_hours": cfg["daily_gpu_hours"], "daily_8gpu_minutes": cfg["daily_gpu_hours"] * 60 / 8})
    print(json.dumps({"dry_run": True, "plans": plans}, indent=2, ensure_ascii=False))
    return 0


def process_token(pid):
    try:
        # Linux starttime protects against PID reuse; comm may contain spaces.
        tail = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if tail[0] == "Z" else f"{pid}:{tail[19]}"
    except (OSError, IndexError):
        return None


def watchdog(root, cfg, token):
    """Independent local janitor; survives controller SIGKILL, reuses persisted ownership.

    It never starts a test. After owner loss it acquires the same lock and cleans
    only unfinished items. A dead host/network cannot be repaired locally.
    """
    pid = int(token.split(":", 1)[0])
    while process_token(pid) == token:
        try:
            daemon = Daemon(root, cfg)
            for r in daemon.state["items"].values():
                if r["status"] in TERMINAL:
                    continue
                end = r.get("hard_deadline", r["queued_at"] + r["spec"].get(
                    "max_admission_wait_minutes", cfg["max_admission_wait_minutes"]) * 60)
                if time.time() >= end - cfg["cleanup_reserve_seconds"]:
                    os.kill(pid, signal.SIGTERM)
                if time.time() >= end - cfg["delete_reserve_seconds"] and process_token(pid) == token:
                    os.kill(pid, signal.SIGKILL)
        except (OSError, ValueError, KeyError, GuardError):
            pass
        time.sleep(min(5, cfg["poll_seconds"]))
    # A clean shutdown has no unfinished item. Never schedule the next test.
    while True:
        try:
            with exclusive(root / "data/trisol_test.lock"):
                daemon = Daemon(root, cfg)
                pending = [(name, r) for name, r in daemon.state["items"].items() if r["status"] not in TERMINAL]
                if not pending:
                    return 0
                for name, r in pending:
                    child = r.get("runner_process")
                    if child:
                        child_pid = int(child.split(":", 1)[0])
                        if process_token(child_pid) == child:
                            try:
                                os.killpg(child_pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                    r.setdefault("outcome", "failed")
                    r.setdefault("reason", "watchdog-controller-exit")
                    daemon.finalize(daemon.queue / name, r)
        except (GuardError, OSError, ValueError, KeyError, TypeError):
            pass
        time.sleep(min(30, cfg["poll_seconds"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--once", action="store_true", help="process at most one approved test or recover cleanup")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--spec", type=Path, help="dry-run one existing queue item's spec")
    parser.add_argument("--enqueue", type=Path, metavar="SPEC")
    parser.add_argument("--slug")
    parser.add_argument("--watchdog", metavar="PID:START", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        cfg = config(root, args.config)
        if args.watchdog:
            return watchdog(root, cfg, args.watchdog)
        if args.dry_run:
            return dry_run(root, cfg, args.spec)
        if args.enqueue:
            if not args.slug:
                parser.error("--enqueue requires --slug")
            print(enqueue(root, args.enqueue, args.slug))
            return 0
        if args.spec:
            parser.error("--spec requires --dry-run")
        with exclusive(root / "data/trisol_test.lock"):
            daemon = Daemon(root, cfg)
            pid = root / "data/trisol_test.pid"
            pid.write_text(str(os.getpid()) + "\n")
            token = process_token(os.getpid())
            if not token:
                raise GuardError("watchdog-requires-linux-proc")
            watcher = [sys.executable, "-B", str(Path(__file__).resolve()), "--root", str(root), "--watchdog", token]
            if args.config:
                watcher += ["--config", str(args.config.resolve())]
            subprocess.Popen(watcher, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
            def stop(*_):
                daemon.stopping = True
            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            try:
                while True:
                    progressed = daemon.tick()
                    if args.once or daemon.stopping or ((root / "tests/queue/STOP").exists() and
                                                         not any(r["status"] == "cleanup" for r in daemon.state["items"].values())):
                        return 0 if not any(r["status"] == "cleanup" for r in daemon.state["items"].values()) else 2
                    if not progressed:
                        # Keep signals responsive even while queue is paused/empty.
                        end = time.monotonic() + cfg["poll_seconds"]
                        while not daemon.stopping and time.monotonic() < end:
                            time.sleep(min(.5, max(0, end - time.monotonic())))
            finally:
                pid.unlink(missing_ok=True)
    except (GuardError, OSError, ValueError, KeyError, TypeError) as exc:
        print("trisol_test_daemon: " + (str(exc) if isinstance(exc, GuardError) else type(exc).__name__), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
