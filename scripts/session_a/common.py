"""Pure session planning, framing, archive and result validation helpers."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import time

FRAME = "SESSION_A_JSON:"
ROLE_ENV = "SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS"
POLICIES = {"stock": ([], "fcfs", False), "baseline": (["000"], "fcfs", False),
            "spf": (["000", "001", "002"], "shortest-prefill-first", False),
            "d1": (["000", "001"], "fcfs", True),
            "spf_d1": (["000", "001", "002"], "shortest-prefill-first", True),
            "hrrn": (["000"], "hrrn", False),
            # D1 v1.2 = patch 004 (final-chunk role split); 004 stacks on 002, so 002 is
            # applied too (inert under fcfs).
            "d1v12": (["000", "001", "002", "004"], "fcfs", True),
            "spf_d1v12": (["000", "001", "002", "004"], "shortest-prefill-first", True)}

# Organizer-base line (T33): L2 configs live in configs.json (edited by scripts/l2.py).
# Each entry: patches applied to the organizer base (000 + optional 1xx), schedule policy,
# D1 switch, and the profile (submission-format JSON) whose command/env it runs.
CONFIGS_FILE = Path(__file__).with_name("configs.json")
# Optional per-config deltas on top of the profile: "args" (flag -> value, or null for a bare
# flag) replace/append launch flags; "env" overrides env (applied after the D1 switch).
PROFILE_OF, EXTRA_ARGS, EXTRA_ENV = {}, {}, {}
if CONFIGS_FILE.exists():
    for _name, _c in json.loads(CONFIGS_FILE.read_text()).items():
        POLICIES[_name] = (list(_c["patches"]), _c["policy"], bool(_c["d1"]))
        PROFILE_OF[_name] = _c.get("profile")
        EXTRA_ARGS[_name] = dict(_c.get("args", {}))
        EXTRA_ENV[_name] = dict(_c.get("env", {}))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def framed(text):
    rows = [line[len(FRAME):] for line in text.splitlines() if line.startswith(FRAME)]
    if len(rows) != 1:
        raise ValueError("missing/ambiguous session protocol frame")
    result = json.loads(rows[0])
    if not isinstance(result, dict):
        raise ValueError("session protocol must be an object")
    return result


def safe_extract(archive, destination):
    """Accept our regular-file archives only; reject links/traversal/duplicates."""
    destination = Path(destination).resolve()
    seen = set()
    for member in archive.getmembers():
        name = Path(member.name)
        if (name.is_absolute() or ".." in name.parts or member.name in seen
                or not member.isfile() or not name.parts):
            raise ValueError("unsafe archive member")
        seen.add(member.name)
        target = destination / name
        if destination not in target.resolve().parents:
            raise ValueError("archive escapes destination")
    for member in archive.getmembers():
        target = destination / member.name
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".incoming")
        # Refuse an existing symlink at the temporary destination too.
        if tmp.is_symlink():
            raise ValueError("unsafe temporary path")
        with archive.extractfile(member) as src, tmp.open("wb") as dst:
            import shutil
            shutil.copyfileobj(src, dst)
        tmp.chmod(member.mode & 0o777)
        tmp.replace(target)


class Budget:
    def __init__(self, seconds, final_seconds, clock=time.monotonic):
        if not all(math.isfinite(v) and v > 0 for v in (seconds, final_seconds)) or final_seconds >= seconds:
            raise ValueError("invalid deadline/final reserve")
        self.clock = clock
        self.end = clock() + seconds
        self.final_seconds = final_seconds

    def remaining(self, final=False):
        return max(0.0, self.end - self.clock() - (0 if final else self.final_seconds))

    def fits(self, cost, reserve=0):
        return cost + reserve <= self.remaining()

    def limit(self, requested, final=False):
        seconds = min(requested, self.remaining(final))
        if seconds <= 0:
            raise TimeoutError("session deadline")
        return seconds


def next_baseline(observations, max_n=30):
    """N6 calibration, N10 holdout, then official +/-4 with result reuse."""
    for n in (6, 10):
        if n not in observations:
            return n, None
    passing = [n for n, ok in observations.items() if ok]
    failing = [n for n, ok in observations.items() if not ok]
    lo, hi = max(passing, default=None), min(failing, default=None)
    if lo is not None and hi is not None and lo >= hi:
        return None, "non_monotone"
    if hi == 2:
        return None, "no_passing_rung"
    if lo is not None and hi == lo + 4:
        return None, "adjacent_bracket"
    n = lo + 4 if observations[10] else hi - 4
    return (None, "max_n_bound") if n > max_n else (n, None)


def level_result(ledger, n):
    levels = ledger.get("levels", [])
    if len(levels) != 1:
        raise ValueError("expected exactly one completed rung")
    r = levels[0]
    if r.get("N") != n or r.get("status") != "completed" or type(r.get("passed")) is not bool:
        raise ValueError("incomplete/mismatched rung is not an SLO failure")
    if len(r.get("dev_gates", {})) != 10 or type(r.get("tpot", {}).get("passed")) is not bool:
        raise ValueError("missing ten dev gates / TPOT gate")
    return r


def candidate_command(candidate, config, port, python="python3"):
    import shlex
    patches, policy, d1 = POLICIES[config]
    command = candidate["command"]
    command = shlex.split(command) if isinstance(command, str) else list(command)
    if command[:3] not in (["python3", "-m", "sglang.launch_server"], ["python", "-m", "sglang.launch_server"]):
        raise ValueError("expected SGLang module command")
    command[0] = python
    for flag, value in (("--port", str(port)), ("--schedule-policy", policy)):
        if flag in command:
            i = command.index(flag)
            command[i+1] = value
        else:
            command += [flag, value]
    for flag, value in EXTRA_ARGS.get(config, {}).items():
        if flag in command:
            i = command.index(flag)
            if value is None:
                continue
            command[i+1] = str(value)
        else:
            command += [flag] if value is None else [flag, str(value)]
    if "--enable-metrics" not in command:
        command.append("--enable-metrics")
    env = dict(candidate["env"])
    env.pop(ROLE_ENV, None)
    env["SGLANG_OPT_USE_TOPK_V2"] = "0"
    if d1:
        env[ROLE_ENV] = "154827,154829"
    env.update(EXTRA_ENV.get(config, {}))
    return command, env, patches


def select_profiles(paths, configs):
    """Match policy/D1 inputs by contents, never by queue filenames/order.

    A single profile is a base for all requested ablations (legacy behavior).
    v1.2 uses the corresponding v1.1 input plus patch 004; HRRN may derive
    from the D0/FCFS input. Ambiguous or missing multi-profile matches fail.
    """
    import shlex
    profiles = []
    for index, path in enumerate(paths):
        path = Path(path).resolve()
        data = json.loads(path.read_text())
        candidate_command(data, "baseline", 30000)  # validate before any remote work
        command = data["command"]
        command = shlex.split(command) if isinstance(command, str) else command
        policy = command[command.index("--schedule-policy") + 1] if "--schedule-policy" in command else "fcfs"
        profiles.append({"path": str(path), "remote": f"profiles/{index:03d}.json",
                         "sha256": digest(path), "policy": policy,
                         "d1": bool(data["env"].get(ROLE_ENV))})
    selected = {}
    for config in configs:
        _, policy, d1 = POLICIES[config]
        if PROFILE_OF.get(config):
            matches = [p for p in profiles if Path(p["path"]).name == Path(PROFILE_OF[config]).name]
            if len(matches) == 1:
                selected[config] = matches[0]
                continue
            raise ValueError(f"profile selection for {config}: {PROFILE_OF[config]} not in --profiles")
        matches = [p for p in profiles if (p["policy"], p["d1"]) == (policy, d1)]
        if not matches and config == "hrrn":
            matches = [p for p in profiles if (p["policy"], p["d1"]) == ("fcfs", False)]
        if len(profiles) == 1:
            matches = profiles
        if len(matches) != 1:
            raise ValueError(f"profile selection for {config}: expected one match, got {len(matches)}")
        selected[config] = matches[0]
    return profiles, selected


def metrics_summary(text):
    categories = {"queue": [], "forward": [], "cache_pool": []}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[ {]", line, 1)[0]
        category = ("queue" if "queue_time" in name else "forward" if "forward_execution" in name
                    else "cache_pool" if any(w in name for w in ("cache_hit", "token_usage", "mamba_", "kv_", "num_queue")) else None)
        if category:
            categories[category].append(line)
    return {k: {"available": bool(v), "samples": v} for k, v in categories.items()}
