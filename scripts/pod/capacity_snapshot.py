#!/usr/bin/env python3
"""Bounded, read-only Pod capacity snapshot; never creates workspace directories."""
import datetime
import json
import os
from pathlib import Path
import subprocess


def read(path, limit=8192):
    try:
        return Path(path).read_text()[:limit].strip()
    except OSError:
        return None


def command(argv, timeout=5):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {"rc": p.returncode, "stdout": p.stdout[:8192], "stderr": p.stderr[:1024]}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": str(e)[:512]}


paths = ["/", "/tmp", "/tmp/ax", "/dev/shm", "/dev/shm/arena-runtime",
         "/root/.cache", "/root/.triton", "/workspace", "/vllm-workspace"]
mounts = []
for line in (read("/proc/self/mountinfo", 262144) or "").splitlines():
    if " - " not in line:
        continue
    left, right = line.split(" - ", 1)
    fields, fs = left.split(), right.split()
    mounts.append({"path": fields[4].replace("\\040", " "), "options": fields[5],
                   "filesystem": fs[0], "source": fs[1]})
observed = {}
for name in paths:
    p = Path(name)
    row = {"exists": p.exists(), "symlink": p.is_symlink(), "resolved": str(p.resolve())}
    if p.exists():
        st = os.statvfs(p)
        row.update(total_bytes=st.f_blocks * st.f_frsize,
                   available_bytes=st.f_bavail * st.f_frsize,
                   free_inodes=st.f_favail)
        candidates = [m for m in mounts if m["path"] == row["resolved"] or
                      Path(m["path"]) in p.resolve().parents]
        row["mount"] = max(candidates, key=lambda m: len(m["path"]), default=None)
    observed[name] = row

cg = Path("/sys/fs/cgroup")
files = ["memory.max", "memory.current", "memory.events", "memory.stat", "cpu.max",
         "memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes",
         "memory/memory.stat", "memory/memory.failcnt"]
result = {
    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "hostname": read("/etc/hostname"),
    "workspace_env": os.environ.get("AX_WORKSPACE_ROOT"),
    "workspace_entries": [{"name": p.name, "is_dir": p.is_dir(), "symlink": p.is_symlink()}
                          for p in sorted(Path("/tmp/ax").iterdir())][:64]
                         if Path("/tmp/ax").is_dir() else [],
    "paths": observed,
    "cgroup": {n: read(cg / n) for n in files if (cg / n).is_file()},
    "directory_usage": {n: command(["du", "-x", "-B1", "--max-depth=1", str(Path(n).resolve())], 5)
                        for n in ["/tmp", "/root/.cache", "/root/.triton", "/dev/shm/arena-runtime"]
                        if Path(n).exists()},
    "gpus": command(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"], 10),
    "runtime_abi": {
        "driver": command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], 10),
        "libc": command(["ldd", "--version"]),
        "python": command(["python3", "--version"]),
        "os_release": read("/etc/os-release"),
    },
    "gpu_processes": command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                              "--format=csv,noheader,nounits"], 10),
    "limits": "statvfs describes backing filesystems, not Pod ephemeral quota/usage; du includes visible image files",
}
body = json.dumps(result, ensure_ascii=False, indent=2)
if len(body.encode()) > 65536:
    raise RuntimeError("capacity snapshot exceeded 64 KiB output budget")
print(body)
