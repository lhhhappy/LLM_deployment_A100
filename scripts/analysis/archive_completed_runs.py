#!/usr/bin/env python3
"""Archive finished Pod runs locally, verify SHA256, then optionally clean Pod.

No tar/gzip snapshot or second full copy is created on Pod. Downloads are resumable
and bounded chunks; only a fully fsynced local receipt authorizes cleanup. --watch
uses a local lock and replaces one status JSON, without a growing daemon log.
"""
import argparse
import base64
import fcntl
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
CHUNK = 120000


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def durable_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def remote(action, *args, on_devbox=False):
    code = base64.b64encode((ROOT / "scripts/pod/archive_run.py").read_bytes()).decode()
    command = shlex.join(["python3", "-B", "-c",
                          "import base64; exec(compile(base64.b64decode(" + repr(code) + "), '<archive_run>', 'exec'))",
                          action, *map(str, args)])
    gpu = "cd /sjtu/linhang/arena/repo && source scripts/pod/common.sh && bexec " + shlex.quote(command)
    if on_devbox and not ROOT.is_relative_to(Path('/sjtu/linhang/arena')):
        raise ValueError('--on-devbox requires the GPU development workspace')
    transport = ['bash', '-c', gpu] if on_devbox else [str(ROOT / "scripts/gssh"), gpu]
    process = subprocess.Popen(transport, cwd=ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True,
                               env={**os.environ, "GSSH_TIMEOUT": "50"})
    try:
        out, err = process.communicate(timeout=55)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise RuntimeError("archive remote call timed out; source retained")
    if process.returncode:
        raise RuntimeError((err or out)[-1600:])
    records = [line.removeprefix("ARCHIVE_RESULT ") for line in out.splitlines()
               if line.startswith("ARCHIVE_RESULT ")]
    if len(records) != 1:
        raise ValueError("missing/ambiguous archive response")
    result = json.loads(records[0])
    if not result.pop("ok"):
        raise ValueError(result["error"])
    return result


def archive_one(name, destination, *, cleanup=False, call=remote):
    meta = call("manifest", name)
    sha = meta["manifest_sha256"]
    # Validate server-provided paths before writing anything locally.
    payload = {"run": meta["run"], "files": meta["files"]}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if sha != expected or meta["run"] != name:
        raise ValueError("invalid source manifest")
    seen = set()
    for row in meta["files"]:
        rel = Path(row["path"])
        if rel.is_absolute() or ".." in rel.parts or not rel.parts or row["path"] in seen:
            raise ValueError("unsafe or duplicate manifest path")
        if type(row["size"]) is not int or row["size"] < 0:
            raise ValueError("invalid source size")
        seen.add(row["path"])
    target = destination / name / sha
    files = target / "files"
    files.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(files).free < sum(r["size"] for r in meta["files"]) + 1024 ** 3:
        raise ValueError("local archive needs source size plus 1 GiB free")
    for row in meta["files"]:
        out = files / row["path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.is_file() and out.stat().st_size == row["size"] and digest(out) == row["sha256"]:
            continue
        partial = out.with_name(out.name + ".archive-part")
        with partial.open("wb") as f:
            for offset in range(0, row["size"], CHUNK):
                length = min(CHUNK, row["size"] - offset)
                reply = call("read", name, row["path"], offset, length)
                data = base64.b64decode(reply["data"], validate=True)
                if len(data) != length:
                    raise ValueError("truncated source chunk")
                f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if digest(partial) != row["sha256"]:
            raise ValueError("download SHA256 mismatch; source retained")
        partial.replace(out)
    # Re-read every local file, including any resumed files, before committing.
    for row in meta["files"]:
        out = files / row["path"]
        if out.stat().st_size != row["size"] or digest(out) != row["sha256"]:
            raise ValueError("local verification failed; source retained")
        fd = os.open(out.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    receipt = {**meta, "verified_local": True, "archive_path": str(target.resolve()),
               "verified_at": time.time(), "source_cleaned": False}
    durable_json(target / "manifest.json", meta)
    durable_json(target / "receipt.json", receipt)
    if cleanup:
        result = call("cleanup", name, sha, str(target.resolve()))
        receipt["cleanup"] = result
        receipt["source_cleaned"] = bool(result["cleaned"])
        durable_json(target / "receipt.json", receipt)
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--destination", type=Path, default=ROOT / "evidence/pod-archives")
    ap.add_argument("--state", type=Path, default=ROOT / "build/scratch/archive-maintenance/status.json")
    ap.add_argument("--cleanup", action="store_true", help="remove verified terminal run copies from Pod")
    ap.add_argument("--on-devbox", action="store_true", help="run on GPU development host, without another SSH hop")
    ap.add_argument("--watch", type=int, default=0, help="poll interval, at least 60 seconds; 0 runs once")
    args = ap.parse_args()
    call = partial(remote, on_devbox=args.on_devbox)
    if args.watch and args.watch < 60:
        ap.error("watch interval must be at least 60 seconds")
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with args.state.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            status = {"pid": os.getpid(), "started_at": time.time(), "cleanup_enabled": args.cleanup,
                      "destination": str(args.destination.resolve()), "runs": []}
            try:
                names = call("list")["runs"]
                for name in names:
                    status["active_run"] = name
                    durable_json(args.state, status)
                    try:
                        result = archive_one(name, args.destination, cleanup=args.cleanup, call=call)
                        status["runs"].append({"run": name, "archive": result["archive_path"],
                                               "cleaned": result["source_cleaned"]})
                    except (OSError, ValueError, RuntimeError) as exc:
                        status["runs"].append({"run": name, "deferred": str(exc)[:1600]})
                status["state"] = "checked"
            except (OSError, ValueError, RuntimeError) as exc:
                status.update(state="retrying", error=str(exc)[:1600])
            status.pop("active_run", None)
            status["finished_at"] = time.time()
            durable_json(args.state, status)
            if not args.watch:
                print(json.dumps(status, ensure_ascii=False))
                return
            time.sleep(args.watch)


if __name__ == "__main__":
    main()
