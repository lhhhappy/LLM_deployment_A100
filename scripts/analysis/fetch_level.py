#!/usr/bin/env python3
"""Fetch one finished dev measurement, verify it, then analyze its failures.

fetch_level.sh RUN N [REFERENCE_RAW]
Output: evidence/L<RUN>/N<N>/ (full run name, never the numeric prefix alone).
Only summary.json selects raw/run; preflight files and other levels are ignored.
Pod reads use pexec_codex; its archive writes stay under /tmp/ax/codex.
Exit 0=valid PASS, 1=valid FAIL, 2=invalid/transfer/analysis failure.
--archive rechecks an already exported archive locally without contacting a pod.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile

REPO = Path(__file__).resolve().parents[2]
CHUNK = 120000
OPTIONAL = ("run_dev.log", "loadgen.log", "score.log", "flush_evidence.json", "rundev_exit_code",
            "verdict_exit_code", "metrics.jsonl", "gpu_util.csv", "server.log", "job.log")

# Run via Python on the pod; no engine calls, no writes to runs/ or queue/.
# Core files must remain unchanged while packing; the live server log may grow.
PACK_CODE = r'''
import hashlib, json, pathlib, sys, tarfile
root, n, archive = pathlib.Path(sys.argv[1]), int(sys.argv[2]), pathlib.Path(sys.argv[3])
level = root / ("N" + str(n))
summary = json.loads((level / "summary.json").read_text())
if summary.get("n") != n:
    raise ValueError("summary N mismatch")
files = {"summary.json": level / "summary.json"}
for key in ("raw", "run"):
    value = summary.get(key)
    if not isinstance(value, str) or not value or value.endswith("/"):
        raise ValueError("missing selected " + key)
    name = pathlib.Path(value).name
    if name in ("", ".", "..") or name in files:
        raise ValueError("invalid/duplicate selected filename")
    files[name] = level / name
metadata = json.loads(files[pathlib.Path(summary["run"]).name].read_text())
if metadata.get("config", {}).get("N") != n:
    raise ValueError("run N mismatch")
core = {name: (path.stat().st_size, path.stat().st_mtime_ns) for name, path in files.items()}
for name in ("run_dev.log", "loadgen.log", "score.log", "flush_evidence.json", "rundev_exit_code",
             "verdict_exit_code", "metrics.jsonl", "gpu_util.csv"):
    if (level / name).is_file(): files[name] = level / name
for name in ("server.log", "job.log"):
    if (root / name).is_file(): files[name] = root / name
archive.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(archive, "w:gz", dereference=True) as tar:
    for name, path in files.items(): tar.add(path, arcname=name, recursive=False)
if any((files[name].stat().st_size, files[name].stat().st_mtime_ns) != stamp for name, stamp in core.items()):
    raise ValueError("measurement files changed during export")
digest = hashlib.sha256()
with archive.open("rb") as f:
    for block in iter(lambda: f.read(1048576), b""): digest.update(block)
print("FETCH_META " + json.dumps({"size": archive.stat().st_size, "sha256": digest.hexdigest()}))
'''

READ_CODE = r'''
import base64, sys
with open(sys.argv[1], "rb") as f:
    f.seek(int(sys.argv[2])); data = f.read(int(sys.argv[3]))
print("FETCH_DATA " + base64.b64encode(data).decode("ascii"))
'''


def level_module():
    spec = importlib.util.spec_from_file_location("fetch_level_verdict", REPO / "scripts/pod/verify/level_verdict.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def destination(run, n, evidence):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run) or n < 1:
        raise ValueError("run must be a directory name; N must be positive")
    return evidence / ("L" + run) / ("N" + str(n))


def remote(code, *args):
    command = shlex.join(["python3", "-c", code, *map(str, args)])
    gpu_command = "cd /sjtu/linhang/arena/repo && " + shlex.join(["scripts/pod/pexec_codex", command])
    result = subprocess.run([str(REPO / "scripts/gssh"), gpu_command], cwd=REPO,
                            env={**os.environ, "GSSH_TIMEOUT": "600"},
                            stdout=subprocess.PIPE, text=True, check=True)
    return result.stdout


def marked(output, prefix):
    lines = [line[len(prefix):] for line in output.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise ValueError(f"missing/ambiguous remote {prefix.strip()} response")
    return lines[0]


def download(run, n, target):
    archive = f"/tmp/ax/codex/fetch_{run}_N{n}.tgz"
    meta = json.loads(marked(remote(PACK_CODE, f"/tmp/ax/runs/{run}", n, archive), "FETCH_META "))
    size, sha = meta["size"], meta["sha256"]
    if not isinstance(size, int) or size <= 0 or not re.fullmatch("[0-9a-f]{64}", sha):
        raise ValueError("invalid archive size/hash")
    digest = hashlib.sha256()
    with target.open("wb") as out:
        for offset in range(0, size, CHUNK):
            payload = marked(remote(READ_CODE, archive, offset, CHUNK), "FETCH_DATA ")
            data = base64.b64decode(payload, validate=True)
            if len(data) != min(CHUNK, size - offset):
                raise ValueError("truncated archive chunk")
            digest.update(data)
            out.write(data)
    if target.stat().st_size != size or digest.hexdigest() != sha:
        raise ValueError("archive size/SHA256 verification failed")
    return {"size": size, "sha256": sha, "source": "pod"}


def extract(archive, staging):
    """Allow only unique flat regular files; no links or path traversal."""
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        names = set()
        for member in members:
            if (not member.isfile() or Path(member.name).name != member.name or
                    member.name in ("", ".", "..") or member.name in names):
                raise ValueError("archive contains unsafe/duplicate/non-file entries")
            names.add(member.name)
        for member in members:
            with tar.extractfile(member) as src, (staging / member.name).open("wb") as dst:
                while block := src.read(1048576):
                    dst.write(block)


def publish(staging, out, n):
    raw, run, _, _ = level_module().selected_files(staging, n)
    # A run/N identifies one measurement. Do not overwrite different raw evidence
    # under an already used name, even if a remote directory was reused.
    for name in ("summary.json", raw.name, run.name):
        existing = out / name
        if existing.is_file() and existing.read_bytes() != (staging / name).read_bytes():
            raise ValueError(f"existing measurement differs: {name}; use a new run directory")
    fetched = {path.name for path in staging.iterdir()}
    out.mkdir(parents=True, exist_ok=True)
    # An absent input in this snapshot must not reuse an older receipt/log.
    for name in OPTIONAL:
        if name not in fetched:
            (out / name).unlink(missing_ok=True)
    for path in staging.iterdir():
        os.replace(path, out / path.name)
    return raw.name


def logged(command, path):
    with path.open("w") as handle:
        result = subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
    print(path.read_text(), end="", flush=True)
    return result.returncode


def analyze(out, n, reference=None, data_root=None, harness_dir=None):
    data_root = Path(data_root or REPO / "s1-dev/data/dev-combined-v1").resolve()
    harness_dir = Path(harness_dir or REPO / "s1-dev/harness").resolve()
    status = logged([sys.executable, "-B", str(REPO / "scripts/pod/verify/level_verdict.py"), str(out), str(n),
                     "--harness-dir", str(harness_dir),
                     "--data-root", str(data_root)], out / "verdict.txt")
    if status not in (0, 1):
        for name in ("badcases.csv", "badcase.txt"):
            (out / name).unlink(missing_ok=True)
        return 2
    raw, _, _, _ = level_module().selected_files(out, n)
    command = [sys.executable, "-B", str(REPO / "scripts/analysis/badcase.py"), str(raw),
               str(out / "server.log"), "--csv", str(out / "badcases.csv"), "--harness-dir", str(harness_dir)]
    if reference:
        command += ["--ref", str(reference)]
    if logged(command, out / "badcase.txt") != 0:
        print("ANALYSIS_FAILED: verdict remains available; bad-case output is incomplete", file=sys.stderr)
        (out / "badcases.csv").unlink(missing_ok=True)
        return 2
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("n", type=int)
    parser.add_argument("reference", nargs="?", type=Path)
    parser.add_argument("--archive", type=Path, help="recheck a previously exported archive; no remote access")
    parser.add_argument("--evidence-root", type=Path, default=REPO / "evidence")
    parser.add_argument("--data-root", type=Path, default=REPO / "s1-dev/data/dev-combined-v1")
    parser.add_argument("--harness-dir", type=Path, default=REPO / "s1-dev/harness")
    args = parser.parse_args(argv)
    out = None
    try:
        out = destination(args.run, args.n, args.evidence_root.resolve())
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".fetch-", dir=out.parent) as work:
            work = Path(work)
            if args.archive:
                archive = args.archive.resolve()
                transfer = {"source": str(archive), "size": archive.stat().st_size,
                            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
            else:
                archive = work / "level.tgz"
                transfer = download(args.run, args.n, archive)
            staging = work / "files"
            staging.mkdir()
            extract(archive, staging)
            fetched = sorted(p.name for p in staging.iterdir())
            raw = publish(staging, out, args.n)
            transfer.update({"run": args.run, "n": args.n, "raw": raw, "files": fetched})
        status = analyze(out, args.n, args.reference.resolve() if args.reference else None,
                         args.data_root, args.harness_dir)
        transfer.update(data_root=str(args.data_root.resolve()), harness_dir=str(args.harness_dir.resolve()))
        transfer["exit_code"] = status
        (out / "fetch_status.json").write_text(json.dumps(transfer, indent=2) + "\n")
        print(f"EVIDENCE {out} exit_code={status}")
        return status
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError, tarfile.TarError) as error:
        print(f"FETCH_FAILED: {error}", file=sys.stderr)
        if out is not None:
            level_module().fail_invalid(out, args.n, [f"fetch failed: {error}"])
            for name in ("verdict.txt", "badcase.txt", "badcases.csv"):
                (out / name).unlink(missing_ok=True)
            (out / "fetch_status.json").write_text(json.dumps({"run": args.run, "n": args.n,
                "exit_code": 2, "error": str(error)}, indent=2) + "\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
