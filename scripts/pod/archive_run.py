#!/usr/bin/env python3
"""Pod side of verified off-pod archival; no temporary archive is created here.

Only terminal runs are eligible. Active engine logs, open files/cwds, incoming
symlinks, non-regular files and young runs block cleanup. Every file, including
dereferenced log aliases, is hashed again before deletion. The caller must have
durably verified the same manifest on local storage before requesting cleanup.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import time

NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
CHUNK = 120000


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def under(path, root):
    return path == root or root in path.parents


def root_for(ax, name):
    if not NAME.fullmatch(name) or name in (".", ".."):
        raise ValueError("invalid run name")
    root = ax / "runs" / name
    if root.is_symlink() or root.parent.is_symlink():
        raise ValueError("run root must not be a symlink")
    return root


def terminal(ax, name):
    states = [s for s in ("pending", "running", "done", "failed", "cancelled")
              if (ax / "queue" / s / (name + ".sh")).exists()]
    return len(states) == 1 and states[0] in ("done", "failed")


def process_paths(proc=Path("/proc")):
    """Conservatively protect all open paths, including reader fds and cwd."""
    paths = set()
    for pid in proc.iterdir():
        if not pid.name.isdigit() or pid.name == str(os.getpid()):
            continue
        try:
            candidates = [pid / "cwd", *(pid / "fd").iterdir()]
        except FileNotFoundError:
            continue
        except PermissionError as exc:
            raise ValueError("cannot inspect process fds; refuse cleanup") from exc
        for link in candidates:
            try:
                target = os.readlink(link)
            except FileNotFoundError:
                continue
            except PermissionError as exc:
                raise ValueError("cannot inspect process fd; refuse cleanup") from exc
            if target.startswith("/"):
                paths.add(Path(target.removesuffix(" (deleted)")).resolve())
    return paths


def walk_files(root):
    for current, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            if (Path(current) / name).is_symlink():
                raise ValueError("directory symlink is not supported")
        for name in sorted(files):
            path = Path(current) / name
            if not path.is_file() and not path.is_symlink():
                raise ValueError("non-regular/dangling entry: " + str(path))
            yield path


def guard(ax, name, *, grace=120, opened=None, now=None):
    root = root_for(ax, name)
    if not root.is_dir() or not terminal(ax, name) or not (root / "exit_code").is_file():
        raise ValueError("run is not terminal with an exit receipt")
    now = time.time() if now is None else now
    if now - (root / "exit_code").stat().st_mtime < grace:
        raise ValueError("terminal run is still inside grace period")
    real = root.resolve()
    opened = process_paths() if opened is None else opened
    if any(under(path, real) for path in opened):
        raise ValueError("run has open file descriptors or a process cwd")
    pointer = ax / "engine_log_path"
    active = Path(pointer.read_text().strip()).resolve() if pointer.is_file() else None
    if active is not None and under(active, real):
        raise ValueError("run owns the current engine log")
    # Log aliases must be included as content, and their source must also be closed.
    for path in walk_files(root):
        target = path.resolve()
        if path.is_symlink():
            if not under(target, (ax / "runs").resolve()):
                raise ValueError("log alias points outside runs")
            if target == active or target in opened:
                raise ValueError("run aliases a live/open log")
    # Delete aliases before their owner, so remaining evidence never dangles.
    runs = ax / "runs"
    for other in runs.iterdir():
        if other == root or not other.is_dir() or other.is_symlink():
            continue
        for current, dirs, files in os.walk(other, followlinks=False):
            for entry in dirs + files:
                path = Path(current) / entry
                if path.is_symlink() and under(path.resolve(), real):
                    raise ValueError("another run still references this run")
    return root


def inventory(root, original_root=None):
    files = []
    for path in walk_files(root):
        source = path
        if original_root is not None and path.is_symlink():
            original = original_root / path.relative_to(root)
            source = (original.parent / os.readlink(path)).resolve()
            if under(source, original_root.resolve()):
                source = root / source.relative_to(original_root.resolve())
        before = source.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("non-regular alias target")
        sha = digest_file(source)
        after = source.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise ValueError("file changed during hashing")
        files.append({"path": path.relative_to(root).as_posix(), "size": after.st_size,
                      "mtime_ns": after.st_mtime_ns, "sha256": sha,
                      "symlink": os.readlink(path) if path.is_symlink() else None})
    return sorted(files, key=lambda row: row["path"])


def manifest(ax, name, **kwargs):
    root = guard(ax, name, **kwargs)
    rows = inventory(root)
    payload = {"run": name, "files": rows}
    sha = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**payload, "manifest_sha256": sha, "total_bytes": sum(r["size"] for r in rows)}


def read_chunk(ax, name, relative, offset, length):
    root = root_for(ax, name)
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or length < 1 or length > CHUNK or offset < 0:
        raise ValueError("invalid chunk request")
    path = root / rel
    if not under(path.resolve(), (ax / "runs").resolve()) or not path.is_file():
        raise ValueError("read target outside runs or not a regular file")
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read(length)
    return {"data": base64.b64encode(data).decode()}


def cleanup(ax, name, expected_sha, archive_location, **kwargs):
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ValueError("invalid verified manifest hash")
    receipt_path = ax / "archive-receipts" / (name + ".json")
    root = root_for(ax, name)
    if not root.exists() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("manifest_sha256") == expected_sha and receipt.get("cleaned"):
            return receipt
        raise ValueError("missing run without matching completed cleanup receipt")
    current = manifest(ax, name, **kwargs)
    if current["manifest_sha256"] != expected_sha:
        raise ValueError("run changed since local archive verification")
    # Repeat guards after hashing, then detach only this exact terminal directory.
    guard(ax, name, **kwargs)
    trash_dir = ax / "codex" / "archive-trash"
    trash_dir.mkdir(parents=True, exist_ok=True)
    if trash_dir.is_symlink() or not under(trash_dir.resolve(), ax.resolve()):
        raise ValueError("cleanup staging must stay inside the workspace")
    trash = trash_dir / (name + "-" + expected_sha)
    if trash.exists() or trash.is_symlink():
        raise ValueError("previous interrupted cleanup requires inspection")
    root.rename(trash)
    try:
        if inventory(trash, original_root=root) != current["files"]:
            raise ValueError("run changed during detach")
    except (OSError, ValueError):
        # No evidence is deleted if a writer changed it during detach, and
        # relative log aliases are resolved against their original location.
        trash.rename(root)
        raise
    shutil.rmtree(trash)
    receipt = {"run": name, "manifest_sha256": expected_sha, "cleaned": True,
               "archive_location": archive_location, "logical_bytes": current["total_bytes"],
               "cleaned_at": time.time()}
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = receipt_path.with_suffix(".part")
    tmp.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    tmp.replace(receipt_path)
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("list", "manifest", "read", "cleanup"))
    ap.add_argument("run", nargs="?")
    ap.add_argument("extra", nargs="*")
    ap.add_argument("--ax", type=Path, default=Path("/tmp/ax"))
    args = ap.parse_args()
    ax = args.ax
    try:
        if args.action == "list":
            if not (ax / "runs").is_dir():
                raise ValueError("workspace is not bootstrapped")
            result = {"runs": sorted((p.name for p in (ax / "runs").iterdir()
                                      if p.is_dir() and terminal(ax, p.name)), reverse=True)}
        elif args.action == "manifest":
            result = manifest(ax, args.run)
        elif args.action == "read":
            result = read_chunk(ax, args.run, args.extra[0], int(args.extra[1]), int(args.extra[2]))
        else:
            result = cleanup(ax, args.run, args.extra[0], args.extra[1])
        print("ARCHIVE_RESULT " + json.dumps({"ok": True, **result}, separators=(",", ":")))
    except (OSError, ValueError) as exc:
        print("ARCHIVE_RESULT " + json.dumps({"ok": False, "error": str(exc)}))


if __name__ == "__main__":
    main()
