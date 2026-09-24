#!/usr/bin/env python3
"""Opt-in fresh-pod workspace on tmpfs; no service lifecycle or file deletion.

AX_WORKSPACE_ROOT=/dev/shm/arena-runtime relocates the existing /tmp/ax API.
Refuse nonempty workspaces, foreign links, non-tmpfs targets and low headroom.
Compiled extension caches stay on their original executable filesystem.
"""
import argparse
import json
import os
from pathlib import Path
import time

GIB = 1024**3


def mount_for(path, mountinfo):
    candidates = []
    for line in mountinfo.splitlines():
        left, right = line.split(" - ", 1)
        fields = left.split()
        mount = Path(fields[4].replace("\\040", " "))
        if path == mount or mount in path.parents:
            candidates.append((len(mount.parts), str(mount), right.split()[0], fields[5]))
    if not candidates:
        raise ValueError("mount not found")
    _, mount, fs, options = max(candidates)
    return dict(mount=mount, filesystem=fs, options=options)


def memory_headroom():
    root = Path('/sys/fs/cgroup')
    for limit, used in [(root/'memory.max', root/'memory.current'),
                        (root/'memory/memory.limit_in_bytes', root/'memory/memory.usage_in_bytes')]:
        if limit.is_file() and used.is_file():
            value = limit.read_text().strip()
            if value != 'max':
                return max(0, int(value)-int(used.read_text()))
    raise ValueError('finite cgroup memory limit/current unavailable')


def prepare(root, ax, *, mountinfo, headroom, minimum=64*GIB, apply=False):
    root, ax = Path(root), Path(ax)
    if not root.is_absolute() or '..' in root.parts:
        raise ValueError('workspace root must be an absolute path without ..')
    # Resolve existing ancestors, so a redirect through a foreign symlink fails.
    resolved = root.resolve()
    mount = mount_for(resolved, mountinfo)
    if mount['filesystem'] != 'tmpfs':
        raise ValueError('workspace root is not on tmpfs')
    existing = resolved
    while not existing.exists():
        existing = existing.parent
    free = os.statvfs(existing).f_bavail * os.statvfs(existing).f_frsize
    if min(free, headroom) < minimum:
        raise ValueError('insufficient tmpfs or cgroup memory headroom')
    target = resolved/'ax'
    linked = ax.is_symlink()
    if target.is_symlink():
        raise ValueError('foreign target symlink; no binding')
    if linked and ax.resolve() != target:
        raise ValueError('foreign workspace symlink; no replacement')
    if not linked and target.exists():
        if not target.is_dir() or any(not p.is_dir() or p.is_symlink() for p in target.rglob('*')):
            raise ValueError('nonempty target workspace; no binding')
    if not linked and ax.exists():
        if not ax.is_dir() or any(not p.is_dir() or p.is_symlink() for p in ax.rglob('*')):
            raise ValueError('nonempty workspace; archive/migrate separately')
    receipt = dict(root=str(resolved), compatibility_path=str(ax), target=str(target),
                   mount=mount, tmpfs_free_bytes=free, cgroup_headroom_bytes=headroom,
                   minimum_free_bytes=minimum, already_linked=linked, applied=False,
                   persistence='RAM only; export evidence externally',
                   jit_caches_relocated=False)
    if apply and not linked:
        target.mkdir(parents=True, exist_ok=True)
        if ax.exists():
            backup = ax.with_name(ax.name+'.empty-before-tmpfs-'+str(time.time_ns()))
            ax.rename(backup)  # preserve even empty directories; never delete
            receipt['empty_directory_backup'] = str(backup)
        ax.parent.mkdir(parents=True, exist_ok=True)
        ax.symlink_to(target, target_is_directory=True)
    receipt['applied'] = apply
    if apply:
        if ax.resolve() != target or target.is_symlink():
            raise ValueError('workspace path changed during setup')
        (target/'workspace-layout.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = os.environ.get('AX_WORKSPACE_ROOT')
    if not root:
        print('WORKSPACE_LAYOUT legacy; AX_WORKSPACE_ROOT unset')
        return
    if Path('/dev/shm') not in Path(root).parents:
        raise ValueError('AX_WORKSPACE_ROOT must be below /dev/shm')
    result = prepare(root, '/tmp/ax', mountinfo=Path('/proc/self/mountinfo').read_text(),
                     headroom=memory_headroom(), apply=args.apply)
    print('WORKSPACE_LAYOUT '+json.dumps(result))


if __name__ == '__main__':
    main()
