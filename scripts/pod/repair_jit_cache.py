#!/usr/bin/env python3
"""Recover published JIT leaves affected by the symlink staging-path bug.

No engine imports or compilation. Revalidate every header, and publish a new
immutable leaf with the identical .so. Never edit or delete an existing leaf.
Use only before measurement; canonical cache paths prevent future occurrences.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

_output_bytes = 0


def emit(row):
    global _output_bytes
    body = json.dumps(row)
    _output_bytes += len(body.encode()) + 1
    assert _output_bytes <= 128*1024, 'total output budget exceeded'
    print(body, flush=True)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def deps_key(entries):
    h = hashlib.sha256()
    for entry in entries:
        h.update(repr(tuple(entry)).encode())
        h.update(b'\0')
    return h.hexdigest()[:16]


def repair_leaf(leaf, anchors, apply=False):
    leaf = leaf.resolve(strict=True)
    scope = leaf.parent
    assert re.fullmatch(r'build-[0-9a-f]{16}', scope.name)
    entries = json.loads((leaf/'sgl_deps.json').read_text())
    assert entries and len(entries) < 10000
    assert leaf.name == 'deps-' + deps_key(entries), 'original manifest hash mismatch'
    retained, removed = [], []
    for root, relative, expected in entries:
        assert re.fullmatch(r'[0-9a-f]{64}', expected)
        if root == 'abs':
            path = Path(relative)
            assert path.is_absolute()
        else:
            base = anchors[root].resolve(strict=True)
            path = (base/relative).resolve(strict=True)
            assert path.is_relative_to(base), 'dependency escapes anchor'
        # Only generated translation units in this exact scope can be dropped.
        generated = (root == 'abs' and path.parent.parent == scope
                     and re.fullmatch(r'\.staging-[0-9a-f]{32}', path.parent.name)
                     and path.name in {'cuda.cu', 'cpp.cpp'})
        if generated:
            assert not path.exists(), 'staging still exists; leave live builds alone'
            assert digest(leaf/path.name) == expected, 'published wrapper changed'
            removed.append(relative)
        else:
            assert digest(path) == expected, 'dependency changed: ' + str(path)
            retained.append([root, relative, expected])
    if not removed:
        return None
    assert retained, 'cannot publish a leaf without real dependencies'
    module = scope.parent.name
    binary = leaf/(module+'.so')
    assert binary.is_file() and not binary.is_symlink()
    binary_hash = digest(binary)
    target = scope/('deps-'+deps_key(retained))
    row = dict(source=str(leaf), target=str(target), removed=removed,
               retained_dependencies=len(retained), binary_sha256=binary_hash,
               status='dry_run')
    if target.exists():
        assert json.loads((target/'sgl_deps.json').read_text()) == retained
        # A previous recovery can come from a different rank's identical input
        # build; it remains self-validating. Do not overwrite that published leaf.
        assert (target/(module+'.so')).is_file()
        row['status'] = 'already_published'
    elif apply:
        temp = Path(tempfile.mkdtemp(prefix='.repair-', dir=scope))
        try:
            os.link(binary, temp/binary.name)
            (temp/'sgl_deps.json').write_text(json.dumps(retained,separators=(',',':')))
            assert digest(temp/binary.name) == binary_hash
            os.rename(temp, target)
        finally:
            if temp.exists():
                shutil.rmtree(temp)
        row['status'] = 'published'
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--engine-pid', type=int, required=True)
    parser.add_argument('--engine-start-ticks', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--watch-seconds', type=int, default=0)
    args = parser.parse_args()
    assert 0 <= args.watch_seconds <= 1800
    assert re.fullmatch(r'[0-9a-f]{40}', args.commit)
    ax = Path('/tmp/ax').resolve(strict=True)
    assert ax == Path('/dev/shm/arena-runtime/ax')
    cache = args.cache_root.resolve(strict=True)
    assert cache == ax/'cache/sglang/jit'
    run = args.run_dir.resolve(strict=True)
    assert run.parent == ax/'runs'
    source = ax/'src'/args.commit
    assert (source/'COMMIT').read_text().strip() == args.commit
    site = Path('/opt/sglang/lib/python3.12/site-packages')
    anchors = {'kernels': source/'sglang/kernels/jit', 'toolkit': Path('/usr/local/cuda'),
               'sys': Path('/usr'), 'sitepkgs': site, 'tvm_ffi': site/'tvm_ffi',
               'pkg:flashinfer': site/'flashinfer', 'pkg:deep_gemm': site/'deep_gemm',
               'pkg:nvidia': site/'nvidia'}
    deadline = time.monotonic() + args.watch_seconds
    seen, count = set(), 0
    while True:
        stat = (Path('/proc')/str(args.engine_pid)/'stat').read_text().rsplit(')',1)[1].split()
        assert stat[0] != 'Z' and stat[19] == args.engine_start_ticks, 'engine identity changed'
        # The queue starts after engine readiness. Stop before any replay work.
        if (ax/'engine.sig').exists() or (run/'job.log').exists():
            emit(dict(status='stopped_before_replay',published=count))
            break
        for scope in sorted((cache/'sm80').glob('sgl_kernel_jit_*/build-*')):
            if scope in seen:
                continue
            leaves = sorted(scope.glob('deps-*'), key=lambda p:p.stat().st_mtime, reverse=True)
            if not leaves:
                continue
            try:
                row = repair_leaf(leaves[0], anchors, apply=args.apply)
                seen.add(scope)
                if row:
                    emit(row)
                    count += row['status'] == 'published'
            except (AssertionError, OSError, KeyError, ValueError) as exc:
                # Reject rather than guess when the defect is not exactly ours.
                seen.add(scope)
                emit(dict(scope=str(scope),status='rejected',reason=str(exc)[:300]))
            assert len(seen) <= 128, 'scope/output budget exceeded'
        if time.monotonic() >= deadline:
            emit(dict(status='finished',published=count))
            break
        time.sleep(min(15, max(0, deadline-time.monotonic())))


if __name__ == '__main__':
    main()
