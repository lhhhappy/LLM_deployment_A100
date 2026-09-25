#!/usr/bin/env python3
"""One-shot compiler progress evidence; read only, no GPU API or stack attach."""
import json
import time
from pathlib import Path

rows = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        comm = (proc/'comm').read_text().strip()
        if comm not in {'cicc', 'ptxas', 'nvcc', 'ninja'}:
            continue
        stat = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        cwd = (proc/'cwd').resolve()
        row = dict(pid=int(proc.name), parent=int(stat[1]), command=comm,
                   state=stat[0], cpu_ticks=int(stat[11])+int(stat[12]),
                   start_ticks=int(stat[19]), cwd=str(cwd),
                   argv=(proc/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')[:4096])
        if cwd.is_relative_to(Path('/dev/shm/arena-runtime')):
            ninja = cwd/'build.ninja'
            if ninja.is_file():
                with ninja.open() as f:
                    row['build_ninja_head'] = f.read(2048)
        rows.append(row)
        if len(rows) >= 8:
            break
    except (OSError, ValueError):
        pass
cache = []
module = Path('/tmp/ax/cache/sglang/jit/sm80/sgl_kernel_jit_gptq_marlin_bf16_t')
if module.is_dir():
    for scope in sorted(module.iterdir())[:8]:
        if not scope.is_dir():
            continue
        for leaf in sorted(scope.iterdir())[:10]:
            if not leaf.is_dir():
                continue
            row = dict(path=str(leaf), mtime=leaf.stat().st_mtime)
            row['files'] = [dict(name=p.name, size=p.stat().st_size, mtime=p.stat().st_mtime)
                            for p in sorted(leaf.iterdir())[:12] if p.is_file()]
            deps = leaf/'sgl_deps.json'
            if deps.is_file():
                with deps.open() as f:
                    row['deps_head'] = f.read(2048)
            cache.append(row)
            if len(cache) >= 10:
                break
        if len(cache) >= 10:
            break
result = dict(observed_at=time.time(), compilers=rows, marlin_cache=cache)
body = json.dumps(result, indent=2)
if len(body.encode()) > 65536:
    for row in rows:
        row.pop('build_ninja_head', None)
    for row in cache:
        row.pop('deps_head', None)
    result['detail_omitted_for_output_budget'] = True
    body = json.dumps(result, indent=2)
assert len(body.encode()) <= 65536, 'compile snapshot exceeded 64 KiB'
print(body)
