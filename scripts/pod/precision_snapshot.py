#!/usr/bin/env python3
"""Bounded source/cache inspection; never imports torch or invokes the GPU."""
import json
import os
from pathlib import Path
import re
import time

base = Path('/opt/sglang/lib/python3.12/site-packages/torch')
source = base/'_inductor/compile_fx.py'
lines = source.read_text().splitlines()
rows = []
for i, line in enumerate(lines):
    if 'TensorFloat32 tensor cores' in line:
        rows.append(dict(path=str(source),first_line=max(0,i-24)+1,
                         source='\n'.join(lines[max(0,i-24):i+12])))
graphs, visited, examined = [], 0, 0
cache = Path('/dev/shm/arena-runtime/ax/cache/inductor')
begin = time.monotonic()
for root, dirs, files in os.walk(cache):
    visited += 1
    if visited > 256 or time.monotonic()-begin > 3 or len(graphs) >= 12:
        break
    for name in files:
        if not name.endswith('.py'):
            continue
        p = Path(root)/name
        if p.stat().st_size > 256*1024:
            continue
        examined += 1
        content = p.read_text(errors='replace').splitlines()
        if not any('extern_kernels.' in line and any(op in line for op in ('.mm(','.bmm(','.addmm(')) for line in content):
            continue
        kept=[]
        for i,line in enumerate(content):
            if any(x in line for x in ('# Topologically','# Original ATen','extern_kernels.','assert_size_stride(', 'torch.float32','torch.bfloat16','def call(','arg0_1,','ALLOW_TF32')):
                kept.append(f'{i+1}: {line[:400]}')
        graphs.append(dict(path=str(p),selected_lines=kept[:60]))
environments = {}
pidfile = Path('/tmp/ax/runs/071-official_b_host64_full_n30_shortwarm/server.pid')
for label, pid in [('image_process',1),('engine',int(pidfile.read_text()))]:
    env = dict(s.decode().split('=',1) for s in (Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0') if b'=' in s)
    environments[label] = {k:v for k,v in env.items() if re.match(r'^(SGLANG_AX_|SGLANG_ARENA_|NCCL_|SGLANG_MAMBA|SGLANG_OPT_|TORCH_ALLOW_TF32|NVIDIA_TF32)',k)}
result = dict(observed_at=time.time(),warning_source=rows,graphs=graphs,environments=environments,
              graph_files_examined=examined,scan_seconds=time.monotonic()-begin,
              scope='bounded cache sample; absence is not proof no FP32 matmul')
body=json.dumps(result,indent=2)
assert len(body.encode()) <= 65536
print(body)
