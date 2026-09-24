#!/usr/bin/env bash
# CPU-only data deployment. Does not enqueue a measurement or touch the engine.
set -euo pipefail
cd /sjtu/linhang/arena/repo
python3 - <<'PY'
from pathlib import Path
import hashlib, tarfile, time
p=Path('build/scratch/full-n30-transfer/full-19a7e5a6.tgz')
expected='265e327857d3d44930a55ee09578273379cae2f438ca0e9c4b56c5db9459e8b6'
for _ in range(90):
    if p.exists() and p.stat().st_size==8526417 and hashlib.sha256(p.read_bytes()).hexdigest()==expected:
        break
    time.sleep(10)
else: raise RuntimeError('full dataset transport not complete; nothing published')
out=Path('build/scratch/full-n30-transfer/extracted');out.mkdir(parents=True,exist_ok=True)
with tarfile.open(p) as t:
    for member in t.getmembers():
        path=Path(member.name)
        assert not path.is_absolute() and '..' not in path.parts and member.isfile(), member.name
    t.extractall(out)
print('GPU_ARCHIVE_VERIFIED',expected,flush=True)
PY
cd build/scratch/full-n30-transfer/extracted
/sjtu/linhang/arena/repo/scripts/pod/ppush /tmp/ax/staging/full-19a7e5a6 data transfer
cd /sjtu/linhang/arena/repo
source scripts/pod/common.sh
bexec 'python3 /tmp/ax/staging/full-19a7e5a6/transfer/full-19a7e5a6/restore.py /tmp/ax/staging/full-19a7e5a6'
