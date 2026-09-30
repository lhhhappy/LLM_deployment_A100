#!/usr/bin/env python3
"""Fetch closed fixed-window evidence with SHA256; never call the engine."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

from fetch_level import CHUNK, READ_CODE, marked, remote


PACK = r'''
import hashlib, json, pathlib, sys, tarfile
name, level, archive = sys.argv[1], int(sys.argv[2]), pathlib.Path(sys.argv[3])
ax = pathlib.Path('/tmp/ax')
assert (ax/'queue/done'/(name+'.sh')).exists(), 'job is not done'
root = ax/'runs'/name/('N'+str(level))
verdict = json.loads((root/'timed_verdict.json').read_text())
assert verdict['status'] == 'DRAINED' and verdict['n'] == level
raw = pathlib.Path(verdict['raw']).name
assert raw == verdict['raw'] and raw.startswith('raw_')
assert hashlib.sha256((root/raw).read_bytes()).hexdigest() == verdict['raw_sha256']
files = [root/raw, root/'timed_verdict.json', root/'timed_score.json',
         root/'timed_window.json', root/'summary.json', root/'flush_evidence.json']
summary = json.loads((root/'summary.json').read_text())
files.append(root/pathlib.Path(summary['run']).name)
before = {p: (p.stat().st_size,p.stat().st_mtime_ns) for p in files}
archive.parent.mkdir(parents=True,exist_ok=True)
with tarfile.open(archive,'w:gz') as tar:
    for p in files: tar.add(p,arcname=p.name,recursive=False)
assert all((p.stat().st_size,p.stat().st_mtime_ns)==stamp for p,stamp in before.items())
print('FETCH_META '+json.dumps(dict(size=archive.stat().st_size,
 sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run) or args.n < 1:
        raise ValueError('invalid run/N')
    archive = f'/tmp/ax/codex/execution-0930/{args.run}-N{args.n}.tgz'
    meta = json.loads(marked(remote(PACK, args.run, args.n, archive), 'FETCH_META '))
    payload = bytearray()
    for offset in range(0, meta['size'], CHUNK):
        block = base64.b64decode(marked(remote(READ_CODE, archive, offset, CHUNK),
                                       'FETCH_DATA '), validate=True)
        assert len(block) == min(CHUNK, meta['size']-offset)
        payload.extend(block)
    assert hashlib.sha256(payload).hexdigest() == meta['sha256']
    args.out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as tar:
        for member in tar.getmembers():
            assert member.isfile() and Path(member.name).name == member.name
            content = tar.extractfile(member).read()
            target = args.out/member.name
            if target.exists():
                existing = target.read_bytes()
                if member.name.endswith('.json'):
                    assert json.loads(existing) == json.loads(content), 'evidence differs: '+str(target)
                else:
                    assert existing == content, 'evidence differs: '+str(target)
            else:
                target.write_bytes(content)
    (args.out/'fetch-receipt.json').write_text(json.dumps(
        dict(run=args.run,n=args.n,archive=meta),indent=2)+'\n')
    print('FETCHED',args.run,'N'+str(args.n),meta['size'],'compressed bytes',flush=True)


if __name__ == '__main__':
    main()
