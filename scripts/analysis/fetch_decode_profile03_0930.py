#!/usr/bin/env python3
"""Bounded read-only profile export; existing fetch_level transport and chunks."""
import base64
import concurrent.futures
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("fetch_level", ROOT / "scripts/analysis/fetch_level.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)
out = ROOT / "evidence/execution-0930/pod/decode-profile"
receipt = json.loads((out / "receipt.json").read_text())
remote_root = "/tmp/ax/codex/profiles/130ezo03-20260930T023613Z"
archive = "/tmp/ax/codex/profile03-reviewed.tgz"
pack = r'''
import hashlib,json,pathlib,sys,tarfile
root,archive=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]); expected=json.loads(sys.argv[3])
files=[root/'profile-analysis.json']
checks=[]
for e in expected:
 p=root/'traces'/e['name']; b=p.read_bytes(); s=hashlib.sha256(b).hexdigest()
 assert len(b)==e['bytes'] and s==e['sha256'],e['name']
 files.append(p);checks.append(dict(name=p.name,bytes=len(b),sha256=s))
assert sum(p.stat().st_size for p in files)<8*1024*1024
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=p.name,recursive=False)
b=archive.read_bytes();print('FETCH_META '+json.dumps(dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),trace_checks=checks,source_files=[dict(name=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])))
'''
meta = json.loads(fetch.marked(fetch.remote(pack, remote_root, archive, json.dumps(receipt['traces'])), "FETCH_META "))
assert len(meta['trace_checks']) == 8
print(json.dumps(dict(stage="packed", bytes=meta['bytes'], all8_remote_hashes_verified=True)), flush=True)
size=meta['bytes']
def read(offset):
    payload=fetch.marked(fetch.remote(fetch.READ_CODE,archive,offset,fetch.CHUNK),'FETCH_DATA ')
    b=base64.b64decode(payload,validate=True)
    assert len(b)==min(fetch.CHUNK,size-offset)
    return offset,b
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    chunks=dict(pool.map(read,range(0,size,fetch.CHUNK)))
b=b''.join(chunks[o] for o in sorted(chunks))
assert len(b)==size and hashlib.sha256(b).hexdigest()==meta['sha256']
target=out/'profile03-reviewed.tgz';target.write_bytes(b)
with tarfile.open(target,'r:gz') as t:
    for m in t.getmembers():
        assert m.isfile() and Path(m.name).name==m.name
        dst=out/('traces' if m.name.endswith('.json.gz') else '')/m.name
        dst.parent.mkdir(parents=True,exist_ok=True)
        dst.write_bytes(t.extractfile(m).read())
for e in receipt['traces']:
    b=(out/'traces'/e['name']).read_bytes()
    assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256']
meta.update(all8_local_hashes_verified=True,archive=str(target.relative_to(ROOT)),remote_root=remote_root,transport='fetch_level.remote / READ_CODE 120000-byte SHA-checked chunks',read_only_scope='Only bounded archive under /tmp/ax/codex written; no API/service/GPU/queue action')
(out/'fetch-receipt.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(dict(stage='complete',all8_local_hashes_verified=True,bytes=size)),flush=True)
