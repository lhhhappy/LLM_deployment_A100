"""Lossless transport compression; the frozen dataset bytes remain unchanged."""
from pathlib import Path
import gzip
import hashlib
import json
import lzma
import shutil
import struct
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
data = ROOT / 'data/s1-dev-longchain'
stage = ROOT / 'build/scratch/full-n30-transfer/staging'
target = stage / 'data/s1-dev-longchain'
transport = stage / 'transfer/full-19a7e5a6'
target.mkdir(parents=True, exist_ok=True)
transport.mkdir(parents=True, exist_ok=True)


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


manifest = json.loads((data/'manifest.json').read_text())
assert sha(data/'manifest.json') == '19a7e5a6827f64a99695cba2d89b7efa2a0b05d207fc95da1568ec1d82280b2c'
for name, digest in manifest['artifacts'].items():
    assert sha(data/name) == digest, name
body = 'bodies/s1-dev-longchain.jsonl.gz'
for name in ['manifest.json', *manifest['artifacts']]:
    if name != body:
        (target/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(data/name, target/name)

with (data/body).open('rb') as f:
    header = f.read(10)
    assert header[:4] == b'\x1f\x8b\x08\x08' and header[8:] == b'\x04\xff'
    filename = bytearray()
    while (b := f.read(1)) != b'\x00':
        assert b, 'truncated gzip header'
        filename += b
mtime = struct.unpack('<I', header[4:8])[0]


class HashSink:
    def __init__(self):
        self.hash = hashlib.sha256()
        self.size = 0

    def flush(self):
        pass

    def write(self, b):
        self.hash.update(b)
        self.size += len(b)
        return len(b)


sink = HashSink()
count = 0
with gzip.open(data/body, 'rb') as source, lzma.open(transport/'body.xz', 'wb', preset=6) as out:
    with gzip.GzipFile(filename=filename.decode('latin1'), fileobj=sink,
                       mode='wb', compresslevel=1, mtime=mtime) as proof:
        for block in iter(lambda: source.read(4 << 20), b''):
            out.write(block)
            proof.write(block)
            count += len(block)
            if count % (512 << 20) == 0:
                print('ENCODED_RAW_BYTES', count, flush=True)
        proof.flush()  # Original TextIOWrapper closes with Z_SYNC_FLUSH before gzip finish.
assert sink.hash.hexdigest() == manifest['artifacts'][body], 'local gzip reconstruction differs'
meta = dict(body=body, filename=filename.decode('latin1'), mtime=mtime,
            compresslevel=1, sha256=sink.hash.hexdigest(), size=sink.size,
            raw_bytes=count, xz_bytes=(transport/'body.xz').stat().st_size)
(transport/'transport.json').write_text(json.dumps(meta, indent=2)+'\n')
shutil.copyfile(Path(__file__).with_name('restore_transport.py'), transport/'restore.py')
archive = stage.parent/'full-19a7e5a6.tgz'
with tarfile.open(archive, 'w:gz') as tar:
    for p in sorted(stage.rglob('*')):
        if p.is_file():
            tar.add(p, arcname=str(p.relative_to(stage)), recursive=False)
meta.update(archive_bytes=archive.stat().st_size, archive_sha256=sha(archive),
            artifact_hashes=manifest['artifacts'])
Path(__file__).with_name('transport-receipt.json').write_text(json.dumps(meta, indent=2)+'\n')
print('TRANSPORT_READY', archive, meta['archive_bytes'], meta['archive_sha256'], flush=True)
