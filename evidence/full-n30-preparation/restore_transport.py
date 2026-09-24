"""Reconstruct and hash-check the frozen gzip; publish only the full valid dataset."""
from pathlib import Path
import gzip
import hashlib
import json
import lzma
import sys

stage = Path(sys.argv[1])
data = stage/'data/s1-dev-longchain'
here = Path(__file__).parent
meta = json.loads((here/'transport.json').read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


assert sha(data/'manifest.json') == '19a7e5a6827f64a99695cba2d89b7efa2a0b05d207fc95da1568ec1d82280b2c'
body = data/meta['body']
body.parent.mkdir(parents=True, exist_ok=True)
if not body.exists() or sha(body) != meta['sha256']:
    tmp = body.with_suffix('.transfer-part')
    with lzma.open(here/'body.xz', 'rb') as src, tmp.open('wb') as raw:
        with gzip.GzipFile(filename=meta['filename'], fileobj=raw, mode='wb',
                           compresslevel=meta['compresslevel'], mtime=meta['mtime']) as dst:
            for block in iter(lambda: src.read(4 << 20), b''):
                dst.write(block)
            dst.flush()  # Match TextIOWrapper close in the frozen generator.
    assert sha(tmp) == meta['sha256'], 'gzip reconstruction differs; not published'
    tmp.replace(body)
manifest = json.loads((data/'manifest.json').read_text())
for name, digest in manifest['artifacts'].items():
    assert sha(data/name) == digest, name
cohort = json.loads((data/'cohort.json').read_text())
assert cohort['n_chains'] == 311 and cohort['n_requests'] == 5601
destination = Path('/tmp/ax/data/s1-dev-longchain')
if destination.exists():
    assert sha(destination/'manifest.json') == sha(data/'manifest.json')
    for name, digest in manifest['artifacts'].items():
        assert sha(destination/name) == digest, name
    print('DATA_ALREADY_READY manifest=19a7e5a6 chains=311 requests=5601', flush=True)
else:
    data.rename(destination)
    print('DATA_TRANSFER_VERIFIED manifest=19a7e5a6 artifacts=8 chains=311 requests=5601', flush=True)
