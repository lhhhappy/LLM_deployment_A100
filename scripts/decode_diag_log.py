#!/usr/bin/env python3
"""Decode a diagnostic-build log (bohr image build-log <id> -o json) that carries a base64 xz tar
between DIAGTGZ_BEGIN/END, verify it against the in-log sha256, and extract it.

Usage: python3 scripts/decode_diag_log.py build/diag5/log.txt build/base_src_full
"""
import base64
import hashlib
import io
import json
import lzma
import re
import sys
import tarfile

log_path, out_dir = sys.argv[1], sys.argv[2]
log = json.load(open(log_path))["data"]["log"]
if "output clipped" in log:
    sys.exit("log was clipped by the platform rate limit; rebuild with slower output")
m = re.search(r"DIAG txz_sha256=([0-9a-f]{64}) bytes=(\d+)", log)
parts = [re.sub(r"^.*?DIAGB64 ", "", l).strip() for l in log.split("\n") if "DIAGB64 " in l]
raw = base64.b64decode("".join(p for p in parts if re.fullmatch(r"[A-Za-z0-9+/=]+", p)))
if not m or hashlib.sha256(raw).hexdigest() != m.group(1):
    sys.exit(f"sha256 mismatch ({len(raw)} bytes); not extracting")
with tarfile.open(fileobj=io.BytesIO(lzma.decompress(raw))) as tar:
    tar.extractall(out_dir, filter="data")
    print(f"ok: {len(tar.getnames())} files, sha256 {m.group(1)[:12]}…, into {out_dir}")
