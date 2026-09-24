"""Port step 4 (of 180): from upstream #38212 take ONLY the compressed-index
ownership and recurrent-checkpoint alignment (unified_radix_cache.py,
mamba_component.py, schedule_batch.py). Its host-pool half is superseded by the
#40913-#40915 declaration stack already applied."""
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from portlib import PKG, REPO, rd, sub_once, wr  # noqa: E402

t = open(f"{REPO}/refs/pr38212.diff").read()
parts = re.split(r"(?m)^(?=diff --git )", t)
keep = "".join(
    p
    for p in parts
    if re.match(
        r"diff --git a/python/sglang/srt/(managers/schedule_batch|mem_cache/unified_cache/"
        r"components/mamba_component|mem_cache/unified_radix_cache)\.py",
        p,
    )
)
r = subprocess.run(
    ["patch", "-p3", "-f", "--no-backup-if-mismatch", "-d", PKG],
    input=keep, text=True, capture_output=True,
)
print(r.stdout[-2000:], r.stderr[-1000:])
assert "FAILED" not in r.stdout, "38212 tree hunks must apply"

p = "srt/mem_cache/unified_radix_cache.py"
s = rd(p)
# Adaptation: widen tree ownership only when the host tier is on. Without
# HiCache the device-only path stays byte-for-byte the pre-180 behaviour (the
# 64/128/192 fork overwrite of shared packed index rows is a separate, pre-existing
# device-path issue; see 180 .md).
s = sub_once(
    s,
    """    if params.disable or params.token_to_kv_pool_allocator is None:
        return params
""",
    """    if params.disable or params.token_to_kv_pool_allocator is None:
        return params
    if not get_memory().enable_hierarchical_cache:
        return params
""",
    "gate widening on hicache",
)
wr(p, s)
print("step3 ok")
