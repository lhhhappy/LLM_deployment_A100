#!/usr/bin/env python3
"""Generate patch 102 (role-boundary KDA checkpoint without splitting the prefill) for a given
schedule_batch.py. The same logic works on the organizer base (build/base_src_full) and on the
v0.5.20 L1 checkout (src/sglang); see plans/active/102-role-track.md.

Usage: make_102_role_track.py <schedule_batch.py> <out.patch>
Output is a unified diff with a/python/sglang/srt/managers/schedule_batch.py paths (patch -p3).
"""
import difflib
import sys

src_path, out_path = sys.argv[1], sys.argv[2]
s = open(src_path).read()


def replace_once(text, old, new):
    assert text.count(old) == 1, f"anchor not unique/absent: {old[:70]!r}"
    return text.replace(old, new)


HELPER = '''

@lru_cache(maxsize=1)
def _role_track_token_ids() -> frozenset:
    """[ax] Role boundary token ids (GLM <|user|>=154827, <|observation|>=154829) from
    SGLANG_ARENA_ROLE_TRACK_TOKEN_IDS. Empty = feature off (stock tracking)."""
    raw = os.environ.get("SGLANG_ARENA_ROLE_TRACK_TOKEN_IDS", "").strip()
    return frozenset(int(x) for x in raw.split(",") if x.strip()) if raw else frozenset()


_ROLE_TRACK_SCAN_WINDOW = int(os.environ.get("SGLANG_ARENA_ROLE_TRACK_SCAN_WINDOW", "32768"))


def _role_track_seqlen(req, prefix_len: int, end_len: int, grid: int):
    """[ax] Last role boundary in (prefix_len, end_len), aligned down to `grid` relative to
    prefix_len, so the tracked KDA state is the one the chain's next turn resumes from."""
    ids = _role_track_token_ids()
    if not ids:
        return None
    fill = req.full_untruncated_fill_ids
    lo = max(prefix_len, end_len - _ROLE_TRACK_SCAN_WINDOW)
    for i in range(min(end_len, len(fill)) - 1, lo - 1, -1):
        if fill[i] in ids:
            aligned = prefix_len + (i - prefix_len) // grid * grid
            return aligned if aligned > prefix_len else None
    return None
'''

# helper goes after the imports block: anchor on the first top-level "logger = " line
anchor = "\nlogger = logging.getLogger(__name__)\n"
s = replace_once(s, anchor, anchor + HELPER)
if "from functools import lru_cache" not in s:
    s = s.replace("\nimport logging\n", "\nimport logging\nfrom functools import lru_cache\n", 1)
if "\nimport os\n" not in s:
    s = s.replace("\nimport logging\n", "\nimport logging\nimport os\n", 1)

old = "            req.kv.mamba_last_track_seqlen = mamba_track_seqlen_aligned\n"
new = """            # [ax] 102: prefer the last role boundary inside this extend (the chain's next turn
            # resumes there, F3); it beats a branch point only when it lies after it.
            role_seqlen = _role_track_seqlen(
                req,
                len(req.prefix_indices),
                len(req.prefix_indices) + req.extend_range.length,
                checkpoint_grid,
            )
            if (
                role_seqlen is not None
                and role_seqlen < len(req.prefix_indices) + req.extend_range.length
                and role_seqlen != mamba_track_seqlen_aligned
                and (
                    req.mamba_branching_seqlen is None
                    or mamba_track_seqlen_aligned != req.mamba_branching_seqlen
                    or role_seqlen > req.mamba_branching_seqlen
                )
            ):
                mamba_track_seqlen = _force_track_h(role_seqlen)
                mamba_track_seqlen_aligned = role_seqlen
""" + old
s = replace_once(s, old, new)

orig = open(src_path).read().splitlines(True)
rel = "python/sglang/srt/managers/schedule_batch.py"
diff = difflib.unified_diff(orig, s.splitlines(True), "a/" + rel, "b/" + rel)
open(out_path, "w").writelines(diff)
print("wrote", out_path)
