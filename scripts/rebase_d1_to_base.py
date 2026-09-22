#!/usr/bin/env python3
"""Produce the organizer-base-image variant of D1 v1.2 (role-boundary KDA checkpoints).

The organizer SGLang base (dev build fe236ea6c3 + organizer fixes, digest f24781f0...) has
an older schedule_policy.py than our v0.5.20 reference (monolithic add_one_req, no
_select_prefill_admission). This script applies the same three D1 v1.2 behaviours to the
base file:
  1. admission-time split: a request that would be admitted whole ends its extend at the
     last role boundary (becomes the round's chunked request);
  2. final-chunk split: the last chunk of a chunked prefill ends at the last role boundary;
  3. guard: at most one partial prefill per round while the feature is on.
Feature is off unless SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS is set (stock behaviour).

Usage: rebase_d1_to_base.py <base schedule_policy.py> patches/101-d1v12-on-base.patch (diff it; see patches/101-*.md)
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()


def replace_once(text, old, new, after=None):
    start = text.index(after) if after else 0
    j = text.index(old, start)
    if after is None:
        assert text.count(old) == 1, f"anchor not unique: {old[:60]!r}"
    return text[:j] + new + text[j + len(old):]


s = replace_once(
    s,
    "from sglang.srt.runtime_context import (\n    get_disagg,\n    get_schedule,\n)",
    "from sglang.srt.runtime_context import (\n    get_disagg,\n    get_schedule,\n    mamba_checkpoint_grid,\n)",
)
for imp in ("import math\n", "from functools import lru_cache\n", "from collections import Counter\n"):
    if imp not in s:
        s = s.replace("\nimport os\n", "\nimport os\n" + imp, 1)

HELPERS = '''@lru_cache(maxsize=1)
def _role_boundary_token_ids() -> frozenset:
    """[ax] Role/message boundary token ids (GLM <|user|>=154827, <|observation|>=154829),
    from SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS. Empty = feature off (stock behaviour)."""
    raw = os.environ.get("SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS", "").strip()
    if not raw:
        return frozenset()
    return frozenset(int(x) for x in raw.split(",") if x.strip())


ROLE_BOUNDARY_STATS: Counter = Counter()
_ROLE_BOUNDARY_SCAN_WINDOW = int(
    os.environ.get("SGLANG_ARENA_ROLE_BOUNDARY_SCAN_WINDOW", "32768")
)


'''
s = replace_once(s, "class AddReqResult(Enum):", HELPERS + "class AddReqResult(Enum):")

METHOD = '''    def _role_split_len(self, req, prefix_len: int, extend_len: int, tag: str):
        """[ax] Length to prefill so this extend ends at the last role boundary
        (grid-aligned relative to prefix_len), or None for stock behaviour."""
        if not _role_boundary_token_ids() or self.dllm_config is not None:
            return None
        stats = ROLE_BOUNDARY_STATS
        if self.rem_chunk_tokens is None:
            stats[tag + "skip_no_chunked_prefill"] += 1
            return None
        if not getattr(self.tree_cache, "supports_mamba", lambda: False)():
            return None
        full_len = prefix_len + extend_len
        branch = getattr(req, "mamba_branching_seqlen", None)
        if branch is not None and prefix_len < branch <= full_len:
            stats[tag + "skip_branch_conflict"] += 1
            return None
        grid = mamba_checkpoint_grid(self.tree_cache.page_size)
        if prefix_len % grid != 0:
            stats[tag + "skip_unaligned_prefix"] += 1
            return None
        ids = req.full_untruncated_fill_ids
        boundary_ids = _role_boundary_token_ids()
        lo = max(prefix_len, full_len - _ROLE_BOUNDARY_SCAN_WINDOW)
        pos = -1
        for i in range(full_len - 1, lo - 1, -1):
            if ids[i] in boundary_ids:
                pos = i
                break
        if pos < 0:
            stats[tag + "skip_no_boundary"] += 1
            return None
        split_len = (pos - prefix_len) // grid * grid
        if split_len < grid or split_len >= extend_len // grid * grid:
            stats[tag + "skip_short"] += 1
            return None
        stats[tag + "taken"] += 1
        return split_len

'''
s = replace_once(s, "    def add_chunked_req(self, req: Req):", METHOD + "    def add_chunked_req(self, req: Req):")

# 2. final-chunk split, scoped to add_chunked_req (the same lines also exist in _add_dllm_req)
s = replace_once(
    s,
    """        truncated = cand_extend_input_len > _rem_tokens
        new_len = min(cand_extend_input_len, _rem_tokens)
""",
    """        truncated = cand_extend_input_len > _rem_tokens
        new_len = min(cand_extend_input_len, _rem_tokens)
        if not truncated:
            # [ax] final chunk: end it at the last role boundary so a reusable KDA state
            # is saved there; the same chunked request finishes the remainder next round.
            role_len = self._role_split_len(
                req, len(req.prefix_indices), cand_extend_input_len, "tail_"
            )
            if role_len is not None:
                new_len, truncated = role_len, True
""",
    after="    def add_chunked_req(self, req: Req):",
)

# 1. admission-time split in the non-chunked commit branch of add_one_req
s = replace_once(
    s,
    """                # Non-chunked prefill — the whole sequence is committed this iter.
                req.set_extend_range(
                    len(req.prefix_indices), len(req.full_untruncated_fill_ids)
                )
                self.can_run_list.append(req)
""",
    """                # [ax] Optionally end this extend at the last role boundary (becomes this
                # round's chunked request; at most one partial per round).
                role_len = None
                if not has_chunked_req and self.new_chunked_req is None:
                    role_len = self._role_split_len(
                        req,
                        len(req.prefix_indices),
                        len(req.full_untruncated_fill_ids) - len(req.prefix_indices),
                        "admit_",
                    )
                if role_len is not None:
                    req.set_extend_range(
                        len(req.prefix_indices), len(req.prefix_indices) + role_len
                    )
                    self.can_run_list.append(req)
                    self.new_chunked_req = req
                    self._req_inc_lock_ref(req)
                    self._update_prefill_budget(
                        prefix_len,
                        role_len,
                        0,
                        req.retracted_stain,
                        mamba_gap_reserve=self._mamba_gap_budget_for_req(req),
                        host_hit_len=req.host_hit_length,
                        storage_hit_len=req.storage_hit_length,
                    )
                    return self.budget_state()
                # Non-chunked prefill — the whole sequence is committed this iter.
                req.set_extend_range(
                    len(req.prefix_indices), len(req.full_untruncated_fill_ids)
                )
                self.can_run_list.append(req)
""",
)

# 3. guard in the chunked branch of add_one_req
s = replace_once(
    s,
    """            else:
                # Make sure at least one page is available
                trunc_len = chunk_tokens_limit // self.page_size * self.page_size
""",
    """            else:
                # [ax] at most one partial prefill per round while the role split is on
                if _role_boundary_token_ids() and self.new_chunked_req is not None:
                    return AddReqResult.OTHER
                # Make sure at least one page is available
                trunc_len = chunk_tokens_limit // self.page_size * self.page_size
""",
)

open(dst, "w").write(s)
print("wrote", dst)
