"""Bounded TP0 observations of sub-grid deferral and planned checkpoints.

CPU metadata only: no cache lookup, allocation, tensor read or synchronization.
Checkpoint records describe batch preparation, not completed/published state.
"""

import json
import logging
import os
import time

ENABLED = os.environ.get("SGLANG_AX_CHUNK_ALIGNMENT_TRACE", "0") == "1"
_logger = logging.getLogger(__name__)
_remaining_bytes = 2 * 1024 * 1024


def _rank_zero():
    from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_rank

    return get_tensor_model_parallel_rank() == 0


def _emit(event, req, **fields):
    global _remaining_bytes
    if _remaining_bytes <= 0:
        return
    record = json.dumps(dict(event=event, rid=req.rid, **fields), separators=(",", ":"))
    size = len(record.encode("utf-8")) + 160
    if size > _remaining_bytes:
        _logger.info('[ax-chunk-alignment] {"event":"budget_exhausted"}')
        _remaining_bytes = 0
        return
    _remaining_bytes -= size
    _logger.info("[ax-chunk-alignment] %s", record)


def defer(req, budget, grid):
    if not _rank_zero():
        return
    pending = getattr(req, "_ax_alignment_pending", None)
    if pending is None:
        req._ax_alignment_pending = [time.monotonic(), 1]
        req._ax_alignment_touched = True
        _emit("defer", req, prefix=len(req.prefix_indices), budget=budget, grid=grid,
              remaining=len(req.full_untruncated_fill_ids) - len(req.prefix_indices),
              checkpoint=getattr(req.kv, "mamba_last_track_seqlen", None))
    else:
        pending[1] += 1


def resume(req, tokens, truncated):
    pending = getattr(req, "_ax_alignment_pending", None)
    if pending is None:
        return
    _emit("resume", req, prefix=len(req.prefix_indices), tokens=tokens,
          truncated=truncated, deferred_decisions=pending[1],
          elapsed_s=time.monotonic() - pending[0])
    req._ax_alignment_pending = None
    req._ax_alignment_await_checkpoint = True


def checkpoint(req, prefix, end, cache_chunk, grid, mask):
    touched = getattr(req, "_ax_alignment_touched", False)
    unaligned = prefix % cache_chunk != 0
    if not (touched or unaligned) or not _rank_zero():
        return
    final = end == len(req.full_untruncated_fill_ids)
    restored = mask and getattr(req, "_ax_alignment_await_checkpoint", False)
    first_unaligned = unaligned and not getattr(req, "_ax_alignment_unaligned_seen", False)
    if first_unaligned or final or restored:
        _emit("checkpoint_plan", req, prefix=prefix, end=end, final=final,
              cache_chunk=cache_chunk, grid=grid, track_mask=mask,
              checkpoint=getattr(req.kv, "mamba_last_track_seqlen", None),
              unaligned_prefix=unaligned)
    if restored:
        req._ax_alignment_await_checkpoint = False
    if unaligned:
        req._ax_alignment_unaligned_seen = True
