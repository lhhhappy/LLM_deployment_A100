"""140: scheduler/cache ownership for two exact KDA checkpoints per extend.

The second slot is optional: allocation pressure drops the extra checkpoint, never
changes request tokens or the scheduling quantum. All slot IDs here are physical
(the startup guard excludes unified memory). The ordinary ping-pong slot remains
owned by the existing extra_buffer machinery.
"""
import math
import os

import torch


def configure(cache, params):
    if os.environ.get("SGLANG_AX_KDA_DUAL_SNAPSHOT", "0") != "1":
        return False
    from sglang.srt.runtime_context import get_server_args, process_model_config

    args = get_server_args()
    model = process_model_config()
    arch = model.hf_config.architectures
    checks = {
        "GLM-5.3-Flash target": "Glm5NextForConditionalGeneration" in arch,
        "enabled FULL+MAMBA Python radix tree": (
            not params.disable and cache.is_mamba_enabled and not cache.is_swa_enabled
            and cache._tree_core_backend == "python" and not params.is_eagle
        ),
        "ordinary extra_buffer": params.enable_mamba_extra_buffer and not params.enable_mamba_extra_buffer_lazy,
        "TP only": params.pp_size == 1 and params.attn_cp_size == 1
                   and not getattr(args, "enable_dp_attention", False),
        "no speculative/mixed/HiCache/unified/int8/ReplaySSM": not any((
            getattr(args, "speculative_algorithm", None),
            getattr(args, "enable_mixed_chunk", False),
            getattr(args, "enable_two_batch_overlap", False),
            getattr(args, "enable_hierarchical_cache", False),
            getattr(args, "enable_unified_memory", False),
            getattr(args, "enable_int8_mamba_checkpoint", False),
            getattr(args, "enable_linear_replayssm", False),
            params.enable_session_radix_cache,
        )),
        "no PD": getattr(args, "disaggregation_mode", "null") == "null",
        "fp32 recurrent states": cache.req_to_token_pool.mamba_pool.mamba_cache.temporal.dtype == torch.float32,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError("SGLANG_AX_KDA_DUAL_SNAPSHOT=1 requires: " + "; ".join(failed))
    return True


def role_depth(ids, grid):
    """Prefix excludes the last user/observation marker; round DOWN, never up."""
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] in (154827, 154829):
            return i // math.lcm(64, grid) * math.lcm(64, grid)
    return 0


def batch_supported(batch, grid):
    # Streaming sessions have separate ownership. A batch containing one uses
    # legacy tracking throughout, avoiding mixed exporter/legacy destinations.
    return getattr(batch.tree_cache, "ax_kda_dual_snapshot", False) and all(
        getattr(req, "session", None) is None
        and len(req.prefix_indices) % math.lcm(64, grid) == 0
        for req in batch.reqs
    )


def prepare(batch, track_indices, track_mask, grid):
    """Allocate optional role slots after normal admission has reserved its slots."""
    if not batch.ax_kda_dual_snapshot_batch:
        return
    pool = batch.req_to_token_pool
    offsets, slots = [], []
    for req, end_slot, active in zip(batch.reqs, track_indices, track_mask):
        assert req.kv.ax_kda_snapshot_slot is None, "previous extend snapshot not consumed"
        prefix = len(req.prefix_indices)
        depth = role_depth(req.full_untruncated_fill_ids, grid)
        req.kv.ax_kda_role_depth = depth
        end = req.kv.mamba_last_track_seqlen if active else prefix
        row_offsets = [end - prefix if active else -1, -1]
        row_slots = [end_slot if active else -1, -1]
        # Reserve one donation replacement per request. On pressure the role
        # snapshot is optional and is skipped without eviction or extra forward.
        if prefix < depth < end and pool.mamba_allocator.available_size() > len(batch.reqs):
            slot = pool.mamba_allocator.alloc(1)
            if slot is not None:
                req.kv.ax_kda_snapshot_slot = slot
                req.kv.ax_kda_snapshot_depth = depth
                row_offsets[1], row_slots[1] = depth - prefix, int(slot.item())
        offsets.append(row_offsets)
        slots.append(row_slots)
    batch.ax_kda_snapshot_offsets = torch.tensor(offsets, dtype=torch.int64, device=batch.device)
    batch.ax_kda_snapshot_slots = torch.tensor(slots, dtype=torch.int64, device=batch.device)


def discard(pool, req):
    slot = req.kv.ax_kda_snapshot_slot
    if slot is not None:
        pool.mamba_allocator.free(slot)
        req.kv.ax_kda_snapshot_slot = None
        req.kv.ax_kda_snapshot_depth = None


def commit(cache, req, end_params, end_result):
    """Insert the additional state using tree-owned KV, with NO duplicate KV free.

    Runs after the normal end insert (both finished and unfinished). That insert
    may already have freed duplicates; do not read stale req_to_token entries.
    Temporarily pin the end while inserting the ancestor, then let the ordinary
    caller perform its original rematch/lock transfer/cleanup.
    """
    slot = req.kv.ax_kda_snapshot_slot
    if slot is None:
        return
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams

    depth = req.kv.ax_kda_snapshot_depth
    if depth >= len(end_params.key) or depth <= 0:
        discard(cache.req_to_token_pool, req)
        return
    lock = cache.inc_lock_ref(end_result.last_device_node)
    try:
        matched = cache.match_prefix(MatchPrefixParams(key=end_params.key))
        assert len(matched.device_indices) >= depth
        params = InsertParams(
            key=end_params.key[:depth],
            value=matched.device_indices[:depth].clone(),
            mamba_value=slot,
            prev_prefix_len=depth,  # all of this KV is already owned by the tree
            chunked=end_params.chunked,
            priority=end_params.priority,
        )
        params.ax_kda_role = True
        result = cache.insert(params)
        # From here the tree owns the slot, or the duplicate can be freed.
        req.kv.ax_kda_snapshot_slot = None
        req.kv.ax_kda_snapshot_depth = None
        if result.mamba_exist:
            cache.req_to_token_pool.mamba_allocator.free(slot)
    finally:
        cache.dec_lock_ref(end_result.last_device_node, lock.to_dec_params())


def eviction_candidate(lru, component_type):
    """Tail checkpoints first, then the existing LRU order; never skip locks."""
    oldest = node = lru.get_lru_no_lock()
    while node is not None:
        if getattr(node, "ax_kda_tail", False) and not getattr(node, "ax_kda_role", False):
            return node
        node = lru.get_prev_no_lock(node)
    return oldest
