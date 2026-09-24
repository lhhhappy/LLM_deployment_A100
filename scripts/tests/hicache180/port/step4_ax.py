"""Port step 5 (of 180): small, local additions not in the upstream stack.

1. DSAIndexerPoolHost.destroy: taken verbatim from upstream #38212 (the index
   mirror's pinned buffer lives outside kv_buffer, so HostKVCache.destroy never
   unregistered it).
2. build_hybrid_mamba_stack: --hicache-size is a per-rank total; carve the
   declared sidecars' (indexer) bytes out of the KV share instead of allocating
   them on top (upstream #38212 notes ratio/sidecar pools exceed the budget).
3. build_hybrid_mamba_stack: packed draft pools are addressed with the target's
   device indices; refuse a packed draft whose device pool is smaller.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from portlib import rd, sub_once, wr  # noqa: E402

# 1 ---------------------------------------------------------------------------
p = "srt/mem_cache/pool_host/dsa.py"
s = rd(p)
s = sub_once(
    s,
    """from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    get_allocator_from_storage,
)""",
    """from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    _cuda_host_unregister,
    get_allocator_from_storage,
)""",
    "dsa import unregister",
)
s = sub_once(
    s,
    """        self.lock = threading.RLock()
        self.clear()

    def get_size_per_token(self):""",
    """        self.lock = threading.RLock()
        self.clear()

    def destroy(self):
        if getattr(self, "_destroyed", False):
            return
        # This pool keeps its registered backing tensor outside kv_buffer,
        # which is the buffer released by HostKVCache.destroy().
        buffer = getattr(self, "index_k_with_scale_buffer", None)
        if buffer is not None and self.pin_memory and (_is_cuda or _is_hip):
            _cuda_host_unregister(buffer)
        self.index_k_with_scale_buffer = None
        super().destroy()

    def get_size_per_token(self):""",
    "dsa destroy",
)
wr(p, s)

# 2 + 3 -----------------------------------------------------------------------
p = "srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py"
s = rd(p)
s = sub_once(
    s,
    """    _validate_host_pool_buffers(configs, page_size=params.page_size)
    kv_host_pool = build_kv_host_pool(
        kv_pool=kv_pool,
        page_size=params.page_size,
        use_mla=use_mla,
        host_size=kv_host_size,
        mtp_draft_device_pools=_root_config(configs).packed_draft_device_pools,
    )
    mamba_host_pool = MambaPoolHost(""",
    """    _validate_host_pool_buffers(configs, page_size=params.page_size)
    # [ax 180] Packed draft layers are addressed with the target's device
    # indices, so every draft buffer must cover the target's slot range.
    for draft in _root_config(configs).packed_draft_device_pools:
        if draft.size < kv_pool.size:
            raise ValueError(
                f"packed draft pool holds {draft.size} slots but the target KV "
                f"pool holds {kv_pool.size}; host restore would index past it"
            )
    if kv_host_size is not None and use_mla:
        kv_host_size = _carve_declared_sidecars(
            kv_host_size, kv_pool=kv_pool, configs=configs
        )
    kv_host_pool = build_kv_host_pool(
        kv_pool=kv_pool,
        page_size=params.page_size,
        use_mla=use_mla,
        host_size=kv_host_size,
        mtp_draft_device_pools=_root_config(configs).packed_draft_device_pools,
    )
    mamba_host_pool = MambaPoolHost(""",
    "mamba stack guards",
)
s = sub_once(
    s,
    """def build_hybrid_mamba_stack(
    *,""",
    '''def _carve_declared_sidecars(
    kv_host_size: float,
    *,
    kv_pool: Any,
    configs: tuple[HostPoolBuildConfig, ...],
) -> float:
    """[ax 180] Shrink the MLA share of --hicache-size so that the MLA host pool
    plus every declared sidecar sized from it (e.g. the DSA indexer mirror, one
    row per KV host slot) fit in the share. Upstream sizes the sidecars on top."""
    root = _root_config(configs)
    root_layers = kv_pool.layer_num + len(root.packed_draft_device_pools)
    root_bpt = kv_pool.kv_cache_dim * kv_pool.store_dtype.itemsize * root_layers
    side_bpt = 0
    for config in configs:
        decl = config.decl
        if decl.is_layout_root:
            continue
        owned = decl.owned_device_layers
        layers = decl.device_pool.layer_num if owned is None else len(owned)
        layers += len(config.packed_draft_device_pools)
        side_bpt += decl.storage_info.bytes_per_token_per_layer * layers
    if side_bpt == 0:
        return kv_host_size
    carved = kv_host_size * root_bpt / (root_bpt + side_bpt)
    logger.info(
        "HiCache size split: %.2f GB for KV + declared sidecars "
        "(%d + %d B/token); KV host pool gets %.2f GB.",
        kv_host_size,
        root_bpt,
        side_bpt,
        carved,
    )
    return carved


def build_hybrid_mamba_stack(
    *,''',
    "carve helper",
)
wr(p, s)
print("step4 ok")
