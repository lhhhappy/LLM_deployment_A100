"""Port steps 2-3: upstream #40914 (separate draft via declarations) and #40915
(hybrid Mamba stack declares the indexer host pool) on top of step 1."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from portlib import PKG, apply, rd, sub_once, wr  # noqa: E402

P = "srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py"


def resolve_40914():
    s = rd(P)
    a = s.index("def build_full_draft_pools(")
    b = s.index("def build_swa_draft_pools(")
    old = s[a:b]
    keep_a = old.index("    # Note(kpham-sgl): DCP x DSpark draft KV is replicated")
    keep_b = old.index('        pool_label="draft",\n    )\n') + len('        pool_label="draft",\n    )\n')
    host_pool_build = old[keep_a:keep_b]
    new = (
        '''def build_full_draft_pools(
    *,
    draft_kv_pool: Any,
    tree_cache: Any,
) -> tuple[list[SidecarPoolSpec], list[PoolEntry]]:
    """Build the separate draft sidecars declared by a full-attention draft pool;
    their indices follow target KV and their layout roots on the draft KV host pool."""
    decls = draft_kv_pool.host_pool_decls()
    # A hybrid draft declares its KV on the full-attention sub-pool.
    pool = layout_root(decls).device_pool
    if pool.layer_num == 0:
        return [], []

    controller = tree_cache.cache_controller
    host_pool_group = controller.mem_pool_host

    configs = prepare_host_pool_configs(
        decls=make_draft_sidecar_decls(decls),
        full_layer_mapping={i: i for i in range(pool.layer_num)},
        transfer_layer_id_max=pool.layer_num,
        index_primary=PoolName.KV,
    )
    _validate_host_pool_buffers(configs, page_size=controller.page_size)
'''
        + host_pool_build
        + '''    entries = _build_declared_entries(configs, root_host_pool=draft_host_pool)
    return [c.decl.sidecar_spec() for c in configs], entries


'''
    )
    s = s[:a] + new + s[b:]
    wr(P, s)
    os.remove(os.path.join(PKG, P + ".rej"))


def resolve_40915():
    """#40915 hunks 2-7 fail only because our base predates the
    transfer_layer_num -> transfer_layer_id_max rename, the
    _stage_local_layer_mapping helper and a later MambaPoolHost layout override.
    Re-express the same change on our base: the hybrid Mamba stack assembles the
    full-attention pool's declarations (KV + INDEXER) and then the Mamba pool."""
    s = rd(P)
    a = s.index("def build_hybrid_mamba_stack(")
    b = s.index("def build_hybrid_mamba_swa_stack(")
    s = s[:a] + '''def build_hybrid_mamba_stack(
    *,
    params: CacheInitParams,
    decls: tuple[HostPoolDecl, ...],
    mamba_pool: Any,
    full_layer_mapping: dict[int, int],
    mamba_layer_mapping: dict[int, int],
    load_cache_event,
    storage_backend: Optional[str],
    use_mla: bool,
    host_mamba_evict_fn: Optional[Callable[[int], Any]] = None,
    device_mamba_evict_fn: Optional[Callable[[int], Any]] = None,
    prefetch_threshold: int = 256,
    model_name: Optional[str] = None,
    storage_backend_extra_config: Optional[dict] = None,
    enable_storage_metrics: bool = False,
) -> HostPoolAssemblyResult:
    """KV plus every pool the hybrid pool declares (e.g. a sparse indexer),
    then the Mamba state pool, which keeps its own host path."""
    kv_pool = layout_root(decls).device_pool
    transfer_layer_num = len(full_layer_mapping | mamba_layer_mapping)
    mamba_allocator = params.req_to_token_pool.mamba_allocator
    packed_drafts = validate_packed_draft_pools(
        target_decls=decls, draft_pools=params.mtp_draft_device_pools
    )
    kv_host_size, mamba_host_size = None, 0
    if get_memory().hicache_size > 0:
        kv_host_size, mamba_host_size = _split_hicache_size(
            get_memory().hicache_size, (kv_pool, mamba_pool)
        )
    if packed_drafts:
        full_layer_mapping = _with_mtp_layer_mapping(
            full_layer_mapping,
            transfer_layer_start=transfer_layer_num,
            target_device_layer_num=kv_pool.layer_num,
            draft_layer_num=len(packed_drafts),
        )
    configs = prepare_host_pool_configs(
        decls=decls,
        full_layer_mapping=full_layer_mapping,
        transfer_layer_id_max=transfer_layer_num + len(packed_drafts),
        packed_draft_decls=packed_drafts,
    )
    _validate_host_pool_buffers(configs, page_size=params.page_size)
    kv_host_pool = build_kv_host_pool(
        kv_pool=kv_pool,
        page_size=params.page_size,
        use_mla=use_mla,
        host_size=kv_host_size,
        mtp_draft_device_pools=_root_config(configs).packed_draft_device_pools,
    )
    mamba_host_pool = MambaPoolHost(
        mamba_pool,
        get_memory().hicache_ratio,
        mamba_host_size,
        allocator_type=_get_allocator_type(),
        layout=get_memory().hicache_mem_layout,
    )
    entries = _build_declared_entries(configs, root_host_pool=kv_host_pool) + [
        build_pool_entry(
            name=PoolName.MAMBA,
            host_pool=mamba_host_pool,
            device_pool=mamba_pool,
            layer_mapping=mamba_layer_mapping,
            transfer_layer_num=transfer_layer_num,
            host_evict_fn=host_mamba_evict_fn,
            device_evict_fn=device_mamba_evict_fn,
            device_alloc_fn=mamba_allocator.alloc,
            device_free_fn=mamba_allocator.free,
        )
    ]
    host_pool_group = HostPoolGroup(entries)
    cache_controller = _build_declared_controller(
        params=params,
        host_pool_group=host_pool_group,
        load_cache_event=load_cache_event,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        transfer_layer_id_max=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
    )
    return HostPoolAssemblyResult(
        host_pool_group=host_pool_group,
        cache_controller=cache_controller,
        configs=configs,
    )


''' + s[b:]

    # _MambaStrategy.build (upstream #40915 hunks 5-6)
    old_build = s[s.index("class _MambaStrategy(StackStrategy):"):]
    old_build = old_build[: old_build.index("def _swa_layer_mappings(")]
    new_build = sub_once(
        old_build,
        """        host_pool_group, cache_controller = build_hybrid_mamba_stack(
            params=params,
            kv_pool=kvcache.full_kv_pool,""",
        """        stack = build_hybrid_mamba_stack(
            params=params,
            decls=kvcache.host_pool_decls(),""",
        "mamba strategy call",
    )
    new_build = sub_once(
        new_build,
        """        return StackBuildResult(
            host_pool_group=host_pool_group,
            cache_controller=cache_controller,
            component_host_pools={
                ComponentType.FULL: host_pool_group.get_pool(PoolName.KV),
                ComponentType.MAMBA: host_pool_group.get_pool(PoolName.MAMBA),
            },
            register_req_to_token_counter=True,
            transfer_layer_num=len(full_layer_mapping | mamba_layer_mapping),
            pools_desc="KV + MAMBA",
        )""",
        """        return StackBuildResult(
            host_pool_group=stack.host_pool_group,
            cache_controller=stack.cache_controller,
            component_host_pools={
                ComponentType.FULL: stack.host_pool_group.get_pool(PoolName.KV),
                ComponentType.MAMBA: stack.host_pool_group.get_pool(PoolName.MAMBA),
            },
            sidecars=stack.sidecars,
            pool_declarations=tuple(c.decl for c in stack.configs),
            register_req_to_token_counter=True,
            transfer_layer_num=len(full_layer_mapping | mamba_layer_mapping),
            pools_desc=" + ".join(
                [c.decl.pool_name.value.upper() for c in stack.configs] + ["MAMBA"]
            ),
        )""",
        "mamba strategy result",
    )
    s = s.replace(old_build, new_build)
    # upstream #40915 hunk 7
    # (hunk 7 applies cleanly; assert it did)
    assert "_DECLARATION_VERIFIED_STRATEGIES: tuple[type, ...] = (_DsaStrategy, _MambaStrategy)" in s
    wr(P, s)


if __name__ == "__main__":
    apply(40914)
    resolve_40914()
    apply(40915)
    rej = os.path.join(PKG, P + ".rej")
    if os.path.exists(rej):
        os.remove(rej)
    resolve_40915()
    print("step2 ok")
