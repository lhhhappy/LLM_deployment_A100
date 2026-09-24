import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from portlib import PKG, apply, rd, sub_once, wr  # noqa: E402

apply(40913)

# ---- pool_host/dsa.py: rejected hunks #4/#5 (constructor). Upstream also has
# an `is_dummy` mode our base lacks; everything else is taken verbatim.
p = "srt/mem_cache/pool_host/dsa.py"
s = rd(p)
a = s.index("    def __init__(\n        self,\n        device_pool: DSATokenToKVPool,")
b = s.index("        buf_elem_size = self.page_num * self.layer_num * self.indexer_page_stride_size\n")
s = s[:a] + '''    def __init__(
        self,
        decl: HostPoolDecl,
        anchor_host: MLATokenToKVPoolHost,
        *,
        packed_draft_device_pools: tuple[DSATokenToKVPool, ...] = (),
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ):
        self.decl = decl
        storage_info = decl.storage_info
        device_pool = decl.device_pool
        self.device_pool = device_pool
        self.page_size = anchor_host.page_size
        self.layout = anchor_host.layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)
        self.dtype = device_pool.store_dtype
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer
        # Host layers are compact: only owned device layers that hold index
        # buffers, then one tail layer per packed draft pool.
        owned_start, owned_end = self._device_owned_layer_range()
        declared = decl.owned_device_layers
        self._live_target_layers = [
            layer
            for layer in range(owned_start, owned_end)
            if declared is None or layer in declared
        ]
        self._device_to_host_layer = {
            layer: i for i, layer in enumerate(self._live_target_layers)
        }
        self.target_layer_num = len(self._live_target_layers)
        self.mtp_draft_device_pools = tuple(packed_draft_device_pools)
        self.layer_num = self.target_layer_num + len(self.mtp_draft_device_pools)

        self.indexer_dtype = storage_info.dtype
        self.size = anchor_host.size
        self.page_num = anchor_host.page_num

        # uint8 storage, so element counts below are byte counts
        self.indexer_page_stride_size = storage_info.page_bytes(self.page_size)
        self.indexer_layout_dim = self.indexer_page_stride_size * self.layer_num
        self.indexer_page_num = (self.size + self.page_size + 1) // self.page_size
        self.size_per_token = storage_info.bytes_per_token_per_layer * self.layer_num

''' + s[b:]
s = sub_once(
    s,
    """        buf_elem_size = self.page_num * self.layer_num * self.indexer_page_stride_size
        requested_bytes = buf_elem_size * self.indexer_dtype.itemsize
""",
    """        requested_bytes = storage_info.host_bytes(
            page_num=self.page_num, layer_num=self.layer_num, page_size=self.page_size
        )
""",
    "dsa requested_bytes",
)
wr(p, s)
os.remove(os.path.join(PKG, p + ".rej"))

# ---- hybrid_pool_assembler.py: rejected hunk #5 (build_anchor_sidecar_stack ->
# declaration assembler) and #8 (old _DsaStrategy removal).
p = "srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py"
rej = open(os.path.join(PKG, p + ".rej")).read().split("\n")
heads = [i for i, l in enumerate(rej) if l.startswith("@@")]
h1 = rej[heads[0] + 1 : heads[1]]
new_side = "\n".join(l[1:] for l in h1 if l[:1] in "+ ")
block = new_side[
    new_side.index("class HostPoolAssemblyResult") : new_side.index(
        "def _build_mha_mla_host_pool("
    )
]
# Our base's PoolEntry/controller keyword is transfer_layer_num (upstream renamed
# it to transfer_layer_id_max later); the value is the same exclusive bound.
block = sub_once(
    block,
    "transfer_layer_id_max=config.layer_binding.transfer_layer_id_max,",
    "transfer_layer_num=config.layer_binding.transfer_layer_id_max,",
    "entry kw",
)
block = sub_once(
    block,
    """        transfer_layer_id_max=transfer_layer_id_max,
        enable_storage_metrics=enable_storage_metrics,
        host_memory_mode=get_memory().hicache_host_memory_mode,""",
    """        transfer_layer_num=transfer_layer_id_max,
        enable_storage_metrics=enable_storage_metrics,
        host_memory_mode=get_memory().hicache_host_memory_mode,""",
    "controller kw",
)
s = rd(p)
a = s.index("def build_anchor_sidecar_stack(")
b = s.index("def _build_mha_mla_host_pool(")
s = s[:a] + block + s[b:]
# old _DsaStrategy (the one built on build_anchor_sidecar_stack)
a = s.index("class _DsaStrategy(StackStrategy):")
b = s.index("class _MiniMaxSparseStrategy(StackStrategy):")
assert "build_anchor_sidecar_stack" in s[a:b]
s = s[:a] + s[b:]
# base linker helper takes (kvcache, page_size)
s = sub_once(
    s,
    """        return _build_dsa_device_pool_group(
            kvcache, page_size, params.mtp_draft_device_pools
        )""",
    """        return _build_dsa_device_pool_group(kvcache, page_size)""",
    "linker call",
)
# Legacy HiRadixCache DSA attach (removed upstream before #40913) still called
# build_anchor_sidecar_stack; route it through the declaration assembler.
a = s.index(
    "        host_pool_group, cache_controller = build_anchor_sidecar_stack(\n"
    "            params=params,\n            kv_pool=kv,"
)
b = s.index("        radix_cache.full_kv_pool_host = host_pool_group.get_pool(PoolName.KV)", a)
s = s[:a] + '''        stack = assemble_host_pools_from_decls(
            params=params,
            decls=kv.host_pool_decls(),
            full_layer_mapping=layer_mapping,
            load_cache_event=load_cache_event,
            storage_backend=get_memory().hicache_storage_backend,
            use_mla=True,
            override_kv_cache_dim=kv.kv_cache_dim,
            prefetch_threshold=prefetch_threshold,
            model_name=get_serving().served_model_name,
            storage_backend_extra_config=extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        host_pool_group = stack.host_pool_group
        cache_controller = stack.cache_controller
''' + s[b:]
wr(p, s)
os.remove(os.path.join(PKG, p + ".rej"))
print("step1 ok")

