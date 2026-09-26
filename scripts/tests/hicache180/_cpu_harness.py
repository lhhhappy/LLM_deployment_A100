"""Harness for patch 180 tests: run the real SGLang host-pool / assembler /
controller / radix-tree code. On CPU (default) ONLY the raw CUDA byte-copy
kernels (sgl_kernel.kvcacheio + the JIT Mamba mover) are replaced by torch
reference copies, and cudaHostRegister by a no-op. With HC180_DEVICE=cuda the
same tests run with the real kernels, pinned host memory and CUDA streams
(dev box; see gpu/README in the 180 .md).

Nothing here decides what is transferred, which layers, which indices or when:
that is all the code under test. The reference kernels only move bytes, and
they assert the row geometry the caller claims (item_size / layout_dim), so a
geometry bug in the code under test fails loudly instead of being papered over.

The tree under test is whatever `sglang` resolves to on PYTHONPATH (see
run_tests.sh), so the same test file runs against an unpatched and a
patched tree; tests that must fail without 180 are marked in their docstrings.
"""

from __future__ import annotations

import contextlib
import os
from typing import Iterable
from unittest.mock import patch

import torch

DEVICE = os.environ.get("HC180_DEVICE", "cpu")

_REGISTRY: dict[int, torch.Tensor] = {}


def sync():
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()


def register_device_tensors(tensors: Iterable[torch.Tensor]) -> None:
    """Pointer tables (tensor of data_ptr) are resolved through this registry."""
    for t in tensors:
        if t is None or t.numel() == 0:
            continue
        _REGISTRY[t.data_ptr()] = t


def clear_registry() -> None:
    _REGISTRY.clear()


def _rows(t: torch.Tensor) -> torch.Tensor:
    t = t if t.is_contiguous() else t.contiguous()
    return t.view(torch.uint8).reshape(t.shape[0], -1)


def _idx(x: torch.Tensor) -> torch.Tensor:
    return x.to(device="cpu", dtype=torch.int64).reshape(-1)


def _resolve(ptr) -> torch.Tensor:
    ptr = int(ptr)
    if ptr not in _REGISTRY:
        raise AssertionError(f"pointer {ptr:#x} not registered as a device buffer")
    return _REGISTRY[ptr]


def ref_pf_lf(src, dst, src_indices, dst_indices, layer_id, item_size, src_layout_dim, **_):
    """page-first host [rows, layers*item] -> one device layer [rows, item]."""
    s, d = _rows(src), dst.view(torch.uint8).reshape(dst.shape[0], -1)
    assert s.shape[1] == src_layout_dim, (s.shape, src_layout_dim)
    assert d.shape[1] == item_size, (d.shape, item_size)
    lo = layer_id * item_size
    d[_idx(dst_indices)] = s[_idx(src_indices), lo : lo + item_size]


def ref_lf_pf(src_layers, dst, src_indices, dst_indices, item_size, dst_layout_dim, num_layers, **_):
    """every device layer (pointer table) -> page-first host [rows, layers*item]."""
    d = dst.view(torch.uint8).reshape(dst.shape[0], -1)
    assert d.shape[1] == dst_layout_dim, (d.shape, dst_layout_dim)
    assert len(src_layers) == num_layers, (len(src_layers), num_layers)
    si, di = _idx(src_indices), _idx(dst_indices)
    for layer in range(num_layers):
        s = _rows(_resolve(src_layers[layer]))
        assert s.shape[1] == item_size, (layer, s.shape, item_size)
        d[di, layer * item_size : (layer + 1) * item_size] = s[si]


def ref_mamba_lf_pf(src_ptrs, dst, src_indices, dst_indices, item_size, dst_layout_dim, num_layers, **_):
    ref_lf_pf(src_ptrs, dst, src_indices, dst_indices, item_size, dst_layout_dim, num_layers)


def ref_mamba_pf_lf(src, dst, src_indices, dst_indices, layer_id, item_size, src_layout_dim, **_):
    ref_pf_lf(src, dst, src_indices, dst_indices, layer_id, item_size, src_layout_dim)


def ref_staged_lf_pf(ptr_src, src_indices, dst_indices, staging, dst, *,
                     page_size, element_size=None, **_):
    """Same byte-copy contract for the staged CUDA D2H entry point."""
    item_size = staging[0, 0].numel() * staging.element_size()
    assert element_size is None or element_size == item_size
    assert len(src_indices) % page_size == 0
    ref_lf_pf(ptr_src, dst, src_indices, dst_indices, item_size,
              dst[0].numel() * dst.element_size(), staging.shape[1])


def ref_one_layer_mla(cache_dst, indices_dst, cache_src, indices_src, *,
                      element_dim=None, **_):
    """JIT H2D byte mover, retaining the page-first source row stride."""
    dim = element_dim or cache_dst.size(-1)
    assert cache_src.element_size() == cache_dst.element_size()
    src = cache_src.view(-1, dim).view(torch.uint8)
    dst = cache_dst.view(-1, dim).view(torch.uint8)
    assert src.shape[1] == dst.shape[1] == dim * cache_dst.element_size()
    dst[_idx(indices_dst)] = src[_idx(indices_src)]


def _unsupported(name):
    def f(*a, **k):
        raise AssertionError(f"{name} is not emulated on CPU; use --hicache-io-backend kernel + page_first")

    return f


@contextlib.contextmanager
def cpu_kernels():
    """Patch the byte movers into the real host-pool modules (create=True: on a
    CPU build these names are never imported by the modules). No-op on CUDA."""
    if DEVICE != "cpu":
        yield
        return
    from sglang.srt.mem_cache.pool_host import base, common, dsa, mamba, mla

    patches = [
        patch.object(common, "_cuda_host_register", lambda *a, **k: None),
        # Registration is simulated, so destruction must not call CUDA on
        # these unregistered tensors. base imported this function by name.
        patch.object(base, "_cuda_host_unregister", lambda *a, **k: None),
        patch.object(dsa, "_cuda_host_unregister", lambda *a, **k: None),
        # small CI boxes: do not keep the 10 GB production host reserve
        patch.object(base, "HICACHE_HOST_MEMORY_RESERVE_BYTES", 0),
    ]
    for mod in (mla, dsa):
        patches += [
            patch.object(mod, "jit_transfer_hicache_all_layer_mla_staged_lf_pf", ref_staged_lf_pf),
            patch.object(mod, "transfer_kv_per_layer_mla_pf_lf", ref_pf_lf, create=True),
            patch.object(mod, "transfer_kv_all_layer_mla_lf_pf", ref_lf_pf, create=True),
            patch.object(mod, "transfer_kv_per_layer_mla", _unsupported("layer_first"), create=True),
            patch.object(mod, "transfer_kv_all_layer_mla", _unsupported("layer_first"), create=True),
            patch.object(mod, "transfer_kv_direct", _unsupported("direct"), create=True),
        ]
    patches += [
        patch.object(mla, "jit_transfer_hicache_one_layer_mla", ref_one_layer_mla),
        patch.object(mamba, "transfer_kv_mamba_lf_pf", ref_mamba_lf_pf, create=True),
        patch.object(mamba, "transfer_kv_mamba_pf_lf", ref_mamba_pf_lf, create=True),
        patch.object(mamba, "transfer_kv_per_layer_mla", _unsupported("mamba layer_first"), create=True),
    ]
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        try:
            yield
        finally:
            clear_registry()


def publish_args(**overrides):
    """Publish ServerArgs for the code under test (mirrors upstream unit tests)."""
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs

    kwargs = dict(
        model_path="dummy",
        page_size=64,
        enable_hierarchical_cache=True,
        hicache_mem_layout="page_first",
        hicache_io_backend="kernel",
        hicache_write_policy="write_through",
        hicache_ratio=2.0,
    )
    kwargs.update(overrides)
    args = ServerArgs(**kwargs)
    args._mamba_cache_chunk_size = 64
    publish(args, role="scheduler")
    return args


def glm_like_pools(
    *, dsa_layers=(3, 7, 11), num_layers=12, size=4096, mamba_size=16, with_draft=False, draft_size=None,
    dcp_width=1,
):
    """Real HybridReqToTokenPool + HybridLinearKVPool (DSA MLA, kpool=4 compressed
    indexer, page 64) on CPU with GLM-5.3-Flash row geometry (kv_lora_rank 512,
    no rope, bf16 latent; 132 B indexer rows) and a small KDA-like state. The
    optional draft pool mirrors the NEXTN draft runner: one DSA layer sharing the
    target's slot space."""
    from sglang.srt.configs.mamba_utils import (
        Mamba2CacheParams,
        Mamba2StateDType,
        Mamba2StateShape,
    )
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool

    shape = Mamba2StateShape.create(
        tp_world_size=1, intermediate_size=16, n_groups=1, num_heads=1,
        head_dim=16, state_size=4, conv_kernel=4,  # conv row (16+2*4)*3*bf16=144 B: CUDA mamba copy needs 16-byte items
    )
    mamba_layers = [i for i in range(num_layers) if i not in dsa_layers]
    req_pool = HybridReqToTokenPool(
        size=8, mamba_size=mamba_size, mamba_spec_state_size=8, max_context_len=size,
        device=DEVICE, enable_memory_saver=False,
        cache_params=Mamba2CacheParams(
            shape=shape, layers=mamba_layers,
            dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
        ),
        mamba_layer_ids=mamba_layers,
        enable_mamba_extra_buffer=True, enable_mamba_extra_buffer_lazy=False,
    )

    def kv_pool(layer_ids, size=size):
        return HybridLinearKVPool(
            page_size=64, size=size, dtype=torch.bfloat16, head_num=1, head_dim=512,
            full_attention_layer_ids=list(layer_ids), device=DEVICE,
            mamba_pool=req_pool.mamba_pool, use_mla=True, kv_lora_rank=512,
            qk_rope_head_dim=0, use_dsa=True, index_head_dim=128, kv_cache_dim=512,
            index_kpool=4, index_kpool_compress=True, max_running_requests=8,
        )

    target = kv_pool(dsa_layers)
    draft = kv_pool([0], size=draft_size or size) if with_draft else None
    if dcp_width > 1:
        from sglang.srt.runtime_context import get_parallel
        assert get_parallel().attn_dcp_size == dcp_width
        # Keep the real construction result. Rebuilding the indexer here would
        # hide a production allocation bug in the hybrid construction path.
    return req_pool, target, draft


def device_tensors(req_pool, *kv_pools):
    out = []
    for pool in kv_pools:
        if pool is None:
            continue
        full = pool.full_kv_pool
        out += list(full.kv_buffer) + list(full.index_k_with_scale_buffer)
    cache = req_pool.mamba_pool.mamba_cache
    out += [cache.temporal[i] for i in range(cache.temporal.shape[0])]
    for conv in cache.conv:
        out += [conv[i] for i in range(conv.shape[0])]
    return out
