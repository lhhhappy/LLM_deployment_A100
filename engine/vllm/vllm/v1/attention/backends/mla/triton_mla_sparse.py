# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-Triton sparse MLA backend for SM8x (A100).

Every other CUDA sparse-MLA backend needs SM90+. This one reuses the XPU
backend's bf16 sparse contract (all tokens go through the top-k MQA path) with
a split-KV Triton kernel. Ported from github.com/wtdcode/vllm-backport
(commit a6ef07a3f, Apache-2.0) and adapted to this tree:

- the top-k width is the shared buffer's, not ``index_topk``: GLM-5.3-Flash's
  kpool indexer appends the in-progress pool's ``kpool - 1`` tokens after the
  2048 selected ones, and converting only 2048 columns would drop them;
- ``record_logical_topk_ready`` is a no-op: it only serves shared index
  groups, which this backend does not support.
"""

from typing import ClassVar

import torch

from vllm.platforms.interface import DeviceCapability
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    flat_kv_row_view,
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.mla.xpu_mla_sparse import (
    XPUMLASparseBackend,
    XPUMLASparseImpl,
    XPUMLASparseMetadata,
    XPUMLASparseMetadataBuilder,
)
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    KV_SPLITS_CANDIDATES,
    triton_mla_sparse_attention,
)


class TritonMLASparseMetadataBuilder(XPUMLASparseMetadataBuilder):
    # The metadata is built into persistent, zero-filled buffers and the kernel
    # picks its split count from the (captured) token count, so uniform decode
    # batches can be captured in full CUDA graphs.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class TritonMLASparseImpl(XPUMLASparseImpl):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sm_count: int | None = None
        if self.topk_indices_buffer is not None:
            self._sm_count = num_compute_units(self.topk_indices_buffer.device.index)
        self._warmup_autotune()

    def _warmup_autotune(self) -> None:
        """Run every split-count specialization once at init so autotuning and
        JIT happen before CUDA graph capture and the first request."""
        if self.topk_indices_buffer is None:
            return
        device = self.topk_indices_buffer.device
        topk = self.topk_indices_buffer.shape[-1]
        q = torch.empty(
            1, self.num_heads, self.head_size, dtype=torch.bfloat16, device=device
        )
        kv = torch.empty(64, 1, self.head_size, dtype=torch.bfloat16, device=device)
        indices = torch.zeros(1, 1, topk, dtype=torch.int32, device=device)
        for splits in KV_SPLITS_CANDIDATES:
            triton_mla_sparse_attention(
                q,
                kv,
                indices,
                sm_scale=self.softmax_scale,
                num_kv_splits=splits,
                sm_count=self._sm_count,
            )

    def record_logical_topk_ready(self) -> None:
        return None

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: XPUMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError("TRITON_MLA_SPARSE supports a bf16 KV cache only")
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        num_tokens = q.shape[0]
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_tokens]

        kv_rows, block_stride_rows = flat_kv_row_view(
            kv_c_and_k_pe_cache, attn_metadata.block_size
        )
        topk_indices_global = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            BLOCK_STRIDE_ROWS=block_stride_rows,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
        )
        out = triton_mla_sparse_attention(
            q,
            kv_rows.view(-1, 1, kv_rows.shape[-1]),
            topk_indices_global.view(num_tokens, 1, -1),
            sm_scale=self.softmax_scale,
            sm_count=self._sm_count,
        )
        return out, None


class TritonMLASparseBackend(XPUMLASparseBackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        # Shares the KV cache group with the sparse indexer, which needs 64 on
        # CUDA; the base MultipleOf(1) would let auto-selection settle on 16.
        return [MultipleOf(64)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # 512 = NoPE latent (GLM-5.3-Flash); 576 = latent + 64 RoPE.
        return [512, 576]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # SM90+ have FlashMLA/FlashInfer sparse backends; keep their selection.
        return capability.major == 8

    @staticmethod
    def get_builder_cls() -> type[TritonMLASparseMetadataBuilder]:
        return TritonMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[TritonMLASparseImpl]:
        return TritonMLASparseImpl
