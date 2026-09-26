"""Opt-in eager-extend routing for the owner-striped GLM DSA latent cache."""

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def uses_local_extend(forward_batch) -> bool:
    # A batch can be reused by speculative workers; metadata alone must not
    # change the meaning of a later verify/draft/decode phase.
    mode = forward_batch.forward_mode
    md = forward_batch.attn_dcp_metadata
    return mode.is_context_parallel_extend() and md is not None and md.dcp_local_extend


@dataclass(frozen=True)
class LocalExtendPolicy:
    heads: int
    width: int
    dim: int
    max_tokens: int = 512
    min_prefix: int = 4096
    large_max_tokens: int = 0

    def select(self, prefix_tokens: int, query_tokens: int) -> bool:
        # Compare collective bytes per rank for ag_rs, bf16 Q/KV and fp32 O/LSE.
        # KV AG: P*D*2*(W-1)/W; Q AG + O RS + LSE AG:
        # T*H*(W-1)*(6*D + 4*W). Include padded query rows, not just live rows.
        # Separate experimental envelope: H=8, KPool=4 measured on two A100s.
        # Sparse head-tile efficiency can outweigh collective bytes here. It
        # costs extra transient memory (about 133 MiB at T=2048 for P=0), so
        # never enable it implicitly with the short-tail optimization.
        if self.large_max_tokens and 2048 <= query_tokens <= self.large_max_tokens:
            return True
        return (
            0 < query_tokens <= self.max_tokens
            and prefix_tokens >= self.min_prefix
            and prefix_tokens * self.dim * 2
            >= query_tokens * self.heads * self.width * (6 * self.dim + 4 * self.width)
        )

    @classmethod
    def from_runner(cls, mr):
        from sglang.srt.runtime_context import get_parallel
        from sglang.srt.utils import get_bool_env_var

        if not get_bool_env_var("SGLANG_AX_DCP_LOCAL_EXTEND"):
            return None
        import torch

        sa, ps, mc = mr.server_args, get_parallel(), mr.model_config
        arch = mc.hf_config.architectures[0]
        checks = [
            (ps.attn_dcp_size != 2, "DCP width must be 2"),
            (ps.attn_cp_size != 1 or sa.dp_size != 1, "prefill CP / DP"),
            (
                arch not in (
                    "Glm5NextForConditionalGeneration",
                    "Glm5NextForConditionalGenerationNextN",
                ),
                "model architecture",
            ),
            (
                sa.dsa_prefill_backend != "tilelang" or sa.dsa_decode_backend != "tilelang",
                "DSA backend",
            ),
            (mr.kv_cache_dtype != torch.bfloat16 or mc.qk_rope_head_dim != 0, "KV dtype / rope"),
            (
                ps.dcp_comm_backend != "ag_rs" or ps.dcp_replicate_q_proj,
                "DCP communication / Q replication",
            ),
            (sa.enable_hisparse, "HiSparse"),
            (sa.pp_size != 1, "pipeline parallelism"),
            (get_bool_env_var("SGLANG_AX_DSA_SPARSE_TRITON"), "118 Triton backend override"),
            (not str(mr.device).startswith("cuda"), "device"),
        ]
        reasons = [reason for bad, reason in checks if bad]
        if not reasons and torch.cuda.get_device_capability()[0] != 8:
            reasons.append("requires SM80-family MLA dispatch")
        if reasons:
            raise ValueError(
                "SGLANG_AX_DCP_LOCAL_EXTEND=1 unsupported: " + ", ".join(reasons)
            )
        max_tokens = int(os.environ.get("SGLANG_AX_DCP_LOCAL_EXTEND_MAX_TOKENS", "512"))
        if not 1 <= max_tokens <= 1024:
            raise ValueError("DCP local extend max tokens must be in [1, 1024]")
        large_max_tokens = int(
            os.environ.get("SGLANG_AX_DCP_LOCAL_EXTEND_LARGE_MAX", "0")
        )
        if large_max_tokens not in (0, 2048, 8192):
            raise ValueError("DCP local extend large max must be 0, 2048 or 8192")
        if large_max_tokens:
            from sglang.srt.configs.model_config import get_dsa_index_kpool

            if (
                mc.num_attention_heads // ps.attn_tp_size != 8
                or mc.kv_lora_rank != 512
                or get_dsa_index_kpool(mc.hf_config) != 4
                or mc.hf_config.index_topk != 2048
            ):
                raise ValueError(
                    "DCP local large extend requires H=8, D=512, KPool=4, topk=2048"
                )
        policy = cls(
            mc.num_attention_heads // ps.attn_tp_size,
            ps.attn_dcp_size,
            mc.kv_lora_rank,
            max_tokens,
            large_max_tokens=large_max_tokens,
        )
        logger.info("[ax] DCP local extend: %s; eager EXTEND/MIXED only", policy)
        return policy
