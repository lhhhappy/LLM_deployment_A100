"""160: GLM NEXTN on sm80, reusing 110/111/112/113 kernels.

Called during argument resolution, before child processes, model construction,
140's cache guard and 101's cached role-ID lookup. Restart to change policy.
"""
import logging
import os

logger = logging.getLogger(__name__)


def policy(cfg, model_arch, sm):
    if model_arch != 'Glm5NextForConditionalGeneration' or sm != 80:
        return {}, {}
    if cfg.speculative_algorithm != 'EAGLE':  # NEXTN has already been resolved.
        return {}, {}
    if cfg.speculative_eagle_topk != 1:
        raise ValueError('160 GLM sm80 NEXTN requires --speculative-eagle-topk 1')
    if not cfg.speculative_draft_model_path:
        raise ValueError('160 GLM NEXTN requires --speculative-draft-model-path')
    # This policy doesn't change sampler thresholds, output budgets, or metrics.
    fields = dict(dsa_prefill_backend='tilelang', dsa_decode_backend='tilelang',
                  kv_cache_dtype='bfloat16', linear_attn_backend='triton',
                  linear_attn_prefill_backend='triton', linear_attn_decode_backend='triton',
                  linear_attn_verify_backend='triton')
    environment = dict(SGLANG_OPT_DEEPGEMM_HC_PRENORM='0', SGLANG_OPT_USE_TOPK_V2='0',
                       SGLANG_AX_SM80_INDEXER='1', SGLANG_AX_SM80_FP8_MOE_MARLIN='1',
                       SGLANG_AX_KDA_DUAL_SNAPSHOT='0', SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS='')
    return fields, environment


def configure(server_args, model_arch):
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.arg_groups.overrides import declare_resolution
    from sglang.srt.runtime_context import get_platform
    from sglang.srt.utils import get_device_sm

    if not get_platform().is_cuda:
        return
    fields, environment = policy(resolving_view(server_args), model_arch, get_device_sm())
    if not fields:
        return
    declare_resolution(server_args, 'ax_mtp_sm80', **fields)
    for key, value in environment.items():
        os.environ[key] = value
    logger.warning('160 GLM sm80 NEXTN: tilelang DSA, Triton KDA, 110-113 kernels; '
                   '101/105 role splitting and 140 dual snapshots disabled. '
                   'Native extra_buffer verify/accept tracking remains enabled.')
