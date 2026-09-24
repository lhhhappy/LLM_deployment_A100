#!/usr/bin/env python3
"""054: add first-MoE stage receipts to a copied formal-A source tree."""
import argparse
from pathlib import Path

from install_numtrace import install, replace_once


def install_moe(package):
    moe_path = package / 'srt/layers/moe/fused_moe_triton/fused_marlin_moe.py'
    model_path = package / 'srt/models/deepseek_v2.py'
    moe = moe_path.read_text()
    model = model_path.read_text()
    if '# [ax-moe-numtrace-054]' in moe:
        raise RuntimeError('MoE trace already installed')
    moe = replace_once(moe, 'from typing import Optional\n',
        'from typing import Optional\nfrom sglang.srt.models import ax_numtrace  # [ax-moe-numtrace-054]\n')
    moe = replace_once(moe, '    M, K = hidden_states.shape\n',
        '    ax_numtrace.capture_moe("input", hidden_states)\n'
        '    ax_numtrace.capture_moe("topk_ids", topk_ids)\n'
        '    ax_numtrace.capture_moe("topk_weights", topk_weights)\n'
        '    ax_numtrace.capture_moe("w1", w1)\n'
        '    ax_numtrace.capture_moe("w2", w2)\n'
        '    ax_numtrace.capture_moe("w1_scale", w1_scale)\n'
        '    ax_numtrace.capture_moe("w2_scale", w2_scale)\n'
        '    M, K = hidden_states.shape\n')
    moe = replace_once(moe, '    if workspace is None:\n',
        '    ax_numtrace.capture_alignment(sorted_token_ids, expert_ids, num_tokens_post_padded, block_size_m)\n'
        '    if workspace is None:\n')
    moe = replace_once(moe, '    if activation == "silu" and is_gated and gemm1_alpha is not None:\n',
        '    ax_numtrace.capture_moe("gemm1", intermediate_cache1)\n'
        '    if activation == "silu" and is_gated and gemm1_alpha is not None:\n')
    moe = replace_once(moe, '    if expert_map is not None:\n        intermediate_cache3.zero_()\n',
        '    ax_numtrace.capture_moe("activation", intermediate_cache2)\n'
        '    if expert_map is not None:\n        intermediate_cache3.zero_()\n')
    moe = replace_once(moe, '    output = zero_copy_context.get_moe_output(hidden_states)\n',
        '    ax_numtrace.capture_moe("gemm2", intermediate_cache3)\n'
        '    output = zero_copy_context.get_moe_output(hidden_states)\n')
    moe = replace_once(moe,
        '        moe_sum_reduce(\n            intermediate_cache3,\n            output,\n            routed_scaling_factor,\n        )\n        return output\n',
        '        moe_sum_reduce(\n            intermediate_cache3,\n            output,\n            routed_scaling_factor,\n        )\n'
        '        ax_numtrace.capture_moe("local_output", output)\n        return output\n')

    start = model.index('    def forward_normal(\n')
    end = model.index('    def forward_cpu(\n', start)
    block = model[start:end]
    block = replace_once(block,
        '            router_logits = self.gate(hidden_states, gemm_output_zero_allocator)\n',
        '            router_logits = self.gate(hidden_states, gemm_output_zero_allocator)\n'
        '            from sglang.srt.models import ax_numtrace\n'
        '            ax_numtrace.capture_moe("router_logits", router_logits)\n')
    model = model[:start] + block + model[end:]
    compile(moe, str(moe_path), 'exec')
    compile(model, str(model_path), 'exec')
    install(package)
    moe_path.write_text(moe)
    model_path.write_text(model)
    print('MOE_NUMTRACE_INSTALLED', package)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('package_dir', type=Path)
    install_moe(p.parse_args().package_dir)
