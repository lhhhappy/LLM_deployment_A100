#!/usr/bin/env python3
"""Install 052 trace hooks into a prepare_src COPY of SGLang (never base_exact)."""

import argparse
from pathlib import Path
import shutil


MARKER = "# [ax-numtrace-052]"


def replace_once(source, old, new):
    count = source.count(old)
    if count != 1:
        raise ValueError(f"numtrace anchor count={count}, wanted 1: {old[:70]!r}")
    return source.replace(old, new, 1)


def install(package_dir):
    model = package_dir / "srt/models/glm5_next.py"
    helper_dst = package_dir / "srt/models/ax_numtrace.py"
    helper_src = Path(__file__).with_name("numtrace_helper.py")
    if not model.is_file() or not helper_src.is_file():
        raise FileNotFoundError("expected prepare_src/sglang and adjacent numtrace_helper.py")
    source = model.read_text()
    if MARKER in source or helper_dst.exists():
        raise RuntimeError("numtrace already installed; refusing double instrumentation")

    source = replace_once(source,
        "class Glm5NextDecoderLayer(nn.Module):",
        "from sglang.srt.models import ax_numtrace\n\n" + MARKER + " import\nclass Glm5NextDecoderLayer(nn.Module):")
    source = replace_once(source,
        "        hidden_states = self.self_attn(\n",
        "        ax_numtrace.capture(self.layer_id, 'attn_input', hidden_states)  " + MARKER + "\n"
        "        hidden_states = self.self_attn(\n")
    source = replace_once(source,
        "        get_attn_tp_context().clear_attn_inputs()\n",
        "        ax_numtrace.capture(self.layer_id, 'attn_output', hidden_states)  " + MARKER + "\n"
        "        get_attn_tp_context().clear_attn_inputs()\n")
    source = replace_once(source,
        "        should_allreduce_fusion = (\n",
        "        ax_numtrace.capture(self.layer_id, 'mlp_input', hidden_states)  " + MARKER + "\n"
        "        should_allreduce_fusion = (\n")
    source = replace_once(source,
        "                hidden_states = self.mlp(\n",
        "                hidden_states = self.mlp(\n")
    source = replace_once(source,
        "\n        if (\n            not (self.dsa_enable_prefill_cp or self.mla_enable_prefill_cp)\n            and should_allreduce_fusion\n",
        "\n        ax_numtrace.capture(self.layer_id, 'mlp_output', hidden_states)  " + MARKER + "\n"
        "        if (\n            not (self.dsa_enable_prefill_cp or self.mla_enable_prefill_cp)\n            and should_allreduce_fusion\n")
    source = replace_once(source,
        "        return hidden_states, residual, topk_indices\n",
        "        ax_numtrace.capture(self.layer_id, 'layer_exit', hidden_states)  " + MARKER + "\n"
        "        ax_numtrace.capture(self.layer_id, 'residual_exit', residual)\n"
        "        return hidden_states, residual, topk_indices\n")
    source = replace_once(source,
        "        device = hidden_states.device\n        zero_allocator = BumpAllocator(\n",
        "        ax_numtrace.begin(self, input_ids, positions, forward_batch, hidden_states)  " + MARKER + "\n"
        "        device = hidden_states.device\n        zero_allocator = BumpAllocator(\n")
    source = replace_once(source,
        "        if not self.pp_group.is_last_rank:\n            return PPProxyTensors(\n",
        "        ax_numtrace.finish(hidden_states, residual)  " + MARKER + "\n"
        "        if not self.pp_group.is_last_rank:\n            return PPProxyTensors(\n")

    # Validate before touching the copy. The helper is a standalone module.
    compile(source, str(model), "exec")
    compile(helper_src.read_text(), str(helper_src), "exec")
    shutil.copyfile(helper_src, helper_dst)
    model.write_text(source)
    print(f"NUMTRACE_INSTALLED {model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    args = parser.parse_args()
    install(args.package_dir)
