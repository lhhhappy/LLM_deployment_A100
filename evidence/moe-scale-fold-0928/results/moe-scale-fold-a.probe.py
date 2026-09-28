#!/usr/bin/env python3
"""Compare native/folded FP8 decoding with exactly the same SM80 MoE configs.

Research only. Both arms use the measured 117 up+down configuration; reduction
stays native. Header SHA stamps and separate constructor/dispatch cache keys
ensure that the two arms load different cubins without changing tile/raster.
"""
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import time

from prepare_humming_scale_fold_probe import cpu_check, prepare, private_compiler, sha


def emit(**record):
    print(json.dumps(record), flush=True)


def canonical(value):
    return json.loads(json.dumps(value))


def measure(torch, fn, inner):
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(inner):
        fn()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / inner


def bit_check(torch, lhs, rhs, **metadata):
    equal = torch.equal(lhs.view(torch.int16), rhs.view(torch.int16))
    finite = bool(torch.isfinite(lhs).all() and torch.isfinite(rhs).all())
    emit(kind="raw_bits", exact=equal, finite=finite, **metadata)
    assert equal and finite, metadata


def scale_allowed(torch, scales):
    # Research dispatch guard, executed outside capture/timing. No source
    # weight/scale tensor is changed, and no unchecked fold launch is exposed.
    return scales.dtype == torch.bfloat16 and bool(
        (torch.isfinite(scales) & (scales >= 0) & (scales <= 255)).all()
    )


def dump_binary(root, name, filename):
    tool = shutil.which("cuobjdump")
    if tool is None:
        return {"sass": "cuobjdump unavailable"}
    result = subprocess.run([tool, "--dump-sass", filename], capture_output=True, text=True)
    destination = root / "results" / "moe-scale-fold-binaries" / Path(filename).parent.name
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "sass.txt").write_text(result.stdout)
    (destination / "sass.stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stderr[-1000:]
    counts = {instruction: len(re.findall(r"\b" + instruction + r"(?:\.|\s)", result.stdout))
              for instruction in ("HMUL2", "HFMA2", "HADD2", "HMMA", "LOP3", "SHF", "F2FP")}
    usage = subprocess.run([tool, "--dump-resource-usage", filename], capture_output=True, text=True)
    (destination / "resource.txt").write_text(usage.stdout + usage.stderr)
    return dict(sass=str(destination / "sass.txt"), sass_mnemonic_counts=counts,
                resource_usage=usage.stdout.strip(), resource_exit=usage.returncode)


def check_gpu_intrinsics(torch):
    """Use the same BF16-pair intrinsics and compiler flags as Humming on SM80."""
    import ctypes
    import dataclasses
    import cuda.bindings.driver as cbd
    from humming.jit.runtime import KernelRuntime

    @dataclasses.dataclass(kw_only=True)
    class ScaleIntrinsicKernel(KernelRuntime):
        name = "ax_scale_intrinsics"

        def init_kernel(self):
            self.code = r'''
#include <cuda_bf16.h>
extern "C" __global__ void ax_scale_intrinsics(
    const unsigned short* scales, unsigned int* output, unsigned int scale_count) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= scale_count * 256) return;
  const unsigned int q = index % 256;
  // Exactly the BF16 bits returned by FP8 bit-field dequantization before
  // mainloop_arith's fixed exponent correction. Exclude FP8 NaN on the host.
  const unsigned int raw16 = ((q & 128) << 8) | ((q & 127) << 4);
  const unsigned int raw_bits = raw16 | (raw16 << 16);
  const unsigned int scale16 = scales[index / 256];
  const unsigned int scale_bits = scale16 | (scale16 << 16);
  const unsigned int factor_bits = 0x7b807b80U;  // BF16 pair 2^120
  const auto raw = *reinterpret_cast<const __nv_bfloat162*>(&raw_bits);
  const auto scale = *reinterpret_cast<const __nv_bfloat162*>(&scale_bits);
  const auto factor = *reinterpret_cast<const __nv_bfloat162*>(&factor_bits);
  auto corrected = __hmul2(raw, factor);
  auto biased = __hmul2(scale, factor);
  auto corrected_bits = *reinterpret_cast<unsigned int*>(&corrected);
  auto biased_bits = *reinterpret_cast<unsigned int*>(&biased);
  // Prevent reassociation/CSE across the explicit native rounding boundary.
  asm volatile("" : "+r"(corrected_bits), "+r"(biased_bits));
  corrected = *reinterpret_cast<const __nv_bfloat162*>(&corrected_bits);
  biased = *reinterpret_cast<const __nv_bfloat162*>(&biased_bits);
  const auto native_value = __hmul2(corrected, scale);
  const auto fold_value = __hmul2(raw, biased);
  output[index * 2] = *reinterpret_cast<const unsigned int*>(&native_value);
  output[index * 2 + 1] = *reinterpret_cast<const unsigned int*>(&fold_value);
}
'''
            self.arg_types = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32)
            self.prepare()

        def __call__(self, scales, output):
            self.check_context()
            config = cbd.CUlaunchConfig()
            config.gridDimX = (scales.numel() * 256 + 255) // 256
            config.gridDimY = config.gridDimZ = 1
            config.blockDimX = 256
            config.blockDimY = config.blockDimZ = 1
            config.hStream = torch.cuda.current_stream().cuda_stream
            result = cbd.cuLaunchKernelEx(config, self.kernel,
                                         ((scales.data_ptr(), output.data_ptr(), scales.numel()), self.arg_types), 0)
            assert int(result[0]) == 0, result

    scales = torch.arange(0x4380, device="cuda", dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    scales = torch.cat([scales, torch.tensor([256, -1, float("inf"), float("nan")],
                                            device="cuda", dtype=torch.bfloat16)])
    output = torch.empty((scales.numel(), 256, 2), device="cuda", dtype=torch.int32)
    kernel = ScaleIntrinsicKernel()
    kernel(scales, output)
    finite_fp8 = (torch.arange(256, device="cuda") & 127) != 127
    mismatch = int((output[:0x4380, finite_fp8, 0] != output[:0x4380, finite_fp8, 1]).sum())
    rejected_mismatches = [int((row[finite_fp8, 0] != row[finite_fp8, 1]).sum())
                           for row in output[0x4380:]]
    emit(kind="gpu_intrinsic_bits", tested_pairs=254 * 0x4380, bf16_lanes_per_pair=2,
         mismatches=mismatch, safe_scale_range=[0, 255], includes_subnormals=True,
         rejected_scale_cases=["256", "-1", "Inf", "NaN"],
         rejected_mismatches=rejected_mismatches, cubin=kernel.kernel_filename,
         cubin_sha256=sha(Path(kernel.kernel_filename).read_bytes()))
    assert mismatch == 0, "SM80 intrinsic/FTZ behavior differs from the CPU dyadic proof"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[8192, 16384])
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--inner", type=int, default=16)
    args = parser.parse_args()
    assert args.rows and all(8192 <= m <= 16384 for m in args.rows)
    assert 3 <= args.rounds <= 15 and 4 <= args.inner <= 32
    os.environ["SGLANG_AX_SM80_FP8_MOE_HUMMING"] = "1"
    os.environ["SGLANG_AX_SM80_MOE_UP_TUNE"] = "1"
    os.environ["SGLANG_AX_SM80_MOE_DOWN_TUNE"] = "1"
    os.environ["SGLANG_AX_SM80_MOE_REDUCE"] = "0"
    os.environ["HUMMING_DISABLE_PARALLEL_BUILD"] = "1"
    import torch
    import humming
    from humming.config import GemmType
    from humming.jit.compiler import Compiler
    from humming.kernel.humming import HummingKernel
    from humming.layer import HummingMethod

    assert torch.cuda.get_device_capability() == (8, 0)
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), Path(__file__).with_name("prepare_humming_scale_fold_probe.py"),
             root / "engine/sglang/srt/layers/quantization/fp8_humming_moe.py",
             root / "engine/sglang/srt/layers/quantization/fp8_humming_tuning.py"]
    emit(kind="environment", gpu=torch.cuda.get_device_name(), torch=torch.__version__,
         humming_source=humming.__file__, arguments=vars(args),
         comparison="current 117 up+down/reduce-off versus identical configs plus scale folding",
         source_sha256={str(p.relative_to(root)): sha(p.read_bytes()) for p in paths})
    emit(**cpu_check())
    metadata = prepare(Compiler.humming_include_dir(), root / "cache/humming_scale_fold_headers")
    emit(kind="headers", **{k: ({n: str(p) for n, p in v.items()} if k == "paths" else v)
                            for k, v in metadata.items()})
    fixture_path = root / "tests/gpu/test_fp8_moe_humming_117.py"
    spec = importlib.util.spec_from_file_location("moe117_scale_fold_fixture", fixture_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    with contextlib.ExitStack() as stack:
        helper.open_runtime(stack)
        checkpoint = helper.make_checkpoint(seed=928117)
        layer = helper.build_layer(checkpoint, True)
        del checkpoint
        core = layer.quant_method.runner_core
        original_get_configs = core.get_humming_gemm_configs
        original_configs = copy.deepcopy(original_get_configs(GemmType.INDEXED))
        common = dict(original_configs)
        # Compile only the two configurations under test, not every startup
        # bucket. The parsed effective configs match the current production
        # table at BOTH benchmark sizes; no shape/order field is repurposed.
        for stage in ("w13", "w2"):
            key = f"{stage}_tuning_config"
            selected = [next(c for lo, hi, c in original_configs[key] if lo < m * 9 <= hi)
                        for m in args.rows]
            assert all(canonical(c) == canonical(selected[0]) for c in selected)
            config = selected[0]
            assert config["block_shape"][0] == 128 and config["block_shape"][2] == 64
            assert not config["use_stream_k"] and not config["use_f16_accum"]
            assert list(config["warp_shape"]) == [64, 64, 64]
            if stage == "w2":
                assert config["block_shape"][1] == 128 and config["num_stages"] == 3 and config["num_ctas_per_sm"] == 2
            else:
                assert config["block_shape"][1] == 256 and config["num_stages"] == 4 and config["num_ctas_per_sm"] == 1
            common[key] = [(0, 1 << 30, copy.deepcopy(config))]
            common[key + "_str"] = json.dumps(common[key])
            emit(kind="effective_config", stage=stage, config=config)
        configs = {name: copy.deepcopy(common) for name in ("native", "fold")}
        for stage in ("w13", "w2"):
            key = f"{stage}_tuning_config_str"
            # The dispatch cache uses raw strings; parsed config is identical.
            configs["fold"][key] = "\n\t" + configs["native"][key]
            assert json.loads(configs["fold"][key]) == json.loads(configs["native"][key])
        selected_variant = {"name": "native"}
        core.get_humming_gemm_configs = lambda mode: configs[selected_variant["name"]]
        stack.callback(setattr, core, "get_humming_gemm_configs", original_get_configs)
        scales = {stage: getattr(layer, layer.humming_metas[stage].weight_scale_name)
                  for stage in ("w13", "w2")}
        scale_hashes = {}
        for stage, tensor in scales.items():
            assert scale_allowed(torch, tensor)
            scale_hashes[stage] = sha(tensor.view(torch.uint8).cpu().numpy().tobytes())
            emit(kind="scale_contract", stage=stage, dtype=str(tensor.dtype),
                 min=float(tensor.min()), max=float(tensor.max()),
                 bytes=tensor.numel() * tensor.element_size(), sha256=scale_hashes[stage])
        boundary = torch.tensor([0, 2.0**-133, 2.0**-126, 1, 255, 256, -1, float("inf"), float("nan")],
                                device="cuda", dtype=torch.bfloat16)
        decisions = [scale_allowed(torch, boundary[i:i+1]) for i in range(boundary.numel())]
        assert decisions == [True] * 5 + [False] * 4
        emit(kind="scale_boundary_dispatch", cases=["zero", "min_subnormal", "min_normal", "one", "max_safe_255",
                                                     "overflow_256", "negative", "infinity", "nan"],
             selected=["fold" if ok else "native" for ok in decisions],
             guard_scope="research dispatch, before capture; original tensors unchanged")

        compile_state, choose = stack.enter_context(private_compiler(metadata, emit))
        check_gpu_intrinsics(torch)
        kernel_receipts = {}
        compiled = False
        for m in args.rows:
            x, topk = helper.make_inputs(m, seed=92820 + m)
            gate = torch.empty(m * 9, 512, device="cuda", dtype=torch.bfloat16)
            activation = torch.empty(m * 9, 256, device="cuda", dtype=torch.bfloat16)
            down = torch.empty(m * 9, 4096, device="cuda", dtype=torch.bfloat16)
            selected_variant["name"] = "native"
            first, second = core._prepare_indexed_gemm_kwargs(topk.topk_ids)
            kwargs = {}
            for name in configs:
                kwargs[name] = {stage: dict(original, tuning_config=configs[name][f"{stage}_tuning_config_str"])
                                for stage, original in (("w13", first), ("w2", second))}

            def up(name):
                return HummingMethod.forward_layer(layer, x, outputs=gate, sublayer_name="w13", **kwargs[name]["w13"])

            def dn(name):
                return HummingMethod.forward_layer(layer, activation, outputs=down, sublayer_name="w2", **kwargs[name]["w2"])

            def full(name):
                selected_variant["name"] = name
                return layer(x, topk)

            for name in ("native", "fold"):
                if not compiled:
                    choose(name)
                up(name)
                if name == "native":
                    gate_reference = gate.clone()
                    core.run_activation(gate, activation)
                else:
                    bit_check(torch, gate, gate_reference, stage="w13", rows=m)
                dn(name)  # Both arms consume the SAME native W13 activation.
                if name == "native":
                    down_reference = down.clone()
                else:
                    bit_check(torch, down, down_reference, stage="w2_same_activation", rows=m)
                if not compiled:
                    for stage in ("w13", "w2"):
                        table = HummingKernel.prepare_kernels(layer.humming_metas[stage].to_str(),
                                                              common["compute_config_str"],
                                                              configs[name][f"{stage}_tuning_config_str"])
                        assert table.numel() == 4
                        kernel_id = int(table[2])
                        kernel = HummingKernel._id2kernel[kernel_id]
                        receipt = dict(variant=name, stage=stage, kernel_id=kernel_id,
                                       cubin=kernel.kernel_filename, cubin_sha256=sha(Path(kernel.kernel_filename).read_bytes()),
                                       parsed_config=canonical(kernel.to_str()),
                                       block_shape=list(kernel.block_shape), warp_shape=list(kernel.warp_shape),
                                       **dump_binary(root, name, kernel.kernel_filename))
                        kernel_receipts[name, stage] = receipt
                        emit(kind="kernel_identity", **receipt)
            if not compiled:
                for stage in ("w13", "w2"):
                    native, fold = (kernel_receipts[name, stage] for name in ("native", "fold"))
                    assert native["kernel_id"] != fold["kernel_id"]
                    assert native["cubin"] != fold["cubin"] and native["cubin_sha256"] != fold["cubin_sha256"]
                    assert native["parsed_config"] == fold["parsed_config"]
                compiled = True
                compile_state["compiles_allowed"] = False
            del gate_reference, down_reference

            # Exercise the research fallback against the actual same input,
            # activation and mutated scale tensor, then restore it exactly.
            if m == args.rows[0]:
                for stage, fn in (("w13", up), ("w2", dn)):
                    tensor = scales[stage]
                    original = tensor.flatten()[0].clone()
                    try:
                        tensor.flatten()[0] = 256
                        allowed = scale_allowed(torch, tensor)
                        name = "fold" if allowed else "native"
                        assert name == "native"
                        reference = fn("native").clone()
                        candidate = fn(name)
                        bit_check(torch, candidate, reference, stage=stage, rows=m, case="scale256_native_fallback")
                        emit(kind="fallback_kernel", stage=stage, selected=name,
                             kernel_id=kernel_receipts[name, stage]["kernel_id"])
                    finally:
                        tensor.flatten()[0].copy_(original)
            with helper.fixed_alignment():
                native_out = full("native")
                fold_out = full("fold")
                bit_check(torch, fold_out, native_out, stage="full_moe_eager", rows=m)
            del native_out, fold_out

            funcs = {"w13": up, "w2": dn, "full_moe": full}
            graphs = {}
            for stage, fn in funcs.items():
                for name in ("native", "fold"):
                    graphs[stage, name] = helper.capture(lambda fn=fn, name=name: fn(name))
            for trial in range(2):
                fresh, fresh_topk = helper.make_inputs(m, seed=110928 + m + trial, x_scale=1 + 3 * trial)
                x.copy_(fresh)
                topk.topk_ids.copy_(fresh_topk.topk_ids)
                topk.topk_weights.copy_(fresh_topk.topk_weights)
                for name in ("native", "fold"):
                    graphs["full_moe", name][0].replay()
                torch.cuda.synchronize()
                bit_check(torch, graphs["full_moe", "fold"][1], graphs["full_moe", "native"][1],
                          stage="full_moe_fresh_graph", rows=m, trial=trial, input_scale=1 + 3 * trial)
                eager = full("native")
                bit_check(torch, graphs["full_moe", "fold"][1], eager, stage="graph_vs_fresh_eager", rows=m, trial=trial)
                del eager, fresh, fresh_topk

            # Return timing to the original uniform inputs/routes. Stage kwargs
            # keep their original fixed alignment; full MoE runs alignment live.
            original_x, original_topk = helper.make_inputs(m, seed=92820 + m)
            x.copy_(original_x)
            topk.topk_ids.copy_(original_topk.topk_ids)
            topk.topk_weights.copy_(original_topk.topk_weights)
            del original_x, original_topk
            for name in ("native", "fold"):
                for _ in range(8):
                    graphs["full_moe", name][0].replay()
            torch.cuda.synchronize()
            times = {(stage, name): [] for stage in funcs for name in configs}
            rng = random.Random(928000 + m)
            for round_id in range(args.rounds):
                keys = list(times)
                rng.shuffle(keys)
                for stage, name in keys:
                    value = measure(torch, graphs[stage, name][0].replay, args.inner)
                    times[stage, name].append(value)
                    emit(kind="timing_round", rows=m, stage=stage, variant=name, round=round_id, graph_ms=value)
            for stage in funcs:
                native, fold = (times[stage, name] for name in ("native", "fold"))
                emit(kind="timing", rows=m, stage=stage, native_ms=statistics.median(native),
                     fold_ms=statistics.median(fold), paired_reduction_pct=statistics.median(
                         [100 * (a - b) / a for a, b in zip(native, fold)]),
                     native_rounds=native, fold_rounds=fold)
            eager_times = {name: [measure(torch, lambda name=name: full(name), 8) for _ in range(3)]
                           for name in ("native", "fold")}
            emit(kind="eager_timing", rows=m, gpu_ms={k: statistics.median(v) for k, v in eager_times.items()})
            del graphs, funcs, gate, down, activation, x, topk, kwargs, first, second
            torch.cuda.synchronize()
        for stage, tensor in scales.items():
            assert sha(tensor.view(torch.uint8).cpu().numpy().tobytes()) == scale_hashes[stage]
        assert original_get_configs(GemmType.INDEXED) == original_configs
        emit(kind="complete", scales_unchanged=True, native_config_cache_unchanged=True,
             no_jit_after_prepare=True, production_files_changed=False)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
