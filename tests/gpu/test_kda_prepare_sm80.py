#!/usr/bin/env python3
"""SM80 KDA prepare: production arm, fallback, raw bits, and warm-JIT contract.

Run on one A100 after installing this worktree's engine. This intentionally
does not measure a complete KDA core; bench_kda_prepare_sm80.py owns its state,
snapshot, eager, and CUDA-graph comparisons.
"""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from unittest import mock

import torch
import triton
import triton.knobs as knobs


FLAG = "SGLANG_AX_KDA_PREFILL_PREPARE"
ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "engine/sglang/kernels/ops/attention/fla/kda_prepare_sm80.py",
    "engine/sglang/srt/layers/attention/linear/kernels/kda_triton.py",
    "engine/sglang/srt/layers/attention/linear/kda_backend.py",
    "tests/gpu/test_kda_prepare_sm80.py",
)


def emit(**record):
    print(json.dumps(record, sort_keys=True), flush=True)


def exact_bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    return torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@contextmanager
def flag(value):
    previous = os.environ.get(FLAG)
    os.environ[FLAG] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = previous


@contextmanager
def watch_triton_cache():
    """Count in-memory misses plus both disk hits and actual compilations.

    Chaining and restoring pre-existing hooks keeps the test usable with the
    engine's optional Triton load diagnostics installed.
    """
    runtime = knobs.runtime
    compilation = knobs.compilation
    assert hasattr(runtime, "jit_cache_hook") and hasattr(compilation, "listener"), (
        "Triton cache hooks are required to prove no first-request compile/load"
    )
    previous_jit = runtime.jit_cache_hook
    previous_listener = compilation.listener
    counts = dict(jit_misses=0, disk_hits=0, compiled=0)

    def jit_hook(*args, **kwargs):
        counts["jit_misses"] += 1
        if previous_jit is not None:
            return previous_jit(*args, **kwargs)

    def listener(**kwargs):
        counts["disk_hits" if kwargs["cache_hit"] else "compiled"] += 1
        if previous_listener is not None:
            return previous_listener(**kwargs)

    runtime.jit_cache_hook = jit_hook
    compilation.listener = listener
    try:
        yield counts
    finally:
        runtime.jit_cache_hook = previous_jit
        compilation.listener = previous_listener


def packed_planes(tokens, device, *, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    packed = torch.randn(
        (tokens, 3, 8, 128), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.2
    packed[::251].zero_()
    packed[1::251].mul_(1e-7)
    q, k, v = (packed[:, plane].unsqueeze(0) for plane in range(3))
    assert all(t.stride()[1:] == (3072, 128, 1) for t in (q, k, v))
    assert all(t.data_ptr() % 16 == 0 for t in (q, k, v))
    return packed, q, k, v


def all_bf16_v_patterns(packed):
    """Fill V with every 16-bit encoding, including +/-0, Inf, and NaNs."""
    tokens = packed.shape[0]
    assert tokens % 64 == 0
    patterns = torch.arange(65536, device=packed.device, dtype=torch.int32)
    patterns = patterns.to(torch.int16).view(torch.bfloat16)
    packed[:, 2] = patterns.reshape(64, 8, 128).repeat(tokens // 64, 1, 1)
    first = packed[:64, 2].contiguous().view(torch.int16).reshape(-1)
    assert torch.equal(first, patterns.view(torch.int16))


def call_extend(kernel, q, k, v, chunk_module):
    """Inspect the actual instance dispatch contract, not downstream numerics."""
    recorded = []
    marker = object()

    def fake_chunk_kda(**kwargs):
        recorded.append(kwargs)
        return marker

    with mock.patch.object(chunk_module, "chunk_kda", side_effect=fake_chunk_kda):
        result = kernel.extend(
            q, k, v,
            g=None, beta=None, ssm_states=None, cache_indices=None,
            query_start_loc=None,
        )
    assert result is marker and len(recorded) == 1
    return recorded[0]


def guard_checks(device, module, prepare):
    from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel

    with flag("0"), mock.patch.object(torch.cuda, "get_device_capability", side_effect=AssertionError("CUDA queried while off")):
        off = TritonKDAKernel()
        assert off.arm_prefill_prepare(device) is False
        assert off._ax_prefill_prepare_device is None
    with flag("1"), mock.patch.object(torch.cuda, "get_device_capability", side_effect=AssertionError("CUDA queried for CPU")):
        unsupported_device = TritonKDAKernel()
        assert unsupported_device.arm_prefill_prepare("cpu") is False
    with flag("1"), mock.patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)), mock.patch.object(
        prepare, "warmup_prepare_sm80", side_effect=AssertionError("warmed unsupported hardware")
    ) as warm:
        unsupported_sm = TritonKDAKernel()
        assert unsupported_sm.arm_prefill_prepare(device) is False
        warm.assert_not_called()
    emit(kind="arm_fallback", off=True, cpu_device=True, unsupported_sm=True)

    # Even a production-shaped input must keep the original FLA invocation off.
    packed, q, k, v = packed_planes(8192, device, seed=928001)
    args = call_extend(off, q, k, v, module)
    assert args["use_qk_l2norm_in_kernel"] is True
    assert args["q"] is q and args["k"] is k and args["v"] is v
    emit(kind="extend_contract", mode="off", native_arguments=True)
    del packed, q, k, v


def unsupported_shapes(armed, device, module, prepare, q, k, v):
    cases = {}

    def reject(label, planes, armed_device=device):
        result = prepare.prepare_qkv_if_supported(*planes, armed_device)
        assert result is None, label
        args = call_extend(armed, *planes, module) if armed_device == device else None
        if args is not None:
            assert args["use_qk_l2norm_in_kernel"] is True
            assert all(args[key] is tensor for key, tensor in zip(("q", "k", "v"), planes))
        cases[label] = True

    reject("other_rows", (q[:, :4096], k[:, :4096], v[:, :4096]))
    reject("wrong_heads", (q[:, :, :4], k[:, :, :4], v[:, :, :4]))
    reject("contiguous_stride", (q.contiguous(), k.contiguous(), v.contiguous()))
    fp16 = torch.empty((q.shape[1], 3, 8, 128), dtype=torch.float16, device=device)
    reject("wrong_dtype", tuple(fp16[:, i].unsqueeze(0) for i in range(3)))
    other_device = torch.device("cuda", device.index + 1)
    reject("other_device", (q, k, v), armed_device=other_device)

    # Offset a flat BF16 allocation by one element, preserving all strides.
    storage = torch.empty((q.shape[1] * 3072 + 1,), dtype=torch.bfloat16, device=device)
    unaligned = storage[1:].view(q.shape[1], 3, 8, 128)
    bad = tuple(unaligned[:, i].unsqueeze(0) for i in range(3))
    assert all(t.stride()[1:] == (3072, 128, 1) and t.data_ptr() % 16 for t in bad)
    reject("unaligned_pointer", bad)
    emit(kind="shape_fallback", cases=cases)


def main():
    assert torch.cuda.is_available(), "Run this GPU test on one A100"
    device = torch.device("cuda", torch.cuda.current_device())
    assert torch.cuda.get_device_capability(device) == (8, 0)
    from sglang.kernels.ops.attention.fla import kda_prepare_sm80 as prepare
    from sglang.kernels.ops.attention.fla import l2norm
    from sglang.srt.layers.attention.linear.kernels import kda_triton as module

    emit(
        kind="environment", gpu=torch.cuda.get_device_name(device),
        torch=torch.__version__, triton=triton.__version__,
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES},
    )
    guard_checks(device, module, prepare)

    with flag("1"), watch_triton_cache() as counts:
        armed = module.TritonKDAKernel()
        assert armed.arm_prefill_prepare(device) is True
        assert armed._ax_prefill_prepare_device == device
        device_cache = prepare._prepare_qkv_kernel.device_caches[device.index][0]
        assert len(device_cache) > 0, "tiny warmup did not retain a loaded kernel"
        warm_counts, warm_variants = counts.copy(), len(device_cache)
        emit(kind="warmup_cache", counts=warm_counts, loaded_variants=warm_variants)

        # A first 8k and first 16k serving-size call must only reuse the tiny
        # warmup specialization: zero JIT misses, disk hits and compilations.
        for tokens in (8192, 16384):
            packed, q, k, v = packed_planes(tokens, device, seed=928000 + tokens)
            all_bf16_v_patterns(packed)
            before = counts.copy()
            before_variants = len(device_cache)
            got = prepare.prepare_qkv_if_supported(q, k, v, device)
            assert got is not None
            torch.cuda.synchronize(device)
            plane_nbytes = v.numel() * v.element_size()
            storage_bytes = [plane.untyped_storage().nbytes() for plane in got]
            assert storage_bytes == [plane_nbytes] * 3, (tokens, storage_bytes, plane_nbytes)
            assert len({plane.untyped_storage().data_ptr() for plane in got}) == 3, (
                tokens, "Q/K/V must not share one backing allocation"
            )
            delta = {key: counts[key] - before[key] for key in counts}
            assert not any(delta.values()) and len(device_cache) == before_variants == warm_variants, (
                tokens, delta, len(device_cache), warm_variants
            )
            emit(kind="serving_cache", tokens=tokens, counts=delta,
                 loaded_variants=len(device_cache), returned_v_storage_bytes=storage_bytes[2])

            q_ref = l2norm.l2norm_fwd(q.contiguous())
            k_ref = l2norm.l2norm_fwd(k.contiguous())
            v_ref = v.contiguous()
            exact = [exact_bits(a, b) for a, b in zip(got, (q_ref, k_ref, v_ref))]
            assert all(exact), (tokens, exact)
            emit(kind="prepare_raw_bits", tokens=tokens, q=exact[0], k=exact[1],
                 v_all_65536_patterns=exact[2])

            before_extend = counts.copy()
            args = call_extend(armed, q, k, v, module)
            torch.cuda.synchronize(device)
            extend_delta = {key: counts[key] - before_extend[key] for key in counts}
            assert not any(extend_delta.values()) and len(device_cache) == warm_variants, (
                tokens, extend_delta, len(device_cache), warm_variants
            )
            assert args["use_qk_l2norm_in_kernel"] is False
            assert all(args[key].is_contiguous() for key in ("q", "k", "v"))
            assert all(exact_bits(args[key], ref) for key, ref in zip(("q", "k", "v"), (q_ref, k_ref, v_ref)))
            emit(kind="extend_contract", mode="armed", tokens=tokens,
                 prepared_arguments=True, cache_counts=extend_delta)
            if tokens == 8192:
                unsupported_shapes(armed, device, module, prepare, q, k, v)
            del packed, q, k, v, q_ref, k_ref, v_ref, got

    emit(kind="complete", status="PASS")


if __name__ == "__main__":
    main()
