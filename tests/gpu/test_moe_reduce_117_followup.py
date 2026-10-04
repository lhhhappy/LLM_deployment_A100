#!/usr/bin/env python3
"""117 reduce follow-up: exact native identity, dispatch, warmup and fresh-input graphs.

Run with the candidate engine on PYTHONPATH and one A100 visible. No pytest is
needed. --only dispatch exercises the metadata guards without GPU launches;
the other groups require SM80. The layer group reuses the existing 117 fixture
to check both reducers on the *same* actual Humming down-GEMM output, so the
GEMM's inherited stream-K nondeterminism cannot hide a reduction mismatch.
"""

import argparse
import contextlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch
import triton

SWITCH = "SGLANG_AX_SM80_MOE_REDUCE"
H, TK = 4096, 9


def emit(**record):
    print(json.dumps(record), flush=True)


def exact(actual, expected, **record):
    # Compare the BF16 representation, including signed zero.
    ok = torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    emit(kind="identity", exact=ok, **record)
    assert ok, record


class LaunchRecorder:
    def __init__(self, kernel=None):
        self.kernel = kernel
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
            if self.kernel is not None:
                return self.kernel[grid](*args, **kwargs)

        return launch


def native(module, inputs, weights, outputs=None, **kwargs):
    with patch.object(module, "_AX_SM80_MOE_REDUCE", False):
        return module.moe_fused_mul_sum(inputs, weights, outputs=outputs, **kwargs)


def run_dispatch(module):
    """Use meta storage and launcher spies; no unsupported kernel is launched."""
    x = torch.empty((4096, TK, H), dtype=torch.bfloat16, device="meta")
    w = torch.empty((4096, TK), dtype=torch.float32, device="meta")
    y = torch.empty((4096, H), dtype=torch.bfloat16, device="meta")
    base = dict(inputs=x, topk_weights=w, outputs=y, routed_scaling_factor=2.5)
    native_spy, fast_spy = LaunchRecorder(), LaunchRecorder()
    with (
        patch.object(module, "moe_fused_mul_sum_kernel", native_spy),
        patch.object(module, "_moe_fused_mul_sum_sm80_kernel", fast_spy),
        patch.object(module, "get_device_capability", return_value=(8, 0)),
    ):
        # Off must not inspect fast-path eligibility or hardware at all.
        with (
            patch.object(module, "_AX_SM80_MOE_REDUCE", False),
            patch.object(module, "_can_use_sm80_moe_reduce", side_effect=AssertionError),
        ):
            assert module.moe_fused_mul_sum(**base) is y
        grid, args, launch = native_spy.calls.pop()
        assert grid == (4, 512)
        assert args[:3] == (x, w, y)
        assert args[3:] == (None, None, 4096, TK * H, False, False, TK, H, 2.5, 8, 1024)
        assert launch == dict(num_warps=16, num_stages=2)
        assert not fast_spy.calls

        with patch.object(module, "_is_sm80_device", return_value=True):
            assert module.moe_fused_mul_sum(**base) is y
            assert len(fast_spy.calls) == 1 and not native_spy.calls
            cases = {
                "below_threshold": dict(inputs=x[:4095], topk_weights=w[:4095], outputs=y[:4095]),
                "fp16_input": dict(inputs=x.to(torch.float16)),
                "fp32_input": dict(inputs=x.to(torch.float32)),
                "bf16_weights": dict(topk_weights=w.to(torch.bfloat16)),
                "fp16_weights": dict(topk_weights=w.to(torch.float16)),
                "fp32_output": dict(outputs=y.to(torch.float32)),
                "strided_output": dict(outputs=torch.empty((H, 4096), device="meta", dtype=y.dtype).T),
                "other_hidden": dict(inputs=x[:, :, :2048].contiguous(), outputs=y[:, :2048].contiguous()),
                "other_topk": dict(inputs=x[:, :8].contiguous(), topk_weights=w[:, :8].contiguous()),
                "ep": dict(is_ep=True, topk_ids=torch.zeros((4096, TK), dtype=torch.int32, device="meta")),
                "expert_map": dict(expert_map=torch.zeros(32, dtype=torch.int32, device="meta")),
                "other_device_weights": dict(topk_weights=torch.empty((4096, TK))),
                "other_device_output": dict(outputs=torch.empty((4096, H), dtype=y.dtype)),
            }
            for scale in (0.0, -1.0, 0.5, float("inf"), float("nan"), torch.tensor(2.5)):
                cases[f"unsupported_scale_{scale}"] = dict(routed_scaling_factor=scale)
            for name, changed in cases.items():
                call = {**base, **changed}
                before = len(native_spy.calls)
                assert module.moe_fused_mul_sum(**call) is call["outputs"]
                assert len(native_spy.calls) == before + 1, name
                assert len(fast_spy.calls) == 1, name

            # Existing layout rejection remains the same with the flag off/on.
            for enabled in (False, True):
                for changed in (
                    dict(inputs=torch.empty((4096, H, TK), device="meta", dtype=x.dtype).transpose(1, 2)),
                    dict(topk_weights=torch.empty((TK, 4096), device="meta").T),
                ):
                    with patch.object(module, "_AX_SM80_MOE_REDUCE", enabled):
                        try:
                            module.moe_fused_mul_sum(**{**base, **changed})
                        except AssertionError:
                            pass
                        else:
                            raise AssertionError("native layout assertion was bypassed")

        # Fake tensors must avoid both launches and hardware queries.
        from torch._subclasses.fake_tensor import FakeTensorMode

        before = len(native_spy.calls), len(fast_spy.calls)
        with FakeTensorMode(), patch.object(module, "_can_use_sm80_moe_reduce", side_effect=AssertionError):
            fake_x = torch.empty((4096, TK, H), device="cuda", dtype=torch.bfloat16)
            fake_w = torch.empty((4096, TK), device="cuda")
            fake_y = module.moe_fused_mul_sum(fake_x, fake_w)
            assert fake_y.shape == (4096, H) and fake_y.dtype == torch.bfloat16
        assert before == (len(native_spy.calls), len(fast_spy.calls))

    # Device capability is keyed by the explicit tensor device, not GPU 0 or
    # the current device, and a second call does not query CUDA again.
    module._is_sm80_device.cache_clear()
    devices = {torch.device("cuda:3"): (8, 0), torch.device("cuda:5"): (8, 6)}
    with (
        patch.object(torch.version, "hip", None),
        patch.object(torch.cuda, "current_device", side_effect=AssertionError),
        patch.object(torch.cuda, "get_device_capability", side_effect=devices.__getitem__) as capability,
    ):
        assert module._is_sm80_device(torch.device("cuda:3"))
        assert module._is_sm80_device(torch.device("cuda:3"))
        assert not module._is_sm80_device(torch.device("cuda:5"))
        assert not module._is_sm80_device(torch.device("cpu"))
        assert not module._is_sm80_device(torch.device("cuda"))
        assert capability.call_count == 2
    module._is_sm80_device.cache_clear()
    emit(kind="dispatch", ok=True, fallback_cases=len(cases), default_off=True, fake=True)


def make_inputs(rows, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    inputs = torch.randn((rows, TK, H), dtype=torch.bfloat16, device="cuda", generator=generator)
    weights = torch.randn((rows, TK), dtype=torch.float32, device="cuda", generator=generator)
    # All-zero weights, negative weights and cancellation with a small residual.
    weights[0].zero_()
    weights[1].fill_(-0.25)
    inputs[2].zero_()
    inputs[2, 0].fill_(256.0)
    inputs[2, 1].fill_(-256.0)
    inputs[2, 2].fill_(2.0**-8)
    weights[2].fill_(1.0)
    inputs[3].zero_()
    weights[3].zero_()
    inputs[3, 8].fill_(4.0)
    weights[3, 8] = 0.4  # Shared-expert slot participates, including routed scale.
    return inputs, weights


def run_numerics(module):
    for rows in (4095, 4096, 4097, 8192, 8193, 16384, 16385):
        inputs, weights = make_inputs(rows, 117000 + rows)
        output = torch.empty((rows, H), device="cuda", dtype=inputs.dtype)
        reference = torch.empty_like(output)
        for scale in (None, 1, 1.0, 2.5):
            native(module, inputs, weights, outputs=reference, routed_scaling_factor=scale)
            output.fill_(float("nan"))
            result = module.moe_fused_mul_sum(inputs, weights, outputs=output, routed_scaling_factor=scale)
            assert result is output
            exact(result, reference, group="numerics", rows=rows, scale=scale, reused=True)
            assert not torch.count_nonzero(output[0]).item()
            effective_scale = 1.0 if scale is None else scale
            assert torch.all(output[2] == 2.0**-8 * effective_scale).item()
            assert torch.all(output[3] == torch.tensor(1.6 * effective_scale, dtype=output.dtype)).item()
        # Exercise allocating output as well as the reusable-buffer contract.
        exact(module.moe_fused_mul_sum(inputs, weights, routed_scaling_factor=2.5), reference,
              group="numerics", rows=rows, scale=2.5, reused=False)

    # Excluded EP/map rows contain NaNs: accidentally taking the new unmasked
    # reducer would propagate them instead of preserving the native semantics.
    inputs, weights = make_inputs(4096, 117999)
    inputs[:, 1].fill_(float("nan"))
    ids = torch.arange(TK, device="cuda", dtype=torch.int32).repeat(4096, 1)
    ep_ids = ids.clone()
    ep_ids[:, 1] = -1
    expert_map = torch.arange(TK, device="cuda", dtype=torch.int32)
    expert_map[1] = -1
    for name, kwargs in (
        ("ep", dict(topk_ids=ep_ids, is_ep=True)),
        ("expert_map", dict(topk_ids=ids, expert_map=expert_map)),
    ):
        reference = native(module, inputs, weights, routed_scaling_factor=2.5, **kwargs)
        result = module.moe_fused_mul_sum(inputs, weights, routed_scaling_factor=2.5, **kwargs)
        assert torch.isfinite(result).all().item()
        exact(result, reference, group="masked_fallback", route=name)


def run_graph(module):
    for rows in (4097, 8192, 16384):
        inputs, weights = make_inputs(rows, 117100 + rows)
        output = torch.empty((rows, H), device="cuda", dtype=inputs.dtype)
        reference = torch.empty_like(output)
        for scale in (1.0, 2.5):
            # Compile before capture and warm on the capture stream.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    module.moe_fused_mul_sum(inputs, weights, outputs=output, routed_scaling_factor=scale)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                module.moe_fused_mul_sum(inputs, weights, outputs=output, routed_scaling_factor=scale)
            for replay in range(3):
                inputs.normal_()
                weights.normal_()
                output.fill_(float("nan"))
                native(module, inputs, weights, outputs=reference, routed_scaling_factor=scale)
                graph.replay()
                exact(output, reference, group="graph", rows=rows, scale=scale, replay=replay)

    # Independent calls on different streams cannot share accumulators or
    # scratch state. Synchronize only after both streams have been enqueued.
    work = []
    for rows in (4097, 8192):
        inputs, weights = make_inputs(rows, 117300 + rows)
        output = torch.empty((rows, H), device="cuda", dtype=inputs.dtype)
        reference = native(module, inputs, weights, routed_scaling_factor=2.5)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        work.append((inputs, weights, output, reference, stream))
    for inputs, weights, output, reference, stream in work:
        with torch.cuda.stream(stream):
            module.moe_fused_mul_sum(inputs, weights, outputs=output, routed_scaling_factor=2.5)
    for inputs, weights, output, reference, stream in work:
        torch.cuda.current_stream().wait_stream(stream)
        exact(output, reference, group="concurrent_streams", rows=inputs.shape[0])


def run_warmup(module):
    device = torch.device("cuda", torch.cuda.current_device())
    module._warmup_sm80_moe_reduce.cache_clear()
    recorder = LaunchRecorder(module._moe_fused_mul_sum_sm80_kernel)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with patch.object(module, "_moe_fused_mul_sum_sm80_kernel", recorder):
        for scale in (1.0, 2.5):
            module.warmup_sm80_moe_reduce(device, torch.bfloat16, H, TK, scale)
            module.warmup_sm80_moe_reduce(device, torch.bfloat16, H, TK, scale)
    torch.cuda.synchronize()
    assert [(grid, args[3]) for grid, args, _ in recorder.calls] == [
        ((4,), 4096), ((4,), 4097), ((4,), 4096), ((4,), 4097)
    ]
    transient = torch.cuda.max_memory_allocated() - before
    assert transient < 2 * 1024**2, transient
    with patch.object(module, "_AX_SM80_MOE_REDUCE", False), patch.object(
        module, "_is_sm80_device", side_effect=AssertionError
    ):
        module.warmup_sm80_moe_reduce(device, torch.bfloat16, H, TK, 2.5)
    # Triton 3.7's JITFunction calls _do_compile only on a specialization miss.
    # Refuse even a disk-cache reload: real prefill shapes must reuse the two
    # warmup specializations already installed in the in-process kernel cache.
    kernel = module._moe_fused_mul_sum_sm80_kernel
    with patch.object(kernel, "_do_compile", side_effect=AssertionError("warmup missed a JIT specialization")):
        for rows in (4096, 4097, 8192, 8193, 16384, 16385):
            inputs = torch.zeros((rows, TK, H), device=device, dtype=torch.bfloat16)
            weights = torch.zeros((rows, TK), device=device)
            output = torch.empty((rows, H), device=device, dtype=torch.bfloat16)
            for scale in (None, 1, 1.0, 2.5):
                module.moe_fused_mul_sum(inputs, weights, outputs=output, routed_scaling_factor=scale)
                assert not torch.count_nonzero(output).item()
    emit(kind="warmup", ok=True, launches=len(recorder.calls), transient_bytes=transient,
         real_prefill_shapes_without_compilation=True)


def run_layer(module):
    fixture = Path(__file__).with_name("test_fp8_moe_humming_117.py")
    spec = importlib.util.spec_from_file_location("moe117_fixture", fixture)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    import sglang.srt.layers.moe.moe_runner.humming as runner

    checks = []

    def checked_reduce(inputs, topk_weights, outputs=None, **kwargs):
        assert module._can_use_sm80_moe_reduce(
            inputs, topk_weights, outputs, kwargs.get("expert_map"),
            kwargs.get("routed_scaling_factor"), kwargs.get("is_ep", False)
        )
        reference = native(module, inputs, topk_weights, **kwargs)
        result = module.moe_fused_mul_sum(inputs, topk_weights, outputs=outputs, **kwargs)
        exact(result, reference, group="humming_down_output", rows=inputs.shape[0])
        checks.append(inputs.shape[0])
        return result

    with contextlib.ExitStack() as stack:
        helper.open_runtime(stack)
        checkpoint = helper.make_checkpoint()
        layer = helper.build_layer(checkpoint, True)
        del checkpoint
        with patch.object(runner, "moe_fused_mul_sum", checked_reduce):
            for rows in (4096, 4097, 8192, 16384):
                inputs, topk = helper.make_inputs(rows, seed=117200 + rows)
                result = layer(inputs, topk)
                assert torch.isfinite(result).all().item()
    assert checks == [4096, 4097, 8192, 16384], checks
    emit(kind="humming_layer", ok=True, checks=len(checks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="dispatch,warmup,numerics,graph,layer")
    args = parser.parse_args()
    # Prove the serving default and the import-time environment contract, then
    # toggle the module constant only inside this isolated regression process.
    os.environ.pop(SWITCH, None)
    module = importlib.import_module("sglang.kernels.ops.moe.moe_fused_mul_sum")
    assert not module._AX_SM80_MOE_REDUCE
    os.environ[SWITCH] = "1"
    assert not module._AX_SM80_MOE_REDUCE
    groups = args.only.split(",")
    tests = dict(dispatch=run_dispatch, warmup=run_warmup, numerics=run_numerics,
                 graph=run_graph, layer=run_layer)
    assert set(groups) <= tests.keys(), groups
    if set(groups) - {"dispatch"}:
        assert torch.cuda.get_device_capability() == (8, 0), "requires an A100/SM80"
        emit(kind="environment", gpu=torch.cuda.get_device_name(), torch=torch.__version__,
             triton=triton.__version__, source=str(Path(module.__file__).resolve()))
    with patch.object(module, "_AX_SM80_MOE_REDUCE", True):
        for group in groups:
            tests[group](module)
    emit(kind="complete", groups=groups)


if __name__ == "__main__":
    main()
