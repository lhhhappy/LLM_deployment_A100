"""A100 safe-gate KDA dispatch probe; mutates only independent synthetic pools."""
import argparse
import contextlib
import json
import time
from pathlib import Path

import torch

from sglang.kernels.ops.attention import kda_packed_decode as cuda_packed
from sglang.kernels.ops.attention.fla.fused_recurrent import fused_recurrent_kda_packed_decode
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel


def error(actual, expected):
    delta = actual.float() - expected.float()
    rms = delta.square().mean().sqrt().item()
    denom = max(expected.float().square().mean().sqrt().item(), 1e-8)
    result = dict(max_abs=delta.abs().max().item(), rms=rms, relative_rms=rms / denom)
    if actual.dtype == expected.dtype == torch.bfloat16:
        def ordered(x):
            bits = x.contiguous().view(torch.int16).to(torch.int32)
            return torch.where(bits < 0, -32768 - bits, bits)
        ulp = (ordered(actual) - ordered(expected)).abs()
        result.update(max_bf16_ulp=ulp.max().item(), fraction_different=(ulp != 0).float().mean().item(), fraction_gt1ulp=(ulp > 1).float().mean().item())
    return result


class Case:
    def __init__(self, batch, pitch=False, padding=True, regime="normal", seed=17, qheads=8):
        torch.manual_seed(seed)
        self.batch, self.h, self.k, self.v = batch, 8, 128, 128
        self.qh = qheads
        self.slots = batch + 7
        self.regime = regime
        def rand(*shape):
            return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        self.frames = []
        for frame in range(16):
            mixed = rand(batch, (2 * self.qh + self.h) * self.k)
            a = rand(batch, self.h * self.k)
            b = rand(1, batch, self.h)
            if regime == "weak":
                a.fill_(-16)
                b.mul_(0.5)
            elif regime == "extreme":
                a[:, ::2] = 100 if frame % 2 else -100
                a[:, 1::2] = -100 if frame % 2 else 100
                b[:, :, ::2] = 20
                b[:, :, 1::2] = -20
            self.frames.append((mixed, a, b))
        self.A = torch.linspace(-2, 1, self.h, device="cuda", dtype=torch.float32)
        self.dt = torch.linspace(-0.5, 0.5, self.h * self.k, device="cuda", dtype=torch.float32)
        self.indices = torch.randperm(self.slots, device="cuda")[:batch].to(torch.int32)
        if padding and batch >= 4:
            self.indices[1::5] = -1
        self.active = self.indices >= 0
        self.inactive = torch.ones(self.slots, device="cuda", dtype=torch.bool)
        self.inactive[self.indices[self.active].long()] = False
        self.pitch = pitch
        self.initial = torch.randn(self.slots, self.h, self.v, self.k, device="cuda") * 0.3
        self.cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
        self.kernel = TritonKDAKernel()

    def pool(self):
        # Same layer view in a page-major all-layer envelope: holes must survive.
        envelope = torch.full((self.slots, 3 if self.pitch else 1, self.h, self.v, self.k), 29.0, device="cuda")
        state = envelope[:, 1 if self.pitch else 0]
        state.copy_(self.initial)
        return envelope, state

    def generic(self, state, frame):
        mixed, a, b = self.frames[frame % len(self.frames)]
        q, k, v = mixed.split((self.qh * self.k, self.qh * self.k, self.h * self.k), dim=-1)
        return fused_sigmoid_gating_delta_rule_update(A_log=self.A, a=a, dt_bias=self.dt,
            softplus_beta=1.0, softplus_threshold=20.0,
            q=q.view(1, self.batch, self.qh, self.k), k=k.view(1, self.batch, self.qh, self.k),
            v=v.view(1, self.batch, self.h, self.v), b=b,
            initial_state_source=state, initial_state_indices=self.indices,
            cu_seqlens=self.cu, use_qk_l2norm_in_kernel=True, is_kda=True, lower_bound=-5.0)

    def packed(self, state, frame):
        mixed, a, b = self.frames[frame % len(self.frames)]
        return self.kernel.packed_decode(mixed, a, b, A_log=self.A, dt_bias=self.dt,
            scale=self.k ** -0.5, ssm_states=state, cache_indices=self.indices,
            num_v_heads=self.h, head_v_dim=self.v, lower_bound=-5.0,
            use_cuda_kernel=getattr(self, "use_cuda_kernel", True))

    def reference(self, state, frame):
        mixed, a, b = self.frames[frame % len(self.frames)]
        parts = mixed.split((self.qh * self.k, self.qh * self.k, self.h * self.k), dim=-1)
        q, k = [x.float().view(self.batch, self.qh, self.k).repeat_interleave(self.h // self.qh, dim=1) for x in parts[:2]]
        v = parts[2].float().view(self.batch, self.h, self.k)
        q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt() * self.k ** -0.5
        k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        decay = (-5 * torch.sigmoid(self.A.exp()[None, :, None] * (a.float().view(self.batch, self.h, self.k) + self.dt.view(self.h, self.k)))).exp()
        rows = state[self.indices[self.active].long()].clone() * decay[self.active, :, None, :]
        residual = (v[self.active] - (rows * k[self.active, :, None, :]).sum(-1)) * b.float().view(self.batch, self.h)[self.active, :, None].sigmoid()
        rows += residual[:, :, :, None] * k[self.active, :, None, :]
        state[self.indices[self.active].long()] = rows
        return (rows * q[self.active, :, None, :]).sum(-1).bfloat16()

    def check_pool(self, envelope, state):
        if not torch.equal(state[self.inactive], self.initial[self.inactive]):
            raise AssertionError("inactive pool slots changed")
        if self.pitch and not bool((envelope[:, (0, 2)] == 29).all()):
            raise AssertionError("another envelope layer changed")
        if not bool(state.isfinite().all()):
            raise AssertionError("nonfinite state")


@contextlib.contextmanager
def cuda_enabled(enabled):
    original = cuda_packed.covered
    if not enabled:
        cuda_packed.covered = lambda *args, **kwargs: False
    try:
        yield
    finally:
        cuda_packed.covered = original


def numerical(case, steps, records, label):
    pools = {name: case.pool() for name in ("generic", "triton", "cuda", "torch")}
    maxima = {}
    for step in range(steps):
        outputs = {"generic": case.generic(pools["generic"][1], step)}
        for name in ("triton", "cuda"):
            with cuda_enabled(name == "cuda"):
                outputs[name] = case.packed(pools[name][1], step)
            if not bool((outputs[name][0, ~case.active] == 0).all()):
                raise AssertionError("padding output was not zero")
        if step < 8:
            ref = case.reference(pools["torch"][1], step)
        if step < 8 or step % 128 == 127 or step == steps - 1:
            for name in ("triton", "cuda"):
                stats = {"output": error(outputs[name][0, case.active], outputs["generic"][0, case.active]),
                         "state": error(pools[name][1], pools["generic"][1])}
                if not all(bool(x.isfinite().all()) for x in outputs.values()):
                    raise AssertionError("nonfinite output")
                if stats["output"]["max_abs"] > 0.01 or stats["state"]["max_abs"] > 0.002 or stats["state"]["relative_rms"] > 2e-3:
                    raise AssertionError(f"unexpected numerical drift {label} {step} {name} {stats}")
                for kind, metric in stats.items():
                    key = f"{name}_{kind}"
                    maxima.setdefault(key, {})
                    for field, value in metric.items():
                        maxima[key][field] = max(value, maxima[key].get(field, 0))
            if step < 8:
                for name in ("generic", "triton", "cuda"):
                    maxima.setdefault(f"{name}_torch_first8", {})
                    for field, value in error(outputs[name][0, case.active], ref).items():
                        maxima[f"{name}_torch_first8"][field] = max(value, maxima[f"{name}_torch_first8"].get(field, 0))
    for name in ("generic", "triton", "cuda"):
        case.check_pool(*pools[name])
    records.append(dict(type="numerical", label=label, batch=case.batch, steps=steps, pitch=case.pitch,
                        regime=case.regime, active=int(case.active.sum()), max_errors=maxima))
    print(json.dumps(records[-1]), flush=True)


def bench(case, records):
    for mode in ("generic", "triton", "cuda"):
        with cuda_enabled(mode == "cuda"):
            _, state = case.pool()
            fn = lambda: case.generic(state, 0) if mode == "generic" else case.packed(state, 0)
            for _ in range(15):
                fn()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(10):
                    fn()
            for graph_mode in (False, True):
                samples = []
                wall = []
                for repeat in range(5):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    t = time.perf_counter()
                    start.record()
                    for _ in range(100):
                        graph.replay() if graph_mode else fn()
                    end.record()
                    end.synchronize()
                    count = 1000 if graph_mode else 100
                    samples.append(start.elapsed_time(end) * 1000 / count)
                    wall.append((time.perf_counter() - t) * 1e6 / count)
                records.append(dict(type="performance", batch=case.batch, mode=mode, graph=graph_mode,
                                    median_us=sorted(samples)[2], device_us=samples, wall_us=wall))
                print(json.dumps(records[-1]), flush=True)


def numerical_graph(case, records):
    pools = {name: case.pool() for name in ("generic", "triton", "cuda")}
    graphs, outputs = {}, {}
    for name in pools:
        with cuda_enabled(name == "cuda"):
            state = pools[name][1]
            fn = (lambda: case.generic(state, 0)) if name == "generic" else (lambda: case.packed(state, 0))
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            graphs[name] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[name]):
                outputs[name] = fn()
    for envelope, state in pools.values():
        state.copy_(case.initial)
    original_indices = case.indices.clone()
    maxima = {}
    for step in range(256):
        # Live buffers change, including padding positions and request ordering.
        case.indices.copy_(original_indices.roll(step % case.batch))
        case.active = case.indices >= 0
        for destination, source in zip(case.frames[0], case.frames[1 + step % 15]):
            destination.copy_(source)
        for name in pools:
            graphs[name].replay()
        if step % 16 == 15:
            for name in ("triton", "cuda"):
                if not bool((outputs[name][0, ~case.active] == 0).all()):
                    raise AssertionError("graph padding output not zero")
                for kind, metric in dict(output=error(outputs[name][0, case.active], outputs["generic"][0, case.active]),
                                         state=error(pools[name][1], pools["generic"][1])).items():
                    if not bool(outputs[name].isfinite().all()) or metric["max_abs"] > (0.01 if kind == "output" else 0.002):
                        raise AssertionError(f"graph drift {name} {kind} {metric}")
                    key = f"{name}_{kind}"
                    maxima.setdefault(key, {})
                    for field, value in metric.items():
                        maxima[key][field] = max(value, maxima[key].get(field, 0))
    for envelope, state in pools.values():
        case.check_pool(envelope, state)
    records.append(dict(type="numerical_graph", batch=case.batch, steps=256, pitch=case.pitch,
                        changing_live_indices=True, max_errors=maxima))
    print(json.dumps(records[-1]), flush=True)


def interleaved_bench(case, records):
    graphs = {}
    # External allocations referenced by a graph must outlive every replay.
    pools = {mode: case.pool() for mode in ("generic", "triton", "cuda")}
    for mode in ("generic", "triton", "cuda"):
        with cuda_enabled(mode == "cuda"):
            _, state = pools[mode]
            fn = (lambda: case.generic(state, 0)) if mode == "generic" else (lambda: case.packed(state, 0))
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            graphs[mode] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[mode]):
                for _ in range(10):
                    fn()
    samples = {name: [] for name in graphs}
    for repeat in range(12):
        order = list(graphs)
        order = order[repeat % 3:] + order[:repeat % 3]
        for name in order:
            graph = graphs[name]
            for _ in range(5):
                graph.replay()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100):
                graph.replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end))
    for name, values in samples.items():
        # elapsed ms / 1000 decode steps * 1000 us/ms equals elapsed ms.
        records.append(dict(type="interleaved_performance", batch=case.batch, mode=name,
                            median_us=sorted(values)[len(values)//2], device_us=values))
        print(json.dumps(records[-1]), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--long-steps", type=int, default=2048)
    parser.add_argument("--graph-only", action="store_true")
    parser.add_argument("--extra-only", action="store_true")
    parser.add_argument("--gqa-only", action="store_true")
    parser.add_argument("--warps4", action="store_true")
    parser.add_argument("--wrapper-fragments")
    args = parser.parse_args()
    if args.warps4:
        # Isolated process/module specialization; production source is unchanged.
        cuda_packed._WARPS = 4
    torch.cuda.set_device(0)
    assert torch.cuda.get_device_capability() == (8, 0)
    records = []
    metadata = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                    numerical_tolerances=dict(output_max_abs=0.01, state_max_abs=0.002, state_relative_rms=0.002),
                    note="synthetic correctness, not model accuracy; timing includes full Python dispatch/allocation, graph amortizes 10 decode steps per replay")
    print(json.dumps(metadata), flush=True)
    try:
        if args.wrapper_fragments:
            import sglang.kernels.ops.attention.fla.fused_recurrent as recurrent_module
            import sglang.srt.layers.attention.linear.kernels.kda_triton as triton_module
            fragments = json.loads(Path(args.wrapper_fragments).read_text())
            exec(fragments["function"], recurrent_module.__dict__)
            triton_module.fused_recurrent_kda_packed_decode = recurrent_module.fused_recurrent_kda_packed_decode
            exec(fragments["method"], triton_module.__dict__)
            TritonKDAKernel.packed_decode = triton_module.packed_decode
            def rejected(*args, **kwargs):
                raise AssertionError("CUDA covered called despite use_cuda_kernel=False")
            cuda_packed.covered = rejected
            for batch in (1, 8, 32):
                case = Case(batch, pitch=True)
                case.use_cuda_kernel = False
                numerical(case, 128, records, f"production_wrapper_B{batch}")
                numerical_graph(case, records)
            return
        if args.gqa_only:
            numerical(Case(32, pitch=True, qheads=4), 256, records, "GQA_H4_HV8")
            numerical_graph(Case(32, pitch=True, qheads=4), records)
            return
        if args.extra_only:
            case = Case(32, pitch=True)
            case.frames = [(x, torch.nn.functional.pad(a, (0, 19))[:, :a.shape[-1]],
                            torch.nn.functional.pad(b, (0, 11))[:, :, :b.shape[-1]]) for x, a, b in case.frames]
            numerical(case, 64, records, "gapped_a_b_B32")
            case = Case(8, pitch=True)
            case.initial.zero_()
            numerical(case, 64, records, "zero_state_B8")
            for batch in (1, 4, 8, 12, 16, 24, 28, 32, 36, 40, 44, 48):
                interleaved_bench(Case(batch, padding=False), records)
            return
        if args.graph_only:
            for batch in (4, 8, 32, 48):
                numerical_graph(Case(batch, pitch=True), records)
            return
        for batch in (1, 2, 4, 7, 8, 12, 16, 24, 28, 32, 36, 40, 44, 48):
            numerical(Case(batch, pitch=batch in (7, 8, 32, 48)), 8, records, f"short_B{batch}")
        for batch, regime in ((8, "normal"), (32, "weak"), (8, "extreme")):
            numerical(Case(batch, pitch=True, regime=regime), args.long_steps, records, f"long_B{batch}_{regime}")
        for batch in (1, 2, 4, 8, 12, 16, 24, 28, 32, 36, 40, 44, 48):
            bench(Case(batch, padding=False), records)
    finally:
        torch.cuda.synchronize()
        Path(args.out).write_text(json.dumps(dict(metadata=metadata, records=records), indent=2) + "\n")


if __name__ == "__main__":
    main()
