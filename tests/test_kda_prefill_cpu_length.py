#!/usr/bin/env python3
"""Focused CPU contract test; --gpu additionally checks the real KDA backend.

CPU mode needs only Python. GPU mode needs the candidate engine in PYTHONPATH
and one CUDA device. It compares unchanged arithmetic/state, then profiles the
per-layer scalar read after warming FLA's separate chunk-index cache. This is
an operator diagnostic, not a whole-model performance or CUDA-graph claim.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
from types import SimpleNamespace as NS
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "engine/sglang/srt/layers/attention/linear/kda_backend.py"


def emit(**record):
    print(json.dumps(record), flush=True)


def source_nodes():
    return ast.parse(SOURCE.read_text()).body


def cpu_namespace():
    enum_path = ROOT / "engine/sglang/srt/model_executor/forward_batch_info.py"
    enum = next(n for n in ast.parse(enum_path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == "ForwardMode")
    import enum as enums

    scope = {"IntEnum": enums.IntEnum, "auto": enums.auto,
             "_AX_KDA_PREFILL_CPU_LENGTH": False}
    definitions = [n for n in source_nodes() if isinstance(n, ast.FunctionDef)
                   and n.name in ("_ax_kda_cpu_logical_tokens", "_ax_kda_resolve_logical_tokens")]
    tree = ast.Module(body=[enum, *definitions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(SOURCE), "exec"), scope)
    return scope


class DeviceEnds:
    """A scalar read is observable; CPU fast-path tests must not read it."""

    def __init__(self, lengths, end=None):
        self.shape = (len(lengths) + 1,)
        self.end = sum(lengths) if end is None else end
        self.reads = 0

    def __getitem__(self, index):
        assert index == -1
        self.reads += 1
        return self.end


class Contracts(unittest.TestCase):
    def setUp(self):
        self.scope = cpu_namespace()
        self.mode = self.scope["ForwardMode"]
        self.host = self.scope["_ax_kda_cpu_logical_tokens"]
        self.resolve = self.scope["_ax_kda_resolve_logical_tokens"]

    def batch(self, lengths, **changes):
        fields = dict(forward_mode=self.mode.EXTEND, batch_size=len(lengths),
                      extend_seq_lens_cpu=list(lengths), attn_cp_metadata=None,
                      tbo_parent_token_range=None)
        fields.update(changes)
        return NS(**fields)

    def test_normal_mixed_and_physical_padding(self):
        rng = random.Random(928)
        lengths_cases = [[8192], [16384], [273, 65], [127, 1, 1], [8191, 0, 0]]
        lengths_cases += [[rng.randrange(0, 100) for _ in range(8)] for _ in range(30)]
        for mode in (self.mode.EXTEND, self.mode.MIXED):
            for lengths in lengths_cases:
                ends = DeviceEnds(lengths)
                fb = self.batch(lengths, forward_mode=mode,
                                extend_num_tokens=sum(lengths) + 8)
                count = self.host(fb, ends)
                self.assertEqual(count, sum(lengths))
                meta = NS(query_start_loc=ends, _ax_cpu_logical_num_tokens=count)
                self.scope["_AX_KDA_PREFILL_CPU_LENGTH"] = True
                self.assertEqual(self.resolve(meta, sum(lengths) + 8), ends.end)
                self.assertEqual(ends.reads, 0)

    def test_default_off_and_fallbacks_read_original_scalar(self):
        ends = DeviceEnds([7, 5], end=19)
        meta = NS(query_start_loc=ends, _ax_cpu_logical_num_tokens=12)
        self.assertEqual(self.resolve(meta, 32), 19)
        self.assertEqual(ends.reads, 1)
        self.scope["_AX_KDA_PREFILL_CPU_LENGTH"] = True
        for count in (None, -1, 33, True):
            meta._ax_cpu_logical_num_tokens = count
            self.assertEqual(self.resolve(meta, 32), 19)
        self.assertEqual(ends.reads, 5)

    def test_unsupported_metadata_is_not_guessed(self):
        ends = DeviceEnds([7, 5])
        cases = [dict(attn_cp_metadata=object()), dict(tbo_parent_token_range=(4, 16)),
                 dict(extend_seq_lens_cpu=None), dict(extend_seq_lens_cpu=[7]),
                 dict(extend_seq_lens_cpu=[7, -1]), dict(extend_seq_lens_cpu=[7, True])]
        cases += [dict(forward_mode=mode) for mode in self.mode
                  if mode not in (self.mode.EXTEND, self.mode.MIXED)]
        for changes in cases:
            self.assertIsNone(self.host(self.batch([7, 5], **changes), ends))
        # GLM populates attn_cp_metadata only after backend metadata init.
        self.assertIsNone(self.host(self.batch([7, 5]), ends, attn_cp_size=2))
        self.assertIsNone(self.host(self.batch([7, 5]), DeviceEnds([12])))
        self.assertEqual(ends.reads, 0)

    def test_real_metadata_hook_populates_only_when_enabled(self):
        class Parent:
            def init_forward_metadata(self, fb):
                self.forward_metadata = NS(query_start_loc=DeviceEnds(fb.extend_seq_lens_cpu),
                                           has_mamba_track_mask=False)

        cls = next(n for n in source_nodes() if isinstance(n, ast.ClassDef)
                   and n.name == "KDAAttnBackend")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == "init_forward_metadata")
        shell = ast.ClassDef(name="Backend", bases=[ast.Name(id="Parent", ctx=ast.Load())],
                             keywords=[], body=[method], decorator_list=[])
        tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), shell], type_ignores=[])
        self.scope["Parent"] = Parent
        self.scope["get_parallel"] = lambda: NS(attn_cp_size=1)
        exec(compile(ast.fix_missing_locations(tree), str(SOURCE), "exec"), self.scope)
        backend = self.scope["Backend"]()
        fb = self.batch([273, 65])
        backend.init_forward_metadata(fb)
        self.assertFalse(hasattr(backend.forward_metadata, "_ax_cpu_logical_num_tokens"))
        self.scope["_AX_KDA_PREFILL_CPU_LENGTH"] = True
        backend.init_forward_metadata(fb)
        self.assertEqual(backend.forward_metadata._ax_cpu_logical_num_tokens, 338)
        self.assertEqual(backend.forward_metadata.query_start_loc.reads, 0)
        fb.attn_cp_metadata = object()
        backend.init_forward_metadata(fb)
        self.assertIsNone(backend.forward_metadata._ax_cpu_logical_num_tokens)
        fb.attn_cp_metadata = None
        self.scope["get_parallel"] = lambda: NS(attn_cp_size=2)
        backend.init_forward_metadata(fb)
        self.assertIsNone(backend.forward_metadata._ax_cpu_logical_num_tokens)


def gpu_checks(rows):
    import torch
    import sglang.srt.layers.attention.linear.kda_backend as module
    from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    assert Path(module.__file__).resolve() == SOURCE.resolve()
    assert torch.cuda.get_device_capability() == (8, 0)
    old_flag = module._AX_KDA_PREFILL_CPU_LENGTH
    emit(kind="environment", probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         backend_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
         device=torch.cuda.get_device_name(), torch=torch.__version__,
         original_flag=old_flag, scope="real KDA backend arithmetic; metadata fixture, no TP communication")
    cases = [([n], 0, ForwardMode.EXTEND) for n in rows]
    cases += [([273, 65], 6, ForwardMode.EXTEND), ([127, 1, 1], 7, ForwardMode.MIXED)]
    try:
        with torch.inference_mode():
            for lengths, pad, mode in cases:
                n, heads, dim = sum(lengths), 8, 128
                width, slots_n = heads * dim, len(lengths)
                generator = torch.Generator(device="cuda").manual_seed(928 + n)
                rand = lambda *shape, dtype=torch.bfloat16: torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
                raw, gate, beta = rand(n + pad, 3 * width), rand(1, n + pad, width), rand(1, n + pad, heads)
                raw[n:] = float("nan")
                cu = torch.tensor([0] + [sum(lengths[:i + 1]) for i in range(slots_n)], dtype=torch.int32, device="cuda")
                slots = torch.arange(slots_n, dtype=torch.int32, device="cuda")
                conv = torch.zeros(slots_n, 3, 3 * width, device="cuda", dtype=torch.bfloat16)
                ssm = torch.zeros(slots_n, heads, dim, dim, device="cuda")
                layer = NS(layer_id=0, conv_weights=rand(3 * width, 4, dtype=torch.float32) * .1,
                           bias=None, q_dim=width, k_dim=width, v_dim=width,
                           head_q_dim=dim, head_k_dim=dim, head_v_dim=dim,
                           A_log=rand(heads, dtype=torch.float32) * .1,
                           dt_bias=rand(width, dtype=torch.float32) * .1, lower_bound=-5.)
                fb = NS(forward_mode=mode, batch_size=slots_n, extend_seq_lens_cpu=lengths,
                        extend_prefix_lens=torch.zeros(slots_n, dtype=torch.int32, device="cuda"),
                        ax_kda_snapshot_offsets=None, ax_kda_snapshot_slots=None,
                        attn_cp_metadata=None, tbo_parent_token_range=None)
                backend = module.KDAAttnBackend.__new__(module.KDAAttnBackend)
                backend.forward_metadata = NS(query_start_loc=cu, mamba_cache_indices=slots,
                                              has_mamba_track_mask=False,
                                              _ax_cpu_logical_num_tokens=module._ax_kda_cpu_logical_tokens(fb, cu))
                backend.req_to_token_pool = NS(mamba2_layer_cache=lambda _: NS(conv=[conv], temporal=ssm))
                backend.kernel_dispatcher = TritonKDAKernel()
                backend.accept_lens_pool = None

                def run(enabled):
                    module._AX_KDA_PREFILL_CPU_LENGTH = enabled
                    return backend.forward_extend(layer, fb, raw, gate, beta)

                # Warm FLA's separate tensor-identity cache before profiling.
                run(False)
                outputs, states, profiles = {}, {}, {}
                for enabled in (False, True):
                    conv.zero_(); ssm.zero_(); torch.cuda.synchronize()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
                        out = run(enabled)
                        torch.cuda.synchronize()
                    outputs[enabled] = out.clone()
                    states[enabled] = (conv.clone(), ssm.clone())
                    profiles[enabled] = sum(e.count for e in prof.key_averages()
                                            if e.key == "aten::_local_scalar_dense")
                equal = torch.equal(outputs[False].view(torch.int16), outputs[True].view(torch.int16))
                conv_equal = torch.equal(states[False][0].view(torch.int16), states[True][0].view(torch.int16))
                ssm_equal = torch.equal(states[False][1].view(torch.int32), states[True][1].view(torch.int32))
                finite = bool(torch.isfinite(outputs[True]).all())
                emit(kind="backend_check", lengths=lengths, padding=pad, mode=mode.name,
                     output_bitexact=equal, conv_bitexact=conv_equal, ssm_bitexact=ssm_equal,
                     finite=finite, local_scalar_dense_counts=profiles)
                assert equal and conv_equal and ssm_equal and finite
                assert profiles[False] == profiles[True] + 1, profiles
                timings = {False: [], True: []}
                rng = random.Random(n)
                for _ in range(7):
                    order = [False, True]
                    rng.shuffle(order)
                    for enabled in order:
                        conv.zero_(); ssm.zero_(); torch.cuda.synchronize()
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        t0 = time.perf_counter()
                        run(enabled)
                        enqueued = time.perf_counter()
                        end.record(); end.synchronize()
                        timings[enabled].append(dict(gpu_ms=start.elapsed_time(end),
                                                     host_ms=(enqueued - t0) * 1000,
                                                     wall_ms=(time.perf_counter() - t0) * 1000))
                emit(kind="backend_timing", lengths=lengths, padding=pad, mode=mode.name,
                     rounds=7, profiler=False, samples=timings,
                     medians={str(enabled): {k: statistics.median(r[k] for r in values)
                                            for k in ("gpu_ms", "host_ms", "wall_ms")}
                              for enabled, values in timings.items()},
                     scope="isolated complete KDA core; excludes projections and TP communication")
    finally:
        module._AX_KDA_PREFILL_CPU_LENGTH = old_flag
    emit(kind="gpu_complete", passed=True, cases=len(cases))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--rows", nargs="+", type=int, default=[8192, 16384])
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    assert result.wasSuccessful()
    emit(kind="cpu_complete", tests=result.testsRun, passed=True)
    if args.gpu:
        gpu_checks(args.rows)


if __name__ == "__main__":
    main()
