#!/usr/bin/env python3
"""T45: actual patched KDA+conv kernels, random projections, 64x128, A100.

Only package initialization/platform probes are stubbed; arithmetic is loaded
unchanged from the generated candidate. This is an operator test, not a service.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import torch
import triton


def bootstrap(root):
    for name, rel in [('sglang', ''), ('sglang.srt', 'srt'), ('sglang.kernels', 'kernels'),
                      ('sglang.srt.utils', 'srt/utils'), ('sglang.kernels.jit', 'kernels/jit')]:
        m = types.ModuleType(name)
        m.__path__ = [str(root / rel)]
        sys.modules[name] = m
    utils = sys.modules['sglang.srt.utils']
    utils.is_cpu = utils.is_npu = utils.cpu_has_amx_support = lambda: False
    utils.cdiv = triton.cdiv
    utils.next_power_of_2 = triton.next_power_of_2
    common = types.ModuleType('sglang.srt.utils.common')
    common.torch_release = tuple(int(v) for v in torch.__version__.split('+')[0].split('.')[:2])
    sys.modules[common.__name__] = common
    jit = types.ModuleType('sglang.kernels.jit.utils')
    jit.is_arch_support_pdl = lambda: False
    sys.modules[jit.__name__] = jit


ORACLE = None
BACKEND = None
DISPATCH = None
TRITON_EXTEND = None
SOURCE = None


def emit(**x):
    print(json.dumps(x, sort_keys=True), flush=True)


def error(a, b):
    delta = (a.float() - b.float()).abs()
    return {'max_abs': float(delta.max()), 'rms': float(delta.square().mean().sqrt()),
            'equal': torch.equal(a, b), 'elements': a.numel()}


@torch.inference_mode()
def run_case(lengths, boundaries, seed, heads=64, strided=False, warm_prefix=False):
    from sglang.kernels.ops.attention.fla.kda import chunk_kda
    from sglang.kernels.ops.attention.fla.kda_snapshot import store_conv
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn
    torch.manual_seed(seed)
    n, dim = len(lengths), 128
    width = 3 * heads * dim
    device = 'cuda'
    total = sum(lengths)
    # Random projection + conv/gate weights at the real KDA H,D dimensions.
    x = torch.randn(total, 64, device=device, dtype=torch.bfloat16)
    projection = torch.randn(64, width, device=device, dtype=torch.bfloat16) * 0.1
    raw = (x @ projection).to(torch.bfloat16)
    conv_weight = torch.randn(width, 4, device=device) * 0.1
    gate = torch.randn(1, total, heads, dim, device=device, dtype=torch.bfloat16)
    beta = torch.randn(1, total, heads, device=device, dtype=torch.bfloat16)
    a_log = torch.randn(heads, device=device) * 0.1
    bias = torch.randn(heads * dim, device=device) * 0.1
    num_slots = 6 * n + 1
    pitch = heads * dim * dim
    backing = torch.zeros(num_slots, pitch + (17 if strided else 0), device=device)
    ssm = backing[:, :pitch].view(num_slots, heads, dim, dim)
    conv_backing = torch.zeros(num_slots, 3 * width + (11 if strided else 0), device=device, dtype=torch.bfloat16)
    conv = conv_backing[:, :3 * width].view(num_slots, 3, width)
    original = torch.randn_like(ssm[:n]) * (0.1 if warm_prefix else 0)
    original_conv = torch.randn_like(conv[:n]) * (0.1 if warm_prefix else 0)

    def tensors(seq_lengths):
        return torch.tensor([0] + list(torch.tensor(seq_lengths).cumsum(0).tolist()), dtype=torch.int32, device=device)

    def forward(inp, g, b, seq_lengths, active, initial, export=None, kernel=None):
        cu = tensors(seq_lengths)
        indices = torch.tensor(active, device=device, dtype=torch.int64)
        data = inp.clone()
        offsets, slots = (None, None) if export is None else export
        if export is not None and BACKEND is not None:
            ns = types.SimpleNamespace
            layer = ns(layer_id=0,conv_weights=conv_weight,bias=None,q_dim=heads*dim,
                       k_dim=heads*dim,v_dim=heads*dim,head_q_dim=dim,head_k_dim=dim,
                       head_v_dim=dim,A_log=a_log,dt_bias=bias,lower_bound=-5.)
            real_triton = ns()
            real_triton.extend = types.MethodType(TRITON_EXTEND, real_triton)
            dispatcher = ns(extend_kernel=real_triton, triton_kernel=real_triton)
            dispatcher.extend = types.MethodType(DISPATCH, dispatcher)
            backend = ns(forward_metadata=ns(query_start_loc=cu,mamba_cache_indices=indices,
                                             has_mamba_track_mask=True),
                         req_to_token_pool=ns(mamba2_layer_cache=lambda _:ns(conv=[conv],temporal=ssm)),
                         kernel_dispatcher=dispatcher,accept_lens_pool=None)
            fb = ns(forward_mode=ns(is_target_verify=lambda:False,is_draft_extend_v2=lambda:False),
                    extend_prefix_lens=torch.full((n,),64 if initial else 0,device=device),
                    extend_seq_lens_cpu=seq_lengths,ax_kda_snapshot_offsets=offsets,
                    ax_kda_snapshot_slots=slots)
            return BACKEND(backend,layer,fb,data,g.reshape(1,-1,heads*dim),b).clone()
        if export is not None:
            store_conv(data, conv, cu, offsets, slots)
        qkv = causal_conv1d_fn(data.T, conv_weight, None, activation='silu',
                              conv_states=conv.transpose(-1, -2),
                              has_initial_state=torch.full((n,), initial, device=device),
                              cache_indices=indices, query_start_loc=cu,
                              seq_lens_cpu=seq_lengths).T
        q, k, v = [t.reshape(1, -1, heads, dim) for t in qkv.split(heads * dim, dim=-1)]
        return (kernel or chunk_kda)(q, k, v, g, b, initial_state=ssm, initial_state_indices=indices,
                         cu_seqlens=cu, A_log=a_log, dt_bias=bias, lower_bound=-5.,
                         beta_is_raw=True, use_qk_l2norm_in_kernel=True,
                         snapshot_offsets=offsets, snapshot_slots=slots).clone()

    ssm[:n].copy_(original); conv[:n].copy_(original_conv)
    offs = torch.tensor([[l // 64 * 64, b] for l, b in zip(lengths, boundaries)], device=device)
    slots = torch.tensor([[n + i, 2*n + i if b > 0 else -1] for i, b in enumerate(boundaries)], device=device)
    full = forward(raw, gate, beta, lengths, list(range(n)), warm_prefix, (offs, slots))
    saved_final = ssm[:n].clone()
    saved_boundary = ssm[2*n:3*n].clone()
    saved_conv = conv[2*n:3*n].clone()
    saved_end = ssm[n:2*n].clone()
    guard_untouched = bool(torch.count_nonzero(ssm[-1]) == 0 and torch.count_nonzero(conv[-1]) == 0)
    # Disabled exporter is the same forward; snapshots must not change active math.
    ssm[:n].copy_(original); conv[:n].copy_(original_conv)
    ordinary = forward(raw, gate, beta, lengths, list(range(n)), warm_prefix)
    on_off_output = error(full, ordinary)
    on_off_final = error(saved_final, ssm[:n])
    assert on_off_output['equal'] and on_off_final['equal'] and guard_untouched
    off_baseline_output = off_baseline_final = None
    if ORACLE is not None:
        os.environ['SGLANG_AX_KDA_DUAL_SNAPSHOT'] = '0'
        ssm[:n].copy_(original); conv[:n].copy_(original_conv)
        disabled = forward(raw, gate, beta, lengths, list(range(n)), warm_prefix)
        disabled_state = ssm[:n].clone()
        ssm[:n].copy_(original); conv[:n].copy_(original_conv)
        baseline = forward(raw, gate, beta, lengths, list(range(n)), warm_prefix, kernel=ORACLE)
        off_baseline_output = error(disabled, baseline)
        off_baseline_final = error(disabled_state, ssm[:n])
        assert off_baseline_output['equal'] and off_baseline_final['equal']
        os.environ['SGLANG_AX_KDA_DUAL_SNAPSHOT'] = '1'
    starts = [0] + list(torch.tensor(lengths).cumsum(0).tolist())

    def sliced(stops, suffix=False):
        ranges = [slice(starts[i] + (stops[i] if suffix else 0),
                        starts[i+1] if suffix else starts[i] + stops[i]) for i in range(n)]
        return (torch.cat([raw[r] for r in ranges]), torch.cat([gate[:, r] for r in ranges], 1),
                torch.cat([beta[:, r] for r in ranges], 1))

    # Exported aligned end must also be exact (partial tails must NOT use bf16 h).
    ends = [l // 64 * 64 for l in lengths]
    ssm[:n].copy_(original); conv[:n].copy_(original_conv)
    forward(*sliced(ends), ends, list(range(n)), warm_prefix)
    end_error = error(saved_end, ssm[:n])
    assert end_error['max_abs'] < 1e-5, end_error
    if all(b > 0 for b in boundaries):
        ssm[:n].copy_(original); conv[:n].copy_(original_conv)
        forward(*sliced(boundaries), boundaries, list(range(n)), warm_prefix)
        boundary_error = error(saved_boundary, ssm[:n])
        conv_error = error(saved_conv, conv[:n])
        assert boundary_error['max_abs'] < 1e-5, boundary_error
        assert conv_error['equal'], conv_error
        suffix_lengths = [l-b for l,b in zip(lengths,boundaries)]
        resumed = forward(*sliced(boundaries, True), suffix_lengths,
                          list(range(2*n, 3*n)), True)
        expect = torch.cat([full[:, starts[i]+boundaries[i]:starts[i+1]] for i in range(n)], 1)
        resume_error = error(resumed, expect)
        final_error = error(ssm[2*n:3*n], saved_final)
        # BF16 output: report exactness, use one BF16-scale tolerance if the
        # base kernel changes its intra-chunk fusion heuristic with sequence size.
        torch.testing.assert_close(resumed, expect, atol=0.003, rtol=0.03)
        assert final_error['max_abs'] < 1e-5, final_error
    else:
        boundary_error = conv_error = resume_error = final_error = None
        assert torch.count_nonzero(saved_boundary) == 0 and torch.count_nonzero(saved_conv) == 0
    emit(kind='numeric', lengths=lengths, boundaries=boundaries, seed=seed, heads=heads,
         dim=dim, strided=strided, warm_prefix=warm_prefix, boundary=boundary_error,
         conv=conv_error, end=end_error, resume=resume_error, final=final_error,
         exporter_output=on_off_output, exporter_final=on_off_final, guard_untouched=guard_untouched,
         off_baseline_output=off_baseline_output, off_baseline_final=off_baseline_final)


def main():
    global ORACLE, BACKEND, DISPATCH, TRITON_EXTEND, SOURCE
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    os.environ["SGLANG_AX_KDA_DUAL_SNAPSHOT"] = "1"
    bootstrap(args.source.resolve())
    SOURCE = args.source
    from sglang.kernels.ops.attention.fla.kda import chunk_kda
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn
    def method(rel, clsname, name):
        path = args.source / rel
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == clsname)
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        tree = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
        ns = dict(torch=torch,chunk_kda=chunk_kda,causal_conv1d_fn=causal_conv1d_fn)
        exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),ns)
        return ns[name]
    BACKEND = method('srt/layers/attention/linear/kda_backend.py','KDAAttnBackend','forward_extend')
    DISPATCH = method('srt/layers/attention/linear/kda_backend.py','KDAKernelDispatcher','extend')
    TRITON_EXTEND = method('srt/layers/attention/linear/kernels/kda_triton.py','TritonKDAKernel','extend')
    if args.baseline:
        def module(name, filename):
            spec = importlib.util.spec_from_file_location(name, args.baseline / filename)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            return mod
        old_h = module('oracle_h', 'chunk_delta_h.py')
        old_kda = module('oracle_kda', 'kda.py')
        old_kda.chunk_gated_delta_rule_fwd_h = old_h.chunk_gated_delta_rule_fwd_h
        ORACLE = old_kda.chunk_kda
    emit(kind='environment', torch=torch.__version__, triton=triton.__version__,
         device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
         files={str(p.relative_to(args.source)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in args.source.rglob('*.py') if p.name in ('chunk_delta_h.py','kda.py','kda_snapshot.py')})
    cases = [([273, 337], [128, 192], 42, 64, False, False)]
    if not args.smoke:
        cases += [([128, 192], [64, 128], 43, 64, False, False),
                  ([65, 129], [64, 128], 44, 64, True, True),
                  ([1025, 833], [512, 640], 45, 64, True, False),
                  ([273, 337], [64, 256], 46, 8, False, True),
                  ([256, 320], [-1, -1], 47, 64, True, False)]
    for case in cases:
        run_case(*case)
    emit(kind='complete', cases=len(cases), passed=True)


if __name__ == '__main__':
    main()
