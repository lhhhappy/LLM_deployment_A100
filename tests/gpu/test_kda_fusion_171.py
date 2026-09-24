#!/usr/bin/env python3
"""171 operator/loader check using the actual candidate SGLang classes.

Run with candidate parent in PYTHONPATH. TP ranks are emulated sequentially on
ONE GPU: this checks slicing, not TP communication or a full model. --config is
the real checkpoint config. JSONL records separate projection timing from state
checks; context is null for a stateless projection, never a fabricated prefix.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.configs.glm5_next import Glm5NextTextConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.models import glm5_next as glm
from sglang.srt.runtime_context import get_parallel, publish
from sglang.srt.server_args import ServerArgs


def emit(**record):
    print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)


def error(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    delta = (a.float() - b.float()).abs()
    return dict(max_abs=delta.max().item(),
                rel_l2=(delta.norm() / b.float().norm().clamp_min(1e-12)).item(),
                exact=torch.equal(a, b))


def check(a, b):
    stats = error(a, b)
    # Screening bound for random BF16 operator inputs, NOT a model tolerance.
    assert stats['rel_l2'] <= .01 and stats['max_abs'] <= .05, stats
    return stats


def checkpoint(config):
    h, d = config.linear_attn_config['num_heads'], config.linear_attn_config['head_dim']
    p, width = h * d, config.hidden_size
    shapes = {name: (p, width) for name in ('q_proj', 'k_proj', 'v_proj')}
    shapes.update(b_proj=(h, width), f_a_proj=(d, width), g_a_proj=(d, width),
                  f_b_proj=(p, d), g_b_proj=(p, d), o_proj=(width, p))
    weights = {name + '.weight': torch.randn(shape, dtype=torch.bfloat16, device='cuda') / shape[1]**.5
               for name, shape in shapes.items()}
    for name in ('q_conv1d', 'k_conv1d', 'v_conv1d'):
        weights[name + '.weight'] = torch.randn(p, 1, 4, dtype=torch.float32, device='cuda') * .1
    weights.update(A_log=torch.randn(h, dtype=torch.float32, device='cuda') * .1,
                   dt_bias=torch.randn(p, dtype=torch.float32, device='cuda') * .1)
    weights['o_norm.weight'] = torch.ones(d, dtype=torch.bfloat16, device='cuda')
    return weights


def make_layer(config, qc, weights, enabled):
    os.environ['SGLANG_AX_KDA_FUSE_PROJ'] = str(int(enabled))
    with torch.device('cuda'):
        layer = glm.Glm5NextLinearAttention(0, config.hidden_size, config, qc,
                                          prefix='model.layers.0.self_attn')
    assert layer.do_fuse_qkvbfg == enabled
    for param in layer.parameters():
        param.fill_(float('nan'))
    # Real model load_weights dispatch on a one-layer shell; no model arithmetic
    # is mocked. DSA postprocessing sees no kv_b_proj weights and is a no-op.
    shell = nn.Module()
    shell.config, shell.quant_config, shell.num_fused_shared_experts = config, qc, 0
    shell.model = nn.Module()
    block = nn.Module()
    block.self_attn = layer
    shell.model.layers = nn.ModuleList([block])
    items = [('model.layers.0.self_attn.' + name, value) for name, value in weights.items()]
    random.Random(171 + enabled).shuffle(items)
    glm.Glm5NextForConditionalGeneration.load_weights(shell, items)
    for name, param in layer.named_parameters():
        assert torch.isfinite(param).all(), ('unloaded', name)
    return layer


def verify_loaded(base, fused, weights, rank, tp):
    p = base.head_dim * base.num_heads // tp
    a = fused.fused_qkvbfg_a_proj.weight
    expected = torch.cat([weights[name + '.weight'][rank*p:(rank+1)*p]
                          for name in ('q_proj', 'k_proj', 'v_proj')] +
                         [weights['b_proj.weight'][rank*base.local_num_heads:(rank+1)*base.local_num_heads],
                          weights['f_a_proj.weight'], weights['g_a_proj.weight']])
    assert torch.equal(a, expected), 'fused load/shard order'
    assert torch.equal(base.qkv_proj.weight, expected[:3*p])
    for i, name in enumerate(('f_b_proj', 'g_b_proj')):
        assert torch.equal(fused.fused_fg_b_proj.weight[i], weights[name + '.weight'][rank*p:(rank+1)*p])
    old_bytes = sum(x.numel()*x.element_size() for x in base.parameters())
    new_bytes = sum(x.numel()*x.element_size() for x in fused.parameters())
    assert old_bytes == new_bytes
    emit(kind='loader', rank=rank, tp=tp, emulated_ranks=True, passed=True,
         old_parameter_bytes=old_bytes, new_parameter_bytes=new_bytes,
         new_persistent_buffer_bytes=0)


def project(layer, x):
    fb = SimpleNamespace(attn_cp_metadata=None)
    fn = layer.forward_qkvbfg_fused if layer.do_fuse_qkvbfg else layer.forward_qkvbfg
    return fn(x, fb)


def state_check(base, fused):
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn, causal_conv1d_update
    from sglang.kernels.ops.attention.fla.kda import chunk_kda
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update
    h, d = base.local_num_heads, base.head_dim
    width = h*d
    states = [(torch.zeros(2,h,d,d,device='cuda',dtype=torch.float32),
               torch.zeros(2,3,3*width,device='cuda',dtype=torch.bfloat16).transpose(-1,-2))
              for _ in range(2)]
    slots = torch.tensor([0,1],device='cuda',dtype=torch.int32)
    history = [0,0]
    # Genuine state continuity: cold ragged extend -> decode -> cached suffix.
    for lengths, decode in (([273,65],False), ([1,1],True), ([37,63],False)):
        x = torch.randn(sum(lengths),base.hidden_size,device='cuda',dtype=torch.bfloat16)
        outputs = []
        for layer, (ssm, conv) in zip((base,fused), states):
            raw,beta,gate,norm_gate = project(layer,x)
            cu = torch.tensor([0,lengths[0],sum(lengths)],device='cuda',dtype=torch.int32)
            if decode:
                qkv = causal_conv1d_update(raw,conv,layer.attn.conv_weights,activation='silu',
                                          conv_state_indices=slots)
            else:
                qkv = causal_conv1d_fn(raw.T,layer.attn.conv_weights,None,activation='silu',
                    conv_states=conv,has_initial_state=torch.tensor([v>0 for v in history],device='cuda'),
                    cache_indices=slots,query_start_loc=cu,seq_lens_cpu=lengths).T
            q,k,v = [part.reshape(1,-1,h,d) for part in qkv.split(width,dim=-1)]
            if decode:
                out = fused_sigmoid_gating_delta_rule_update(layer.A_log,gate.unsqueeze(0),layer.dt_bias,
                    1.,20.,q,k,v,beta.unsqueeze(0),ssm,slots,use_qk_l2norm_in_kernel=True,
                    cu_seqlens=cu,is_kda=True,lower_bound=-5.)
            else:
                out = chunk_kda(q,k,v,gate.reshape(1,-1,h,d),beta.unsqueeze(0),initial_state=ssm,
                    initial_state_indices=slots,cu_seqlens=cu,A_log=layer.A_log,dt_bias=layer.dt_bias,
                    lower_bound=-5.,beta_is_raw=True,use_qk_l2norm_in_kernel=True)
            normalized = layer.o_norm(out,norm_gate.unflatten(-1,(-1,d)))
            # Evaluate the real local GEMM without RowParallelLinear's group
            # allocator context: this probe has no distributed process group.
            outputs.append(layer.o_proj.quant_method.apply(
                layer.o_proj, normalized.squeeze(0).flatten(-2)))
        emit(kind='state', history_tokens=history[:], new_tokens=lengths, decode=decode,
             local_layer_output=check(outputs[1],outputs[0]),
             ssm=check(states[1][0],states[0][0]), conv=check(states[1][1],states[0][1]))
        history = [p+c for p,c in zip(history,lengths)]

    # MTP's non-fused verification kernel (official A sets fused verify off).
    # Verify keeps committed SSM unchanged; acceptance selects a scratch step.
    initial_states = [(ssm.clone(),conv.clone()) for ssm,conv in states]
    x = torch.randn(8,base.hidden_size,device='cuda',dtype=torch.bfloat16)
    verified=[]
    for layer,(ssm,conv) in zip((base,fused),states):
        committed=ssm.clone()
        raw,beta,gate,_ = project(layer,x)
        scratch=torch.zeros(2,4,h,d,d,device='cuda',dtype=torch.float32)
        windows=torch.zeros(2,4,3,3*width,device='cuda',dtype=torch.bfloat16)
        scratch_ids=torch.arange(2,device='cuda',dtype=torch.int32)
        qkv=causal_conv1d_update(raw.view(2,4,3*width).transpose(1,2),conv,
            layer.attn.conv_weights,activation='silu',conv_state_indices=slots,
            intermediate_conv_window=windows.transpose(-1,-2),intermediate_state_indices=scratch_ids
        ).transpose(1,2).reshape(8,3*width)
        q,k,v=[part.reshape(1,-1,h,d) for part in qkv.split(width,dim=-1)]
        out=fused_sigmoid_gating_delta_rule_update(layer.A_log,gate.unsqueeze(0),layer.dt_bias,
            1.,20.,q,k,v,beta.unsqueeze(0),ssm,slots,use_qk_l2norm_in_kernel=True,
            cu_seqlens=torch.tensor([0,4,8],device='cuda',dtype=torch.int32),is_kda=True,
            lower_bound=-5.,disable_state_update=True,intermediate_states_buffer=scratch,
            intermediate_state_indices=scratch_ids,cache_steps=4)
        assert torch.equal(ssm,committed), 'verify overwrote committed state'
        verified.append((out,scratch,windows))
    emit(kind='verify',tokens_per_request=4,history_tokens=history,committed_ssm_unchanged=True,
         output=check(verified[1][0],verified[0][0]),
         scratch=check(verified[1][1],verified[0][1]),windows=check(verified[1][2],verified[0][2]))
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import scatter_mamba_states_after_mtp_verify
    for accepted in (1,2,3,4):
        committed=[]
        for (_,scratch,windows),(ssm0,conv0) in zip(verified,initial_states):
            ssm,conv=ssm0.clone(),conv0.clone()
            cache=SimpleNamespace(temporal=ssm.unsqueeze(0),conv=[conv.transpose(-1,-2).unsqueeze(0)],
                intermediate_ssm=scratch.unsqueeze(0),intermediate_conv_window=[windows.unsqueeze(0)])
            scatter_mamba_states_after_mtp_verify(cache,slots,
                torch.full((2,),accepted-1,device='cuda',dtype=torch.int32),
                torch.full((2,),-1,device='cuda',dtype=torch.int32),torch.zeros_like(slots))
            assert torch.equal(ssm,scratch[:,accepted-1])
            assert torch.equal(conv.transpose(-1,-2),windows[:,accepted-1])
            committed.append((ssm,conv))
        emit(kind='accept',accepted_tokens=accepted,ssm=check(committed[1][0],committed[0][0]),
             conv=check(committed[1][1],committed[0][1]))


def percentiles(values):
    xs = sorted(values)
    return dict(p50=statistics.median(xs), p95=xs[min(len(xs)-1,int(.95*len(xs)))],
                min=min(xs), max=max(xs))


def measure(fn, repeats, inner):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    wall, gpu = [], []
    for _ in range(repeats):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        torch.cuda.synchronize()
        before = time.perf_counter()
        start.record()
        for _ in range(inner): fn()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter()-before)*1000/inner)
        gpu.append(start.elapsed_time(end)/inner)
    return dict(wall_ms=percentiles(wall), cuda_event_ms=percentiles(gpu),
                wall_samples_ms=wall,cuda_event_samples_ms=gpu,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                peak_extra_allocated_bytes=torch.cuda.max_memory_allocated()-allocated)


def benchmark(base, fused, tokens, repeats):
    for count in tokens:
        x = torch.randn(count,base.hidden_size,device='cuda',dtype=torch.bfloat16)
        ref, new = project(base,x), project(fused,x)
        errors = [check(a,b) for a,b in zip(new,ref)]
        # Must reject a real wiring mistake (f/g swapped), even with random input.
        assert error(new[2],ref[3])['rel_l2'] > .1
        emit(kind='projection_numeric', tokens=count, outputs=errors,
             candidate_strides=[list(v.stride()) for v in new], negative_control_rejected=True)
        graphs = []
        for layer in (base,fused):
            for _ in range(3): project(layer,x)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            before = torch.cuda.memory_allocated()
            with torch.cuda.graph(graph):
                out = project(layer,x)
            graph_bytes = torch.cuda.memory_allocated()-before
            for _ in range(3):
                x.normal_()
                graph.replay()
                eager = project(layer,x)
                for a,b in zip(out,eager):
                    assert torch.equal(a,b), 'projection graph reused stale input'
            graphs.append((graph,out,graph_bytes))
        for mode in ('eager','cuda_graph'):
            # Alternating paired order detects drift; all comparisons same input.
            for order in range(2):
                arms = [('base',base,graphs[0]),('fused',fused,graphs[1])]
                if order: arms.reverse()
                for arm,layer,(graph,_,graph_bytes) in arms:
                    fn = (lambda layer=layer:project(layer,x)) if mode=='eager' else graph.replay
                    timing=measure(fn,repeats,10 if count<=1024 else 3)
                    emit(kind='timing',scope='kda_projection_only',arm=arm,mode=mode,order=order,
                         new_tokens=count, context_tokens=None, prefill_requests=None,decode_requests=None,
                         real_rows=count,padded_rows=count,tp=base.tp_size,emulated_ranks=True,
                         dtype='bfloat16',scatter=None,mtp=None,graph=mode=='cuda_graph',
                         rank_max_ms=None,kv_capacity=None,kda_slots=None,repeats=repeats,
                         graph_allocated_bytes=graph_bytes,**timing)
        del graphs, ref, new, graph, out, eager, fn


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--patch',type=Path,required=True)
    parser.add_argument('--tokens',default='1,4,33,37,63,65,256,1024,4096,8192,16384')
    parser.add_argument('--ranks',default='0,1,2,3,4,5,6,7')
    parser.add_argument('--repeats',type=int,default=11)
    args=parser.parse_args()
    torch.manual_seed(171)
    torch.set_default_dtype(torch.bfloat16)
    publish(ServerArgs(model_path=str(args.config.parent),tp_size=8,
                       disable_custom_all_reduce=True),role='scheduler')
    config_json=json.loads(args.config.read_text())
    config=Glm5NextTextConfig(**config_json['text_config'])
    qc=Fp8Config.from_config(config_json['quantization_config'])
    qc.packed_modules_mapping=glm.Glm5NextForConditionalGeneration.packed_modules_mapping
    source=Path(glm.__file__)
    emit(kind='environment',torch=torch.__version__,device=torch.cuda.get_device_name(),
         capability=list(torch.cuda.get_device_capability()),tp=8,emulated_ranks=True,
         source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
         patch_sha256=hashlib.sha256(args.patch.read_bytes()).hexdigest(),
         config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest())
    weights=checkpoint(config)
    ranks=[int(v) for v in args.ranks.split(',')]
    for rank in ranks:
        with get_parallel().override(tp_size=8,tp_rank=rank,attn_tp_size=8,attn_tp_rank=rank):
            base=make_layer(config,qc,weights,False)
            fused=make_layer(config,qc,weights,True)
            verify_loaded(base,fused,weights,rank,8)
            x=torch.randn(37,config.hidden_size,device='cuda',dtype=torch.bfloat16)
            emit(kind='rank_numeric',rank=rank,errors=[check(a,b) for a,b in zip(project(fused,x),project(base,x))])
            if rank==ranks[0]:
                # A checkpoint's dtype string must not force stage B away from
                # the dtype actually used to allocate stage A's parameters.
                torch.set_default_dtype(torch.float16)
                half=make_layer(config,qc,weights,True)
                assert half.fused_fg_b_proj.weight.dtype==half.fused_qkvbfg_a_proj.weight.dtype==torch.float16
                assert all(v.dtype==torch.float16 for v in project(half,x.half()))
                emit(kind='dtype_override',dtype='float16',passed=True)
                del half
                torch.set_default_dtype(torch.bfloat16)
                state_check(base,fused)
                for arm,layer in (('base',base),('fused',fused)):
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                        project(layer,x)
                    emit(kind='calls',arm=arm,ops={event.key:event.count for event in prof.key_averages()
                         if event.key in ('aten::linear','aten::mm','aten::bmm','aten::clone','aten::contiguous')})
                benchmark(base,fused,[int(v) for v in args.tokens.split(',')],args.repeats)
            del base,fused
    emit(kind='complete',passed=True,ranks=ranks)


if __name__=='__main__':
    main()
