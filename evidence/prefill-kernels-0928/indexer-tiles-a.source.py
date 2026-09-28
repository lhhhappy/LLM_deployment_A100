#!/usr/bin/env python3
"""Research the 113 scorer tile after 114 shards queries across eight ranks.

Use actual FP8 unpack, scorer and pooled top-k kernels. This excludes projection,
KV-cache gathering and TP collectives. No serving dispatch changes. The existing
head-wise BF16 rounding, FP32 weighting and masking remain unchanged.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import statistics

import torch
import triton

from bench_moe_candidate_sm80 import DeviceTelemetry


def emit(**record):
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, nargs='+', default=[1024, 2048])
    parser.add_argument('--keys', type=int, default=65536)
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--inner', type=int, default=8)
    args = parser.parse_args()
    assert args.rounds >= 3 and args.inner >= 1
    assert torch.cuda.get_device_capability() == (8, 0)
    from sglang.srt.layers.attention.dsa import sm80_indexer_kernels as ops
    from sglang.srt.layers.attention.dsa.kpool_fp8_index import topk_from_pooled_history_logits

    source = Path(ops.__file__)
    emit(kind='environment', arguments=vars(args), torch=torch.__version__,
         triton=triton.__version__, gpu=torch.cuda.get_device_name(),
         source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
         probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         scope='per-rank 114 rows; FP8 unpack + 113 logits + pooled top-k only')
    # BQ, BK, GROUP, LOOP, warps. Keep the existing implementation and math.
    configs = [(2, 128, 32, 4, 4), (4, 64, 32, 4, 4),
               (4, 128, 32, 4, 4), (8, 64, 32, 4, 4),
               (4, 128, 16, 8, 4), (4, 256, 32, 4, 8)]
    telemetry = DeviceTelemetry()
    try:
        with torch.inference_mode():
            for nq in args.rows:
                nk, h, d = args.keys, 32, 128
                assert nk > nq * 2 + 512
                torch.manual_seed(113928 + nq)
                q = torch.randn(nq, h, d, device='cuda').to(torch.float8_e4m3fn)
                k = torch.randn(nk, d, device='cuda').to(torch.float8_e4m3fn)
                scale = torch.rand(nk, device='cuda') * 0.2 + 0.01
                weight = torch.randn(nq, h, device='cuda') * 0.03
                ks = torch.zeros(nq, device='cuda', dtype=torch.int32)
                # Rank zero's rows in a final 8-rank chunk, pool_size=4.
                ke = (nk - nq * 2 + torch.arange(nq, device='cuda', dtype=torch.int32) // 4).clamp_min(0)
                seq_lens = ke * 4 + 3
                qb = torch.empty(nq, h, d, device='cuda', dtype=torch.bfloat16)
                kb = torch.empty(nk, d, device='cuda', dtype=torch.bfloat16)

                def score(cfg, out):
                    bq, bk, group, loop, warps = cfg
                    ops._unpack_prefill[(triton.cdiv(nq*h*d, 1024),)](
                        q.view(torch.uint8), qb, nq, h, d, *q.stride(), 1024, num_warps=4)
                    ops._unpack_prefill[(triton.cdiv(nk*d, 1024),)](
                        k.view(torch.uint8), kb, nk, 1, d, k.stride(0), 0, k.stride(1), 1024, num_warps=4)
                    return ops._prefill[(triton.cdiv(nq, bq)*triton.cdiv(nk, bk*loop),)](
                        qb, kb, scale, weight, ks, ke, out, nq, nk, h, *qb.stride(), *kb.stride(),
                        scale.stride(0), *weight.stride(), ks.stride(0), ke.stride(0), True,
                        bq, bk, h, group, loop, num_warps=warps, num_stages=1, enable_fp_fusion=False)

                def select(out):
                    return topk_from_pooled_history_logits(
                        out, ke, 4, 2048, seq_lens=seq_lens, row_starts=ks)

                expected = ops.fp8_mqa_logits(q, (k, scale), weight, ks, ke, clean_logits=True)
                expected_indices = select(expected)
                variants = {}
                for cfg in configs:
                    name = 'native' if cfg == configs[0] else '_'.join(map(str, cfg))
                    out = torch.empty_like(expected)
                    kernel = score(cfg, out)
                    indices = select(out)
                    torch.cuda.synchronize()
                    scores_exact = torch.equal(out.view(torch.int32), expected.view(torch.int32))
                    indices_exact = torch.equal(indices, expected_indices)
                    emit(kind='numerics', rows=nq, keys=nk, variant=name,
                         scores_raw_bits=scores_exact, indices_exact=indices_exact,
                         regs=kernel.n_regs, spills=kernel.n_spills, shared=kernel.metadata.shared)
                    if not (scores_exact and indices_exact):
                        # Do not time or promote a numerically different candidate.
                        continue
                    graphs, retained = {}, []
                    for phase in ('scorer', 'scorer_topk'):
                        for _ in range(3):
                            score(cfg, out)
                            if phase == 'scorer_topk':
                                indices = select(out)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            score(cfg, out)
                            if phase == 'scorer_topk':
                                indices = select(out)
                        graphs[phase] = graph
                        retained.append(indices)
                    # External buffers and capture outputs must outlive replay.
                    variants[name] = (graphs, out, retained)
                assert 'native' in variants

                # Fresh values and shorter histories must overwrite masked columns.
                weight.mul_(-0.71)
                ke.sub_(129)
                seq_lens.copy_(ke * 4 + 1)
                expected = ops.fp8_mqa_logits(q, (k, scale), weight, ks, ke, clean_logits=True)
                expected_indices = select(expected)
                for name, (graphs, out, retained) in variants.items():
                    graphs['scorer_topk'].replay()
                    exact = torch.equal(out.view(torch.int32), expected.view(torch.int32))
                    topk_exact = torch.equal(retained[-1], expected_indices)
                    emit(kind='fresh_graph_numerics', rows=nq, keys=nk, variant=name,
                         scores_raw_bits=exact, indices_exact=topk_exact)
                    assert exact and topk_exact

                for phase in ('scorer', 'scorer_topk'):
                    samples = {name: [] for name in variants}
                    rng = random.Random(113 + nq + len(phase))
                    for iteration in range(args.rounds):
                        order = list(variants)
                        rng.shuffle(order)
                        for name in order:
                            graph = variants[name][0][phase]
                            for _ in range(3):
                                graph.replay()
                            torch.cuda.synchronize()
                            begin = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            before = telemetry.read()
                            begin.record()
                            for _ in range(args.inner):
                                graph.replay()
                            end.record()
                            end.synchronize()
                            elapsed = begin.elapsed_time(end) / args.inner
                            samples[name].append(elapsed)
                            emit(kind='timing_round', rows=nq, keys=nk, phase=phase, round=iteration,
                                 variant=name, ms=elapsed, telemetry_before=before, telemetry_after=telemetry.read())
                    for name, values in samples.items():
                        gains = [(a-b)/a*100 for a, b in zip(samples['native'], values)]
                        emit(kind='timing', rows=nq, keys=nk, phase=phase, variant=name,
                             median_ms=statistics.median(values), paired_median_gain_percent=statistics.median(gains),
                             samples=values, paired_gains=gains)
                variants.clear()
                del expected, expected_indices, q, k, qb, kb, out, indices, retained, graph, graphs
                gc.collect()
        emit(kind='complete')
    finally:
        telemetry.close()


if __name__ == '__main__':
    main()
