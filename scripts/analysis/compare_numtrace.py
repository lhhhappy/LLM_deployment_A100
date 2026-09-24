#!/usr/bin/env python3
"""Locate the earliest unequal recorded stage across repeated TP8 prefills.

Input: directory with rank-0.jsonl through rank-7.jsonl. Does not infer which
kernel caused a stage to differ. Parameter checks are samples, not full weights.
"""
import argparse
import json
from pathlib import Path


def read_rank(path, rank):
    forwards = []
    active = None
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row['rank'] != rank:
            raise ValueError(f'{path}: mixed rank')
        if row['kind'] == 'begin':
            if active is not None:
                raise ValueError(f'{path}: incomplete forward')
            active = dict(begin=row, stages=[])
        elif row['kind'] == 'stage':
            if active is None or row['forward'] != active['begin']['forward']:
                raise ValueError(f'{path}: orphan stage')
            active['stages'].append(row)
        elif row['kind'] == 'end':
            if active is None or row['forward'] != active['begin']['forward']:
                raise ValueError(f'{path}: orphan end')
            if row['missing_layers'] or row['observed_layers'] != active['begin']['expected_layers']:
                raise ValueError(f'{path}: missing layers')
            keys = [(s['layer'], s['stage']) for s in active['stages']]
            if len(keys) != len(set(keys)):
                raise ValueError(f'{path}: duplicate stage')
            expected = [(-1, 'embedding')]
            for layer in active['begin']['expected_layers']:
                expected.extend((layer, stage) for stage in (
                    'attn_input', 'attn_output', 'mlp_input', 'mlp_output',
                    'layer_exit', 'residual_exit'))
            expected.extend([(-1, 'pre_final_norm'), (-1, 'model_residual')])
            if [key for key in keys if not key[1].startswith('moe_')] != expected:
                raise ValueError(f'{path}: incomplete or reordered stage coverage')
            forwards.append(active)
            active = None
        else:
            raise ValueError(f'{path}: unknown kind')
    if active is not None:
        raise ValueError(f'{path}: truncated last forward')
    return forwards


MOE_STAGES = ["router_logits", "input", "topk_ids", "topk_weights", "w1", "w2",
              "w1_scale", "w2_scale", "sorted_ids", "expert_ids", "padded_count",
              "gemm1", "activation", "gemm2", "local_output"]


def compare(root, require_moe=False):
    reports = []
    identities = None
    for rank in range(8):
        forwards = read_rank(root / f'rank-{rank}.jsonl', rank)
        groups = {}
        for f in forwards:
            if require_moe:
                moe_keys = [(s['layer'], s['stage']) for s in f['stages'] if s['stage'].startswith('moe_')]
                if moe_keys != [(3, 'moe_' + s) for s in MOE_STAGES]:
                    raise ValueError(f'rank {rank}: incomplete first-MoE coverage')
            b = f['begin']
            key = (b['tokens'], b['input_ids']['sha256'], b['positions']['sha256'])
            groups.setdefault(key, []).append(f)
        if {k[0] for k in groups} != {37, 256} or len(groups) != 2:
            raise ValueError(f'rank {rank}: expected exactly the two frozen input cases')
        if identities is not None and set(groups) != identities:
            raise ValueError(f'rank {rank}: inputs differ across ranks')
        identities = set(groups)
        for key, repeats in sorted(groups.items()):
            if len(repeats) != 3:
                raise ValueError(f'rank {rank}: expected 3 repeats for {key[0]}, got {len(repeats)}')
            anchor = repeats[0]
            for candidate in repeats[1:]:
                a, b = anchor['stages'], candidate['stages']
                if [(x['layer'], x['stage']) for x in a] != [(x['layer'], x['stage']) for x in b]:
                    raise ValueError(f'rank {rank}: different stage coverage')
                diffs = []
                for x, y in zip(a, b):
                    tx, ty = x['tensor'], y['tensor']
                    if tx is None or ty is None:
                        different = tx != ty
                    else:
                        if tx['shape'] != ty['shape'] or tx['dtype'] != ty['dtype']:
                            raise ValueError(f'rank {rank}: tensor shape/dtype mismatch')
                        different = tx['sha256'] != ty['sha256']
                    if different:
                        diffs.append(dict(layer=x['layer'], stage=x['stage']))
                reports.append(dict(rank=rank, tokens=key[0],
                    reference_forward=anchor['begin']['forward'],
                    candidate_forward=candidate['begin']['forward'],
                    input_sha256=key[1],
                    parameter_samples_equal=(anchor['begin']['parameters']['sha256'] ==
                                             candidate['begin']['parameters']['sha256']),
                    first_different_stage=diffs[0] if diffs else None,
                    different_moe_stages=[x['stage'] for x in diffs if x['stage'].startswith('moe_')],
                    differing_stages=len(diffs), stages=len(a)))
    return dict(status='COMPLETE_TP8_TRACE', comparisons=reports,
                scope='Synchronous diagnostic; first unequal stage is a location, not a kernel cause. '
                      'Parameters were sampled, not fully hashed; no performance/quality verdict.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace_dir', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--require-moe', action='store_true')
    a = p.parse_args()
    result = compare(a.trace_dir, a.require_moe)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    for row in result['comparisons']:
        print(f"rank={row['rank']} tokens={row['tokens']} repeat={row['candidate_forward']} "
              f"first={row['first_different_stage']} weights_sample_equal={row['parameter_samples_equal']}")


if __name__ == '__main__':
    main()
