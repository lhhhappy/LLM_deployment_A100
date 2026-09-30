#!/usr/bin/env python3
"""CPU audit of saved first-MoE tensors; no equivalence tolerance or score."""
import argparse
import json
from pathlib import Path
import torch


def load(root, rank, forward, stage):
    return torch.load(root / f'rank-{rank}-forward-{forward}-layer-3-moe_{stage}.pt',
                      map_location='cpu', weights_only=True)


def route(root, rank, forward):
    ids=load(root,rank,forward,'topk_ids').reshape(-1).long()
    order=load(root,rank,forward,'sorted_ids').long()
    expert=load(root,rank,forward,'expert_ids').long()
    count=int(load(root,rank,forward,'padded_count').item())
    if count!=order.numel() or not expert.numel() or count%expert.numel():
        raise ValueError('invalid recorded alignment extent')
    block=count//expert.numel();valid=order<ids.numel()
    if not torch.equal(order[valid].sort().values,torch.arange(ids.numel())):
        raise ValueError('missing, repeated or negative token assignment')
    if not torch.all(order[~valid]==ids.numel()):raise ValueError('invalid padding sentinel')
    if not torch.equal(ids[order[valid]],expert.repeat_interleave(block)[valid]):
        raise ValueError('token assigned to wrong expert')
    # Marlin reads valid rows first in each block.
    masks=valid.reshape(-1,block)
    if torch.any(masks[:,1:] & ~masks[:,:-1]):raise ValueError('interleaved padding')
    inverse=torch.empty(ids.numel(),dtype=torch.int64)
    inverse[order[valid]]=torch.arange(count)[valid]//block
    return inverse


def audit(root):
    torch.set_num_threads(4)
    reports=[]
    for rank in range(8):
        for anchor,others in [(1,[2,3]),(4,[5,6])]:
            inv=route(root,rank,anchor)
            for other in others:
                moved=route(root,rank,other)!=inv
                row={'rank':rank,'reference_forward':anchor,'candidate_forward':other,
                     'routing_permutation_valid':True,'tokens_moved_to_different_block':int(moved.sum()),'stages':{}}
                for stage in ['gemm1','activation','gemm2','local_output']:
                    a,b=(load(root,rank,i,stage) for i in (anchor,other))
                    if a.shape!=b.shape:raise ValueError('stage shape mismatch')
                    if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('nonfinite stage')
                    delta=a.float()-b.float();differ=a!=b
                    result={'elements':a.numel(),'different':int(differ.sum()),
                            'max_abs':delta.abs().max().item(),'mean_abs':delta.abs().mean().item(),
                            'relative_l2':(delta.norm()/a.float().norm().clamp_min(1e-30)).item()}
                    if stage=='gemm1':
                        rows=differ.reshape(moved.numel(),-1).any(dim=1)
                        result.update(different_rows=int(rows.sum()),different_rows_same_block=int((rows & ~moved).sum()))
                    row['stages'][stage]=result
                reports.append(row)
    return {'status':'COMPLETE_SAVED_TENSOR_AUDIT','comparisons':reports,
            'scope':'Routing coverage and error magnitudes only; no tolerance or causal verdict.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('trace_dir',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit(args.trace_dir);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
