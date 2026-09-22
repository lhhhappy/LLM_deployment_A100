#!/usr/bin/env python3
"""T43 bounded decode warp/decoder sweep, graphs of 20 calls; preserves emitted sources."""
from pathlib import Path
import statistics
import sys
import torch
import test_sm80_indexer_112 as t

root=Path(sys.argv[1])
torch.manual_seed(112)
for name in ('initial', 'bits'):
    for warps in (4,8,16):
        src=(root/(name+'.py')).read_text().replace('num_warps=4, enable_fp_fusion=False',
                                                  f'num_warps={warps}, enable_fp_fusion=False')
        path=root/f'{name}_decode{warps}.py'
        path.write_text(src)
        m=t.module(f'{name}_decode{warps}',path)
        for L in (32000,190000):
            args=t.decode_case(6,L,1)
            t.compare(m.fp8_paged_mqa_logits(*args),t.old.fp8_paged_mqa_logits(*args),f'{name}/{warps}/{L}')
            samples=t.graph_measure(m.fp8_paged_mqa_logits,args,{})
            t.emit(kind='decode_tune',name=name,warps=warps,L=L,ms=statistics.median(samples),samples=samples)
t.emit(kind='complete',mode='decode_tune')
