#!/usr/bin/env python3
"""Record compiled kernels actually dispatched by final 113; no alternate launch path."""
import hashlib
import json
from pathlib import Path
import sys
import torch
import test_sm80_indexer_113 as t

outdir=Path(sys.argv[1]);outdir.mkdir(exist_ok=True,parents=True)
compiled={}
for name in ('_unpack_prefill','_prefill','_paged'):
    obj=getattr(t.new,name)
    def decorate(original,label):
        def run(*args,**kwargs):
            kernel=original(*args,**kwargs)
            compiled[(label,kernel.hash)]=kernel
            return kernel
        return run
    obj.run=decorate(obj.run,name)
args,kw=t.t.ragged_case(8192,190000,True,'causal')
t.new.fp8_mqa_logits(*args,**kw)
t.new.fp8_paged_mqa_logits(*t.t.decode_case(6,32000,1))
torch.cuda.synchronize()
receipt={'source_sha256':hashlib.sha256(t.SRC.read_bytes()).hexdigest(),'kernels':[]}
for idx,((name,key),kernel) in enumerate(compiled.items()):
    ptx=kernel.asm['ptx']
    assert '.target sm_80' in ptx
    assert '.e4m3' not in ptx
    if name!='_unpack_prefill':assert '.bf16.bf16' in ptx
    stem=f'{idx}{name}'
    (outdir/(stem+'.ptx')).write_text(ptx)
    (outdir/(stem+'.ttgir')).write_text(kernel.asm['ttgir'])
    receipt['kernels'].append(dict(name=name,registers=kernel.n_regs,spills=kernel.n_spills,
                       shared=kernel.metadata.shared,num_warps=kernel.metadata.num_warps,
                       ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),
                       target_sm80=True,bf16_mma='.bf16.bf16' in ptx,fp8_instruction=False))
(outdir/'compiler_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
