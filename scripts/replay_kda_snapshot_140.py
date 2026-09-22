#!/usr/bin/env python3
"""T45 offline estimate on real dev chains and exact rendered GLM token IDs.

No future-request oracle: checkpoint choice sees current prompt only. Unlimited
cache, serial within each chain, no generated decode tokens, no retraction or
resource pressure. Models actual 101 branch-priority/final split and 140 two-point
tracking. This is a token-hit/scheduling estimate, never measured serving gain.
"""
import argparse
from array import array
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

GRID=64
ROLE=(154827,154829)


def floor(x): return x//GRID*GRID


def lcp(a,b):
    n=min(len(a),len(b)); i=0
    while i<n and a[i]==b[i]: i+=1
    return i


def role_depth(ids):
    for i in range(len(ids)-1,-1,-1):
        if ids[i] in ROLE: return floor(i)
    return 0


def split_101(ids,prefix,extend,branch):
    end=prefix+extend
    if branch is not None and prefix<branch<=end: return None
    if prefix%GRID: return None
    for i in range(end-1,max(prefix,end-32768)-1,-1):
        if ids[i] in ROLE:
            length=floor(i-prefix)
            return length if GRID<=length<floor(extend) else None
    return None


def simulate(requests,token_ids,on,chunk):
    by_chain=defaultdict(list)
    for r in requests: by_chain[r['chain_id']].append(r)
    rows=[]
    for chain,rs in by_chain.items():
        rs.sort(key=lambda r:r['dispatch_offset_ms'])
        # Every entry represents real retained KV and one checkpoint at its end.
        saved=[]
        for r in rs:
            rid=f"{r['pack']}:{r['view']}:{r['logical_call_id']}"
            ids=token_ids[rid]; length=len(ids)
            hit=full=0
            for old,depth in saved:
                common=floor(lcp(ids,old[:depth]))
                full=max(full,common)
                if depth<=common: hit=max(hit,depth)
            branch=full if full>hit else None
            pos=hit; calls=splits=0; depths=[]; role=role_depth(ids)
            while pos<length:
                ext=min(chunk,length-pos)
                if not on and ext==length-pos:
                    cut=split_101(ids,pos,ext,branch)
                    if cut is not None: ext=cut; splits+=1
                end=pos+ext; tracked=pos+floor(ext)
                if ext>=GRID:
                    if not on and branch is not None and pos<branch<end and (branch-pos)%GRID==0:
                        tracked=branch
                    if tracked>0:
                        saved.append((ids,tracked)); depths.append(tracked)
                    if on and pos<role<tracked:
                        saved.append((ids,role)); depths.append(role)
                calls+=1;pos=end
            rows.append(dict(req_id=rid,chain_id=chain,phase=r['phase'],tokens=length,
                             hit=hit,uncached=length-hit,full_kv_hit=full,branch=branch,
                             role=role,extends=calls,splits=splits,inserted=depths,
                             fast_intra=r['phase']=='intra' and r['uncached_expected']<=4096))
    return rows


def stats(rows):
    fast=[r['uncached'] for r in rows if r['fast_intra']]
    fast.sort()
    return dict(requests=len(rows),hit_tokens=sum(r['hit'] for r in rows),
                uncached_tokens=sum(r['uncached'] for r in rows),
                extends=sum(r['extends'] for r in rows),extra_101_splits=sum(r['splits'] for r in rows),
                fast_intra_requests=len(fast),fast_intra_uncached_p95=fast[int(.95*(len(fast)-1))])


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.dataset/'harness'))
    from s1_common import Renderer
    import transformers,tokenizers
    renderer=Renderer(str(args.dataset/'glm_tok'))
    data=args.dataset/'data/dev-combined-v1'
    requests=list(map(json.loads,(data/'requests.jsonl').read_text().splitlines()))
    metadata={f"{r['pack']}:{r['view']}:{r['logical_call_id']}":r for r in requests}
    ids={}; hashes=[]
    with gzip.open(data/'bodies/dev-combined-v1.jsonl.gz','rt') as f:
        for line in f:
            b=json.loads(line);rid=b['req_id']
            t=renderer.tokenizer.encode(renderer.render(b),add_special_tokens=False)
            assert len(t)==metadata[rid]['glm_tokens'],(rid,len(t),metadata[rid]['glm_tokens'])
            ids[rid]=array('q',t)
            hashes.append(dict(req_id=rid,tokens=len(t),sha256=hashlib.sha256(ids[rid].tobytes()).hexdigest()))
    (args.out/'token_receipts.json').write_text(json.dumps(hashes,indent=2))
    result=dict(assumptions='unlimited cache; chains independent and serial; prompt tokens only; no decode checkpoints/eviction/retraction/admission pressure; current-prompt-only boundary selection',
                transformers=transformers.__version__,tokenizers=tokenizers.__version__,chains=len(set(r['chain_id'] for r in requests)),chunks={})
    for chunk in (8192,2048):
        off=simulate(requests,ids,False,chunk); on=simulate(requests,ids,True,chunk)
        old={r['req_id']:r for r in off}
        paired=[dict(req_id=r['req_id'],off=old[r['req_id']],on=r,hit_delta=r['hit']-old[r['req_id']]['hit']) for r in on]
        (args.out/f'replay_{chunk}.json').write_text(json.dumps(paired,indent=2))
        result['chunks'][str(chunk)]=dict(off=stats(off),on=stats(on),
                        improved=sum(r['hit_delta']>0 for r in paired),regressed=sum(r['hit_delta']<0 for r in paired),
                        hit_delta=sum(r['hit_delta'] for r in paired))
    (args.out/'replay_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
