#!/usr/bin/env python3
"""T45 real scheduler/PrefillAdder AST parity using the existing T41 fixture."""
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('sched_fixture',ROOT/'tests/test_sched_protect_chain.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
BASE=ROOT/'build/p140/control/sglang';CAND=ROOT/'build/p140/stack/sglang'


def run(root,on,seed,protect=True,tail=False):
    with patch.dict(os.environ,{'SGLANG_AX_KDA_DUAL_SNAPSHOT':str(int(on)),
                               'SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS':'99',
                               'SGLANG_AX_SCHED_PROTECT':str(int(protect))}):
        rng=random.Random(seed)
        chunk=f.Req('continuation',11008,cached=2560,boundary=12600) if tail else None
        waits=[f.Req('cold',25000,boundary=21100), f.Req('short',1024,cached=8192,boundary=8600)]
        s,ns=f.make_scheduler(root,waiting=waits,chunk=chunk,running=[f.Req('decode',1)])
        # T41 fixture normally replaces this function; restore actual production
        # helper so the new startup flag is exercised, including 101/105 guards.
        f.compile_nodes(root/'srt/managers/schedule_policy.py',{'_role_boundary_token_ids'},ns)
        ns['mamba_checkpoint_grid']=lambda p:64
        traces=[]
        for step in range(30):
            arrival=([f.Req(f's{step}',rng.choice([256,1024,4096]),cached=8192,boundary=8200)]
                     if step%3==0 else [])
            traces.append(f.step(s,arrival))
        return traces


def main():
    parity=[]
    for seed in range(8):
        for protect in (False,True):
            for tail in (False,True):
                baseline=run(BASE,False,seed,protect,tail)
                candidate=run(CAND,False,seed,protect,tail)
                assert json.dumps(baseline,separators=(',',':'))==json.dumps(candidate,separators=(',',':'))
                parity.append(dict(seed=seed,protect=protect,tail=tail,trace=baseline))
    # Targeted one-extend example: role at 512 inside a 1024-token prompt.
    examples={}
    for on in (False,True):
        with patch.dict(os.environ,{'SGLANG_AX_KDA_DUAL_SNAPSHOT':str(int(on)),
                                   'SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS':'99',
                                   'SGLANG_AX_SCHED_PROTECT':'0'}):
            s,ns=f.make_scheduler(CAND,waiting=[f.Req('r',1024,boundary=600)])
            f.compile_nodes(CAND/'srt/managers/schedule_policy.py',{'_role_boundary_token_ids'},ns)
            ns['mamba_checkpoint_grid']=lambda p:64
            traces=[f.step(s) for _ in range(3)]
            count=sum(t['mode']=='prefill' for t in traces)
            assert count==(1 if on else 2),(on,traces)
            examples[str(int(on))]=traces
    # Production scheduler runs on for all previous parity scenarios without
    # any extra role partial; 120's independent one-partial guard remains.
    enabled=[dict(seed=i,trace=run(CAND,True,i,True,True)) for i in range(8)]
    out=ROOT/'evidence/T45'
    (out/'scheduler_off_traces.json').write_text(json.dumps(parity,indent=2))
    (out/'scheduler_on_traces.json').write_text(json.dumps(enabled,indent=2))
    (out/'scheduler_summary.json').write_text(json.dumps(dict(off_cases=len(parity),off_rounds=len(parity)*30,
              off_json_bytes_equal=True,on_cases=len(enabled),single_request=examples),indent=2)+'\n')
    print('PASS:',len(parity),'off traces /',len(parity)*30,'rounds; 8 on traces; role prompt 2 -> 1 extends')


if __name__=='__main__':main()
