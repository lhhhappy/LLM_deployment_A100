#!/usr/bin/env python3
"""CPU-only, two real Gloo ranks: a non-leader pool failure must reach both ranks."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, module_path, rendezvous):
    io=ModuleType('sglang.srt.managers.io_struct'); io.GenerateReqInput=NS
    sys.modules[io.__name__]=io
    spec=importlib.util.spec_from_file_location('ax150', module_path)
    warm=importlib.util.module_from_spec(spec); spec.loader.exec_module(warm)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    req=NS(size=8,available_size=lambda:8,mamba_pool=NS(size=16),
           mamba_allocator=NS(available_size=lambda:16))
    tree=NS(total_size=lambda:(0,0),**{k:lambda:0 for k in (
        'full_evictable_size','full_protected_size','mamba_evictable_size','mamba_protected_size')})
    scheduler=NS(is_fully_idle=lambda:True,req_to_token_pool=req,
                 token_to_kv_pool_allocator=NS(size=1024,page_size=64,available_size=lambda:1024),
                 tree_cache=tree,tp_cpu_group=dist.group.WORLD)
    try:
        results=[]
        for phase in ('all-clean','rank1-leak','rank1-refused'):
            req.mamba_allocator.available_size=lambda:15 if rank==1 and phase=='rank1-leak' else 16
            failed=False
            try:
                warm.verify_empty(scheduler, not(rank==1 and phase=='rank1-refused'))
            except RuntimeError:
                failed=True
            assert failed==(phase!='all-clean'),(rank,phase,failed)
            results.append({'phase':phase,'failed':failed})
        print(json.dumps({'rank':rank,'status':'PASS','results':results}),flush=True)
    finally:
        dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--module',type=Path,required=True)
    parser.add_argument('--workdir',type=Path,required=True)
    args=parser.parse_args()
    rendezvous=args.workdir.resolve()/f'gloo_init_{os.getpid()}'
    try:
        mp.spawn(worker,args=(str(args.module.resolve()),str(rendezvous)),nprocs=2,join=True)
    finally:
        rendezvous.unlink(missing_ok=True)


if __name__=='__main__': main()
