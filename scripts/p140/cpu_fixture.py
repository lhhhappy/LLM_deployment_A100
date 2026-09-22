"""CPU fixture: execute production tree/controller/components without CUDA imports.

AST removes import-time dependency loading and the rank-consensus decorator only.
All cache methods, DTOs, locks, LRU, split/insert/evict actions are production code.
Allocators are checked CPU fakes; unsupported HiCache/session services are stubs.
"""
import ast
from array import array
import collections
import dataclasses
import enum
import hashlib
import heapq
import logging
import math
from pathlib import Path
import sys
import time
import types
import typing
from abc import ABC, abstractmethod

import msgspec
import numpy as np
import torch

NS = types.SimpleNamespace
CREATED = []


class Env:
    def __getattr__(self, name):
        return NS(get=lambda: False)


def load(root):
    prod = types.ModuleType('t45_production')
    sys.modules[prod.__name__] = prod
    g = prod.__dict__
    g.update(vars(typing)); g['__name__'] = 't45_production'
    g.update(dict(torch=torch, msgspec=msgspec, dataclasses=dataclasses, dataclass=dataclasses.dataclass,
                  ABC=ABC, abstractmethod=abstractmethod, enum=enum, Enum=enum.Enum, IntFlag=enum.IntFlag,
                  auto=enum.auto, defaultdict=collections.defaultdict, array=array, math=math,
                  heapq=heapq, logging=logging, logger=logging.getLogger('T45'), time=time, sys=sys,
                  float64=np.float64, hashlib=hashlib, envs=Env(), rank_consensus=lambda **kw: lambda f: f,
                  get_observability=lambda: NS(enable_metrics=False), NodeId=int,
                  ceil_align=lambda x, a: (x+a-1)//a*a))
    def execute(rel, names=None):
        tree = ast.parse((root / rel).read_text())
        nodes = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.If)):
                continue
            if names is not None and getattr(node, 'name', None) not in names:
                continue
            if isinstance(node, ast.ClassDef):
                # Rank consensus is a multi-rank diagnostic, not cache logic.
                for method in node.body:
                    if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method.decorator_list = [d for d in method.decorator_list
                                                 if not isinstance(d, ast.Call) or
                                                 getattr(d.func, 'id', '') != 'rank_consensus']
            nodes.append(node)
        tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes, type_ignores=[])
        ast.fix_missing_locations(tree)
        exec(compile(tree, str(root / rel), 'exec'), g)

    execute('srt/mem_cache/unified_cache/component_type.py')
    execute('srt/mem_cache/base_prefix_cache.py')
    execute('srt/mem_cache/cache_init_params.py')
    execute('srt/mem_cache/evict_policy.py')
    execute('srt/mem_cache/hicache_storage.py', {'PoolName','PoolTransfer','PoolTransferResult','PoolHitPolicy'})
    execute('srt/mem_cache/utils.py', {'get_eviction_strategy','get_hash_str','compute_node_hash_values',
                                      'split_node_hash_value','compute_node_event_hash_values','hash_str_to_int64'})
    g['_EVICTION_POLICY_FACTORIES'] = {'lru': g['LRUStrategy'], 'fifo': g['FIFOStrategy']}
    execute('srt/disaggregation/kv_events.py', {'StorageMedium', 'KVCacheEvent', 'BlockRemoved', 'BlockStored', 'BlockStoredMetadata', 'BlockStoredWithMetadata', 'AllBlocksCleared'})
    execute('srt/mem_cache/events.py')
    execute('srt/mem_cache/radix_cache.py', {'RadixKey'})
    execute('srt/mem_cache/unified_cache/cache_action.py')
    execute('srt/mem_cache/unified_cache/unified_tree_core_interface.py')
    execute('srt/mem_cache/unified_cache/components/tree_component.py')
    execute('srt/mem_cache/unified_cache/components/full_component.py')
    execute('srt/mem_cache/unified_cache/components/mamba_component.py')
    execute('srt/mem_cache/unified_cache/unified_tree_core.py')
    execute('srt/mem_cache/unified_radix_cache.py', {'UnifiedRadixCache'})
    # Real req state and real tracking preparation.
    execute('srt/managers/schedule_batch.py', {'ReqKvInfo', '_MambaRadixCacheV2TrackEntry'})
    tree = ast.parse((root / 'srt/managers/schedule_batch.py').read_text())
    batch = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ScheduleBatch')
    prep = next(n for n in batch.body if isinstance(n, ast.FunctionDef) and n.name == '_mamba_radix_cache_v2_req_prepare_for_extend')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')],level=0), prep],type_ignores=[])), '<real tracking prepare>', 'exec'), g)
    g['mamba_cache_chunk_size'] = lambda: 64
    g['mamba_checkpoint_grid'] = lambda page: math.lcm(64, page)
    g['mamba_extra_buffer_lazy_enabled'] = lambda: False
    g['_MAMBA_DEBUG_ASSERTS'] = False
    tree = ast.parse((root / 'srt/managers/scheduler.py').read_text())
    scheduler = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
    flush = next(n for n in scheduler.body if isinstance(n, ast.FunctionDef) and n.name == 'flush_cache')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[flush],type_ignores=[])), '<real flush>', 'exec'), g)
    # Pool ownership methods, including production fallback/abort cleanup.
    tree = ast.parse((root / 'srt/mem_cache/memory_pool.py').read_text())
    pool = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HybridReqToTokenPool')
    for method in pool.body:
        if isinstance(method, ast.FunctionDef) and method.name in {
            'free_mamba_cache', 'get_mamba_ping_pong_other_idx', 'get_mamba_ping_pong_keep_idx',
            'donate_mamba_ping_pong_slot', 'set_mamba_ping_pong_slot'}:
            exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method],type_ignores=[])), '<real pool>', 'exec'),g)
    # Dynamic imports in actual methods resolve to these same production DTOs.
    for name in ['sglang','sglang.srt','sglang.srt.mem_cache']:
        if name not in sys.modules:
            mod=types.ModuleType(name); mod.__path__=[]; sys.modules[name]=mod
    for suffix in ['base_prefix_cache', 'radix_cache']:
        sys.modules['sglang.srt.mem_cache.'+suffix] = prod
    return prod


class Slots:
    device = 'cpu'
    def __init__(self, size):
        self.size=size; self.live=set(); self.free_log=[]
    def alloc(self,n):
        free=[i for i in range(1,self.size+1) if i not in self.live][:n]
        if len(free)<n: return None
        self.live.update(free)
        return torch.tensor(free,dtype=torch.int64)
    def free(self, slots):
        ids=slots.reshape(-1).tolist()
        assert len(ids)==len(set(ids)), ('duplicate batch free',ids)
        assert set(ids) <= self.live, ('free of unowned slots',ids,self.live)
        self.live.difference_update(ids); self.free_log += ids
    def free_segment(self,slots,start_pos=0): self.free(slots)
    def free_segments(self,segments):
        for slots,start in segments: self.free_segment(slots,start)
    def available_size(self): return self.size-len(self.live)
    schedulable_available_size=available_size
    def clear(self): self.live.clear()


class Pool:
    def __init__(self, p, size=64):
        self.mamba_allocator=Slots(size); self.mamba_ckpt_pool=None
        self.enable_mamba_extra_buffer=True; self.enable_mamba_extra_buffer_lazy=False
        self.mamba_ping_pong_track_buffer_size=2
        self.req_to_token=torch.zeros((16,4096),dtype=torch.int64)
        self.req_index_to_mamba_ping_pong_track_buffer_mapping=torch.zeros((16,2),dtype=torch.int64)
        for name in ['free_mamba_cache','get_mamba_ping_pong_other_idx','get_mamba_ping_pong_keep_idx',
                     'donate_mamba_ping_pong_slot','set_mamba_ping_pong_slot']:
            setattr(self,name,types.MethodType(getattr(p,name),self))
    def reset_aux_cache_allocator(self): pass
    def write(self,indices,value): self.req_to_token[indices]=value
    def clear(self):
        self.mamba_allocator.clear(); self.req_to_token.zero_()
        self.req_index_to_mamba_ping_pong_track_buffer_mapping.zero_()


class Req:
    def __init__(self,p,cache,ids,row=0,prefix=None):
        self.kv=p.ReqKvInfo(req_pool_idx=row)
        self.origin_input_ids=list(ids); self.output_ids=[]
        self.full_untruncated_fill_ids=list(ids)
        self.prefix_indices=torch.tensor([],dtype=torch.int64) if prefix is None else prefix
        self.extra_key=None; self.cache_salt=None; self.priority=0
        self.session=None; self.mamba_branching_seqlen=None
        self.extend_range=NS(start=len(self.prefix_indices),end=len(ids),length=len(ids)-len(self.prefix_indices))
        self.skip_lock_node_ids={}; self.swa_uuid_for_lock=None; self.swa_prefix_lock_released=False
        self.last_node=cache.tree_core.root_node.id
        self.kv.mamba_pool_idx=cache.req_to_token_pool.mamba_allocator.alloc(1)[0]
        self.kv.mamba_ping_pong_track_buffer=cache.req_to_token_pool.mamba_allocator.alloc(2)
        self.kv.mamba_next_track_idx=0; self.kv.mamba_last_track_idx=1
        cache.req_to_token_pool.req_index_to_mamba_ping_pong_track_buffer_mapping[row]=self.kv.mamba_ping_pong_track_buffer
        cache.req_to_token_pool.req_to_token[row,:len(ids)]=cache.token_to_kv_pool_allocator.alloc(len(ids))
    def get_fill_ids(self): return self.full_untruncated_fill_ids[:self.extend_range.end]
    def __getattr__(self,name):
        if name.startswith('mamba_'): return getattr(self.kv,name)
        raise AttributeError(name)
    def __setattr__(self,name,value):
        if name.startswith('mamba_') and name != 'mamba_branching_seqlen' and 'kv' in self.__dict__:
            setattr(self.kv,name,value)
        else: object.__setattr__(self,name,value)


def cache_fixture(p,on=True,cap=-1,slots=64):
    c=p.UnifiedRadixCache.__new__(p.UnifiedRadixCache)
    c.req_to_token_pool=Pool(p,slots); c.token_to_kv_pool_allocator=Slots(65536)
    c.disable=False; c.enable_mamba_extra_buffer=True; c.enable_session_radix_cache=False
    c.ax_kda_dual_snapshot=on; c.is_mamba_enabled=True; c.is_swa_enabled=False
    c.linker=None; c.metrics_collector=None; c.cache_controller=None; c.buffer_pipeline=None
    c.host_pool_group=None; c.sidecar_pool_specs=[]
    c.session=NS(try_match_prefix=lambda *a: None,try_cache_unfinished_req=lambda *a,**k: False,
                 try_cache_finished_req=lambda *a,**k:False,try_inc_lock_ref=lambda *a,**k:None,
                 try_dec_lock_ref=lambda *a,**k:None,slots={})
    c.session_refs=NS(reset=lambda:None)
    params=p.CacheInitParams(False,c.req_to_token_pool,c.token_to_kv_pool_allocator,64,
                            tree_components=(p.ComponentType.FULL,p.ComponentType.MAMBA),enable_mamba_extra_buffer=True)
    full=p.FullComponent(c,params)
    mamba=p.MambaComponent.__new__(p.MambaComponent)
    p.TreeComponent.__init__(mamba,c,params)
    mamba.mamba_cache_chunk_size=64; mamba.mamba_checkpoint_grid=64
    mamba.mamba_max_states_per_path=cap; mamba._mamba_pool_host=None
    c.components={p.ComponentType.FULL:full,p.ComponentType.MAMBA:mamba}
    c._components_tuple=tuple(c.components.values()); c.tree_components=tuple(c.components)
    c.tree_core=p.UnifiedTreeCore(params,c.components)
    CREATED.append(c)
    return c
