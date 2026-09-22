#!/usr/bin/env python3
"""T45 CPU regression on real UnifiedRadixCache methods/tree/components."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch
from p140.cpu_fixture import load, cache_fixture, Req

P=None; helper=None


def prepared(cache, ids=None, row=0):
    if ids is None:
        ids=[1]*337; ids[150]=154827
    req=Req(P,cache,ids,row)
    batch=NS(reqs=[req],tree_cache=cache,req_to_token_pool=cache.req_to_token_pool,device='cpu',
             model_config=NS(hf_text_config=NS(mamba_chunk_size=64)), ax_kda_dual_snapshot_batch=cache.ax_kda_dual_snapshot)
    entry=P._mamba_radix_cache_v2_req_prepare_for_extend(batch,req)
    helper.prepare(batch,[entry.track_index],[entry.track_mask],64)
    return req,batch,entry


class Tests(unittest.TestCase):
    def test_alignment(self):
        for pos,want in [(0,0),(63,0),(64,64),(65,64),(127,64),(128,128)]:
            ids=[2]*300; ids[pos]=154829
            self.assertEqual(helper.role_depth(ids,64),want)
        ids=[2]*300; ids[199]=154827
        self.assertEqual(helper.role_depth(ids,128),128)
        ids[250]=154829
        self.assertEqual(helper.role_depth(ids,128),128)
        self.assertEqual(helper.role_depth([2]*500,64),0)

    def test_insert_match_restore_refcounts(self):
        c=cache_fixture(P); req,batch,entry=prepared(c)
        slot=req.kv.ax_kda_snapshot_slot.clone()
        self.assertEqual(batch.ax_kda_snapshot_offsets.tolist(),[[320,128]])
        c.cache_unfinished_req(req)
        self.assertIsNone(req.kv.ax_kda_snapshot_slot)
        # Same chain next turn diverges at the marker; exact reusable state=128.
        key=P.RadixKey([1]*150+[154829]+[3]*186)
        hit=c.match_prefix(P.MatchPrefixParams(key))
        self.assertEqual(len(hit.device_indices),128)
        node=c.tree_core.node_by_id(hit.last_device_node)
        self.assertTrue(node.ax_kda_role)
        self.assertTrue(torch.equal(node.component_data[P.ComponentType.MAMBA].value,slot))
        original=slot.item()
        # Exercise production deferred COW: slot allocation and source identity.
        req2=NS(kv=P.ReqKvInfo())
        c.match_prefix(P.MatchPrefixParams(key,cow_mamba=True,req=req2))
        self.assertEqual(req2.kv.mamba_cow_src_index.item(),original)
        self.assertNotEqual(req2.kv.mamba_pool_idx.item(),original)
        lock=c.inc_lock_ref(hit.last_device_node)
        self.assertEqual(node.component_data[P.ComponentType.MAMBA].lock_ref,1)
        c.dec_lock_ref(hit.last_device_node,lock.to_dec_params())
        self.assertEqual(node.component_data[P.ComponentType.MAMBA].lock_ref,0)

    def test_duplicate_role_no_kv_double_free(self):
        c=cache_fixture(P)
        r,_,_=prepared(c); c.cache_unfinished_req(r)
        live=set(c.req_to_token_pool.mamba_allocator.live)
        r2,_,_=prepared(c,row=1)
        c.cache_unfinished_req(r2)
        self.assertIsNone(r2.kv.ax_kda_snapshot_slot)
        self.assertEqual(len(c.req_to_token_pool.mamba_allocator.live),len(live)+3)
        self.assertEqual(len(c.token_to_kv_pool_allocator.live),354) #320 tree +17x2 tails

    def test_finished_insert(self):
        c=cache_fixture(P); r,_,_=prepared(c)
        c.cache_finished_req(r,kv_len_to_handle=337)
        self.assertEqual(len(c.req_to_token_pool.mamba_allocator.live),2)
        self.assertEqual(len(c.token_to_kv_pool_allocator.live),320)
        self.assertIsNone(r.kv.mamba_pool_idx)
        hit=c.match_prefix(P.MatchPrefixParams(P.RadixKey([1]*150+[3]*187)))
        self.assertEqual(len(hit.device_indices),128)

    def test_abort_discards_slot(self):
        c=cache_fixture(P); r,_,_=prepared(c)
        c.cache_finished_req(r,is_insert=False,kv_len_to_handle=337)
        self.assertEqual(c.req_to_token_pool.mamba_allocator.live,set())
        self.assertEqual(c.token_to_kv_pool_allocator.live,set())
        self.assertIsNone(r.kv.ax_kda_snapshot_slot)

    def test_pressure_skips_optional(self):
        c=cache_fixture(P,slots=4); r,b,_=prepared(c)
        self.assertIsNone(r.kv.ax_kda_snapshot_slot)
        self.assertEqual(b.ax_kda_snapshot_slots[0,1].item(),-1)
        c.cache_unfinished_req(r)
        self.assertEqual(len(c.req_to_token_pool.mamba_allocator.live),4)

    def test_mamba_eviction_tail_before_role(self):
        c=cache_fixture(P); r,_,_=prepared(c)
        c.cache_finished_req(r,kv_len_to_handle=337)
        # Touch tail so it is newer than role; priority must beat LRU.
        c.match_prefix(P.MatchPrefixParams(P.RadixKey(r.origin_input_ids[:320])))
        c.evict(P.EvictParams(mamba_num=1))
        hit=c.match_prefix(P.MatchPrefixParams(P.RadixKey(r.origin_input_ids)))
        self.assertEqual(len(hit.device_indices),128)
        self.assertEqual(len(c.req_to_token_pool.mamba_allocator.live),1)

    def test_full_eviction_tail_before_other_leaf(self):
        c=cache_fixture(P)
        other,_,_=prepared(c,[9]*192,row=1); c.cache_finished_req(other,kv_len_to_handle=192)
        r,_,_=prepared(c); c.cache_finished_req(r,kv_len_to_handle=337)
        c.evict(P.EvictParams(num_tokens=64))
        self.assertEqual(len(c.match_prefix(P.MatchPrefixParams(P.RadixKey(other.origin_input_ids))).device_indices),192)
        self.assertEqual(len(c.match_prefix(P.MatchPrefixParams(P.RadixKey(r.origin_input_ids))).device_indices),128)

    def test_locked_tail_not_evicted(self):
        c=cache_fixture(P); r,_,_=prepared(c); c.cache_unfinished_req(r)
        c.evict(P.EvictParams(mamba_num=8))
        end=c.tree_core.node_by_id(r.last_node)
        self.assertIsNotNone(end.component_data[P.ComponentType.MAMBA].value)
        self.assertEqual(end.component_data[P.ComponentType.MAMBA].lock_ref,1)

    def test_flush_resets_tree_and_pools(self):
        c=cache_fixture(P); r,_,_=prepared(c); c.cache_finished_req(r,kv_len_to_handle=337)
        # Scheduler's idle flush order: cache reset, req pool and token pool clear.
        c.reset(); c.req_to_token_pool.clear(); c.token_to_kv_pool_allocator.clear()
        self.assertEqual(c.req_to_token_pool.mamba_allocator.available_size(),64)
        self.assertEqual(c.token_to_kv_pool_allocator.available_size(),65536)
        self.assertEqual(len(c.tree_core._node_arena),1)
        self.assertEqual(len(c.match_prefix(P.MatchPrefixParams(P.RadixKey(r.origin_input_ids))).device_indices),0)
        for ct in c.tree_components:
            self.assertEqual(c.tree_core.component_evictable_size_[ct],0)
            self.assertEqual(c.tree_core.component_protected_size_[ct],0)
            self.assertFalse(c.tree_core.lru_lists[ct].cache)

    def test_path_cap_preserves_role(self):
        c=cache_fixture(P,cap=2); r,_,_=prepared(c)
        c.cache_unfinished_req(r)
        role=[n for n in c.tree_core._node_arena.values() if getattr(n,'ax_kda_role',False)]
        self.assertEqual(len(role),1)
        self.assertIsNotNone(role[0].component_data[P.ComponentType.MAMBA].value)


def main():
    global P,helper
    parser=argparse.ArgumentParser(); parser.add_argument('--source',type=Path,required=True)
    args=parser.parse_args(); P=load(args.source)
    path=args.source/'srt/mem_cache/kda_dual_snapshot.py'
    spec=importlib.util.spec_from_file_location('sglang.srt.mem_cache.kda_dual_snapshot',path)
    helper=importlib.util.module_from_spec(spec); sys.modules[spec.name]=helper; spec.loader.exec_module(helper)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    print(json.dumps({'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors)}))
    raise SystemExit(not result.wasSuccessful())


if __name__=='__main__': main()
