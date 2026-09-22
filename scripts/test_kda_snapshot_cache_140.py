#!/usr/bin/env python3
"""T45 CPU regression on real UnifiedRadixCache methods/tree/components."""
import argparse
import importlib.util
import json
import os
from unittest.mock import patch
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch
from p140.cpu_fixture import load, cache_fixture, Req, CREATED

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
    def setUp(self): CREATED.clear()
    def tearDown(self):
        for c in CREATED: c.tree_core.sanity_check([], [])
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
        scheduler=NS(is_fully_idle=lambda:True,tree_cache=c,req_to_token_pool=c.req_to_token_pool,
                     token_to_kv_pool_allocator=c.token_to_kv_pool_allocator,
                     grammar_manager=NS(clear=lambda:None),
                     metrics_reporter=NS(reset_metrics=lambda:None,is_stats_logging_rank=False),draft_worker=None)
        self.assertTrue(P.flush_cache(scheduler,empty_cache=False))
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


    def test_flush_busy_keeps_pending_slot(self):
        c=cache_fixture(P); r,_,_=prepared(c)
        live=set(c.req_to_token_pool.mamba_allocator.live)
        s=NS(is_fully_idle=lambda:False,waiting_queue=[r],running_batch=NS(reqs=[]))
        self.assertFalse(P.flush_cache(s,empty_cache=False))
        self.assertEqual(c.req_to_token_pool.mamba_allocator.live,live)
        self.assertIsNotNone(r.kv.ax_kda_snapshot_slot)
        c.cache_finished_req(r,is_insert=False,kv_len_to_handle=337)

    def test_branch_priority_replaced_by_two_points(self):
        for on in (False,True):
            c=cache_fixture(P,on=on); r,_,_=prepared(c)
            # Re-prepare the same range after discarding the provisional extra.
            helper.discard(c.req_to_token_pool,r)
            r.mamba_branching_seqlen=64
            b=NS(tree_cache=c,req_to_token_pool=c.req_to_token_pool,
                 model_config=NS(hf_text_config=NS(mamba_chunk_size=64)),ax_kda_dual_snapshot_batch=on)
            entry=P._mamba_radix_cache_v2_req_prepare_for_extend(b,r)
            self.assertEqual(r.kv.mamba_last_track_seqlen,320 if on else 64)
            self.assertTrue(entry.track_mask)

    def test_role_exactly_at_end_uses_single_slot(self):
        c=cache_fixture(P); ids=[1]*337;ids[325]=154827
        r,b,_=prepared(c,ids)
        self.assertIsNone(r.kv.ax_kda_snapshot_slot)
        c.cache_finished_req(r,kv_len_to_handle=337)
        node=c.tree_core.node_by_id(c.match_prefix(P.MatchPrefixParams(P.RadixKey(ids))).last_device_node)
        self.assertTrue(node.ax_kda_role)
        self.assertFalse(node.ax_kda_tail)
        self.assertEqual(len(c.req_to_token_pool.mamba_allocator.live),1)

    def test_no_role_and_short_extend(self):
        for size in (17,63,64,65,320):
            c=cache_fixture(P);r,b,e=prepared(c,[1]*size)
            self.assertIsNone(r.kv.ax_kda_snapshot_slot)
            self.assertEqual(b.ax_kda_snapshot_slots[0,1].item(),-1)
            self.assertEqual(e.track_mask,size>=64)
            c.cache_finished_req(r,kv_len_to_handle=size)
            self.assertEqual(len(c.req_to_token_pool.mamba_allocator.live),int(size>=64))

    def test_streaming_and_unaligned_batch_fallback(self):
        c=cache_fixture(P);r,_,_=prepared(c)
        b=NS(tree_cache=c,reqs=[r])
        self.assertTrue(helper.batch_supported(b,64))
        r.session=object();self.assertFalse(helper.batch_supported(b,64))
        r.session=None;r.prefix_indices=torch.arange(63)
        self.assertFalse(helper.batch_supported(b,64))

    def test_namespace_preserved(self):
        c=cache_fixture(P);r,_,_=prepared(c);r.cache_salt='private-A';r.extra_key='adapter-A'
        c.cache_finished_req(r,kv_len_to_handle=337)
        ids=[1]*150+[2]*187
        hit=c.match_prefix(P.MatchPrefixParams(P.RadixKey(ids,'adapter-A',cache_salt='private-A')))
        self.assertEqual(len(hit.device_indices),128)
        for adapter,salt in [('adapter-B','private-A'),('adapter-A','private-B')]:
            self.assertEqual(len(c.match_prefix(P.MatchPrefixParams(P.RadixKey(ids,adapter,cache_salt=salt))).device_indices),0)

    def test_full_pressure_frees_every_tree_slot(self):
        c=cache_fixture(P);r,_,_=prepared(c);c.cache_finished_req(r,kv_len_to_handle=337)
        c.evict(P.EvictParams(num_tokens=10000,mamba_num=100))
        self.assertEqual(c.req_to_token_pool.mamba_allocator.live,set())
        self.assertEqual(c.token_to_kv_pool_allocator.live,set())

    def test_two_requests_never_share_snapshot_slots(self):
        c=cache_fixture(P);r,b,_=prepared(c);r2,b2,_=prepared(c,row=1)
        self.assertTrue(set(b.ax_kda_snapshot_slots.flatten().tolist()).isdisjoint(b2.ax_kda_snapshot_slots.flatten().tolist()))
        c.cache_finished_req(r,kv_len_to_handle=337)
        c.cache_finished_req(r2,kv_len_to_handle=337)
        self.assertEqual(len(c.req_to_token_pool.mamba_allocator.live),2)

    def test_middle_chunk_exports_last_global_role(self):
        c=cache_fixture(P); ids=[1]*900;ids[150]=154827;ids[280]=154829
        r,b,_=prepared(c,ids)
        helper.discard(c.req_to_token_pool,r)
        r.extend_range=NS(start=0,end=512,length=512)
        e=P._mamba_radix_cache_v2_req_prepare_for_extend(b,r)
        helper.prepare(b,[e.track_index],[e.track_mask],64)
        self.assertEqual(b.ax_kda_snapshot_offsets.tolist(),[[512,256]])
        c.cache_unfinished_req(r,chunked=True)
        self.assertIsNone(r.kv.ax_kda_snapshot_slot)
        # Continue the original chunk without another role snapshot.
        r.extend_range=NS(start=512,end=900,length=388)
        e=P._mamba_radix_cache_v2_req_prepare_for_extend(b,r)
        helper.prepare(b,[e.track_index],[e.track_mask],64)
        self.assertEqual(b.ax_kda_snapshot_offsets.tolist(),[[384,-1]])
        c.cache_unfinished_req(r)
        c.cache_finished_req(r,kv_len_to_handle=900)
        hit=c.match_prefix(P.MatchPrefixParams(P.RadixKey([1]*150+[154827]+[1]*129+[2]*620)))
        self.assertEqual(len(hit.device_indices),256)


    def test_configuration_guard_and_off_bypass(self):
        import types
        runtime=types.ModuleType('sglang.srt.runtime_context')
        args=NS(disaggregation_mode='null')
        runtime.get_server_args=lambda:args
        runtime.process_model_config=lambda:NS(hf_config=NS(architectures=['Glm5NextForConditionalGeneration']))
        sys.modules[runtime.__name__]=runtime
        c=NS(is_mamba_enabled=True,is_swa_enabled=False,_tree_core_backend='python',
             req_to_token_pool=NS(mamba_pool=NS(mamba_cache=NS(temporal=torch.zeros(1)))))
        params=NS(disable=False,is_eagle=False,enable_mamba_extra_buffer=True,
                  enable_mamba_extra_buffer_lazy=False,pp_size=1,attn_cp_size=1,enable_session_radix_cache=False)
        with patch.dict(os.environ,{'SGLANG_AX_KDA_DUAL_SNAPSHOT':'1'}):
            self.assertTrue(helper.configure(c,params))
            for flag in ('enable_unified_memory','enable_two_batch_overlap','enable_mixed_chunk',
                         'enable_hierarchical_cache','enable_int8_mamba_checkpoint','enable_linear_replayssm',
                         'enable_dp_attention','speculative_algorithm'):
                setattr(args,flag,True)
                with self.assertRaises(ValueError):helper.configure(c,params)
                setattr(args,flag,False)
            c.req_to_token_pool.mamba_pool.mamba_cache.temporal=torch.zeros(1,dtype=torch.bfloat16)
            with self.assertRaises(ValueError):helper.configure(c,params)
        with patch.dict(os.environ,{'SGLANG_AX_KDA_DUAL_SNAPSHOT':'0'}):
            self.assertFalse(helper.configure(None,None))


def off_trace():
    rows=[]
    for branch in (None,64,128):
        c=cache_fixture(P,on=False,cap=2)
        for i in range(3):
            ids=[1]*337;ids[150]=154827;ids[200]=i+2
            r=Req(P,c,ids,i)
            r.mamba_branching_seqlen=branch
            b=NS(tree_cache=c,req_to_token_pool=c.req_to_token_pool,
                 model_config=NS(hf_text_config=NS(mamba_chunk_size=64)),ax_kda_dual_snapshot_batch=False)
            entry=P._mamba_radix_cache_v2_req_prepare_for_extend(b,r)
            c.cache_finished_req(r,kv_len_to_handle=337)
            hit=c.match_prefix(P.MatchPrefixParams(P.RadixKey(ids)))
            nodes=sorted((list(n.key), None if n.component_data[P.ComponentType.MAMBA].value is None
                          else n.component_data[P.ComponentType.MAMBA].value.tolist(),
                          n.component_data[P.ComponentType.MAMBA].lock_ref,
                          n.component_data[P.ComponentType.FULL].lock_ref)
                         for n in c.tree_core._node_arena.values() if n.parent is not None)
            rows.append(dict(branch=branch,step=i,entry=list(entry),hit=len(hit.device_indices),nodes=nodes,
                             slots=sorted(c.req_to_token_pool.mamba_allocator.live),
                             kv=sorted(c.token_to_kv_pool_allocator.live)))
            c.tree_core.sanity_check([],[])
        c.evict(P.EvictParams(num_tokens=9999,mamba_num=999))
        rows.append(dict(branch=branch,empty_slots=sorted(c.req_to_token_pool.mamba_allocator.live),
                         empty_kv=sorted(c.token_to_kv_pool_allocator.live)))
    return rows


def main():
    global P,helper
    parser=argparse.ArgumentParser(); parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--off-trace',type=Path)
    args=parser.parse_args(); P=load(args.source)
    if args.off_trace:
        args.off_trace.write_text(json.dumps(off_trace(),sort_keys=True,separators=(',',':'))+'\n')
        print('OFF trace complete'); return
    path=args.source/'srt/mem_cache/kda_dual_snapshot.py'
    spec=importlib.util.spec_from_file_location('sglang.srt.mem_cache.kda_dual_snapshot',path)
    helper=importlib.util.module_from_spec(spec); sys.modules[spec.name]=helper; spec.loader.exec_module(helper)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    print(json.dumps({'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors)}))
    raise SystemExit(not result.wasSuccessful())


if __name__=='__main__': main()
