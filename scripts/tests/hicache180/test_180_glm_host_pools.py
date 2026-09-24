"""180 — GLM-5.3-Flash host tier: what a host restore must bring back.

Runs the REAL hybrid-pool strategy selection (`_select_strategy`), the REAL
`HybridCacheController` write/load paths (write queue merge, index moves,
`_l2_transfers` / `_l2_load_transfers`, the L2 engine's per-layer loop,
sidecar index resolution, Mamba device-slot allocation) and the REAL host pool
classes on CPU, over REAL device pools (`HybridLinearKVPool` with a compressed
kpool=4 DSA indexer, `HybridReqToTokenPool` Mamba state). Only raw byte copies
are emulated (see _cpu_harness.py).

Discriminating: on a tree WITHOUT 180 the hybrid Mamba stack has no INDEXER
host pool, so `test_round_trip_restores_every_component_byte_exact` fails on the
indexer (and draft-indexer) bytes, and the structure test fails. With 180 all
pass.

Unit of restore (what "same semantic position" means here): a radix node of key
length L owns, for positions [0, L), the MLA latent rows, the packed indexer
rows of every 256-token group fully inside [0, L), the same for the MTP draft
layer, and the KDA/conv state AFTER consuming tokens [0, L). The test moves one
such node (4 pages = one 256-token group) and one state slot.
"""

import os
import sys
import threading
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cpu_harness as H  # noqa: E402

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer  # noqa: E402
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType  # noqa: E402

POISON = 0xAB


def _fill_random(tensors, seed):
    g = torch.Generator().manual_seed(seed)
    for t in tensors:
        b = t.view(torch.uint8)
        b.copy_(torch.randint(0, 256, b.shape, generator=g, dtype=torch.uint8).to(b.device))


def _same_bytes(a, b):
    # random bytes include NaN patterns; compare storage, not values
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _poison(t, rows):
    t.view(torch.uint8).reshape(t.shape[0], -1)[rows] = POISON


class _FakeAllocator:
    """Device token allocator stand-in: load() takes the next preset pages."""

    def __init__(self, kvcache):
        self._kv = kvcache
        self.next_alloc = None

    def get_kvcache(self):
        return self._kv

    def alloc(self, n):
        out, self.next_alloc = self.next_alloc, None
        assert out is not None and len(out) == n, (n, out)
        return out

    def free(self, x):
        pass


class TestGlmHostTier(unittest.TestCase):
    def setUp(self):
        from sglang.srt.runtime_context import reset_context

        self.addCleanup(reset_context)
        H.publish_args()
        self._kernels = H.cpu_kernels()
        self._kernels.__enter__()
        self.addCleanup(self._kernels.__exit__, None, None, None)

    def _build(self, *, with_draft):
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as asm

        req_pool, kv, draft = H.glm_like_pools(with_draft=with_draft)
        allocator = _FakeAllocator(kv)
        params = SimpleNamespace(
            page_size=64,
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            mtp_draft_device_pools=(draft,) if draft is not None else (),
            tp_cache_group=None,
            attn_cp_cache_group=None,
            attn_tp_cache_group=None,
            pp_cache_group=None,
        )
        cache = SimpleNamespace(evict_host=lambda *a, **k: 0, _enable_metrics_flag=False)
        strategy = asm._select_strategy(kv, {ComponentType.FULL, ComponentType.MAMBA})
        self.assertEqual(type(strategy).__name__, "_MambaStrategy")
        result = strategy.build(
            cache=cache,
            kvcache=kv,
            params=params,
            server_args=None,
            load_cache_event=threading.Event(),
        )
        H.register_device_tensors(H.device_tensors(req_pool, kv, draft))
        return req_pool, kv, draft, allocator, result

    def test_stack_declares_indexer_next_to_kv_and_mamba(self):
        """Fails without 180 (no INDEXER entry, no sidecar)."""
        for with_draft in (False, True):
            with self.subTest(with_draft=with_draft):
                _, kv, draft, _, result = self._build(with_draft=with_draft)
                names = [e.name for e in result.host_pool_group.entries]
                self.assertIn(PoolName.INDEXER, names)
                self.assertEqual(
                    [(s.pool_name, s.indices_from_pool) for s in result.sidecars],
                    [(PoolName.INDEXER, PoolName.KV)],
                )
                idx = result.host_pool_group.entry_map[PoolName.INDEXER]
                n_dsa = kv.full_kv_pool.layer_num
                self.assertEqual(idx.host_pool.layer_num, n_dsa + (1 if with_draft else 0))
                # every DSA transfer layer (global id) maps to its device layer
                for gid, lid in kv.full_attention_layer_id_mapping.items():
                    self.assertEqual(idx.layer_mapper(gid), lid)
                if with_draft:
                    self.assertEqual(
                        [id(p) for p in idx.packed_draft_device_pools],
                        [id(draft.full_kv_pool)],
                    )

    def test_round_trip_restores_every_component_byte_exact(self):
        """Backup one 256-token group + one KDA slot through the real controller,
        poison every device copy (and the freed pages), load into DIFFERENT
        device pages/slot, and compare bytes. Pure copies: tolerance is zero.
        Fails without 180 on the indexer rows."""
        for with_draft in (False, True):
            with self.subTest(with_draft=with_draft):
                self._round_trip(with_draft)

    def _round_trip(self, with_draft):
        req_pool, kv, draft, allocator, result = self._build(with_draft=with_draft)
        ctrl, group = result.cache_controller, result.host_pool_group
        full = kv.full_kv_pool
        pools = [full] + ([draft.full_kv_pool] if draft is not None else [])
        cache = req_pool.mamba_pool.mamba_cache
        state_tensors = [cache.temporal] + list(cache.conv)
        _fill_random([b for p in pools for b in p.kv_buffer], 1)
        _fill_random([b for p in pools for b in p.index_k_with_scale_buffer], 2)
        _fill_random(state_tensors, 3)

        dev = H.DEVICE
        src_pages = torch.arange(2, 6, device=dev)  # one 256-token compression group
        dst_pages = torch.arange(10, 14, device=dev)
        src_tok = (src_pages[:, None] * 64 + torch.arange(64, device=dev)).reshape(-1)
        dst_tok = (dst_pages[:, None] * 64 + torch.arange(64, device=dev)).reshape(-1)
        src_slot = req_pool.mamba_allocator.alloc(1)

        ref = {
            "mla": [p.kv_buffer[i][src_tok].clone() for p in pools for i in range(p.layer_num)],
            "idx": [p.index_k_with_scale_buffer[i][src_pages].clone() for p in pools for i in range(p.layer_num)],
            "state": [t[:, src_slot].clone() for t in state_tensors],
        }

        # ---- D2H through the real write path (write-through of one node) ----
        sidecars = [
            PoolTransfer(name=s.pool_name, indices_from_pool=s.indices_from_pool, hit_policy=s.hit_policy)
            for s in result.sidecars
        ]
        mamba_w = PoolTransfer(name=PoolName.MAMBA, device_indices=src_slot)
        # HybridCacheController.write resolves the side-pool host slots in
        # place and submits (start_writing) immediately.
        host = ctrl.write(src_tok, node_id=1, extra_pools=[mamba_w] + sidecars)
        self.assertIsNotNone(host)
        mamba_host = mamba_w.host_indices
        self.assertIsNotNone(mamba_host)
        self.assertEqual(len(ctrl.write_queue), 0)
        self.assertEqual(len(ctrl.ack_write_queue), 1)
        H.sync()

        # ---- evict: poison the source copies and every page we may reuse ----
        for p in pools:
            for i in range(p.layer_num):
                _poison(p.kv_buffer[i], src_tok)
                _poison(p.kv_buffer[i], dst_tok)
                _poison(p.index_k_with_scale_buffer[i], src_pages)
                _poison(p.index_k_with_scale_buffer[i], dst_pages)
        for t in state_tensors:
            t.view(torch.uint8)[:, src_slot] = POISON
        req_pool.mamba_allocator.free(src_slot)

        # ---- H2D through the real load path into different pages / slot ----
        allocator.next_alloc = dst_tok
        mamba_l = PoolTransfer(name=PoolName.MAMBA, host_indices=mamba_host, nodes_to_load=[1])
        loaded = ctrl.load(
            host,
            node_id=1,
            extra_pools=[mamba_l]
            + [
                PoolTransfer(name=s.pool_name, indices_from_pool=s.indices_from_pool, hit_policy=s.hit_policy)
                for s in result.sidecars
            ],
        )
        self.assertTrue(torch.equal(loaded, dst_tok))
        dst_slot = mamba_l.device_indices  # allocated by the real Mamba allocator
        self.assertIsNotNone(dst_slot)
        for t in state_tensors:  # make sure the destination slot starts poisoned
            t.view(torch.uint8)[:, dst_slot] = POISON
        ctrl.start_loading()
        H.sync()

        k = 0
        for p in pools:
            for i in range(p.layer_num):
                self.assertTrue(
                    _same_bytes(p.kv_buffer[i][dst_tok], ref["mla"][k]),
                    f"MLA latent layer {k} not restored",
                )
                self.assertTrue(
                    _same_bytes(p.index_k_with_scale_buffer[i][dst_pages], ref["idx"][k]),
                    f"DSA indexer rows layer {k} ({'draft' if p is not full else 'target'}) not restored",
                )
                k += 1
        for t, r in zip(state_tensors, ref["state"]):
            self.assertTrue(_same_bytes(t[:, dst_slot], r), "KDA/conv state not restored")

    def test_flush_clears_every_host_pool(self):
        """/flush_cache -> UnifiedRadixCache.reset -> HostPoolGroup.clear():
        every entry, including the indexer mirror, must return all slots."""
        _, _, _, _, result = self._build(with_draft=True)
        group = result.host_pool_group
        self.assertIn(PoolName.INDEXER, group.entry_map)  # fails without 180
        full = {e.name: e.host_pool.available_size() for e in group.entries}
        group.alloc(256)
        group.alloc(2, pool=PoolName.MAMBA)
        self.assertLess(group.available_size(), full[PoolName.KV])
        group.clear()
        self.assertEqual({e.name: e.host_pool.available_size() for e in group.entries}, full)

    def test_hicache_size_is_a_total_that_includes_the_indexer(self):
        """--hicache-size (per rank) must bound MLA + indexer + Mamba host bytes.
        Upstream sizes the indexer mirror on top of the KV share; 180 carves it."""
        from sglang.srt.runtime_context import reset_context

        reset_context()
        H.publish_args(hicache_size=1)  # 1 GB per rank
        _, _, _, _, result = self._build(with_draft=True)
        group = result.host_pool_group
        total = 0
        for e in group.entries:
            hp = e.host_pool
            total += hp.size * hp.size_per_token if e.name != PoolName.INDEXER else (
                hp.page_num * hp.indexer_layout_dim
            )
        self.assertLessEqual(total, 1e9 * 1.01)
        self.assertGreater(total, 1e9 * 0.95)

    def test_packed_draft_must_cover_target_slots(self):
        """A packed draft pool is addressed with target device indices."""
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as asm

        req_pool, kv, small_draft = H.glm_like_pools(with_draft=True, draft_size=1024)
        params = SimpleNamespace(
            page_size=64, req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=_FakeAllocator(kv),
            mtp_draft_device_pools=(small_draft,), tp_cache_group=None,
            attn_cp_cache_group=None, attn_tp_cache_group=None, pp_cache_group=None,
        )
        with self.assertRaisesRegex(ValueError, "packed draft pool holds"):
            asm._select_strategy(kv, {ComponentType.FULL, ComponentType.MAMBA}).build(
                cache=SimpleNamespace(evict_host=lambda *a, **k: 0, _enable_metrics_flag=False),
                kvcache=kv, params=params, server_args=None, load_cache_event=threading.Event(),
            )


if __name__ == "__main__":
    unittest.main()
