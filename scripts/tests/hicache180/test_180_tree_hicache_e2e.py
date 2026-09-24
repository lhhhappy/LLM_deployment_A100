"""180 — packed DSA indexer rows through the REAL radix tree + HiCache on CPU.

GLM-5.3-Flash's kpool=4 compressed indexer writes the 64 pooled rows of every
256-token group into the FIRST 64-token page of that group
(kpool_fp8_index.compute_pooled_write_locs, used verbatim below). A radix-tree
node boundary inside a group therefore makes a page that the tree already owns
(and may already have backed up to host) receive rows later. Two hazards:

  (a) stale host copy: a node ending mid-group is written through to host; the
      same request keeps producing the rest of the group (next chunk / decode)
      into the tree-owned group-head page; the host copy never sees those rows,
      so a host restore brings back a group whose tail rows are wrong.
  (b) fork overwrite: a request that forks at a 64/128/192 offset inside a group
      writes its own pooled rows into the group-head page shared with the other
      branch, corrupting that branch (on device, and later on host).

180 widens tree ownership to lcm(page, page*index_kpool)=256 tokens when HiCache
is on (upstream #38212), so every node owns whole groups and a group is only
published once it is complete.

Everything here is the real code: UnifiedRadixCache + TreeCore, init_hicache,
_MambaStrategy host pools, HybridCacheController write-through / load-back,
L2TransferEngine. A tiny "model" writes MLA latent rows and pooled index rows
exactly where the real kernels would (compute_pooled_write_locs). Only raw byte
copies are emulated (_cpu_harness). Each test says whether it must FAIL without
180.
"""

import os
import sys
import unittest
from array import array

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cpu_harness as H  # noqa: E402

from sglang.srt.layers.attention.dsa.kpool_fp8_index import (  # noqa: E402
    compute_pooled_write_locs,
)
from sglang.srt.mem_cache.base_prefix_cache import (  # noqa: E402
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType  # noqa: E402

PAGE, KPOOL, HEAD = 64, 4, 128
ROW_BYTES = HEAD + 4  # fp8 key + fp32 scale per pooled row
POISON = 0xAB


def _row_value(tokens, pool_id):
    """Deterministic 132-byte payload for pooled row `pool_id` of this token
    prefix: depends on every token up to the pool's last token (like a real
    indexer key, which depends on the whole prefix through the hidden state)."""
    end = (pool_id + 1) * KPOOL
    h = 1469598103934665603
    for t in tokens[:end]:
        h = (h * 1099511628211 + int(t) + 1) % (1 << 61)
    g = torch.Generator().manual_seed(h ^ pool_id)
    return torch.randint(0, 256, (ROW_BYTES,), generator=g, dtype=torch.uint8)


def _mla_value(tokens, pos):
    h = 7
    for t in tokens[: pos + 1]:
        h = (h * 31 + int(t) + 3) % (1 << 61)
    g = torch.Generator().manual_seed(h)
    return torch.randint(0, 256, (1024,), generator=g, dtype=torch.uint8)


class _Model:
    """Writes what one DSA layer's kernels would write for positions [a, b)."""

    def __init__(self, dsa_pool):
        self.pool = dsa_pool  # DSATokenToKVPool (full_kv_pool)

    def _index_rows(self, layer):
        buf = self.pool.index_k_with_scale_buffer[layer]  # [pages, 64*132] uint8
        return buf

    def run(self, tokens, slots, a, b):
        """slots: token -> device slot for positions [0, b)."""
        page_table_64 = slots[::PAGE] // PAGE
        for layer in range(self.pool.layer_num):
            kv = self.pool.kv_buffer[layer].view(torch.uint8).reshape(
                self.pool.kv_buffer[layer].shape[0], -1
            )
            for pos in range(a, b):
                kv[slots[pos]] = _mla_value(tokens, pos).to(kv.device)
            # pools that close inside [a, b)
            closed = [q for q in range(b // KPOOL) if a <= q * KPOOL + KPOOL - 1 < b]
            if not closed:
                continue
            locs = compute_pooled_write_locs(
                page_table_64,
                torch.tensor(closed, dtype=torch.int64, device=page_table_64.device),
                KPOOL,
            )
            buf = self._index_rows(layer)
            for q, loc in zip(closed, locs.tolist()):
                page, r = divmod(int(loc), PAGE)
                row = _row_value(tokens, q)
                row = row.to(buf.device)
                buf[page, r * HEAD : (r + 1) * HEAD] = row[:HEAD]
                buf[page, PAGE * HEAD + r * 4 : PAGE * HEAD + (r + 1) * 4] = row[HEAD:]

    def check(self, tokens, slots, n):
        """Every pooled row of the first n tokens (n multiple of 256) and every
        latent row must equal what the model wrote for THIS token sequence."""
        page_table_64 = slots[::PAGE] // PAGE
        pools = list(range(n // KPOOL))
        locs = compute_pooled_write_locs(
            page_table_64,
            torch.tensor(pools, dtype=torch.int64, device=page_table_64.device),
            KPOOL,
        ).tolist()
        bad_rows, bad_mla = [], []
        for layer in range(self.pool.layer_num):
            buf = self._index_rows(layer)
            for q, loc in zip(pools, locs):
                page, r = divmod(int(loc), PAGE)
                got = torch.cat(
                    [
                        buf[page, r * HEAD : (r + 1) * HEAD],
                        buf[page, PAGE * HEAD + r * 4 : PAGE * HEAD + (r + 1) * 4],
                    ]
                ).cpu()
                if not torch.equal(got, _row_value(tokens, q)):
                    bad_rows.append((layer, q))
            kv = self.pool.kv_buffer[layer].view(torch.uint8).reshape(
                self.pool.kv_buffer[layer].shape[0], -1
            )
            for pos in range(n):
                if not torch.equal(kv[slots[pos]].cpu(), _mla_value(tokens, pos)):
                    bad_mla.append((layer, pos))
        return bad_rows, bad_mla


class TestTreeHiCacheIndexerOwnership(unittest.TestCase):
    def setUp(self):
        from sglang.srt.runtime_context import reset_context

        self.addCleanup(reset_context)
        self.args = H.publish_args()
        self._k = H.cpu_kernels()
        self._k.__enter__()
        self.addCleanup(self._k.__exit__, None, None, None)

        from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        self.req_pool, self.kv, _ = H.glm_like_pools(size=8192, mamba_size=32)
        self.alloc = PagedTokenToKVPoolAllocator(
            size=8192, page_size=PAGE, dtype=torch.bfloat16, device=H.DEVICE,
            kvcache=self.kv, need_sort=False,
        )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=self.req_pool,
            token_to_kv_pool_allocator=self.alloc,
            page_size=PAGE,
            tree_components=(ComponentType.FULL, ComponentType.MAMBA),
            enable_mamba_extra_buffer=True,
        )
        self.cache = UnifiedRadixCache(params)
        self.cache.init_hicache(self.args, params)
        H.register_device_tensors(H.device_tensors(self.req_pool, self.kv))
        self.model = _Model(self.kv.full_kv_pool)
        self.tree_page = self.cache.page_size

    # -- helpers ------------------------------------------------------------
    def _state_slot(self):
        slot = self.req_pool.mamba_allocator.alloc(1)
        self.assertIsNotNone(slot)
        return slot

    def _publish(self, tokens, slots, cache_len):
        """What cache_unfinished_req/cache_finished_req do for a checkpoint at
        cache_len: insert the page-aligned key with its KV and a state slot,
        then let write-through run to completion."""
        n = cache_len // self.tree_page * self.tree_page
        if n == 0:
            return 0
        res = self.cache.insert(
            InsertParams(
                key=RadixKey(array("q", tokens[:n])),
                value=slots[:n].clone(),
                mamba_value=self._state_slot(),
            )
        )
        self._drain_writes()
        return n

    def _drain_writes(self):
        self.cache.cache_controller.start_writing()
        H.sync()
        self.cache.writing_check()

    def _evict_all_to_host(self):
        self.cache.evict(EvictParams(num_tokens=1 << 30, mamba_num=1 << 20))
        # poison every device page so a restore can never pass by reading
        # stale device memory
        for layer in range(self.kv.full_kv_pool.layer_num):
            self.kv.full_kv_pool.kv_buffer[layer].view(torch.uint8).fill_(POISON)
            self.kv.full_kv_pool.index_k_with_scale_buffer[layer].fill_(POISON)

    def _load_back(self, tokens):
        m = self.cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
        self.assertGreater(m.host_hit_length, 0, "expected a host hit")
        ok = self.cache.load_back(m.best_match_node)
        self.assertTrue(ok)
        self.cache.ready_to_load_host_cache()
        H.sync()
        self.cache.loading_check()
        m2 = self.cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
        self.assertEqual(m2.host_hit_length, 0)
        return m2.device_indices

    def _alloc_slots(self, n):
        s = self.alloc.alloc(n)
        self.assertIsNotNone(s)
        return s

    # -- tests --------------------------------------------------------------
    def test_tree_page_is_widened_to_one_compression_group(self):
        """Fails without 180: the tree keeps 64-token ownership."""
        self.assertEqual(self.tree_page, 256)

    def test_host_restore_after_chunk_then_decode_keeps_group_rows(self):
        """Hazard (a). A 1024-token request is prefilled to 960 (checkpoint,
        write-through), then produces tokens 960..1023 (decode) and is
        published at 1024. Evict to host, poison device, load back: every
        pooled row of all four groups must be the request's own.
        Fails without 180 (rows of pools 240..255 come back stale)."""
        g = torch.Generator().manual_seed(0)
        tokens = torch.randint(10, 1000, (1024,), generator=g).tolist()
        slots = self._alloc_slots(1024)
        self.model.run(tokens, slots, 0, 960)
        self._publish(tokens, slots, 960)
        self.model.run(tokens, slots, 960, 1024)
        self._publish(tokens, slots, 1024)
        self._evict_all_to_host()
        restored = self._load_back(tokens)
        self.assertEqual(len(restored), 1024)
        bad_rows, bad_mla = self.model.check(tokens, restored, 1024)
        self.assertEqual(bad_mla, [], "latent rows")
        self.assertEqual(bad_rows, [], "pooled indexer rows restored from host")

    def test_fork_inside_group_does_not_overwrite_sibling(self):
        """Hazard (b). Branch A = X[0:128] + Y, branch B = X[0:128] + Z; a state
        exists at 128 (an earlier prompt ended there). B reuses the longest
        prefix the tree offers and computes the rest. A's rows must still be
        A's, on device and after a host round trip.
        Fails without 180 (B writes its group-0 rows into A's page)."""
        g = torch.Generator().manual_seed(1)
        x = torch.randint(10, 1000, (128,), generator=g).tolist()
        a = x + torch.randint(10, 1000, (384,), generator=g).tolist()
        b = x + torch.randint(1000, 2000, (384,), generator=g).tolist()
        slots_a = self._alloc_slots(512)
        self.model.run(a, slots_a, 0, 512)
        self._publish(a, slots_a, 128)  # an earlier prompt ended at 128
        self._publish(a, slots_a, 512)
        # B: reuse whatever the tree matches (tree-page aligned), compute the rest
        m = self.cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", b))))
        hit = len(m.device_indices)
        self.assertEqual(hit % self.tree_page, 0)
        own = self._alloc_slots(512 - hit)
        slots_b = torch.cat([m.device_indices, own]) if hit else own
        self.model.run(b, slots_b, hit, 512)
        self._publish(b, slots_b, 512)
        # device check for A
        ma = self.cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", a))))
        self.assertEqual(len(ma.device_indices), 512)
        bad_rows, _ = self.model.check(a, ma.device_indices, 512)
        self.assertEqual(bad_rows, [], "A's pooled rows after B ran (device)")
        # and after a host round trip
        self._evict_all_to_host()
        restored = self._load_back(a)
        bad_rows, bad_mla = self.model.check(a, restored, 512)
        self.assertEqual((bad_rows, bad_mla), ([], []), "A after host restore")
        restored_b = self._load_back(b)
        bad_rows, bad_mla = self.model.check(b, restored_b, 512)
        self.assertEqual((bad_rows, bad_mla), ([], []), "B after host restore")

    def test_flush_empties_device_and_host_tiers(self):
        """True flush: after reset nothing matches (device or host) and every
        host pool, including the indexer mirror, has all slots free."""
        g = torch.Generator().manual_seed(2)
        tokens = torch.randint(10, 1000, (512,), generator=g).tolist()
        slots = self._alloc_slots(512)
        self.model.run(tokens, slots, 0, 512)
        self._publish(tokens, slots, 512)
        self._evict_all_to_host()
        group = self.cache.cache_controller.mem_pool_host
        self.assertLess(group.available_size(), group.anchor_entry.host_pool.size)
        self.cache.reset()
        m = self.cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
        self.assertEqual((len(m.device_indices), m.host_hit_length), (0, 0))
        for e in group.entries:
            self.assertEqual(e.host_pool.available_size(), e.host_pool.size, e.name)

    def test_flush_with_write_through_in_flight(self):
        """/flush_cache while a write-through D2H is still unacknowledged (on
        CUDA it is really in flight). The stale ack must not republish anything
        after the flush, the freed host slots must be reusable, and a new
        prefix written into the same slots must restore byte-exact."""
        g = torch.Generator().manual_seed(3)
        old = torch.randint(10, 1000, (512,), generator=g).tolist()
        slots = self._alloc_slots(512)
        self.model.run(old, slots, 0, 512)
        self.cache.insert(
            InsertParams(
                key=RadixKey(array("q", old)), value=slots.clone(),
                mamba_value=self._state_slot(),
            )
        )
        ctrl = self.cache.cache_controller
        ctrl.start_writing()  # submitted, NOT acknowledged (no writing_check)
        self.assertGreaterEqual(len(ctrl.ack_write_queue), 1)
        # what Scheduler.flush_cache does once idle
        self.cache.reset()
        self.req_pool.clear()
        self.alloc.clear()
        self.cache.writing_check()  # a stale ack must be gone, not committed
        m = self.cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", old))))
        self.assertEqual((len(m.device_indices), m.host_hit_length), (0, 0))
        for e in ctrl.mem_pool_host.entries:
            self.assertEqual(e.host_pool.available_size(), e.host_pool.size, e.name)
        # new content reuses the same device and host slots
        new = torch.randint(1000, 2000, (512,), generator=g).tolist()
        slots = self._alloc_slots(512)
        self.model.run(new, slots, 0, 512)
        self._publish(new, slots, 512)
        self._evict_all_to_host()
        restored = self._load_back(new)
        self.assertEqual(self.model.check(new, restored, 512), ([], []))


class TestNoHiCacheKeepsBaseBehaviour(unittest.TestCase):
    """Without --enable-hierarchical-cache, 180 must not change the device tree."""

    def test_tree_page_stays_physical(self):
        from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
        from sglang.srt.runtime_context import reset_context

        self.addCleanup(reset_context)
        H.publish_args(enable_hierarchical_cache=False)
        req_pool, kv, _ = H.glm_like_pools(size=4096)
        alloc = PagedTokenToKVPoolAllocator(
            size=4096, page_size=PAGE, dtype=torch.bfloat16, device=H.DEVICE,
            kvcache=kv, need_sort=False,
        )
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False, req_to_token_pool=req_pool,
                token_to_kv_pool_allocator=alloc, page_size=PAGE,
                tree_components=(ComponentType.FULL, ComponentType.MAMBA),
                enable_mamba_extra_buffer=True,
            )
        )
        self.assertEqual(cache.page_size, PAGE)


if __name__ == "__main__":
    unittest.main()
