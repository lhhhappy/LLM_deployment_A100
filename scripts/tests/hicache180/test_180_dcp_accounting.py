"""Independent DCP accounting checks on the actual host-pool constructors."""
from types import SimpleNamespace as NS
import unittest

import torch

import _cpu_harness as H
import test_180_glm_host_pools as base


class TestDCPAccounting(unittest.TestCase):
    def build(self, width):
        from sglang.srt.runtime_context import reset_context

        fixture = base.TestGlmHostTier()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        reset_context()
        H.publish_args(hicache_size=1)
        return fixture._build(with_draft=True, dcp_width=width)[-1]

    def test_fixed_budget_includes_actual_backing_storages(self):
        from sglang.srt.mem_cache.hicache_storage import PoolName

        for width in (1, 2, 4, 8):
            with self.subTest(width=width):
                result = self.build(width)
                storages = {}
                for entry in result.host_pool_group.entries:
                    hp = entry.host_pool
                    if entry.name == PoolName.INDEXER:
                        tensors = [hp.index_k_with_scale_buffer]
                    elif entry.name == PoolName.KV:
                        tensors = [hp.kv_buffer]
                    else:
                        tensors = hp.get_hybrid_pool_buffer()
                    for tensor in tensors:
                        storage = tensor.untyped_storage()
                        storages[storage.data_ptr()] = storage.nbytes()
                actual_bytes = sum(storages.values())
                self.assertGreater(actual_bytes, .95e9)
                self.assertLessEqual(actual_bytes, 1.01e9)  # whole-page rounding
                print(f"DCP_HOST_BUDGET width={width} backing_bytes={actual_bytes}", flush=True)
            self.doCleanups()

    def test_transfer_bytes_match_physical_copy_geometry(self):
        from sglang.srt.mem_cache.hicache_storage import PoolName

        for width in (1, 2, 4, 8):
            with self.subTest(width=width):
                result = self.build(width)
                group = result.host_pool_group
                anchor = group.anchor_entry.host_pool
                indexer = group.entry_map[PoolName.INDEXER].host_pool
                # Four physical MLA pages, W times as many replicated index pages.
                physical_tokens = anchor.page_size * 4
                logical_tokens = physical_tokens * width
                op = NS(device_indices=torch.arange(logical_tokens),
                        pool_transfers=[NS(name=PoolName.INDEXER,
                                           indices_from_pool=PoolName.KV,
                                           host_indices=None)])
                # Derive row bytes from the allocated tensor shapes, including
                # the packed NextN layer, rather than trusting size_per_token.
                # MLA page_first is token-major [tokens, layers, 1, dim];
                # the indexer is page-major [pages, layers, 1, page_bytes].
                mla_row_bytes = anchor.kv_buffer[0].numel() * anchor.kv_buffer.element_size()
                index_page_bytes = indexer.index_k_with_scale_buffer[0].numel()
                expected = physical_tokens * mla_row_bytes + 4 * width * index_page_bytes
                self.assertEqual(result.cache_controller._transfer_num_bytes(op), expected)
                print(f"DCP_TRANSFER_BYTES width={width} expected={expected}", flush=True)
            self.doCleanups()


if __name__ == "__main__":
    unittest.main(verbosity=2)
