"""Engine 127: which KDA checkpoint a request's prefill publishes when its radix
match found a full-KV hit beyond its deepest KDA state (a branch point).

One state is saved per extend. The base saves the branch point instead of the
extend-end checkpoint. With SGLANG_AX_KDA_TAIL_FIRST=1 the last extend of a
fresh request keeps its prompt-end checkpoint when the branch point is deep
(>= SGLANG_AX_KDA_TAIL_FIRST_MIN_SHARE of the prompt, pinned to 0.5 here); a
shallow branch point and a retracted request (fill = prompt + output) keep the
base choice (engine/docs/127-kda-tail-first.md).

Real code: Req.init_next_round_input (radix match, branch point), ScheduleBatch
track selection, the backend's snapshot index math, UnifiedRadixCache insert and
match, HybridReqToTokenPool on CPU. Fixture and chunk runner are those of
test_180_upstream_38212_checkpoint (same directory): the DSA pool object only
supplies the compressed-index geometry (tree page, checkpoint grid 256), and the
kernel's states are written as exact prefix markers, so a published slot is
checked to hold the state of the prefix its depth claims.

The switch comes from the run's environment; run it both ways:
  SGLANG_AX_KDA_TAIL_FIRST=0|1 run_tests.sh TREE PYTHON test_127_tail_first
On a tree without 127 the run with the switch on fails (verified).
"""

import os
import unittest
from array import array
from itertools import count
from unittest.mock import patch

from sglang.srt.managers import schedule_batch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.runtime_context import reset_context
from sglang.srt.sampling.sampling_params import SamplingParams

import test_180_upstream_38212_checkpoint as ckpt

TAIL_FIRST = os.environ.get("SGLANG_AX_KDA_TAIL_FIRST", "0") == "1"
_RIDS = count()


class TestKdaTailFirst(unittest.TestCase):
    def setUp(self):
        env = patch.dict(
            os.environ,
            {
                "SGLANG_AX_KDA_TAIL_FIRST": "1" if TAIL_FIRST else "0",
                "SGLANG_AX_KDA_TAIL_FIRST_MIN_SHARE": "0.5",
            },
        )
        env.start()
        self.addCleanup(env.stop)
        switch = getattr(schedule_batch, "ax_kda_tail_first", None)
        if switch is not None:  # read once per process: re-read this case's environment
            switch.cache_clear()
            self.addCleanup(switch.cache_clear)
        self.h = ckpt.TestCompressedDsaCheckpointRoundtrip()
        self.cache, self.allocator, self.req_pool = self.h._fixture()

    def tearDown(self):
        reset_context()

    def _request(self, tokens, output=()):
        """A request after the scheduler's real radix match (no KDA state on its
        path here, so its whole fill is extended). `output` is what a retracted
        request keeps: it re-prefills prompt + output."""
        sampling = SamplingParams(max_new_tokens=len(output) + 1)
        sampling.normalize(None)
        req = Req(
            rid=f"r{next(_RIDS)}",
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=sampling,
            vocab_size=128,
        )
        req.output_ids = array("q", output)
        self.req_pool.alloc([req])
        req.init_next_round_input(self.cache)
        self.assertEqual(len(req.prefix_indices), 0)
        n = len(req.full_untruncated_fill_ids)
        self.assertEqual(n, len(tokens) + len(output))
        self.req_pool.write((req.kv.req_pool_idx, slice(0, n)), self.allocator.alloc(n))
        req.kv.kv_allocated_len = n
        return req

    def _prefill(self, req, *ends):
        """Run the extends ending at `ends`; return each one's published checkpoint depth."""
        return [self.h._run_chunk(self.cache, self.req_pool, req, end)[0] for end in ends]

    def _reusable(self, tokens, depth):
        """A later request with `tokens` resumes from a KDA state at exactly `depth`."""
        self.h._assert_cached_state_matches_prefix(
            self.cache, self.req_pool, array("q", tokens), depth
        )

    def test_deep_branch_point_in_the_last_extend(self):
        # A leaves KV [0, 768) with its state at 768. B shares 512 tokens with A:
        # its match finds KV to 512 but no state, so the branch point is 512.
        self._prefill(self._request([1] * 1000), 1000)
        b_tokens = [1] * 512 + [2] * 400
        b = self._request(b_tokens)
        self.assertEqual(b.mamba_branching_seqlen, 512)  # deep: 512 >= 0.5 * 912
        # One extend 0 -> 912 reaches B's prompt end (grid checkpoint 768).
        self.assertEqual(self._prefill(b, 912), [768 if TAIL_FIRST else 512])
        # B's chain successor (B's prompt + a new turn) resumes at B's prompt-end
        # checkpoint only with the switch on; the base leaves it at the branch point.
        self._reusable(b_tokens + [3] * 200, 768 if TAIL_FIRST else 512)
        # The cost: another request forking at the same point finds no state there.
        self._reusable([1] * 512 + [4] * 300, 0 if TAIL_FIRST else 512)

    def test_shallow_branch_point_stays_with_siblings(self):
        # S shares only 256 tokens (a system prompt, say) with A: 256 < 0.5 * 1100.
        self._prefill(self._request([1] * 1000), 1000)
        s = self._request([1] * 256 + [2] * 844)
        self.assertEqual(s.mamba_branching_seqlen, 256)
        self.assertEqual(self._prefill(s, 1100), [256])
        self._reusable([1] * 256 + [3] * 300, 256)

    def test_retracted_request_keeps_the_branch_point(self):
        # A leaves KV [0, 1024). R shares 768 tokens with A (deep for its 912-token
        # prompt and for its 1062-token fill) and re-prefills 150 output tokens
        # after a retraction: its fill-end checkpoint 1024 lies in its output,
        # which the successor (prompt + new turn) never matches.
        self._prefill(self._request([1] * 1100), 1100)
        r_tokens = [1] * 768 + [2] * 144
        r = self._request(r_tokens, output=[9] * 150)
        self.assertEqual(r.mamba_branching_seqlen, 768)
        self.assertEqual(self._prefill(r, 1062), [768])
        self._reusable(r_tokens + [3] * 200, 768)

    def test_branch_point_in_an_earlier_chunk_is_unchanged(self):
        self._prefill(self._request([1] * 1000), 1000)
        e_tokens = [1] * 512 + [5] * 600
        e = self._request(e_tokens)
        self.assertEqual(e.mamba_branching_seqlen, 512)
        # The branch point lies in the first chunk (0 -> 832), the prompt end in
        # the second (832 -> 1112): both are saved whatever the switch.
        self.assertEqual(self._prefill(e, 832, 1112), [512, 1024])
        self._reusable(e_tokens + [6] * 100, 1024)
        self._reusable([1] * 512 + [7] * 300, 512)


if __name__ == "__main__":
    unittest.main()
