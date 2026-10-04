#!/usr/bin/env python3
"""118 on a real A100: the Triton DSA sparse attention against an fp32 reference and the served TileLang kernel.

GLM-5.3-Flash TP8 per-rank shape: 8 heads, 512-dim latent, qk_rope_head_dim 0, 2048 top-k + 3 kpool tail = 2051
index columns. Index rows are laid out like the kpool top-k transform (complete groups of 4, in order while a query
sees at most 512 groups, else 512 random groups in random order; then the 1-3 tail tokens; then -1) over a random
page table; q and KV are random bf16. An operator test, not a model or service test.

Run with the candidate tree first on PYTHONPATH, on one A100:
  python -m unittest -v test_dsa_sparse_118           numerics, dispatch, CUDA graph, no-recompile, negative control
  python test_dsa_sparse_118.py --bench               CUDA-graph timing of both dispatch paths (JSONL)
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import triton

from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd
from sglang.srt.layers.attention.dsa import sparse_attention_triton as k118

H, D, TOPK, KPOOL, PAGE = 8, 512, 2048, 4, 64
WIDTH = TOPK + KPOOL - 1
SM_SCALE = D**-0.5
LOG2E = math.log2(math.e)
DEV = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = False  # the fp32 reference must be fp32


def emit(**record):
    print(json.dumps(record, sort_keys=True), flush=True)


# ----------------------------------------------------------------------------- inputs
def page_table(ctx, pool_rows, gen):
    """Token position -> pool row: 64-token pages scattered over the pool; page 0 is the allocator's dummy page."""
    pages = torch.randperm(pool_rows // PAGE - 1, generator=gen)[: triton.cdiv(ctx, PAGE)] + 1
    return (pages[:, None] * PAGE + torch.arange(PAGE)).reshape(-1)[:ctx]


def kpool_rows(positions, table, gen, width=WIDTH, kpool=KPOOL):
    """Index rows [len(positions), width] as the kpool top-k transform emits them for queries at `positions`."""
    rows = torch.full((len(positions), width), -1, dtype=torch.int64)
    ngroups = TOPK // kpool
    for r, p in enumerate(positions):
        groups = (p + 1) // kpool
        sel = torch.arange(groups) if groups <= ngroups else torch.randperm(groups, generator=gen)[:ngroups]
        history = (sel[:, None] * kpool + torch.arange(kpool)).reshape(-1)
        tokens = torch.cat([history, torch.arange(groups * kpool, p + 1)])  # + the tail after the last group
        rows[r, : tokens.numel()] = table[tokens]
    return rows.to(torch.int32)


def make_case(requests, pool_rows=1 << 20, heads=H, seed=0, width=WIDTH, kpool=KPOOL):
    """requests: list of (context_len, query_tokens); each request's queries are its last `query_tokens` positions."""
    gen = torch.Generator().manual_seed(seed)
    rows = [kpool_rows(range(ctx - n, ctx), page_table(ctx, pool_rows, gen), gen, width, kpool) for ctx, n in requests]
    idx = torch.cat(rows).to(DEV)
    g = torch.Generator(device=DEV).manual_seed(seed + 1)
    q = torch.randn(idx.shape[0], heads, D, generator=g, device=DEV).to(torch.bfloat16)
    kv = torch.randn(pool_rows, 1, D, generator=g, device=DEV).to(torch.bfloat16)
    return q, kv, idx


def pad64(idx):
    """What _forward_tilelang does before the TileLang call: -1-pad the columns to a multiple of 64."""
    pad = (-idx.shape[-1]) % 64
    return torch.cat([idx, idx.new_full((idx.shape[0], pad), -1)], dim=1) if pad else idx


# ----------------------------------------------------------------------------- reference
def reference(q, kv, idx, sm_scale=SM_SCALE, chunk=32):
    """fp32 softmax over valid entries (0 <= index < P). Returns out [M, H, D], LSE [M, H] in log2 units (-inf for
    rows with no valid entry, whose out is 0) and vmax [M]: the largest |KV| value a row reads."""
    kv2 = kv.view(kv.shape[0], -1).float()
    out = torch.zeros(q.shape, dtype=torch.float32, device=DEV)
    lse = torch.full(q.shape[:2], float("-inf"), device=DEV)
    vmax = torch.zeros(q.shape[0], device=DEV)
    for s in range(0, q.shape[0], chunk):
        ii = idx[s : s + chunk].long()
        valid = (ii >= 0) & (ii < kv.shape[0])
        keys = kv2[ii.clamp(0, kv.shape[0] - 1)] * valid[..., None]
        scores = torch.einsum("chd,ckd->chk", q[s : s + chunk].float(), keys) * sm_scale
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
        has = valid.any(1)
        m = torch.where(has[:, None, None], scores.amax(-1, keepdim=True), torch.zeros_like(scores[..., :1]))
        p = torch.exp(scores - m)
        l = p.sum(-1, keepdim=True)
        out[s : s + chunk] = torch.einsum("chk,ckd->chd", p, keys) / l.clamp_min(1e-30)
        lse[s : s + chunk] = torch.where(has[:, None], (torch.log(l) + m).squeeze(-1) * LOG2E, float("-inf"))
        vmax[s : s + chunk] = keys.abs().amax((1, 2))
    return out, lse, vmax


def bound_ratio(x, ref, vmax, rows):
    """max |x - ref| / (2^-8 * (|ref| + vmax_row)) over `rows`: bf16 rounds the output and the softmax weights to
    8 significant bits, so an error of 2^-9 of either term per rounding is expected; <= 1 passes."""
    x, ref = x[rows].float(), ref[rows]
    tol = 2.0**-8 * (ref.abs() + vmax[rows][:, None, None])
    return float(torch.nan_to_num((x - ref).abs() / tol, nan=float("inf")).max())


def lse_ratio(lse, ref):
    """max |lse - ref| / (1e-4 + 1e-6 |ref|): both are fp32 sums over 512-wide dot products in different orders, a few
    fp32 ulps apart at large logits (q x 30 gives LSE ~ 2e3 in log2 units); <= 1 passes."""
    return float(((lse - ref).abs() / (1e-4 + 1e-6 * ref.abs())).max())


def served(q, kv, idx, return_lse=False):
    return tilelang_sparse_fwd(q=q, kv=kv, indices=pad64(idx).unsqueeze(1), sm_scale=SM_SCALE, d_v=D,
                               return_lse=return_lse)


def triton118(q, kv, idx, return_lse=False, sm_scale=SM_SCALE):
    return k118.sparse_attention_fwd(q, kv, idx, sm_scale, d_v=D, return_lse=return_lse)


def splits_for(rows, heads=H, width=WIDTH):
    return k118._num_splits(rows * triton.cdiv(heads, 16), k118._wave(heads, D, width, DEV))


# ----------------------------------------------------------------------------- cases
def with_holes(case, frac=0.3, seed=3):
    """-1 at random positions plus whole 32/64-column tiles of -1 in the middle of every row."""
    q, kv, idx = case
    g = torch.Generator(device=DEV).manual_seed(seed)
    idx = torch.where(torch.rand(idx.shape, generator=g, device=DEV) < frac, -1, idx)
    idx[:, 64:128] = -1
    idx[:, 1024:1056] = -1
    return q, kv, idx


def pool_boundary():
    """Rows reading the last pool row (valid) and rows holding indices >= P (out of range, must count as -1), both
    in place of a valid entry and in the -1 column after a short tail."""
    q, kv, idx = make_case([(40000, 64)], pool_rows=1 << 18, seed=5)
    P = kv.shape[0]
    idx[0::2, 5] = P - 1
    idx[1::4, 100] = P + 7
    after_tail = (idx[:, WIDTH - 1] < 0).nonzero().flatten()
    idx[after_tail[::2], WIDTH - 1] = P
    return q, kv, idx


def masked_rows():
    """Target-verify shape whose last 3 rows are fully -1, like padding rows from _pad_topk_indices."""
    q, kv, idx = make_case([(30000 + 997 * i, 4) for i in range(5)], seed=6)
    idx[-3:] = -1
    return q, kv, idx


def peaky_last_split():
    """q x 30 (near one-hot softmax) and, for head 0 of every row, a key aligned with q at the row's last valid
    index, so that head's maximum lands in the last non-empty split (8 decode rows: 16 splits)."""
    q, kv, idx = make_case([(20000 + 937 * i, 1) for i in range(8)], seed=16)
    last = torch.where(idx >= 0, torch.arange(idx.shape[1], device=DEV), -1).amax(1)
    rows = idx[torch.arange(idx.shape[0], device=DEV), last].long()
    kv[rows, 0] = torch.where(q[:, 0] >= 0, 3.0, -3.0).to(kv.dtype)
    return (q.float() * 30).to(torch.bfloat16), kv, idx


CASES = {
    "prefill_long": lambda: make_case([(49152, 1024)]),
    "prefill_first_chunk": lambda: make_case([(1000, 1000)], seed=1),
    "prefill_group_boundary": lambda: make_case([(2128, 128)], seed=2),  # queries cross 512 complete groups
    "prefill_odd_M": lambda: make_case([(30011, 333)], seed=4),
    "verify_32x4": lambda: make_case([(20000 + 937 * i, 4) for i in range(32)], seed=7),
    "decode_32x1": lambda: make_case([(20000 + 937 * i, 1) for i in range(32)], seed=8),
    "decode_M1": lambda: make_case([(5, 1)], seed=9),
    "holes_and_empty_tiles": lambda: with_holes(make_case([(40000, 257)], seed=10)),
    "pool_boundary": pool_boundary,
    "masked_rows": masked_rows,
    "peaky_last_split": peaky_last_split,
    "width_2048_kpool1": lambda: make_case([(9000, 200)], seed=11, width=2048, kpool=1),
    "heads64_split": lambda: make_case([(30000, 37)], heads=64, seed=12),
    "heads64": lambda: make_case([(30000, 300)], heads=64, seed=13),
}
SERVED_EXCLUDED = {"pool_boundary", "heads64_split", "heads64"}  # served counts index P; 64 heads is another kernel


class Numerics(unittest.TestCase):
    """Both kernels against fp32 on every row with a valid index; LSE in the served kernel's log2 units."""

    def run_case(self, name):
        q, kv, idx = CASES[name]()
        ref, ref_lse, vmax = reference(q, kv, idx)
        out, lse = triton118(q, kv, idx, return_lse=True)
        out, lse = out[0], lse[0]
        rows = ((idx >= 0) & (idx < kv.shape[0])).any(1)
        rec = dict(case=name, M=q.shape[0], heads=q.shape[1], splits=splits_for(q.shape[0], q.shape[1], idx.shape[1]),
                   triton_bound=bound_ratio(out, ref, vmax, rows),
            triton_max_abs=float((out[rows].float() - ref[rows]).abs().max()),
            triton_lse_max_abs=float((lse[rows] - ref_lse[rows]).abs().max()))
        self.assertLessEqual(rec["triton_bound"], 1.0, rec)
        self.assertLessEqual(lse_ratio(lse[rows], ref_lse[rows]), 1.0, rec)
        # The LSE and plain specializations produce the same output.
        self.assertTrue(torch.equal(triton118(q, kv, idx)[0], out))
        # A row without a valid index gives 0 and LSE -inf.
        self.assertTrue(bool((out[~rows] == 0).all()) and bool((lse[~rows] == float("-inf")).all()))
        if name not in SERVED_EXCLUDED:
            s_out, s_lse = served(q, kv, idx, return_lse=True)
            s_out, s_lse = s_out[0], s_lse[0]
            rec.update(served_bound=bound_ratio(s_out, ref, vmax, rows),
                       served_max_abs=float((s_out[rows].float() - ref[rows]).abs().max()),
                       served_lse_max_abs=float((s_lse[rows] - ref_lse[rows]).abs().max()),
                       triton_vs_served_max_abs=float((out[rows].float() - s_out[rows].float()).abs().max()))
            rec.update(triton_mean_abs=float((out[rows].float() - ref[rows]).abs().mean()),
                       served_mean_abs=float((s_out[rows].float() - ref[rows]).abs().mean()))
            self.assertLessEqual(rec["served_bound"], 1.0, rec)
            self.assertLessEqual(lse_ratio(s_lse[rows], ref_lse[rows]), 1.0, rec)
            # Same accuracy as the served kernel on average (a systematic error, e.g. a skipped tile, fails this).
            self.assertLessEqual(rec["triton_mean_abs"], 1.25 * rec["served_mean_abs"] + 1e-7, rec)
            # Documented difference: TileLang returns NaN for a fully masked row (same LSE, -inf).
            self.assertTrue(bool(s_out[~rows].isnan().all()) and bool((s_lse[~rows] == float("-inf")).all()))
        emit(kind="numerics", **rec)

    def test_cases(self):
        for name in CASES:
            with self.subTest(name):
                self.run_case(name)

    def test_index_at_pool_size_counts_as_masked(self):
        q, kv, idx = pool_boundary()
        masked = torch.where(idx >= kv.shape[0], -1, idx)
        self.assertTrue(bool((idx[:, WIDTH - 1] == kv.shape[0]).any()) and bool((idx[:, 100] > kv.shape[0]).any()))
        self.assertTrue(torch.equal(triton118(q, kv, idx), triton118(q, kv, masked)))

    def test_nan_input_is_not_hidden(self):
        q, kv, idx = make_case([(3000, 40)], seed=14)
        q[3, 2, 7] = float("nan")
        out = triton118(q, kv, idx)[0]
        self.assertTrue(bool(out[3, 2].isnan().all()))
        self.assertFalse(bool(out[:3].isnan().any() or out[4:].isnan().any()))


class NegativeControl(unittest.TestCase):
    """The numerics check must reject a kernel that reads the wrong KV rows."""

    def test_dropped_mask_is_rejected(self):
        q, kv, idx = make_case([(1000, 1000)], seed=1)
        ref, _, vmax = reference(q, kv, idx)
        rows = (idx >= 0).any(1)
        wrong = triton118(q, kv, idx.clamp_min(0))[0]  # -1 read as pool row 0
        ratio = bound_ratio(wrong, ref, vmax, rows)
        emit(kind="negative_control", defect="mask_dropped", bound=ratio)
        self.assertGreater(ratio, 1.0)

    def test_off_by_one_row_is_rejected(self):
        q, kv, idx = make_case([(49152, 256)], seed=15)
        ref, _, vmax = reference(q, kv, idx)
        rows = (idx >= 0).any(1)
        wrong = triton118(q, kv, torch.where(idx >= 0, idx + 1, idx))[0]
        ratio = bound_ratio(wrong, ref, vmax, rows)
        emit(kind="negative_control", defect="off_by_one_row", bound=ratio)
        self.assertGreater(ratio, 1.0)


class Dispatch(unittest.TestCase):
    """DeepseekSparseAttnBackend._forward_tilelang: switch off = the served call unchanged, on = the 118 kernel."""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention import dsa_backend

        cls.backend = dsa_backend

    def call(self, switch, q, kv, idx, return_lse, *, prefill_switch=False, is_prefill=False):
        prev = self.backend._AX_DSA_SPARSE_TRITON
        prev_prefill = self.backend._AX_DSA_SPARSE_TRITON_PREFILL
        self.backend._AX_DSA_SPARSE_TRITON = switch
        self.backend._AX_DSA_SPARSE_TRITON_PREFILL = prefill_switch
        try:
            return self.backend.DeepseekSparseAttnBackend._forward_tilelang(
                NS(_ax118_prefill_seen=True), q_all=q, kv_cache=kv, v_head_dim=D,
                page_table_1=idx, sm_scale=SM_SCALE,
                return_lse=return_lse, is_prefill=is_prefill)
        finally:
            self.backend._AX_DSA_SPARSE_TRITON = prev
            self.backend._AX_DSA_SPARSE_TRITON_PREFILL = prev_prefill

    def test_prefill_switch_does_not_replace_partial_or_decode_calls(self):
        q, kv, idx = masked_rows()
        for is_prefill in (False, True):
            for return_lse in (False, True):
                with self.subTest(is_prefill=is_prefill, return_lse=return_lse):
                    got = self.call(False, q, kv, idx, return_lse,
                                    prefill_switch=True, is_prefill=is_prefill)
                    use_triton = is_prefill and not return_lse
                    want = triton118(q, kv, idx) if use_triton else served(q, kv, idx, return_lse)
                    if return_lse:
                        self.assertTrue(torch.equal(got[1], want[1][0]))
                        got, want = got[0], want[0]
                    self.assertTrue(torch.equal(got.isnan(), want.isnan()))
                    self.assertTrue(torch.equal(got.nan_to_num(), want.nan_to_num()))

    def test_off_is_served_on_is_118(self):
        q, kv, idx = masked_rows()
        for return_lse in (False, True):
            off, on = self.call(False, q, kv, idx, return_lse), self.call(True, q, kv, idx, return_lse)
            want_off, want_on = served(q, kv, idx, return_lse), triton118(q, kv, idx, return_lse)
            if return_lse:  # [1, M, H] -> [M, H] for the DCP combine, both paths
                self.assertEqual(off[1].shape, (q.shape[0], H))
                self.assertEqual(on[1].shape, (q.shape[0], H))
                self.assertTrue(torch.equal(off[1], want_off[1][0]) and torch.equal(on[1], want_on[1][0]))
                off, on, want_off, want_on = off[0], on[0], want_off[0], want_on[0]
            self.assertEqual(on.shape, (1, q.shape[0], H, D))
            self.assertTrue(torch.equal(off.nan_to_num(), want_off.nan_to_num()))
            self.assertTrue(bool(off[0, -3:].isnan().all()))  # served path really ran
            self.assertTrue(torch.equal(on, want_on))

    def test_state_report(self):
        prev = self.backend._AX_DSA_SPARSE_TRITON, self.backend._ax118_engaged
        try:
            self.backend._AX_DSA_SPARSE_TRITON, self.backend._ax118_engaged = False, False
            self.assertEqual(self.backend.ax118_state(), "off:SGLANG_AX_DSA_SPARSE_TRITON_unset")
            self.backend._AX_DSA_SPARSE_TRITON = True
            self.assertEqual(self.backend.ax118_state(), "off:no_tilelang_dsa_backend")
            self.backend._ax118_engaged = True
            self.assertEqual(self.backend.ax118_state(), "on")
        finally:
            self.backend._AX_DSA_SPARSE_TRITON, self.backend._ax118_engaged = prev

    def test_prefill_state_report(self):
        with patch.object(self.backend, "_AX_DSA_SPARSE_TRITON", False), \
                patch.object(self.backend, "_AX_DSA_SPARSE_TRITON_PREFILL", True), \
                patch.object(self.backend, "_ax118_engaged", False):
            self.assertEqual(self.backend.ax118_state(), "off:no_tilelang_dsa_prefill")
            self.backend._ax118_engaged = True
            self.assertEqual(self.backend.ax118_state(), "on:prefill")


def graph_replay(make):
    """Capture one call on static buffers, then refill q, KV and indices with new content and replay; each replay
    must equal an eager call on the same inputs bit for bit and stay within the fp32 bound."""
    q, kv, idx = make(0)
    out = triton118(q, kv, idx)  # warm (also compiled by the warmup in serving)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = triton118(q, kv, idx)
    for trial in range(1, 4):
        q2, kv2, idx2 = make(trial)
        q.copy_(q2)
        kv.copy_(kv2)
        idx.copy_(idx2)
        g.replay()
        torch.cuda.synchronize()
        eager = triton118(q, kv, idx)
        assert torch.equal(out, eager), "replay differs from eager"
        ref, _, vmax = reference(q, kv, idx)
        ratio = bound_ratio(out[0], ref, vmax, (idx >= 0).any(1))
        assert ratio <= 1.0, ratio
    return 3


class CudaGraph(unittest.TestCase):
    def test_replay_with_fresh_inputs(self):
        shapes = {
            "decode_M1": [(3000, 1)],
            "decode_32x1": [(20000 + 937 * i, 1) for i in range(32)],
            "verify_32x4": [(20000 + 937 * i, 4) for i in range(32)],
            "prefill_512": [(30000, 512)],
        }
        splits = set()
        for name, reqs in shapes.items():
            with self.subTest(name):
                M = sum(n for _, n in reqs)

                def make(trial, reqs=reqs):  # other contexts, pages, q and KV per trial; same shapes
                    return make_case([(c - 97 * trial, n) for c, n in reqs], pool_rows=1 << 19, seed=100 + trial)

                replays = graph_replay(make)
                splits.add(splits_for(M))
                emit(kind="graph", case=name, M=M, splits=splits_for(M), replays=replays, status="PASS")
        self.assertIn(1, splits)  # both the single-split and the split + combine path were captured
        self.assertGreater(max(splits), 1)


# GLM-5.3-Flash at TP8: 64 heads / 8 ranks, 512-wide latent, top-k 2048, kpool 4 (served) and 1.
GLM_TP8 = dict(dsa_prefill_impl="tilelang", dsa_decode_impl="tilelang", kv_cache_dtype=torch.bfloat16,
               qk_rope_head_dim=0, hisparse_coordinator=None, num_q_heads=64 // 8, kv_lora_rank=512,
               dsa_index_topk=2048, dsa_index_kpool=4, device="cuda")

NO_RECOMPILE_PROBE = r"""
import torch, triton.knobs as knobs
from types import SimpleNamespace as NS
from sglang.srt.layers.attention import dsa_backend as b
from sglang.srt.layers.attention.dsa import sparse_attention_triton as k
loaded = []  # Triton loads each specialization onto the device once per process, at its first launch
knobs.runtime.kernel_load_start_hook.add(lambda module, function, name, *_: loaded.append(name))
ours = lambda: len([n for n in loaded if "sparse_attention" in n])
b._AX_DSA_SPARSE_TRITON = True
b._AX_DSA_SPARSE_TRITON_PREFILL = False
b.get_parallel = lambda: NS(dcp_enabled=False)
b.get_exec = lambda: NS(deterministic=NS(enable_deterministic_inference=False))
cfg = dict(CFG, device_sm_major=torch.cuda.get_device_capability()[0])
MODE_SETUP
for kpool in (4, 1):  # warm up through the backend's own code, as at server start
    b.DeepseekSparseAttnBackend._ax118_init(NS(**dict(cfg, dsa_index_kpool=kpool)))
warm = ours()
pool = torch.randn(1 << 16, 1, 512, device="cuda").to(torch.bfloat16)
for kpool in (4, 1):
    width = cfg["dsa_index_topk"] + kpool - 1  # the indexer's output (dsa_indexer_kpool.py: index_topk + tail_pool)
    for M in (1, 2, 3, 5, 8, 13, 16, 31, 32, 64, 100, 128, 161, 162, 163, 215, 333, 1000, 4096, 8191, 8192):
        q = torch.randn(M, cfg["num_q_heads"], 512, device="cuda").to(torch.bfloat16)
        buf = torch.randint(0, 1 << 16, (M, width + 61), device="cuda", dtype=torch.int32)
        for idx in (buf[:, :width].contiguous(), buf[:, :width]):  # packed and a column slice of a wider table
            if b._AX_DSA_SPARSE_TRITON_PREFILL:
                b.DeepseekSparseAttnBackend._forward_tilelang(
                    NS(_ax118_prefill_seen=True), q, pool[:1000 + M], 512, idx, 0.04,
                    is_prefill=True)
            else:
                k.sparse_attention_fwd(q, pool[: 1000 + M], idx, 0.04, 512)
torch.cuda.synchronize()
print(warm, ours() - warm)
"""


class Warmup(unittest.TestCase):
    def run_probe(self, mode_setup=""):
        """In a fresh process: after the backend's warmup at the GLM TP8 config, no call with the indexer's output
        shape (any M, pool size or index row stride) compiles or loads another specialization."""
        probe = NO_RECOMPILE_PROBE.replace("CFG", repr(GLM_TP8)).replace("MODE_SETUP", mode_setup)
        res = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=os.environ)
        self.assertEqual(res.returncode, 0, res.stderr[-3000:])
        warm, late = map(int, res.stdout.split()[-2:])
        emit(kind="warmup", loaded_in_warmup=warm, loaded_after_warmup=late)
        self.assertEqual(warm, 5 + 4 + 5)  # 5 split counts per width; the 4 combine kernels do not depend on it
        self.assertEqual(late, 0)

    def test_backend_warmup_covers_every_serving_shape(self):
        self.run_probe()

    def test_prefill_mode_warmup_covers_every_serving_shape(self):
        self.run_probe("b._AX_DSA_SPARSE_TRITON = False\n"
                       "b._AX_DSA_SPARSE_TRITON_PREFILL = True\n"
                       "b.get_parallel = lambda: NS(dcp_enabled=True)")


class Startup(unittest.TestCase):
    """DeepseekSparseAttnBackend._ax118_init engages only with the switch and a TileLang impl, and refuses to start
    outside the validated envelope instead of falling back to TileLang."""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention import dsa_backend

        cls.b = dsa_backend

    def init(self, switch=True, cuda=True, dcp=False, deterministic=False,
             prefill_switch=False, **overrides):
        cfg = dict(GLM_TP8, device_sm_major=torch.cuda.get_device_capability()[0])
        cfg = NS(**dict(cfg, **overrides))
        exec_ctx = NS(deterministic=NS(enable_deterministic_inference=deterministic))
        with patch.object(self.b, "_AX_DSA_SPARSE_TRITON", switch), patch.object(self.b, "_ax118_engaged", False), \
                patch.object(self.b, "_AX_DSA_SPARSE_TRITON_PREFILL", prefill_switch), \
                patch.object(self.b, "is_cuda", lambda: cuda), \
                patch.object(self.b, "get_parallel", lambda: NS(dcp_enabled=dcp)), \
                patch.object(self.b, "get_exec", lambda: exec_ctx):
            self.b.DeepseekSparseAttnBackend._ax118_init(cfg)
            return self.b._ax118_engaged

    def test_engages_with_switch_and_a_tilelang_impl(self):
        self.assertTrue(self.init())
        self.assertTrue(self.init(dsa_prefill_impl="flashmla_sparse"))  # decode still TileLang
        self.assertFalse(self.init(dsa_prefill_impl="flashmla_sparse", dsa_decode_impl="flashmla_kv"))
        self.assertFalse(self.init(switch=False, dcp=True, kv_cache_dtype=torch.float8_e4m3fn))

    def test_refuses_outside_the_envelope(self):
        cases = {"not CUDA": dict(cuda=False), "sm7x": dict(device_sm_major=7),
                 "float8_e4m3fn KV cache": dict(kv_cache_dtype=torch.float8_e4m3fn),
                 "qk_rope_head_dim=64": dict(qk_rope_head_dim=64), "DCP": dict(dcp=True),
                 "HiSparse": dict(hisparse_coordinator=object()),
                 "deterministic inference": dict(deterministic=True)}
        for reason, kw in cases.items():
            with self.subTest(reason), self.assertRaisesRegex(ValueError, "118.*got: .*" + re.escape(reason)):
                self.init(**kw)

    def test_prefill_mode_accepts_dcp_but_keeps_other_guards(self):
        self.assertTrue(self.init(switch=False, prefill_switch=True, dcp=True))
        self.assertFalse(self.init(switch=False, prefill_switch=True, dcp=True,
                                   dsa_prefill_impl="flashmla_sparse"))
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.init(prefill_switch=True)
        for kw in (dict(kv_cache_dtype=torch.float8_e4m3fn), dict(qk_rope_head_dim=64),
                   dict(hisparse_coordinator=object()), dict(deterministic=True)):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.init(switch=False, prefill_switch=True, dcp=True, **kw)


# ----------------------------------------------------------------------------- timing (--bench)
_FLUSH = None


def l2_flush():
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(96 << 20, dtype=torch.uint8, device=DEV)
    _FLUSH.fill_(1)


def graph_ms(fn, n=8, reps=7):
    """Per-call GPU time inside a CUDA graph with L2 flushed before every call (as decode/verify run in serving):
    (graph of n x [flush, fn] - graph of n x [flush]) / n, median of `reps` replays."""
    fn()
    torch.cuda.synchronize()
    graphs = []
    for with_fn in (True, False):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n):
                l2_flush()
                if with_fn:
                    fn()
        graphs.append(g)

    def replay_ms(g):
        g.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b))
        return sorted(ts)[reps // 2]

    return (replay_ms(graphs[0]) - replay_ms(graphs[1])) / n


BENCH = {
    "prefill_8192_ctx49k": [(49152, 8192)],
    "prefill_8192_first_chunk": [(8192, 8192)],
    "prefill_333_ctx30k": [(30011, 333)],
    "verify_32x4": [(20000 + 937 * i, 4) for i in range(32)],
    "verify_8x4": [(20000 + 937 * i, 4) for i in range(8)],
    "verify_1x4": [(30000, 4)],
    "decode_32x1": [(20000 + 937 * i, 1) for i in range(32)],
    "decode_8x1": [(20000 + 937 * i, 1) for i in range(8)],
    "decode_1x1": [(30000, 1)],
}


def bench(rounds):
    from sglang.srt.layers.attention import dsa_backend

    fwd = dsa_backend.DeepseekSparseAttnBackend._forward_tilelang
    k118.warmup_sparse_attention_fwd(H, D, WIDTH, DEV, False)
    emit(kind="env", gpu=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__,
         sms=torch.cuda.get_device_properties(DEV).multi_processor_count)
    for name, reqs in BENCH.items():
        q, kv, idx = make_case(reqs, seed=21)

        def call(switch):
            def f():
                dsa_backend._AX_DSA_SPARSE_TRITON = switch
                return fwd(None, q_all=q, kv_cache=kv, v_head_dim=D, page_table_1=idx, sm_scale=SM_SCALE)
            return f

        times = {"served": [], "triton": []}
        for _ in range(rounds):  # interleaved rounds; report the median round
            for label, switch in (("served", False), ("triton", True)):
                times[label].append(graph_ms(call(switch)))
        dsa_backend._AX_DSA_SPARSE_TRITON = False
        med = {k: sorted(v)[len(v) // 2] for k, v in times.items()}
        emit(kind="bench", case=name, M=q.shape[0], splits=splits_for(q.shape[0]),
             valid_per_row=float((idx >= 0).sum(1).float().mean()), served_ms=med["served"], triton_ms=med["triton"],
             speedup=med["served"] / med["triton"], served_rounds=times["served"], triton_rounds=times["triton"])
    for key, entry in k118._sparse_attention_fwd_kernel.device_caches[torch.cuda.current_device()][0].items():
        emit(kind="kernel", specialization=str(key)[:200], n_regs=entry.n_regs, n_spills=entry.n_spills,
             shared=entry.metadata.shared)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--rounds", type=int, default=5)
    args, rest = ap.parse_known_args()
    if args.bench:
        bench(args.rounds)
    else:
        unittest.main(argv=[sys.argv[0], *rest])
