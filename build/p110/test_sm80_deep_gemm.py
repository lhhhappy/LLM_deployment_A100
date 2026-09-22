"""Unit test for the patch-110 shim: paged and ragged MQA logits vs a naive per-row/per-head loop."""
import sys

import torch

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
import sm80_deep_gemm as m  # noqa: E402

torch.manual_seed(0)
dev = "cuda"
H, D, BLK = 32, 128, 64


def naive(q, k, kscale, w):
    # q [H,D] f32, k [n,D] f32 -> [n]
    return (torch.relu(k @ q.t()) * w.view(1, H)).sum(1) * kscale


# ---- ragged (prefill) ----
nq, nk = 37, 300
q = (torch.randn(nq, H, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
k = (torch.randn(nk, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
ks_ = torch.rand(nk, device=dev) + 0.5
w = torch.randn(nq, H, device=dev)
kst = torch.randint(0, 50, (nq,), device=dev, dtype=torch.int32)
ke = torch.randint(100, nk, (nq,), device=dev, dtype=torch.int32)
out = m.fp8_mqa_logits(q, (k, ks_), w, kst, ke, clean_logits=True)
err = 0.0
for i in range(nq):
    ref = naive(q[i].float(), k.float(), ks_, w[i])
    a, b = int(kst[i]), int(ke[i])
    err = max(err, (out[i, a:b] - ref[a:b]).abs().max().item() / (ref[a:b].abs().max().item() + 1e-6))
    assert torch.isinf(out[i, :a]).all() and torch.isinf(out[i, b:]).all()
print(f"ragged max rel err {err:.3e}")
assert err < 2e-2

# ---- paged (decode) ----
B, P, NB = 5, 7, 40
cache = torch.zeros(NB, BLK, 1, D + 4, dtype=torch.uint8, device=dev)
vals = (torch.randn(NB, BLK, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
scales = torch.rand(NB, BLK, device=dev) + 0.5
flat = cache.view(NB, BLK * (D + 4))
flat[:, : BLK * D] = vals.view(torch.uint8).reshape(NB, BLK * D)
flat[:, BLK * D :] = scales.contiguous().view(torch.uint8).reshape(NB, BLK * 4)
bt = torch.stack([torch.randperm(NB, device=dev)[:P] for _ in range(B)]).to(torch.int32)
ctx = torch.randint(1, P * BLK, (B,), device=dev, dtype=torch.int32)
qd = (torch.randn(B, 1, H, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
wd = torch.randn(B, H, device=dev)
maxlen = P * BLK + 10
outp = m.fp8_paged_mqa_logits(qd, cache, wd, ctx, bt, None, maxlen)
err = 0.0
for b in range(B):
    kk = vals[bt[b].long()].reshape(P * BLK, D).float()
    ss = scales[bt[b].long()].reshape(P * BLK)
    ref = naive(qd[b, 0].float(), kk, ss, wd[b])
    L = int(ctx[b])
    err = max(err, (outp[b, :L] - ref[:L]).abs().max().item() / (ref[:L].abs().max().item() + 1e-6))
    assert (outp[b, L:] == 0).all()
print(f"paged max rel err {err:.3e}  out shape {tuple(outp.shape)}")
assert err < 2e-2
# CUDA-graph capture of the paged path (no host sync allowed)
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    m.fp8_paged_mqa_logits(qd, cache, wd, ctx, bt, None, maxlen)
torch.cuda.synchronize()
with torch.cuda.graph(g):
    o2 = m.fp8_paged_mqa_logits(qd, cache, wd, ctx, bt, None, maxlen)
g.replay(); torch.cuda.synchronize()
assert torch.allclose(o2, outp)
print("cuda graph capture OK; ALL PASS")
