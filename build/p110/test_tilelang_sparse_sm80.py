# sm80 check: DSA sparse MLA via tilelang v1 kernel (GLM: kv_lora_rank=512, rope=0 -> tail_dim=0),
# heads/rank=8, topk=2048 + kpool tail padded to 2112. Numeric vs torch reference + CUDA graph capture.
import sys, torch
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "/tmp/ax/src/b110")
from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd
torch.manual_seed(0); dev = "cuda"
H, D, N, TOPK = 8, 512, 200_000, 2112
scale = D ** -0.5
def ref(q, kv, idx):
    out = torch.empty(q.shape[0], H, D, device=dev, dtype=torch.float32)
    for t in range(q.shape[0]):
        ii = idx[t, 0]; valid = ii >= 0; k = kv[ii.clamp(min=0), 0].float()
        s = (q[t].float() @ k.T) * scale; s[:, ~valid] = float("-inf")
        out[t] = torch.softmax(s, -1) @ k
    return out
kv = torch.randn(N, 1, D, device=dev, dtype=torch.bfloat16)
ok = True
for T, nvalid in [(1, 2051), (8, 700), (64, 2112), (512, 2048), (4096, 1500)]:
    q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
    idx = torch.full((T, 1, TOPK), -1, device=dev, dtype=torch.int32)
    idx[:, :, :nvalid] = torch.randint(0, N, (T, 1, nvalid), device=dev, dtype=torch.int32)
    o = tilelang_sparse_fwd(q=q, kv=kv, indices=idx, sm_scale=scale, d_v=D)
    o = o.reshape(T, H, D).float()
    r = ref(q[:64], kv, idx[:64]); e = ((o[:64] - r).abs().max() / r.abs().max()).item()
    torch.cuda.synchronize(); t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True); t0.record()
    for _ in range(10): tilelang_sparse_fwd(q=q, kv=kv, indices=idx, sm_scale=scale, d_v=D)
    t1.record(); torch.cuda.synchronize()
    print(f"T={T} valid={nvalid} rel_err={e:.2e} time={t0.elapsed_time(t1)/10:.3f}ms"); ok &= e < 2e-2
q = torch.randn(32, H, D, device=dev, dtype=torch.bfloat16)
idx = torch.randint(0, N, (32, 1, TOPK), device=dev, dtype=torch.int32)
tilelang_sparse_fwd(q=q, kv=kv, indices=idx, sm_scale=scale, d_v=D)
g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
with torch.cuda.stream(s):
    with torch.cuda.graph(g): out = tilelang_sparse_fwd(q=q, kv=kv, indices=idx, sm_scale=scale, d_v=D)
g.replay(); torch.cuda.synchronize()
e = ((out.reshape(32, H, D).float() - ref(q, kv, idx)).abs().max()).item(); print("graph replay abs_err", e); ok &= e < 5e-2
print("PASS" if ok else "FAIL")
