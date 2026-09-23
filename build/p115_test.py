# Patch 115 check: tilelang sparse MLA fwd on sm80 with H=8 (TP8 path) and H=64 (DCP path).
# argv: <old_tree_parent> <new_tree_parent>. H=8: new must be bit-identical to old. H=64: new vs torch reference
# (old is expected to fail with "dynamic shared memory 262144"). Also CUDA-graph replay for the new kernel.
import importlib, sys, torch
def load(root):
    for m in [k for k in sys.modules if k.startswith("sglang")]: del sys.modules[m]
    sys.path.insert(0, root); mod = importlib.import_module("sglang.kernels.ops.attention.dsa.tilelang_kernel"); sys.path.pop(0); return mod
torch.manual_seed(0); dev = "cuda"; D, N, TOPK = 512, 100_000, 2112; scale = D ** -0.5
kv = torch.randn(N, 1, D, device=dev, dtype=torch.bfloat16)
def ref(q, idx, H):
    out = torch.empty(q.shape[0], H, D, device=dev)
    for t in range(q.shape[0]):
        ii = idx[t, 0]; v = ii >= 0; k = kv[ii.clamp(min=0), 0].float(); s_ = (q[t].float() @ k.T) * scale; s_[:, ~v] = float("-inf")
        out[t] = torch.softmax(s_, -1) @ k
    return out
old = load(sys.argv[1]).tilelang_sparse_fwd; new = load(sys.argv[2]).tilelang_sparse_fwd
ok = True
for H in (8, 64):
    for T in (1, 16, 512):
        q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
        idx = torch.full((T, 1, TOPK), -1, device=dev, dtype=torch.int32); idx[:, :, :2051] = torch.randint(0, N, (T, 1, 2051), device=dev, dtype=torch.int32)
        o_new = new(q=q, kv=kv, indices=idx, sm_scale=scale, d_v=D).reshape(T, H, D)
        r = ref(q[:16], idx[:16], H); e = ((o_new[:16].float() - r).abs().max() / r.abs().max()).item()
        line = f"H={H} T={T} rel_err_vs_ref={e:.2e}"
        if H == 8:
            o_old = old(q=q, kv=kv, indices=idx, sm_scale=scale, d_v=D).reshape(T, H, D); same = torch.equal(o_old, o_new); line += f" bit_identical_to_old={same}"; ok &= same
        torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
        for _ in range(5): new(q=q, kv=kv, indices=idx, sm_scale=scale, d_v=D)
        b.record(); torch.cuda.synchronize(); line += f" ms={a.elapsed_time(b)/5:.3f}"
        print(line, flush=True); ok &= e < 2e-2
    q = torch.randn(16, H, D, device=dev, dtype=torch.bfloat16); idx = torch.randint(0, N, (16, 1, TOPK), device=dev, dtype=torch.int32)
    new(q=q, kv=kv, indices=idx, sm_scale=scale, d_v=D); g = torch.cuda.CUDAGraph(); st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        with torch.cuda.graph(g): out = new(q=q, kv=kv, indices=idx, sm_scale=scale, d_v=D)
    g.replay(); torch.cuda.synchronize(); e = ((out.reshape(16, H, D).float() - ref(q, idx, H)).abs().max()).item()
    print(f"H={H} graph abs_err={e:.2e}"); ok &= e < 5e-2
try:
    q = torch.randn(4, 64, D, device=dev, dtype=torch.bfloat16); idx = torch.randint(0, N, (4, 1, TOPK), device=dev, dtype=torch.int32)
    old(q=q, kv=kv, indices=idx, sm_scale=scale, d_v=D); print("old H=64: ran (unexpected)")
except Exception as e: print("old H=64 fails as expected:", str(e)[:90])
print("PASS" if ok else "FAIL")
