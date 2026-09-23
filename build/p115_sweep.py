# Find a tilelang v1 sparse-attn config that compiles on sm80 for 64 heads/rank (DCP) and is fastest.
import importlib, sys, torch
sys.path.insert(0, sys.argv[1]); tk = importlib.import_module("sglang.kernels.ops.attention.dsa.tilelang_kernel")
torch.manual_seed(0); dev = "cuda"; D, N, TOPK, H = 512, 100_000, 2112, 64; scale = D ** -0.5
kv = torch.randn(N, 1, D, device=dev, dtype=torch.bfloat16)
def ref(q, idx):
    out = torch.empty(q.shape[0], H, D, device=dev)
    for t in range(q.shape[0]):
        ii = idx[t, 0]; v = ii >= 0; k = kv[ii.clamp(min=0), 0].float(); s_ = (q[t].float() @ k.T) * scale; s_[:, ~v] = float("-inf"); out[t] = torch.softmax(s_, -1) @ k
    return out
for HB, BI, ST, TH in [(64, 32, 2, 256), (64, 64, 1, 256), (32, 64, 1, 256), (32, 32, 2, 128), (64, 32, 1, 256), (32, 64, 2, 128), (16, 64, 2, 128)]:
    try:
        ker = tk.sparse_attention_fwd_kernel_v1(H, D, 0, TOPK, sm_scale=scale, return_lse=False, max_heads_per_block=HB, block_I=BI, num_stages=ST, threads=TH)
        res = []
        for T in (16, 512):
            q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16); idx = torch.full((T, 1, TOPK), -1, device=dev, dtype=torch.int32)
            idx[:, :, :2051] = torch.randint(0, N, (T, 1, 2051), device=dev, dtype=torch.int32)
            lse = torch.empty((1, T, H), dtype=torch.float32, device=dev)
            o = ker(q.unsqueeze(0), kv.unsqueeze(0), idx.unsqueeze(0), lse).reshape(T, H, D); r = ref(q[:8], idx[:8])
            e = ((o[:8].float() - r).abs().max() / r.abs().max()).item()
            torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
            for _ in range(5): ker(q.unsqueeze(0), kv.unsqueeze(0), idx.unsqueeze(0), lse)
            b.record(); torch.cuda.synchronize(); res.append(f"T={T} err={e:.1e} ms={a.elapsed_time(b)/5:.3f}")
        print(f"OK   HB={HB} BI={BI} stages={ST} threads={TH}: " + "  ".join(res), flush=True)
    except Exception as ex:
        print(f"FAIL HB={HB} BI={BI} stages={ST} threads={TH}: {type(ex).__name__}: {str(ex).splitlines()[0][:100]}", flush=True)
