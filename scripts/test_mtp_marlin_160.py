# T48 extends F58: clip=10, draft/verify batch shapes, dense FP8 Marlin.
# sm80 check for patch 111: block-fp8 MoE experts through Marlin (fp8 e4m3 weights, bf16 activations).
# GLM-5.3 TP8 shapes: hidden k=4096, intermediate/rank n=256, block 128x128, top-8. Also CUDA graph capture.
import sys, types, torch
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "/tmp/ax/src/b111")
from sgl_kernel.scalar_type import scalar_types
from sglang.srt.layers.quantization.marlin_utils_fp8 import prepare_moe_fp8_layer_for_marlin
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
torch.manual_seed(0); dev = "cuda"
E, K, N, TOPK, B = 33, 4096, 256, 8, 128
def qblock(w):  # w [E, n, k] bf16 -> fp8 [E,n,k], scale_inv [E, n/B, k/B]
    e, n, k = w.shape; wb = w.float().view(e, n // B, B, k // B, B)
    s = wb.abs().amax(dim=(2, 4)).clamp(min=1e-6) / 448.0
    q = (wb / s[:, :, None, :, None]).view(e, n, k).to(torch.float8_e4m3fn)
    return q, s
def deq(q, s):
    e, n, k = q.shape
    return (q.float().view(e, n // B, B, k // B, B) * s[:, :, None, :, None]).view(e, n, k)
w13 = torch.randn(E, 2 * N, K, device=dev) * 0.02; w2 = torch.randn(E, K, N, device=dev) * 0.02
q13, s13 = qblock(w13); q2, s2 = qblock(w2); d13, d2 = deq(q13, s13), deq(q2, s2)
layer = types.SimpleNamespace(num_experts=E, hidden_size=K, intermediate_size_per_partition=N,
                              weight_block_size=[B, B], orig_dtype=torch.bfloat16)
layer.w13_weight = torch.nn.Parameter(q13, requires_grad=False); layer.w2_weight = torch.nn.Parameter(q2, requires_grad=False)
layer.w13_weight_scale_inv = torch.nn.Parameter(s13, requires_grad=False); layer.w2_weight_scale_inv = torch.nn.Parameter(s2, requires_grad=False)
class L(torch.nn.Module): pass
m = L(); [setattr(m, k, v) for k, v in vars(layer).items()]
prepare_moe_fp8_layer_for_marlin(m, size_k_first=False)
def ref(x, tw, ti):
    out = torch.zeros(x.shape[0], K, device=dev)
    for t in range(x.shape[0]):
        for j in range(TOPK):
            e = ti[t, j]; h = d13[e] @ x[t].float(); a = torch.nn.functional.silu(h[:N].clamp(max=10.)) * h[N:].clamp(-10.,10.)
            out[t] += tw[t, j] * (d2[e] @ a)
    return out
def run(x, tw, ti):
    return fused_marlin_moe(hidden_states=x, w1=m.w13_weight, w2=m.w2_weight, w1_scale=m.w13_weight_scale,
        w2_scale=m.w2_weight_scale, gating_output=torch.empty(x.shape[0], E, device=dev), topk_weights=tw,
        topk_ids=ti, num_bits=8, fp8_weights=True, clamp_limit=10.)
ok = True
for M in [1, 6, 24, 32]:
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 10
    ti = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).int()
    tw = torch.softmax(torch.randn(M, TOPK, device=dev), -1)
    o = run(x, tw, ti).float(); r = ref(x[:32], tw[:32], ti[:32])
    e = ((o[:32] - r).norm() / r.norm()).item()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
    for _ in range(10): run(x, tw, ti)
    b.record(); torch.cuda.synchronize()
    print(f"M={M} rel_err={e:.2e} time={a.elapsed_time(b)/10:.3f}ms"); ok &= e < 2e-2
x = torch.randn(16, K, device=dev, dtype=torch.bfloat16) * 10; ti = torch.randint(0, E, (16, TOPK), device=dev).int()
tw = torch.softmax(torch.randn(16, TOPK, device=dev), -1); run(x, tw, ti)
g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
with torch.cuda.stream(s):
    with torch.cuda.graph(g): out = run(x, tw, ti)
g.replay(); torch.cuda.synchronize(); r = ref(x, tw, ti)
e = ((out.float() - r).norm() / r.norm()).item(); print(f"graph replay rel_err={e:.2e}"); ok &= e < 2e-2
assert ok, "MoE numeric failure"
print("MOE_CLIP10_PASS")

# Ordinary FP8 projections use the native dense Marlin path on sm80.
from sglang.srt.layers.quantization.marlin_utils_fp8 import prepare_fp8_layer_for_marlin, apply_fp8_marlin_linear
for NN, KK in [(1536,4096),(512,4096),(4096,512)]:
    w=torch.randn(1,NN,KK,device=dev)*.02
    qw,sw=qblock(w); dw=deq(qw,sw)[0]
    lin=L(); lin.input_size_per_partition=KK;lin.output_size_per_partition=NN
    lin.weight_block_size=[128,128];lin.orig_dtype=torch.bfloat16
    lin.weight=torch.nn.Parameter(qw[0],requires_grad=False)
    lin.weight_scale_inv=torch.nn.Parameter(sw[0],requires_grad=False)
    prepare_fp8_layer_for_marlin(lin,size_k_first=False)
    for MM in (1,6,24):
        xx=torch.randn(MM,KK,device=dev,dtype=torch.bfloat16)
        def dense():return apply_fp8_marlin_linear(xx,lin.weight,lin.weight_scale,lin.workspace,NN,KK,None)
        yy=dense(); rr=xx.float()@dw.T
        err=float((yy.float()-rr).norm()/rr.norm());assert err<.02,err
        for _ in range(3):dense()
        gg=torch.cuda.CUDAGraph()
        with torch.cuda.graph(gg):oo=dense()
        for _ in range(3):
            xx.copy_(torch.randn_like(xx));gg.replay()
            rr=xx.float()@dw.T; ge=float((oo.float()-rr).norm()/rr.norm());assert ge<.02,ge
        print(f"DENSE M={MM} N={NN} K={KK} rel_err={err:.6g} graph_rel_err={ge:.6g} PASS")
print("MARLIN_ALL_PASS")
