# INT8 W8A8 MoE feasibility on A100 (R8 direction A1): speed + accuracy vs the served Marlin W8A16 path (patch 111).
# GLM-5.3-Flash TP8 per-rank MoE shapes: E=288 experts, hidden K=4096, intermediate/rank N=256, top-8, FP8 blocks 128x128.
# Paths (all from the patched sglang tree on PYTHONPATH):
#   marlin_fp8_w8a16  : fused_marlin_moe (fp8 e4m3 weights, bf16 activations)      <- what we serve now
#   triton_int8_block : fused_experts_impl use_int8_w8a8, block 128x128 weights, per-token-group(128) activations
#   triton_int8_chan  : fused_experts_impl use_int8_w8a8, per-channel weights, per-token activations
#   triton_bf16       : fused_experts_impl unquantized bf16 weights (speed reference only)
# Accuracy: relative L2 error of the MoE output vs an fp32 reference computed from the dequantized FP8 weights
# (the model's "true" weights). Prints one JSON line per (path, M) and a final table.
import json, sys, time, types
import torch

torch.manual_seed(0)
dev = "cuda"
E, K, N, TOPK, B = 288, 4096, 256, 8, 128
MS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else "64,512,2048,8192,16384".split(","))]

from sgl_kernel.scalar_type import scalar_types  # noqa: F401  (import check)
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts_impl
from sglang.srt.layers.quantization.marlin_utils_fp8 import prepare_moe_fp8_layer_for_marlin

# The Triton MoE path reads process config that a running server publishes; publish defaults the way
# upstream's test/registered/quant/test_block_int8.py does.
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

# Single-GPU TP group (world_size=1) so MoE code paths that query TP state work offline.
import os as _os
from sglang.srt.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
_port = 29500 + (_os.getpid() % 1000)
init_distributed_environment(world_size=1, rank=0, local_rank=0,
                             distributed_init_method=f"tcp://127.0.0.1:{_port}", backend="nccl")
initialize_model_parallel(tensor_model_parallel_size=1)


def qblock_fp8(w):  # [E, n, k] -> fp8 [E,n,k], scale_inv [E, n/B, k/B]
    e, n, k = w.shape
    wb = w.float().view(e, n // B, B, k // B, B)
    s = wb.abs().amax(dim=(2, 4)).clamp(min=1e-8) / 448.0
    return (wb / s[:, :, None, :, None]).view(e, n, k).to(torch.float8_e4m3fn), s


def deq_block(q, s):
    e, n, k = q.shape
    return (q.float().view(e, n // B, B, k // B, B) * s[:, :, None, :, None]).view(e, n, k)


def int8_block(wf):  # fp32 weights [E,n,k] -> int8 + per-block scale [E, n/B, k/B]
    e, n, k = wf.shape
    wb = wf.view(e, n // B, B, k // B, B)
    s = wb.abs().amax(dim=(2, 4)).clamp(min=1e-8) / 127.0
    return torch.round(wb / s[:, :, None, :, None]).clamp(-127, 127).view(e, n, k).to(torch.int8), s.float()


def int8_chan(wf):  # per output channel: scale [E, n, 1]
    s = wf.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 127.0
    return torch.round(wf / s).clamp(-127, 127).to(torch.int8), s.float()


# Weights with a realistic heavy-tailed spread inside blocks (random normal * per-row magnitude jitter).
w13 = torch.randn(E, 2 * N, K, device=dev) * 0.02 * (1 + 3 * torch.rand(E, 2 * N, 1, device=dev))
w2 = torch.randn(E, K, N, device=dev) * 0.02 * (1 + 3 * torch.rand(E, K, 1, device=dev))
q13, s13 = qblock_fp8(w13); q2, s2 = qblock_fp8(w2)
d13, d2 = deq_block(q13, s13), deq_block(q2, s2)          # the model's true (fp8-dequantized) weights
del w13, w2

# Marlin layer (patch 111 path)
layer = torch.nn.Module()
for k_, v_ in dict(num_experts=E, hidden_size=K, intermediate_size_per_partition=N,
                   weight_block_size=[B, B], orig_dtype=torch.bfloat16).items():
    setattr(layer, k_, v_)
layer.w13_weight = torch.nn.Parameter(q13.clone(), requires_grad=False)
layer.w2_weight = torch.nn.Parameter(q2.clone(), requires_grad=False)
layer.w13_weight_scale_inv = torch.nn.Parameter(s13.clone(), requires_grad=False)
layer.w2_weight_scale_inv = torch.nn.Parameter(s2.clone(), requires_grad=False)
prepare_moe_fp8_layer_for_marlin(layer, size_k_first=False)

i13b, is13b = int8_block(d13); i2b, is2b = int8_block(d2)
i13c, is13c = int8_chan(d13); i2c, is2c = int8_chan(d2)
b13, b2w = d13.to(torch.bfloat16), d2.to(torch.bfloat16)


def ref(x, tw, ti, rows=64):
    x = x[:rows].float(); out = torch.zeros(x.shape[0], K, device=dev)
    for t in range(x.shape[0]):
        for j in range(TOPK):
            e = ti[t, j]; h = d13[e] @ x[t]; a = torch.nn.functional.silu(h[:N]) * h[N:]
            out[t] += tw[t, j] * (d2[e] @ a)
    return out


def run(path, x, tw, ti):
    if path == "marlin_fp8_w8a16":
        return fused_marlin_moe(hidden_states=x, w1=layer.w13_weight, w2=layer.w2_weight, w1_scale=layer.w13_weight_scale,
                                w2_scale=layer.w2_weight_scale, gating_output=torch.empty(x.shape[0], E, device=dev),
                                topk_weights=tw, topk_ids=ti, num_bits=8, fp8_weights=True)
    common = dict(topk_weights=tw, topk_ids=ti, inplace=False, gate_up_interleaved=False)
    if path == "triton_int8_block":
        return fused_experts_impl(x, i13b, i2b, use_int8_w8a8=True, w1_scale=is13b, w2_scale=is2b, block_shape=[B, B], **common)
    if path == "triton_int8_chan":
        return fused_experts_impl(x, i13c, i2c, use_int8_w8a8=True, per_channel_quant=True, w1_scale=is13c, w2_scale=is2c, **common)
    if path == "triton_bf16":
        return fused_experts_impl(x, b13, b2w, **common)


def timeit(f, n=10):
    f(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
    for _ in range(n): f()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / n


rows = []
for M in MS:
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    ti = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).int()
    tw = torch.softmax(torch.randn(M, TOPK, device=dev), -1)
    r = ref(x, tw, ti)
    for path in ("marlin_fp8_w8a16", "triton_int8_block", "triton_int8_chan", "triton_bf16"):
        try:
            o = run(path, x, tw, ti).float()[: r.shape[0]]
            err = ((o - r).norm() / r.norm()).item()
            ms = timeit(lambda: run(path, x, tw, ti))
            flops = 2 * M * TOPK * (K * 2 * N + N * K)
            row = dict(path=path, M=M, ms=round(ms, 3), tflops=round(flops / ms / 1e9, 1), rel_err=float(f"{err:.3e}"))
        except Exception as e:  # keep going; a failing path is itself a finding
            row = dict(path=path, M=M, error=f"{type(e).__name__}: {str(e)[:160]}")
        rows.append(row); print(json.dumps(row), flush=True)

print("\n| M | " + " | ".join(p for p in ("marlin_fp8_w8a16", "triton_int8_block", "triton_int8_chan", "triton_bf16")) + " |")
for M in MS:
    cells = []
    for p in ("marlin_fp8_w8a16", "triton_int8_block", "triton_int8_chan", "triton_bf16"):
        r = next(x for x in rows if x["path"] == p and x["M"] == M)
        cells.append("ERR" if "error" in r else f"{r['ms']}ms/{r['tflops']}TF/err{r['rel_err']:.1e}")
    print(f"| {M} | " + " | ".join(cells) + " |")
bad = [r for r in rows if "error" in r and r["path"] != "triton_bf16"]
print("FAIL" if any(r["path"] == "marlin_fp8_w8a16" for r in bad) else "DONE")
