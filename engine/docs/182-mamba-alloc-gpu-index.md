# 182: reuse one asynchronous CUDA index for static Mamba allocation

Default off: `SGLANG_AX_MAMBA_ALLOC_GPU_INDEX=1`. This candidate starts from `86b82497` and does not change179.

`HybridReqToTokenPool.alloc` previously used the same Python slot list for two CUDA advanced-index assignments. Each conversion can wait on the current CUDA stream. The enabled path constructs one fresh pinned int64 index, uploads it nonblocking, and shares the CUDA index between both original advanced-index assignments. It still returns the original CPU list. The pinned source is immutable; PyTorch's pinned allocator fences reuse after the queued copy. No persistent staging buffer or allocator mirror is introduced.

The gate requires the exact static pool type, ordinary extra buffer (not lazy), matching CUDA devices, PP size1, no speculative algorithm, and no CUDA graph capture. Flag off and unsupported modes keep the original Python indices. Existing scheduler WAR and forward stream waits remain required. This review does not generalize the path to unified/migration, speculation, or PP.

## Development probe

Exact proposed `Hybrid.alloc` and committed179 `prepare_for_extend` were AST extracted. CUDA tensors and the Mamba slot allocator were real; Req-slot admission, KV-location allocation, and sampling were scaffolds. Synthetic tokens/slot IDs, no model.60 interleaved repetitions per condition. Times are medians in microseconds; completion explicitly calls CUDA synchronize after the full preparation boundary. Queued work is a simulated CUDA sleep of about7ms, not measured model execution.

| Batch | Idle CPU enqueue | Idle GPU complete | Queued CPU enqueue | Queued GPU complete |
| --- | ---: | ---: | ---: | ---: |
| 1 | 414.3 → 394.2 | 436.1 → 416.4 | 7408.9 → 7256.2 | 7432.6 → 7280.0 |
| 2 | 419.4 → 408.0 | 441.7 → 431.7 | 7418.3 → 7296.4 | 7442.2 → 7320.8 |
| 8 | 555.9 → 533.0 | 580.5 → 557.5 | 7535.8 → 579.4 | 7561.7 → 7192.3 |
| 12 | 601.3 → 582.6 | 626.8 → 607.1 | 7541.6 → 606.8 | 7567.7 → 7194.7 |
| 32 | 849.7 → 823.5 | 874.5 → 848.6 | 7705.7 → 850.9 | 7731.3 → 7196.1 |

B1/B2 retain179's scalar track lookup and wait on it under queued work. Larger batches can submit preparation without waiting for simulated GPU work; GPU completion still includes that work. This is a preparation boundary measurement, not TPOT, chain, or capacity evidence.

Raw receipts and reproducible scripts: `evidence/execution-0930/tokenspeed-host-review/pool_index_candidate_exact_probe.{py,json}` and `pool_index_candidate_lifecycle_probe.{py,json}`. The lifecycle probe compares mask false/true, helper position swaps, continuing chunks, donation, free/reallocation, flag/context fallbacks, and fenced two-stream mapping generations. Independent source and receipt review passed; service enablement still requires TP8 verification.

A separate probe of direct stacking of pre-helper frozen per-Req CUDA track slots gave smaller B1/B2 preparation times. It is an AST hypothesis only and is not part of this patch.
