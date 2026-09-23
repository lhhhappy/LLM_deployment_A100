# [ax] T50/116: DCP address-protocol check on SGLang's real model code (random weights, TP2, optional --dcp-size 2).
# Sequence per run (rank0 logits saved to $AX_CHECK_OUT as a dict of fp32 CPU tensors):
#   1. optional filler: allocate AX_FILL_FRAC of the KV allocator first, so the requests land on HIGH virtual locs
#      (under DCP W=2 the virtual space is 2x the per-rank rows -> locs >= local rows; stock stack reads OOB there)
#   2. cold prefill of bs requests (prefix lengths AX_PREFIX, multiples of 512)       -> "cold"
#   3. extend the SAME requests with AX_EXT new tokens on top of the cached prefix     -> "ext"  (prefix-hit path)
#   4. AX_DECODE greedy decode steps                                                  -> "dec"  [steps, bs, vocab]
# Compare a DCP run against a non-DCP run of the same stack with dcp_compare.py.
# Usage: python dcp_check.py <one_batch CLI args ...> (--batch-size/--input-len are ignored except for arg parsing)
import os
import numpy as np
import torch
import sglang.benchmark.one_batch as ob


def work(server_args, port_args, bench_args, gpu_id, tp_rank):
    ob.publish(server_args, role="scheduler")
    ob.initialize_moe_config(); ob.initialize_fp8_gemm_config(); ob.initialize_fp4_gemm_config()
    runner, _ = ob.load_model(server_args, port_args, gpu_id, tp_rank)
    mr = runner.torch_runner
    alloc = mr.token_to_kv_pool_allocator
    pre = [int(x) for x in os.environ.get("AX_PREFIX", "8192,6144").split(",")]
    ext = [int(x) for x in os.environ.get("AX_EXT", "1000,700").split(",")]
    n_dec = int(os.environ.get("AX_DECODE", "4"))
    frac = float(os.environ.get("AX_FILL_FRAC", "0"))
    info = {"alloc_size": int(alloc.size), "alloc_page": int(alloc.page_size),
            "pool_rows": int(mr.token_to_kv_pool.size)}
    if frac > 0:
        n = int(alloc.size * frac) // alloc.page_size * alloc.page_size
        filler = alloc.alloc(n)
        assert filler is not None, "filler alloc failed"
        info["filler"] = n
    np.random.seed(1234)
    ids = [np.random.randint(0, 10000, (p + e,), dtype=np.int32) for p, e in zip(pre, ext)]
    reqs = ob.prepare_synthetic_inputs_for_latency_test(len(pre), 0, custom_inputs=[x[:p] for x, p in zip(ids, pre)])
    out = {}
    nxt, lg, batch = runner.extend(reqs)
    out["cold"] = lg.float().cpu()
    r2t = mr.req_to_token_pool.req_to_token
    locs = torch.cat([r2t[r.kv.req_pool_idx, :p] for r, p in zip(reqs, pre)])
    info["max_loc_cold"] = int(locs.max().item())
    for r, x, p in zip(reqs, ids, pre):  # prefix hit: reuse the prefix KV, extend with the new tokens
        r.full_untruncated_fill_ids.extend(x[p:].tolist())
        r.prefix_indices = r2t[r.kv.req_pool_idx, :p].to(r.prefix_indices.dtype)
        r.logprob_start_len = -1
        r.set_extend_range(len(r.prefix_indices), len(r.full_untruncated_fill_ids))
    nxt, lg, batch = runner.extend(reqs)
    out["ext"] = lg.float().cpu()
    locs = torch.cat([r2t[r.kv.req_pool_idx, : p + e] for r, p, e in zip(reqs, pre, ext)])
    info["max_loc_ext"] = int(locs.max().item())
    dec = []
    for _ in range(n_dec):
        nxt, lg = runner.decode(nxt, batch)
        dec.append(lg.float().cpu())
    if dec:
        out["dec"] = torch.stack(dec)
    torch.cuda.synchronize()
    if tp_rank == 0:
        out["info"] = info
        torch.save(out, os.environ["AX_CHECK_OUT"])
        print("AX_CHECK done", info, {k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)}, flush=True)


if __name__ == "__main__":  # spawned TP ranks re-import this module; only the parent launches
    ob.latency_test = work
    ob.cli_main()
