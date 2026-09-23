# [ax] T50/116: DCP address-protocol check on SGLang's real model code (random weights, TP2, optional --dcp-size 2).
# Sequence per run (rank0 logits saved to $AX_CHECK_OUT as a dict of fp32 CPU tensors):
#   1. optional filler: allocate AX_FILL_FRAC of the KV allocator first, so the requests land on HIGH virtual locs
#      (under DCP W=2 the virtual space is 2x the per-rank rows -> locs >= local rows; stock stack reads OOB there)
#   2. cold prefill of bs requests (prefix lengths AX_PREFIX, multiples of 512)       -> "cold"
#   3. extend the SAME requests with AX_EXT new tokens on top of the cached prefix     -> "ext"  (prefix-hit path)
#   4. AX_DECODE greedy decode steps                                                  -> "dec"  [steps, bs, vocab]
# Random dummy weights make the logits almost independent of attention (09-23: stock DCP2 matched non-DCP bit-exactly),
# so the SENSITIVE oracle is "attn": the input of each DSA layer's o_proj (this rank's heads, after the DCP LSE combine),
# captured by a forward pre-hook in eager passes (rows subsampled for the cold prefill). AX_ATTN_BOOST (default 1) scales
# the DSA o_proj outputs identically in both arms (09-23: the logits stayed insensitive even so; attn is the oracle).
# CUDA-graph decode: the hook copies the o_proj input into a static buffer during capture and it is read after each
# replay (info.graph_dec_steps > 0 proves the graph ran). Pass --batch-size >= number of requests so decode is captured.
# Compare a DCP run against a non-DCP run of the same stack with dcp_compare.py.
# Usage: python dcp_check.py <one_batch CLI args ...> (--batch-size/--input-len are ignored except for arg parsing)
import os
import numpy as np
import torch
import sglang.benchmark.one_batch as ob


def _ax_well_scaled_init(orig):
    """Stock dummy init (+-1e-3, one seed for all params) collapses the signal (DSA o_proj input ~1e-13, pure
    cancellation noise). After it, re-init: DSA attention 2-D weights ~N(0,1/fan_in), 1-D norms 1, biases 0; every other
    RMSNorm weight 1; embedding ~N(0,1). Seeded by parameter NAME so both arms and replicated params agree.
    Runs before _post_load_weights, so w_kc/w_vc are derived from the new kv_b_proj."""
    import zlib
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

    def init(model, *a, **k):
        orig(model, *a, **k)
        attn = [n + "." for n, m in model.named_modules() if isinstance(m, DeepseekV2AttentionMLA)]
        with torch.no_grad():
            for name, p in model.named_parameters():
                if not torch.is_floating_point(p) or torch.finfo(p.dtype).bits < 16:
                    continue
                g = torch.Generator(device=p.device); g.manual_seed(zlib.crc32(name.encode()))
                leaf = name.rsplit(".", 1)[-1]
                in_attn = any(name.startswith(x) for x in attn)
                if "embed_tokens" in name and p.dim() == 2:
                    p.copy_(torch.randn(p.shape, generator=g, device=p.device, dtype=torch.float32).to(p.dtype))
                elif p.dim() == 1 and "norm" in name and leaf == "weight":
                    p.fill_(1.0)
                elif in_attn and leaf == "bias":
                    p.zero_()
                elif in_attn and p.dim() == 2 and leaf == "weight":
                    w = torch.randn(p.shape, generator=g, device=p.device, dtype=torch.float32) / p.shape[1] ** 0.5
                    p.copy_(w.to(p.dtype))
    return init


def work(server_args, port_args, bench_args, gpu_id, tp_rank):
    ob.publish(server_args, role="scheduler")
    if os.environ.get("AX_WELL_SCALED", "1") == "1":
        import sglang.srt.model_loader.loader as _ld
        _ld.initialize_dummy_weights = _ax_well_scaled_init(_ld.initialize_dummy_weights)
    ob.initialize_moe_config(); ob.initialize_fp8_gemm_config(); ob.initialize_fp4_gemm_config()
    runner, _ = ob.load_model(server_args, port_args, gpu_id, tp_rank)
    mr = runner.torch_runner
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
    cap, stage, static, seen, names = {}, ["load"], {}, set(), []
    boost = float(os.environ.get("AX_ATTN_BOOST", "1"))
    for name, m in mr.model.named_modules():
        if isinstance(m, DeepseekV2AttentionMLA):
            if boost != 1:  # o_proj is quantized (int-packed): scale its OUTPUT (hook is captured into graphs too)
                def post(mod, args, out, _b=boost):
                    return (out[0] * _b, *out[1:]) if isinstance(out, tuple) else out * _b
                m.o_proj.register_forward_hook(post)
            names.append(name)
            def pre(mod, args, _n=name):
                if torch.cuda.is_current_stream_capturing():  # graph: record a copy into a static buffer
                    b = static.get((_n, args[0].shape[0]))
                    if b is None:
                        b = static[(_n, args[0].shape[0])] = torch.empty_like(args[0])
                    b.copy_(args[0])
                    return
                seen.add(_n)
                x = args[0].detach().float()
                if stage[0] == "cold":
                    x = torch.cat([x[::16], x[-64:]])
                cap.setdefault(stage[0], {}).setdefault(_n, []).append(x.cpu())
            m.o_proj.register_forward_pre_hook(pre)
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
    stage[0] = "cold"
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
    stage[0] = "ext"
    nxt, lg, batch = runner.extend(reqs)
    out["ext"] = lg.float().cpu()
    locs = torch.cat([r2t[r.kv.req_pool_idx, : p + e] for r, p, e in zip(reqs, pre, ext)])
    info["max_loc_ext"] = int(locs.max().item())
    dec = []
    stage[0] = "dec"
    info["graph_dec_steps"] = 0
    for _ in range(n_dec):
        seen.clear()
        nxt, lg = runner.decode(nxt, batch)
        dec.append(lg.float().cpu())
        for n in names:  # replayed CUDA graph: no Python hook ran -> read the captured static buffer
            b = static.get((n, len(reqs)))
            if n not in seen and b is not None:
                cap.setdefault("dec", {}).setdefault(n, []).append(b[: len(reqs)].float().cpu())
                info["graph_dec_steps"] += 1
    if dec:
        out["dec"] = torch.stack(dec)
    torch.cuda.synchronize()
    if tp_rank == 0:
        out["info"] = info
        out["attn"] = {st: {n: torch.cat(v) for n, v in d.items()} for st, d in cap.items() if st != "load"}
        torch.save(out, os.environ["AX_CHECK_OUT"])
        print("AX_CHECK done", info, {k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)}, flush=True)


if __name__ == "__main__":  # spawned TP ranks re-import this module; only the parent launches
    ob.latency_test = work
    ob.cli_main()
