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
import json
import time
from dataclasses import replace
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
    # hooks must exist BEFORE the decode CUDA graphs are captured (init_cuda_graphs runs inside load_model)
    from sglang.srt.model_executor.model_runner import ModelRunner
    _orig_icg = ModelRunner.init_cuda_graphs
    def _icg(self, *a, **k):
        install(self.model)
        return _orig_icg(self, *a, **k)
    ModelRunner.init_cuda_graphs = _icg
    cap, stage, static, seen, names = {}, ["load"], {}, set(), []
    routes = []
    input_records = []
    indexer_records = []
    if os.environ.get("AX_CAPTURE_INDEXER") == "1":
        from dcp_indexer_capture import install as install_indexer_capture
        install_indexer_capture(stage, indexer_records)
    if os.environ.get("AX_CAPTURE_INPUTS") == "1":
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
        original_extend = DeepseekSparseAttnBackend.forward_extend
        input_oracle = {}
        if os.environ.get("AX_TOPK_ROW_ORACLE"):
            oracle_file = os.environ["AX_TOPK_ROW_ORACLE"] + (f".rank{tp_rank}" if tp_rank else "") + ".inputs"
            input_oracle = {r["layer"]: r for r in torch.load(oracle_file)}
        def inspected_extend(self, q, k, v, layer, forward_batch, *args, **kwargs):
            if stage[0] == "ext" and q.shape[0] > 4000:
                row = 3950
                record = {"layer": layer.layer_id, "row": row, "q": q[row:row+1].float().cpu()}
                for name, value in (("k", k), ("v", v), ("topk", kwargs.get("topk_indices"))):
                    if value is not None:
                        record[name] = value[row:row+1].cpu().clone()
                if layer.layer_id in input_oracle:
                    topk = kwargs["topk_indices"].clone()
                    wanted = input_oracle[layer.layer_id]["topk"]
                    topk[row:row+1].copy_(wanted)
                    kwargs["topk_indices"] = topk
                    record["replayed_topk"] = wanted
                input_records.append(record)
            return original_extend(self, q, k, v, layer, forward_batch, *args, **kwargs)
        DeepseekSparseAttnBackend.forward_extend = inspected_extend
    def install(model):
        from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
        if names:
            return
        boost = float(os.environ.get("AX_ATTN_BOOST", "1"))
        for name, m in model.named_modules():
            if isinstance(m, DeepseekV2AttentionMLA):
                if boost != 1:  # o_proj is quantized (int-packed): scale its OUTPUT (hook is captured into graphs too)
                    def post(mod, args, out, _b=boost):
                        return (out[0] * _b, *out[1:]) if isinstance(out, tuple) else out * _b
                    m.o_proj.register_forward_hook(post)
                names.append(name)
                original_core = m.forward_absorb_core
                def core(*args, _core=original_core, _name=name, **kwargs):
                    fb = args[4]
                    if stage[0] in ("cold", "ext"):
                        md = fb.attn_dcp_metadata
                        routes.append({"stage": stage[0], "layer": _name,
                                       "local": bool(md and md.dcp_local_extend),
                                       "q_shape": list(args[2].shape),
                                       "prefix": sum(fb.extend_prefix_lens_cpu),
                                       "buffer_bytes": (md.dcp_kv_buffer.nbytes if md and md.dcp_kv_buffer is not None else 0)})
                    return _core(*args, **kwargs)
                m.forward_absorb_core = core
                def pre(mod, args, _n=name):
                    if torch.cuda.is_current_stream_capturing():  # graph: record a copy into a static buffer
                        b = static.get((_n, args[0].shape[0]))
                        if b is None:
                            b = static[(_n, args[0].shape[0])] = torch.empty_like(args[0])
                        b.copy_(args[0])
                        return
                    if stage[0] == "timing":
                        return
                    seen.add(_n)
                    x = args[0].detach().float()
                    if stage[0] == "cold":
                        x = torch.cat([x[::16], x[-64:]])
                    elif stage[0] == "ext" and os.environ.get("AX_ATTN_CAPTURE_ROW"):
                        row = int(os.environ["AX_ATTN_CAPTURE_ROW"])
                        if not 0 <= row < x.shape[0]:
                            raise ValueError("attention capture row outside extend batch")
                        x = x[row:row+1]
                    cap.setdefault(stage[0], {}).setdefault(_n, []).append(x.cpu())
                m.o_proj.register_forward_pre_hook(pre)

    runner, _ = ob.load_model(server_args, port_args, gpu_id, tp_rank)
    mr = runner.torch_runner
    captured = {}
    original_forward = mr.forward
    def forward(fb, *args, **kwargs):
        if stage[0] == "ext":
            if os.environ.get("AX_MIXED") == "1":
                from sglang.srt.model_executor.forward_batch_info import ForwardMode
                fb.forward_mode = ForwardMode.MIXED
            captured["extend"] = replace(fb, **{k: v.clone() for k, v in vars(fb).items() if torch.is_tensor(v)})
        return original_forward(fb, *args, **kwargs)
    mr.forward = forward
    alloc = mr.token_to_kv_pool_allocator
    pre = [int(x) for x in os.environ.get("AX_PREFIX", "8192,6144").split(",")]
    ext = [int(x) for x in os.environ.get("AX_EXT", "1000,700").split(",")]
    n_dec = int(os.environ.get("AX_DECODE", "4"))
    frac = float(os.environ.get("AX_FILL_FRAC", "0"))
    from sglang.srt.layers.quantization.fp8_humming_moe import humming_moe_layer_count
    info = {"alloc_size": int(alloc.size), "alloc_page": int(alloc.page_size),
            "pool_rows": int(mr.token_to_kv_pool.size),
            "prefix_lengths": pre, "extend_lengths": ext,
            "attention_capture_row": (int(os.environ["AX_ATTN_CAPTURE_ROW"])
                                      if os.environ.get("AX_ATTN_CAPTURE_ROW") else None),
            "mixed_input": os.environ.get("AX_MIXED") == "1",
            "heads_total": mr.model_config.num_attention_heads,
            "humming_layers": humming_moe_layer_count()}
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
    timing = None
    repeats = int(os.environ.get("AX_TIMING_REPEATS", "0"))
    if repeats:
        # Numerical observations are already frozen. Repeated execution uses
        # the same shape/addresses; mutated KDA state is NOT a numerical oracle.
        # Disable every diagnostic tensor copy in the measured forwards.
        stage[0] = "timing"
        fb = captured["extend"]
        with torch.inference_mode():
            for _ in range(5):
                original_forward(replace(fb))
            torch.cuda.synchronize()
            wall_ms, device_ms = [], []
            for _ in range(repeats):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start = time.perf_counter()
                begin.record()
                original_forward(replace(fb))
                end.record(); end.synchronize()
                wall_ms.append((time.perf_counter() - start) * 1000)
                device_ms.append(begin.elapsed_time(end))
        timing = {"wall_ms": wall_ms, "device_ms": device_ms,
                  "scope": "ModelRunner.forward: metadata + all layers + logits; excludes scheduling/tokenization"}
    if os.environ.get("AX_PAIRED_TIMING") == "1":
        # Use one process to separate routing cost from drift across launches.
        # Policy selection and the full runner still execute in the timed region.
        stage[0] = "timing"
        fb = captured["extend"]
        policy = mr.eager_runner.dcp_local_extend_policy
        assert policy is not None
        arms = (("gather_kv", None), ("local_kv", policy))
        paired = {name: {"wall_ms": [], "device_ms": [], "peak_temporary_bytes": 0}
                  for name, _ in arms}
        with torch.inference_mode():
            for _, option in arms:
                mr.eager_runner.dcp_local_extend_policy = option
                for _ in range(5):
                    original_forward(replace(fb))
            for iteration in range(40):
                for name, option in (arms if iteration % 2 == 0 else arms[::-1]):
                    mr.eager_runner.dcp_local_extend_policy = option
                    torch.cuda.synchronize()
                    base_memory = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start = time.perf_counter()
                    begin.record()
                    original_forward(replace(fb))
                    end.record(); end.synchronize()
                    result = paired[name]
                    result["wall_ms"].append((time.perf_counter() - start) * 1000)
                    result["device_ms"].append(begin.elapsed_time(end))
                    result["peak_temporary_bytes"] = max(result["peak_temporary_bytes"],
                        torch.cuda.max_memory_allocated() - base_memory)
        mr.eager_runner.dcp_local_extend_policy = policy
        timing = {"paired": paired, "scope": "Interleaved full ModelRunner.forward; mutated state only for timing, not numerics"}
    if os.environ.get("AX_COMPONENT_BENCH"):
        stage[0] = "timing"
        from dcp_local_extend_bench import run
        run(os.environ["AX_COMPONENT_BENCH"] + f".rank{tp_rank}.json")
    info["routes"] = routes
    from sglang.srt.layers.dcp.local_extend import local_extend_mechanism_tokens
    info["mechanisms"] = local_extend_mechanism_tokens(mr)
    stats = mr.eager_runner.dcp_local_extend_stats
    info["route_counts"] = stats.snapshot() if stats is not None else None
    if os.environ.get("SGLANG_AX_DCP_LOCAL_EXTEND", "0") == "1" and not any(r["local"] for r in routes):
        raise RuntimeError("No local extend executed; this arm cannot validate the mechanism")
    if tp_rank >= 0:
        out["info"] = info
        out["attn"] = {st: {n: torch.cat(v) for n, v in d.items()} for st, d in cap.items() if st != "load"}
        dest = os.environ["AX_CHECK_OUT"] + (f".rank{tp_rank}" if tp_rank else "")
        torch.save(out, dest)
        if input_records:
            torch.save(input_records, dest + ".inputs")
        if os.environ.get("AX_CAPTURE_INDEXER") == "1":
            from dcp_indexer_capture import cpu_records
            if not indexer_records:
                raise RuntimeError("No pooled indexer row captured; diagnostic is incomplete")
            torch.save(cpu_records(indexer_records), dest + ".indexer")
        with open(dest + ".json", "w") as handle:
            json.dump({"rank": tp_rank, "info": info, "timing": timing}, handle, indent=2)
        print("AX_CHECK done", info, {k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)}, flush=True)


if __name__ == "__main__":  # spawned TP ranks re-import this module; only the parent launches
    ob.latency_test = work
    ob.cli_main()
    # one_batch can return success when a spawned TP worker crashes.
    # No output means this diagnostic failed, never a passing numerical run.
    if not os.path.isfile(os.environ["AX_CHECK_OUT"]):
        raise SystemExit("DCP check failed: TP workers produced no result")
