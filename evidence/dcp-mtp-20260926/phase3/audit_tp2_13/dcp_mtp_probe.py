#!/usr/bin/env python3
"""Real NEXTN/EAGLE V2 eager/graph diagnostic on the two-A100 developer box.

Run separately with DCP=1 and DCP=2 against the same source/model. Well-scaled
dummy initialization and per-layer DSA signals follow dcp_check.py. This runner
intentionally synchronizes observations; its timings are not performance data.
By default proposals are natural. --accept-oracle injects controlled proposals
from a natural reference; target verification and state commit stay untouched.
Missing acceptance lengths must not be reported as passed. Output budget: at most 256 sampled
attention observations per role/rank (roughly 64 MiB per scheduler process for
this scaled model).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def install_diagnostics():
    import torch
    import sglang.srt.model_loader.loader as loader
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import FullCudaGraphBackend
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
    from dcp_check import _ax_well_scaled_init

    loader.initialize_dummy_weights = _ax_well_scaled_init(loader.initialize_dummy_weights)
    original_init = ModelRunner.init_cuda_graphs
    trace_root = Path(os.environ["AX_MTP_TRACE_DIR"])
    recording = trace_root / "RECORDING"
    active_capture = [None]
    original_capture = FullCudaGraphBackend.capture_one
    original_replay = FullCudaGraphBackend.replay

    def emit(state, record, *, graph_key=None, replay_id=None):
        if state["count"] >= 256:
            raise RuntimeError("MTP diagnostic output budget exhausted")
        record = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in record.items()}
        record["attn"] = record["attn"].float()
        label = trace_root / "CURRENT_CASE"
        record["case"] = label.read_text() if label.exists() else None
        record.update(graph_key=graph_key, replay_id=replay_id)
        torch.save(record, trace_root / f"{record['role']}{record['rank']}_{state['count']:04d}.pt")
        state["count"] += 1

    def capture(backend, shape_key, *args, **kwargs):
        if not hasattr(backend, "_dcp_observations"):
            backend._dcp_observations = {}
            backend._dcp_replay_id = 0
        observations = backend._dcp_observations[shape_key] = []
        previous = active_capture[0]
        active_capture[0] = observations
        try:
            return original_capture(backend, shape_key, *args, **kwargs)
        finally:
            active_capture[0] = previous

    def replay(backend, shape_key, *args, **kwargs):
        out = original_replay(backend, shape_key, *args, **kwargs)
        if recording.exists():
            records = backend._dcp_observations.get(shape_key, [])
            if not records:
                raise RuntimeError(f"Missing captured DSA observations for {shape_key}")
            backend._dcp_replay_id += 1
            for state, record in records:
                emit(state, record, graph_key=repr(shape_key), replay_id=backend._dcp_replay_id)
        return out

    FullCudaGraphBackend.capture_one = capture
    FullCudaGraphBackend.replay = replay

    def init_graphs(runner, *args, **kwargs):
        role = "draft" if runner.is_draft_worker else "target"
        rank = runner.ps.tp_rank
        state = {"count": 0}
        pool = runner.token_to_kv_pool
        pool = getattr(pool, "full_kv_pool", pool)

        def storages_bytes(tensors):
            storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                        for t in tensors if t.numel()}
            return sum(storages.values())

        indexer = pool.index_key_cache.buffer
        capacity = {"role": role, "rank": rank,
                    "dcp_size": runner.ps.attn_dcp_size,
                    "physical_latent_rows": pool.size,
                    "physical_page_size": pool.page_size,
                    "allocator_logical_size": runner.token_to_kv_pool_allocator.size,
                    "latent_storage_bytes": storages_bytes(pool.kv_buffer),
                    "indexer_storage_bytes": storages_bytes(indexer),
                    "latent_shapes": [list(t.shape) for t in pool.kv_buffer],
                    "indexer_shapes": [list(t.shape) for t in indexer],
                    "attention_parameter_dtypes": {
                        n: str(p.dtype) for n, p in runner.model.named_parameters()
                        if n.endswith(("kv_b_proj.weight", "q_a_layernorm.weight"))}}
        (trace_root.parent / f"capacity_{role}{rank}.json").write_text(json.dumps(capacity, indent=2) + "\n")
        for name, attention in runner.model.named_modules():
            if not isinstance(attention, DeepseekV2AttentionMLA):
                continue
            current = {}

            def enter(module, call_args, call_kwargs, _current=current):
                _current["batch"] = call_kwargs.get("forward_batch")
                if _current["batch"] is None:
                    _current["batch"] = call_args[2]

            def observe(module, call_args, _current=current, _name=name):
                capturing = torch.cuda.is_current_stream_capturing()
                if not capturing and (active_capture[0] is not None or not recording.exists()):
                    return  # Exclude graph warmups and engine startup.
                fb = _current["batch"]
                x = call_args[0].detach()
                # Fixed beginning/end samples preserve meaningful DSA signal.
                rows = list(range(x.shape[0])) if x.shape[0] <= 16 else list(range(8)) + list(range(x.shape[0] - 8, x.shape[0]))
                sample = x if x.shape[0] <= 16 else torch.cat((x[:8], x[-8:]))
                record = {"role": role, "rank": rank, "layer": _name,
                          "mode": fb.forward_mode.name, "shape": tuple(x.shape),
                          "rows": rows, "attn": sample,
                          "seq_lens": fb.seq_lens,
                          "positions": fb.positions}
                if fb.forward_mode.is_draft_extend_v2():
                    spec = fb.spec_info
                    accepted = spec.num_accept_tokens
                    width = spec.num_tokens_per_req
                    front = spec.num_front_tokens
                    assert width > 0 and x.shape[0] == len(accepted) * width
                    # The worker selects accept_lens-1; KPool also limits its
                    # tail updates by num_accept_tokens. Rejected window rows
                    # are computed but are not part of the accepted history.
                    # Capture the real mask, so lengths 2/3/4 are checked too.
                    valid = (torch.arange(width, device=x.device)[None, :]
                             < (accepted[:, None] + front)).reshape(-1)
                    valid_sample = valid if x.shape[0] <= 16 else torch.cat((valid[:8], valid[-8:]))
                    record.update(valid_rows=valid_sample, accepted_tokens=accepted,
                                  draft_window_width=width, num_front_tokens=front)
                if capturing:
                    if active_capture[0] is None:
                        raise RuntimeError("Unobserved graph backend; use full decode graphs")
                    # Every capture-time call gets its own copy, including both
                    # inner draft decode steps. Copies replay with the graph.
                    record = {k: v.clone() if torch.is_tensor(v) else v for k, v in record.items()}
                    active_capture[0].append((state, record))
                else:
                    emit(state, record)

            attention.register_forward_pre_hook(enter, with_kwargs=True)
            attention.o_proj.register_forward_pre_hook(observe)
        return original_init(runner, *args, **kwargs)

    ModelRunner.init_cuda_graphs = init_graphs


# Multiprocessing spawn reimports this module before the scheduler loads models.
if os.environ.get("AX_MTP_TRACE_DIR"):
    install_diagnostics()
    if os.environ.get("AX_MTP_ACCEPT_ORACLE"):
        from dcp_mtp_acceptance import install
        install()


def main():
    import random
    import sglang

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--dcp", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hicache", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--restore", action="store_true", help="HiCache churn, host restore and true flush")
    parser.add_argument("--accept-oracle", type=Path,
                        help="Natural reference responses.json; inject proposals for lengths 1/2/3/4")
    args = parser.parse_args()
    if args.tp < 1 or args.dcp < 1 or args.tp % args.dcp:
        parser.error("--dcp must be a positive divisor of --tp")
    if args.restore and not args.hicache:
        parser.error("--restore requires --hicache")
    if args.restore and args.accept_oracle:
        parser.error("Run host restoration and controlled acceptance as separate diagnostics")
    args.output.mkdir(parents=True, exist_ok=False)
    trace = args.output / "trace"
    trace.mkdir()
    os.environ["AX_MTP_TRACE_DIR"] = str(trace.resolve())
    if args.accept_oracle:
        os.environ["AX_MTP_ACCEPT_ORACLE"] = str(args.accept_oracle.resolve(strict=True))
    # Parent imports above predate AX_MTP_TRACE_DIR; workers import with it set.
    # Keep all churn on the host while exceeding device KV capacity. At TP1
    # this fixture's KDA share leaves a 1 GB host budget too little KV space;
    # 4 GB per rank covers target/draft latent, indexer and KDA checkpoints.
    extra = dict(enable_hierarchical_cache=True, hicache_size=4,
                 hicache_io_backend="kernel", hicache_mem_layout="page_first",
                 hicache_write_policy="write_through") if args.hicache else {}
    engine = None
    try:
        engine = sglang.Engine(
            model_path=args.model, load_format="dummy", tp_size=args.tp, dcp_size=args.dcp,
            skip_tokenizer_init=True, random_seed=1234, kv_cache_dtype="bfloat16",
            dsa_prefill_backend="tilelang", dsa_decode_backend="tilelang",
            linear_attn_backend="triton", mem_fraction_static=0.60,
            max_total_tokens=32768, max_mamba_cache_size=64, max_running_requests=8,
            mamba_radix_cache_strategy="extra_buffer", chunked_prefill_size=8192,
            disable_cuda_graph=not args.graph, cuda_graph_max_bs_decode=2, disable_custom_all_reduce=True,
            speculative_algorithm="NEXTN", speculative_draft_model_path=args.model,
            speculative_num_steps=3, speculative_eagle_topk=1,
            speculative_num_draft_tokens=4, enable_cache_report=True, **extra,
        )
        (trace / "RECORDING").touch()
        rng = random.Random(1234)
        ids = [rng.randrange(10000) for _ in range(4164)]
        results = []

        def generate(label, prompt):
            (trace / "CURRENT_CASE").write_text(label)
            out = engine.generate(input_ids=prompt, sampling_params={
                "temperature": 0.0, "max_new_tokens": 12, "ignore_eos": True,
            }, return_logprob=True)
            results.append({"case": label, "prompt_length": len(prompt), "result": out,
                            "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest()})
            (args.output / "responses.json").write_text(json.dumps(results, indent=2) + "\n")
            return out

        # Prefix ends chosen so W=1 (64) and W=2 (128) save the same checkpoint.
        # Otherwise different cache granularity changes the KDA prefill split
        # and the two traces are not the same numerical operation.
        for label, prompt in (("short_local_empty", ids[:1]),
                              ("cold", ids[:3073]), ("prefix", ids[:3204]),
                              ("longer_prefix", ids)):
            generate(label, prompt)
        if args.accept_oracle:
            from dcp_mtp_acceptance import verdict
            verdict(args.output, results, args.accept_oracle, args.tp)
        restore = None
        if args.restore:
            warm = generate("device_repeat", ids)
            (trace / "RECORDING").unlink()
            churn = []
            # Both arms receive the same churn: twice DCP2's logical capacity.
            for i in range(32):
                prompt = [rng.randrange(10000) for _ in range(4096)]
                out = engine.generate(input_ids=prompt, sampling_params={
                    "temperature": 0.0, "max_new_tokens": 1, "ignore_eos": True})
                churn.append({"case": i, "tokens": len(prompt),
                              "input_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                              "meta_info": out["meta_info"]})
                (args.output / "churn.json").write_text(json.dumps(churn, indent=2) + "\n")
            (trace / "RECORDING").touch()
            restored = generate("host_restore", ids)
            restore = {"warm_cache": warm["meta_info"].get("cached_tokens_details"),
                       "restored_cache": restored["meta_info"].get("cached_tokens_details"),
                       "same_tokens": warm["output_ids"] == restored["output_ids"]}
            (args.output / "host_restore_verdict.json").write_text(json.dumps(restore, indent=2) + "\n")
            if not restore["restored_cache"] or restore["restored_cache"].get("host", 0) <= 0:
                raise RuntimeError(f"No proven host restore: {restore}")
        if not list(trace.glob("target*.pt")) or not list(trace.glob("draft*.pt")):
            raise RuntimeError("Missing target/draft sensitive observations")
        import torch
        from dcp_mtp_compare import compare_record
        phases = {}
        restore_records = {"device_repeat": [], "host_restore": []}
        for path in sorted(trace.glob("*.pt")):
            record = torch.load(path)
            if record.get("case") in restore_records:
                restore_records[record["case"]].append((path.name, record))
            if record.get("graph_key") is not None:
                key = f"{record['role']}:{record['mode']}"
                phases.setdefault(key, set()).add((record["rank"], record["graph_key"], record["replay_id"]))
        if args.graph and not {"target:TARGET_VERIFY", "draft:DECODE", "draft:DRAFT_EXTEND_V2"}.issubset(phases):
            raise RuntimeError(f"Missing MTP graph phases; observed {sorted(phases)}")
        if args.restore:
            warm_records, host_records = (restore_records[k] for k in ("device_repeat", "host_restore"))
            # These two named cases intentionally run the same request via
            # device hits and host restoration. Other metadata must match.
            checks = [{"test_file": a[0], "ref_file": b[0],
                       **compare_record(a[1], b[1], compare_case=False)}
                      for a, b in zip(host_records, warm_records)]
            restore.update(same_trace_count=len(host_records) == len(warm_records), records=checks)
            restore["passed"] = bool(checks) and restore["same_tokens"] and restore["same_trace_count"] and all(c["passed"] for c in checks)
            (args.output / "host_restore_verdict.json").write_text(json.dumps(restore, indent=2) + "\n")
            if not restore["passed"]:
                raise RuntimeError("Host restore sensitive numerical comparison failed")
            (trace / "RECORDING").unlink()
            flush = engine.flush_cache()
            flush = {"success": flush.success, "message": flush.message}
            if not flush["success"]:
                (args.output / "flush.json").write_text(json.dumps(flush, indent=2) + "\n")
                raise RuntimeError(f"Engine did not flush its caches: {flush}")
            cold = engine.generate(input_ids=ids, sampling_params={
                "temperature": 0., "max_new_tokens": 1, "ignore_eos": True})
            flush_receipt = {"flush_result": flush, "after_flush_meta": cold["meta_info"]}
            (args.output / "flush.json").write_text(json.dumps(flush_receipt, indent=2) + "\n")
            if cold["meta_info"].get("cached_tokens") != 0:
                raise RuntimeError("Flush left cached prefix tokens")
        summary = {"validity": "MTP_DIAGNOSTIC", "tp": args.tp, "dcp": args.dcp,
                   "hicache": args.hicache, "graph_phase_rank_replays": {k: len(v) for k, v in phases.items()},
                   "host_restore_and_flush": bool(restore and restore["passed"]),
                   "controlled_proposals": bool(args.accept_oracle),
                   "trace_files": len(list(trace.glob("*.pt"))),
                   "limitations": "Scaled model; only observed acceptance lengths; no TP8 or SLO claim."}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
