#!/usr/bin/env python3
"""Real NEXTN/EAGLE V2 eager/graph diagnostic on the two-A100 developer box.

Run separately with DCP=1 and DCP=2 against the same source/model. Well-scaled
dummy initialization and per-layer DSA signals follow dcp_check.py. This runner
intentionally synchronizes observations; its timings are not performance data.
No proposal/acceptance decisions are forced. Missing acceptance lengths remain
uncovered and must not be reported as passed. Output budget: at most 256 sampled
attention observations per process (roughly 64 MiB for this scaled model).
"""
from __future__ import annotations

import argparse
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


def main():
    import random
    import sglang

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dcp", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hicache", action="store_true")
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    trace = args.output / "trace"
    trace.mkdir()
    os.environ["AX_MTP_TRACE_DIR"] = str(trace.resolve())
    # Parent imports above predate AX_MTP_TRACE_DIR; workers import with it set.
    extra = dict(enable_hierarchical_cache=True, hicache_size=1,
                 hicache_io_backend="kernel", hicache_mem_layout="page_first",
                 hicache_write_policy="write_through") if args.hicache else {}
    engine = None
    try:
        engine = sglang.Engine(
            model_path=args.model, load_format="dummy", tp_size=2, dcp_size=args.dcp,
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
        for label, prompt in (("short_local_empty", ids[:1]),
                              ("cold", ids[:3073]), ("prefix", ids[:3140]),
                              ("longer_prefix", ids)):
            out = engine.generate(input_ids=prompt, sampling_params={
                "temperature": 0.0, "max_new_tokens": 12, "ignore_eos": True,
            }, return_logprob=True)
            results.append({"case": label, "prompt_length": len(prompt), "result": out})
            (args.output / "responses.json").write_text(json.dumps(results, indent=2) + "\n")
        if not list(trace.glob("target*.pt")) or not list(trace.glob("draft*.pt")):
            raise RuntimeError("Missing target/draft sensitive observations")
        import torch
        phases = {}
        for path in sorted(trace.glob("*.pt")):
            record = torch.load(path)
            if record.get("graph_key") is not None:
                key = f"{record['role']}:{record['mode']}"
                phases.setdefault(key, set()).add((record["rank"], record["graph_key"], record["replay_id"]))
        if args.graph and not {"target:TARGET_VERIFY", "draft:DECODE", "draft:DRAFT_EXTEND_V2"}.issubset(phases):
            raise RuntimeError(f"Missing MTP graph phases; observed {sorted(phases)}")
        summary = {"validity": "MTP_DIAGNOSTIC", "dcp": args.dcp,
                   "hicache": args.hicache, "graph_phase_rank_replays": {k: len(v) for k, v in phases.items()},
                   "trace_files": len(list(trace.glob("*.pt"))),
                   "limitations": "Scaled model; natural acceptance only; host eviction/restore not proven by this model probe."}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
