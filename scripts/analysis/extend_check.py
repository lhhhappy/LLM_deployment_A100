# Exactness + timing check for engine-level patches using SGLang's real model code (random weights, any TP size).
# Replaces one_batch.latency_test with: load model -> seeded synthetic prompt -> extend (prefill) -> save rank0
# next-token logits to $AX_CHECK_OUT (.pt) and print median extend time. Run twice with a feature flag off/on and
# compare the two .pt files (bit-exact expected for pure work-partitioning patches such as 114).
# Usage: python extend_check.py <one_batch CLI args ...>   (e.g. --model-path M --load-format dummy --tp-size 2 ...
#        --batch-size 1 --input-len 8192 --output-len 1)
import os, sys, time
import numpy as np, torch
import sglang.benchmark.one_batch as ob


def work(server_args, port_args, bench_args, gpu_id, tp_rank):
    ob.publish(server_args, role="scheduler")
    ob.initialize_moe_config(); ob.initialize_fp8_gemm_config(); ob.initialize_fp4_gemm_config()
    model_runner, _ = ob.load_model(server_args, port_args, gpu_id, tp_rank)
    times, logits = [], None
    for it in range(4):
        np.random.seed(1234); torch.manual_seed(1234)
        reqs = ob.prepare_synthetic_inputs_for_latency_test(bench_args.batch_size[0], bench_args.input_len[0])
        torch.cuda.synchronize(); t = time.perf_counter()
        _, logits, batch = ob.extend(reqs, model_runner)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t)
        model_runner.req_to_token_pool.clear(); model_runner.token_to_kv_pool_allocator.clear()
    if tp_rank == 0:
        torch.save(logits.float().cpu(), os.environ["AX_CHECK_OUT"])
        print(f"AX_CHECK extend_ms median(3 after warmup) = {1000*sorted(times[1:])[1]:.1f}  all={[round(1000*x,1) for x in times]}", flush=True)


ob.latency_test = work
sys.argv = [sys.argv[0]] + sys.argv[1:]
ob.cli_main()
