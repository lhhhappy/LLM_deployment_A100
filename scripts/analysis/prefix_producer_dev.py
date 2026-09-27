#!/usr/bin/env python3
"""Real TP2/DCP2/MTP/HiCache scheduler diagnostic, scaled dummy weights.

No SLO/accuracy claim. The normal engine executes synthetic shared prefixes;
its ax-prefix trace must prove actual producer and READY admission. Outputs
and every mechanism decision are retained. No model/scheduler monkey patch,
except the existing reproducible, well-scaled dummy initializer.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random


def install_dummy():
    from dcp_check import _ax_well_scaled_init
    import sglang.srt.model_loader.loader as loader

    loader.initialize_dummy_weights = _ax_well_scaled_init(loader.initialize_dummy_weights)


if os.environ.get("AX_PREFIX_DEV_DUMMY") == "1":
    install_dummy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mtp', action='store_true')
    parser.add_argument('--graph', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['AX_PREFIX_DEV_DUMMY'] = '1'
    import sglang

    extra = dict(speculative_algorithm='NEXTN', speculative_draft_model_path=args.model,
        speculative_num_steps=3, speculative_eagle_topk=1, speculative_num_draft_tokens=4) if args.mtp else {}
    engine = None
    records = []
    try:
        engine = sglang.Engine(model_path=args.model, load_format='dummy', tp_size=2, dcp_size=2,
            log_level='info', skip_tokenizer_init=True, random_seed=1234, kv_cache_dtype='bfloat16',
            dsa_prefill_backend='tilelang', dsa_decode_backend='tilelang', linear_attn_backend='triton',
            mem_fraction_static=0.60, max_total_tokens=65536, max_mamba_cache_size=96,
            max_running_requests=12, mamba_radix_cache_strategy='extra_buffer',
            chunked_prefill_size=8192, max_prefill_tokens=16384, schedule_policy='lpm',
            prefill_decode_interval=1, disable_cuda_graph=not args.graph,
            cuda_graph_max_bs_decode=8, disable_custom_all_reduce=True, enable_cache_report=True,
            enable_hierarchical_cache=True, hicache_size=4, hicache_io_backend='kernel',
            hicache_mem_layout='page_first', hicache_write_policy='write_through', **extra)
        rng = random.Random(128)
        prefix = [rng.randrange(10000) for _ in range(16384)]
        prompts = [prefix + [rng.randrange(10000) for _ in range(n)] for n in (3072, 512, 1024, 1536)]
        prompts += [[rng.randrange(10000) for _ in range(14080)]]
        def generate(label, inputs):
            output = engine.generate(input_ids=inputs, sampling_params={
                'temperature': 0.0, 'max_new_tokens': 16, 'ignore_eos': True},
                return_logprob=True, logprob_start_len=-1)
            outputs = output if isinstance(output, list) else [output]
            for result in outputs:
                if len(result['output_ids']) != 16:
                    raise RuntimeError('incomplete generation')
                if not all(math.isfinite(p[0]) for p in result['meta_info']['output_token_logprobs']):
                    raise RuntimeError('non-finite output logprobs')
            records.append(dict(case=label, result=output))
            (args.output / 'responses.json').write_text(json.dumps(records, indent=2) + '\n')
        generate('warmup', prompts)
        for cycle in range(2):
            flush = engine.flush_cache()
            if not flush.success:
                raise RuntimeError(f'flush failed: {flush}')
            generate(f'cold_family_{cycle}', prompts)
            generate(f'warm_repeat_{cycle}', prompts[1:4])
        (args.output / 'summary.json').write_text(json.dumps(dict(
            validity='DIAGNOSTIC', requests=5 + 2 * (5 + 3), tp=2, dcp=2,
            mtp=args.mtp, graph=args.graph, hicache=True,
            limitations='Scaled dummy model; no TP8 accuracy or SLO claim. Require mechanism log coverage.'), indent=2) + '\n')
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == '__main__':
    main()
