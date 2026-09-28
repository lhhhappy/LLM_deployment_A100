"""117 follow-up: bounded, immutable overrides of measured SM80 MoE tables.

The caller checks the model/device/dtype contract. Humming table intervals count
routed rows (tokens * 9), with an exclusive lower and inclusive upper bound.
"""

import json


_NATIVE_DOWN = {
    "block_shape": [128, 256, 64],
    "warp_shape": [64, 64, 64],
    "use_stream_k": False,
    "use_f16_accum": False,
    "num_sms": 108,
    "num_stages": 4,
    "num_ctas_per_sm": 1,
    "num_write_splits": 1,
}


def _retune_prefill_table(configs: dict, table_key: str, native: dict, changes: dict) -> dict:
    """Retune only the 8k–16k range and exact native kernel family.

    Keep the M block height: W1 and W2 share the expert block index. Never
    mutate Humming's cached table or nested configuration.
    Unknown configurations and all sizes outside the measured range are kept.
    """
    lower, upper = 8192 * 9 - 1, 16384 * 9
    table = []
    changed = False
    for lo, hi, config in configs[table_key]:
        start, stop = max(lo, lower), min(hi, upper)
        normalized = dict(config)
        # Native Python heuristics use tuples; the serialized table uses lists.
        # Treat those two shape representations alike, retaining every key so
        # a future option still prevents this exact-family override.
        for key in ("block_shape", "warp_shape"):
            shape = normalized.get(key)
            if isinstance(shape, (tuple, list)):
                normalized[key] = list(shape)
        # An exact match also leaves future Humming options such as a different
        # raster order alone until they have their own numerical/perf evidence.
        if start >= stop or normalized != native:
            table.append((lo, hi, config))
            continue
        if lo < start:
            table.append((lo, start, config))
        tuned = dict(config, **changes)
        table.append((start, stop, tuned))
        if stop < hi:
            table.append((stop, hi, config))
        changed = True
    if not changed:
        return configs
    return dict(configs, **{table_key: table, table_key + "_str": json.dumps(table)})


def tune_sm80_prefill_down(configs: dict) -> dict:
    """Keep W2's K order; change only its N tile, pipeline depth and CTA count."""
    return _retune_prefill_table(
        configs, "w2_tuning_config", _NATIVE_DOWN,
        {"block_shape": [128, 128, 64], "num_stages": 3, "num_ctas_per_sm": 2},
    )


def tune_sm80_prefill_up(configs: dict) -> dict:
    """Avoid W13's cross-CTA split-K sum for these large shapes.

    This keeps the dot product in FP32 through K and removes stream-K's BF16
    partial-output sums. Require an independent numerical reference rather
    than a bitwise match with the split-K baseline.
    """
    return _retune_prefill_table(
        configs, "w13_tuning_config", dict(_NATIVE_DOWN, use_stream_k=True),
        {"use_stream_k": False},
    )
