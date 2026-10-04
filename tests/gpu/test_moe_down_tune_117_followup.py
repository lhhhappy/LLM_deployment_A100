#!/usr/bin/env python3
"""Check production 117 W13/W2 tuning on one A100 with independent layer caches.

This is a wiring and exact-stage regression, not a performance benchmark or
another FP32 model oracle. The existing 117 fixture loads one checkpoint into
four real FusedMoE layers: off, down-only, up-only, and both switches. It observes
the effective production tables, checks converted weights, and compares W2 on
the same actual activation and routing alignment. W13 changes accumulation
order, so its repeatability is reported without an identity-to-baseline gate.
No pytest is required.
"""

import contextlib
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

SWITCHES = ("SGLANG_AX_SM80_MOE_DOWN_TUNE", "SGLANG_AX_SM80_MOE_UP_TUNE")
CASES = {"off": (False, False), "down": (True, False), "up": (False, True), "both": (True, True)}
ROWS = (4096, 8191, 8192, 8193, 16384, 16385)
NATIVE = dict(
    block_shape=[128, 256, 64], warp_shape=[64, 64, 64],
    use_stream_k=False, use_f16_accum=False, num_sms=108,
    num_stages=4, num_ctas_per_sm=1, num_write_splits=1,
)
TUNED = dict(NATIVE, block_shape=[128, 128, 64], num_stages=3, num_ctas_per_sm=2)
NATIVE_UP = dict(NATIVE, use_stream_k=True)
TUNED_UP = dict(NATIVE_UP, use_stream_k=False)


def emit(**record):
    print(json.dumps(record), flush=True)


def selected(table, rows):
    hits = [config for lo, hi, config in table if lo < rows * 9 <= hi]
    assert len(hits) == 1, (rows, hits)
    return hits[0]


def canonical(config):
    # Native Humming uses tuples for shapes; serialized launch tables use lists.
    # Compare their values while still requiring every expected option/value.
    return json.loads(json.dumps(config))


def bitwise_equal(left, right):
    return torch.equal(left.view(torch.uint8), right.view(torch.uint8))


@contextlib.contextmanager
def tune_flags(module, down, up):
    with (
        patch.object(module, "_AX_SM80_MOE_DOWN_TUNE", down),
        patch.object(module, "_AX_SM80_MOE_UP_TUNE", up),
    ):
        yield


def expected_config(base_configs, stage, rows, down, up):
    native = canonical(selected(base_configs[f"{stage}_tuning_config"], rows))
    enabled = down if stage == "w2" else up
    if enabled and 8192 <= rows <= 16384:
        assert native == (NATIVE if stage == "w2" else NATIVE_UP), (stage, rows, native)
        return TUNED if stage == "w2" else TUNED_UP
    return native


def check_guards(module, layer, native_configs, gemm_type, stage):
    """Use real converted metadata, but stub the base table and launch nothing."""
    from humming import dtypes
    from humming.config import GemmType

    meta_fields = (
        "a_dtype", "b_dtype", "c_dtype", "bs_dtype",
        "weight_scale_group_size", "weight_scale_group_size_n",
    )

    def probe(layer_changes=None, meta_changes=None, runner_changes=None):
        runner = object.__new__(module._AxFp8HummingRunnerCore)
        runner.layer = SimpleNamespace(
            hidden_size=layer.hidden_size,
            intermediate_size_per_partition=layer.intermediate_size_per_partition,
            params_dtype=layer.params_dtype,
            w2_weight=layer.w2_weight,
            humming_metas={
                name: SimpleNamespace(**{key: getattr(layer.humming_metas[name], key) for key in meta_fields})
                for name in ("w13", "w2")
            },
        )
        runner.num_experts = 289
        runner.config = SimpleNamespace(top_k=9)
        runner.humming_gemm_configs = {}
        for key, value in (layer_changes or {}).items():
            setattr(runner.layer, key, value)
        for key, value in (meta_changes or {}).items():
            setattr(runner.layer.humming_metas[stage], key, value)
        for key, value in (runner_changes or {}).items():
            setattr(runner, key, value)
        return runner

    cases = [
        ("hidden", probe(layer_changes=dict(hidden_size=2048)), gemm_type),
        ("intermediate", probe(layer_changes=dict(intermediate_size_per_partition=512)), gemm_type),
        ("experts", probe(runner_changes=dict(num_experts=288)), gemm_type),
        ("topk", probe(runner_changes=dict(config=SimpleNamespace(top_k=8))), gemm_type),
        ("params_dtype", probe(layer_changes=dict(params_dtype=torch.float16)), gemm_type),
        ("cpu", probe(layer_changes=dict(w2_weight=SimpleNamespace(device=torch.device("cpu")))), gemm_type),
        ("grouped", probe(), GemmType.GROUPED_CONTIGUOUS),
        ("activation_dtype", probe(meta_changes=dict(a_dtype=dtypes.float16)), gemm_type),
        ("weight_dtype", probe(meta_changes=dict(b_dtype=dtypes.float16)), gemm_type),
        ("output_dtype", probe(meta_changes=dict(c_dtype=dtypes.float16)), gemm_type),
        ("scale_dtype", probe(meta_changes=dict(bs_dtype=dtypes.float16)), gemm_type),
        ("group_k", probe(meta_changes=dict(weight_scale_group_size=64)), gemm_type),
        ("group_n", probe(meta_changes=dict(weight_scale_group_size_n=128)), gemm_type),
    ]

    def base_table(runner, mode):
        runner.humming_gemm_configs[mode.value] = table
        return table

    with (
        tune_flags(module, down=stage == "w2", up=stage == "w13"),
        patch.object(module.HummingRunnerCore, "get_humming_gemm_configs", new=base_table),
    ):
        for name, runner, mode in cases:
            table = copy.deepcopy(native_configs)
            before = copy.deepcopy(table)
            with patch.object(torch.cuda, "get_device_capability", side_effect=AssertionError("early fallback queried CUDA")):
                result = runner.get_humming_gemm_configs(mode)
            assert result is table and table == before, name

        # A valid tensor device is queried explicitly, and a non-SM80 result
        # returns the original table. No global/current-device lookup is used.
        table = copy.deepcopy(native_configs)
        with patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)) as capability:
            assert probe().get_humming_gemm_configs(gemm_type) is table
            capability.assert_called_once_with(layer.w2_weight.device)

        # An otherwise valid native family with an unknown option is left alone.
        table = copy.deepcopy(native_configs)
        key = f"{stage}_tuning_config"
        table[key] = [
            (lo, hi, dict(config, raster_group_m=2))
            for lo, hi, config in table[key]
        ]
        table[key + "_str"] = json.dumps(table[key])
        before = copy.deepcopy(table)
        assert probe().get_humming_gemm_configs(gemm_type) is table
        assert table == before

        # With both switches enabled, invalid metadata in one stage must not
        # disable tuning for the other eligible stage or modify the bad stage.
        table = copy.deepcopy(native_configs)
        before = copy.deepcopy(table)
        with tune_flags(module, down=True, up=True):
            result = probe(meta_changes=dict(weight_scale_group_size_n=128)).get_humming_gemm_configs(gemm_type)
        other = "w13" if stage == "w2" else "w2"
        assert result is not table and result[key] is table[key]
        assert canonical(selected(result[f"{other}_tuning_config"], 8192)) == (TUNED_UP if other == "w13" else TUNED)
        assert table == before
    emit(kind="guards", stage=stage, ok=True, early_fallback_cases=len(cases), unsupported_device=True,
         unknown_native_config=True, independent_metadata_gates=True)


def main():
    # The switch is startup-only. Controlled test patches make two independent
    # layers; toggling flags on an already-tuned core is not a baseline.
    for switch in SWITCHES:
        os.environ.pop(switch, None)
    os.environ.pop("SGLANG_AX_SM80_MOE_REDUCE", None)
    module = importlib.import_module("sglang.srt.layers.quantization.fp8_humming_moe")
    assert not module._AX_SM80_MOE_DOWN_TUNE and not module._AX_SM80_MOE_UP_TUNE
    for switch in SWITCHES:
        os.environ[switch] = "1"
    assert not module._AX_SM80_MOE_DOWN_TUNE and not module._AX_SM80_MOE_UP_TUNE
    from humming import dtypes
    from humming.config import GemmType
    from humming.jit.compiler import Compiler
    from humming.layer import HummingMethod

    assert torch.cuda.get_device_capability() == (8, 0)
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__), Path(module.__file__), Path(module.__file__).with_name("fp8_humming_tuning.py")]
    emit(kind="environment", gpu=torch.cuda.get_device_name(), torch=torch.__version__,
         source_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    spec = importlib.util.spec_from_file_location("moe117_fixture", Path(__file__).with_name("test_fp8_moe_humming_117.py"))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    warmups = {}
    original_run = module._AxFp8HummingRunnerCore.run

    def observed_run(core, runner_input, *args, **kwargs):
        tokens = runner_input.hidden_states.shape[0]
        seen = warmups.setdefault(id(core), [])
        seen.append(tokens)
        if len(seen) == 1:
            assert tokens == 1, seen
            return original_run(core, runner_input, *args, **kwargs)
        # Compiler.compile includes disk-cache lookup: forbidding it also
        # proves all later shapes reuse initialized in-process kernels.
        with patch.object(Compiler, "compile", side_effect=AssertionError("Humming compiled after the first warmup token")):
            return original_run(core, runner_input, *args, **kwargs)

    with contextlib.ExitStack() as stack:
        helper.open_runtime(stack)
        checkpoint = helper.make_checkpoint()
        layers, tables = {}, {}
        with patch.object(module._AxFp8HummingRunnerCore, "run", new=observed_run):
            for name, flags in CASES.items():
                with tune_flags(module, *flags):
                    layers[name] = helper.build_layer(checkpoint, True)
                    core = layers[name].quant_method.runner_core
                    tables[name] = core.get_humming_gemm_configs(GemmType.INDEXED)
                if name == "off":
                    base_snapshot = copy.deepcopy(tables[name])
                assert tables["off"] == base_snapshot
        del checkpoint
        baseline = layers["off"]
        base_core = baseline.quant_method.runner_core
        assert len({id(layer.quant_method.runner_core) for layer in layers.values()}) == 4
        assert len({id(configs) for configs in tables.values()}) == 4
        snapshots = copy.deepcopy(tables)
        assert tables["up"]["w13_tuning_config"] == tables["both"]["w13_tuning_config"]
        assert tables["down"]["w2_tuning_config"] == tables["both"]["w2_tuning_config"]
        assert tables["up"]["w2_tuning_config"] == base_snapshot["w2_tuning_config"]
        assert tables["down"]["w13_tuning_config"] == base_snapshot["w13_tuning_config"]

        base_tensors = dict(baseline.named_parameters()) | dict(baseline.named_buffers())
        for name, layer in layers.items():
            configs, flags = tables[name], CASES[name]
            if name != "off":
                candidate_tensors = dict(layer.named_parameters()) | dict(layer.named_buffers())
                assert base_tensors.keys() == candidate_tensors.keys()
                for tensor_name, tensor in base_tensors.items():
                    other = candidate_tensors[tensor_name]
                    assert tensor.dtype == other.dtype and tensor.shape == other.shape
                    assert bitwise_equal(tensor, other), (name, tensor_name)
                emit(kind="converted_weights", path=name, exact=True, tensors=len(base_tensors))
            assert configs["compute_config"] == base_snapshot["compute_config"]
            for stage in ("w13", "w2"):
                meta = layer.humming_metas[stage]
                assert (meta.a_dtype, meta.b_dtype, meta.c_dtype, meta.bs_dtype) == (
                    dtypes.bfloat16, dtypes.float8e4m3, dtypes.bfloat16, dtypes.bfloat16
                )
                assert (meta.weight_scale_group_size, meta.weight_scale_group_size_n) == (128, 0)
                key = f"{stage}_tuning_config"
                assert json.loads(configs[key + "_str"]) == canonical(configs[key])
                for rows in ROWS:
                    effective = selected(configs[key], rows)
                    assert canonical(effective) == expected_config(base_snapshot, stage, rows, *flags), (name, stage, rows, effective)
                    emit(kind="effective_table", path=name, stage=stage, rows=rows, config=effective)
            module._check_block_heights(configs, f"test_{name}")
            assert warmups[id(layer.quant_method.runner_core)] == list(module._WARMUP_TOKENS)
            emit(kind="startup", path=name, warmup_tokens=warmups[id(layer.quant_method.runner_core)],
                 no_humming_compile_after_first_token=True, block_heights_match=True)
        emit(kind="actual_metadata", group_k=128, group_n=0, fp8_bf16=True, stages=["w13", "w2"])
        for stage in ("w13", "w2"):
            check_guards(module, layers["both"], base_snapshot, GemmType.INDEXED, stage)

        original_forward = HummingMethod.forward_layer
        observed = []
        layer_names = {id(layer): name for name, layer in layers.items()}

        def checked_forward(layer, inputs, **kwargs):
            result = original_forward(layer, inputs, **kwargs)
            stage = kwargs["sublayer_name"]
            assert stage in ("w13", "w2")
            name = layer_names[id(layer)]
            configs, flags = tables[name], CASES[name]
            rows = kwargs["valid_shape_m"] // 9
            actual = kwargs["tuning_config"]
            key = f"{stage}_tuning_config"
            assert actual == configs[key + "_str"]
            effective = selected(configs[key], rows)
            assert canonical(effective) == expected_config(base_snapshot, stage, rows, *flags)
            observed.append((name, stage, rows))
            emit(kind="forwarded_table", path=name, stage=stage, rows=rows, config=effective)
            if stage == "w13":
                if rows in (8192, 16384):
                    # Reuse the same input and the exact same alignment. This
                    # reports determinism; it is not an accuracy acceptance gate.
                    first = kwargs["outputs"].clone()
                    original_forward(layer, inputs, **kwargs)
                    second = kwargs["outputs"]
                    differing = torch.count_nonzero(first.view(torch.int16) != second.view(torch.int16)).item()
                    emit(kind="w13_repeatability", path=name, rows=rows,
                         stream_k=effective["use_stream_k"], exact=differing == 0,
                         differing_values=differing, informational_only=True)
                return result
            if name == "off":
                return result
            if not effective["use_stream_k"]:
                # The two W2 launches consume identical actual W1+SwiGLU
                # output and alignment, isolating W2 from W1 nondeterminism.
                reference = torch.empty_like(kwargs["outputs"])
                native_kwargs = dict(kwargs, outputs=reference, tuning_config=base_snapshot["w2_tuning_config_str"])
                original_forward(baseline, inputs, **native_kwargs)
                equal = bitwise_equal(kwargs["outputs"], reference)
                emit(kind="same_activation_w2", path=name, rows=rows, exact=equal)
                assert equal, (name, rows)
            else:
                assert not 8192 <= rows <= 16384
                emit(kind="same_activation_w2", path=name, rows=rows, unchanged_stream_k=True)
            return result

        with (
            patch.object(HummingMethod, "forward_layer", new=staticmethod(checked_forward)),
            patch.object(Compiler, "compile", side_effect=AssertionError("first real prefill compiled Humming")),
        ):
            for rows in ROWS:
                inputs, topk = helper.make_inputs(rows, seed=117400 + rows)
                for name in ("down", "up", "both"):
                    with tune_flags(module, *CASES[name]):
                        output = layers[name](inputs, topk)
                        assert torch.isfinite(output).all().item()
                    assert tables == snapshots
                if rows in (8192, 16384):
                    with tune_flags(module, *CASES["off"]):
                        output = baseline(inputs, topk)
                        assert torch.isfinite(output).all().item()
                        assert base_core.get_humming_gemm_configs(GemmType.INDEXED) is tables["off"]
                    assert tables == snapshots
        expected_calls = {
            (name, stage, rows) for name in ("down", "up", "both")
            for stage in ("w13", "w2") for rows in ROWS
        } | {("off", stage, rows) for stage in ("w13", "w2") for rows in (8192, 16384)}
        assert len(observed) == len(expected_calls) and set(observed) == expected_calls, observed
        emit(kind="default_off_isolation", ok=True, native_table_unchanged=True,
             all_four_tables_unchanged=True, independent_runner_caches=True,
             independent_switch_tables=True, no_first_prefill_humming_compile=True)
        torch.cuda.synchronize()
        torch.distributed.destroy_process_group()
    emit(kind="complete")


if __name__ == "__main__":
    main()
