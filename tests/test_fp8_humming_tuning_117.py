"""CPU checks for the routed-row table boundary shared by W1/W2 dispatch."""

import copy
import importlib.util
import json
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "engine/sglang/srt/layers/quantization/fp8_humming_tuning.py"
spec = importlib.util.spec_from_file_location("fp8_humming_tuning_117", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TuningTableTests(unittest.TestCase):
    def setUp(self):
        self.native = dict(block_shape=[128, 256, 64], warp_shape=[64, 64, 64],
                           use_stream_k=False, use_f16_accum=False, num_sms=108,
                           num_stages=4, num_ctas_per_sm=1, num_write_splits=1)
        small = dict(self.native, block_shape=[64, 256, 64], use_stream_k=True)
        self.configs = dict(w13_tuning_config=[(0, 40000, small), (40000, 1 << 30, self.native)],
                            w2_tuning_config=[(0, 40000, small), (40000, 1 << 30, self.native)],
                            compute_config={"use_f16_accum": False, "gemm_type": "indexed"})
        self.configs["w2_tuning_config_str"] = json.dumps(self.configs["w2_tuning_config"])

    @staticmethod
    def selected(table, routed_rows):
        hits = [cfg for lo, hi, cfg in table if lo < routed_rows <= hi]
        assert len(hits) == 1, (routed_rows, hits)
        return hits[0]

    def test_all_boundaries_preserve_coverage_and_block_height(self):
        before = copy.deepcopy(self.configs)
        tuned = module.tune_sm80_prefill_down(self.configs)
        points = {1, 40000, 40001, 1 << 30}
        for tokens in (4096, 4097, 8191, 8192, 8193, 16383, 16384, 16385):
            points.update((tokens * 9 - 1, tokens * 9, tokens * 9 + 1))
        for rows in sorted(points):
            old = self.selected(self.configs["w2_tuning_config"], rows)
            new = self.selected(tuned["w2_tuning_config"], rows)
            w1 = self.selected(tuned["w13_tuning_config"], rows)
            self.assertEqual(new["block_shape"][0], w1["block_shape"][0])
            self.assertEqual(new["block_shape"][2], old["block_shape"][2])
            if 8192 * 9 <= rows <= 16384 * 9:
                self.assertEqual((new["block_shape"][1], new["num_stages"], new["num_ctas_per_sm"]),
                                 (128, 3, 2))
            else:
                self.assertEqual(new, old)
        self.assertEqual(self.configs, before)
        self.assertEqual(json.loads(tuned["w2_tuning_config_str"]),
                         json.loads(json.dumps(tuned["w2_tuning_config"])))
        self.assertIs(tuned["w13_tuning_config"], self.configs["w13_tuning_config"])

    def test_unknown_native_configuration_is_untouched(self):
        for change in ({"raster_group_m": 2}, {"use_f16_accum": True}, {"use_stream_k": True},
                       {"num_sms": 80}, {"num_write_splits": 2}, {"warp_shape": [32, 64, 64]}):
            configs = copy.deepcopy(self.configs)
            configs["w2_tuning_config"] = [(0, 1 << 30, dict(self.native, **change))]
            self.assertIs(module.tune_sm80_prefill_down(configs), configs)

    def test_live_tuple_shapes_match_serialized_list_shapes(self):
        configs = copy.deepcopy(self.configs)
        for _, _, config in configs["w2_tuning_config"]:
            for key in ("block_shape", "warp_shape"):
                config[key] = tuple(config[key])
        before = copy.deepcopy(configs)
        tuned = module.tune_sm80_prefill_down(configs)
        self.assertIsNot(tuned, configs)
        list_tuned = module.tune_sm80_prefill_down(self.configs)
        self.assertEqual(json.loads(tuned["w2_tuning_config_str"]),
                         json.loads(list_tuned["w2_tuning_config_str"]))
        self.assertEqual(configs, before)

    def test_multiple_intervals_and_repeat_application(self):
        configs = copy.deepcopy(self.configs)
        configs["w2_tuning_config"] = [(0, 90000, self.native), (90000, 1 << 30, self.native)]
        tuned = module.tune_sm80_prefill_down(configs)
        for rows in (8192 * 9 - 1, 8192 * 9, 89999, 90000, 90001, 16384 * 9, 16384 * 9 + 1):
            self.selected(tuned["w2_tuning_config"], rows)
        self.assertIs(module.tune_sm80_prefill_down(tuned), tuned)

    def test_up_preserves_every_option_except_stream_k_and_composes_with_down(self):
        configs = copy.deepcopy(self.configs)
        native_up = dict(self.native, use_stream_k=True,
                         block_shape=(128, 256, 64), warp_shape=(64, 64, 64))
        configs["w13_tuning_config"] = [(0, 1 << 30, native_up)]
        configs["w13_tuning_config_str"] = json.dumps(configs["w13_tuning_config"])
        original = copy.deepcopy(configs)
        up = module.tune_sm80_prefill_up(configs)
        for tokens in (4096, 8191, 8192, 8193, 16384, 16385):
            selected = self.selected(up["w13_tuning_config"], tokens * 9)
            self.assertEqual(selected, dict(native_up, use_stream_k=not (8192 <= tokens <= 16384)))
        self.assertIs(up["w2_tuning_config"], configs["w2_tuning_config"])
        self.assertEqual(configs, original)
        both_a = module.tune_sm80_prefill_down(up)
        both_b = module.tune_sm80_prefill_up(module.tune_sm80_prefill_down(configs))
        self.assertEqual(both_a, both_b)
        self.assertEqual(json.loads(up["w13_tuning_config_str"]),
                         json.loads(json.dumps(up["w13_tuning_config"])))
        self.assertIs(module.tune_sm80_prefill_up(up), up)
        for change in ({"use_f16_accum": True}, {"use_stream_k": False}, {"raster_group_m": 2}):
            unknown = dict(configs, w13_tuning_config=[(0, 1 << 30, dict(native_up, **change))])
            self.assertIs(module.tune_sm80_prefill_up(unknown), unknown)


if __name__ == "__main__":
    unittest.main()
