# 8 卡试验本·事实部分（自动生成 2026-09-23 15:37 UTC，`scripts/pod/ledger.py`）

## 001-b113_start_probe  [done]
- 补丁：000 101 105 110 111 112 113
- 参数：``
- 环境：``
  - （无关键结果行）

## 002-cap_smoke  [done]
- 补丁：
- 参数：``
- 环境：``
  - `CAP_SMOKE correct=12/12 finish=['stop'] mean_completion_tokens=2209`

## 003-verify_kernels  [done]
- 补丁：000 101 105 110 111 112 113 140 120 130 150 160
- 参数：``
- 环境：``
  - （无关键结果行）

## 012-dev_b113_n6  [done]
- 补丁：000 101 105 110 111 112 113
- 参数：``
- 环境：``
  - （无关键结果行）

## 012b-coldprobe_b113  [done]
- 补丁：000 101 105 110 111 112 113
- 参数：``
- 环境：``
  - `COLDPROBE {"len": 20000, "ttft_s": 1.979, "wall_s": 2.059, "prompt_tokens": 20000, "cached": 0, "first_ids": [3226, 309, 197, 197]}`
  - `COLDPROBE {"len": 60000, "ttft_s": 6.371, "wall_s": 6.455, "prompt_tokens": 60000, "cached": 0, "first_ids": [271, 197, 197, 322]}`
  - `COLDPROBE {"len": 190000, "ttft_s": 19.289, "wall_s": 19.384, "prompt_tokens": 190000, "cached": 0, "first_ids": [197, 197, 29, 197]}`

## 012c-coldprobe_b114  [done]
- 补丁：000 101 105 110 111 112 113 114
- 参数：``
- 环境：``
  - （无关键结果行）

## 012d-coldprobe_b113_cp8  [failed]
- 补丁：000 101 105 110 111 112 113
- 参数：``
- 环境：``
  - `ENGINE_DIED`

## 012e-coldprobe_b113_dcp8  [failed]
- 补丁：000 101 105 110 111 112 113
- 参数：``
- 环境：``
  - `ENGINE_DIED`

## 012f-coldprobe_b114  [done]
- 补丁：000 101 105 110 111 112 113 114
- 参数：``
- 环境：``
  - `COLDPROBE {"len": 20000, "ttft_s": 9.221, "wall_s": 9.312, "prompt_tokens": 20000, "cached": 0, "first_ids": [3226, 309, 197, 197]}`
  - `COLDPROBE {"len": 60000, "ttft_s": 6.178, "wall_s": 6.262, "prompt_tokens": 60000, "cached": 0, "first_ids": [271, 197, 197, 1]}`
  - `COLDPROBE {"len": 190000, "ttft_s": 15.443, "wall_s": 15.537, "prompt_tokens": 190000, "cached": 0, "first_ids": [197, 197, 8004, 2184]}`

## 012g-coldprobe_b113_cp8  [failed]
- 补丁：000 101 105 110 111 112 113
- 参数：``
- 环境：``
  - `ENGINE_DIED`

## 013-dev_b120on_n6  [done]
- 补丁：000 101 105 110 111 112 113 120
- 参数：``
- 环境：``
  - （无关键结果行）

## 013b-coldprobe_b115_dcp8  [done]
- 补丁：000 101 105 110 111 112 113 114 115
- 参数：``
- 环境：``
  - `COLDPROBE {"len": 20000, "ttft_s": 2.036, "wall_s": 2.138, "prompt_tokens": 20000, "cached": 0, "first_ids": [3226, 309, 197, 197]}`
  - `COLDPROBE {"len": 60000, "ttft_s": 5.891, "wall_s": 5.997, "prompt_tokens": 60000, "cached": 0, "first_ids": [271, 197, 197, 322]}`
  - `COLDPROBE {"len": 190000, "ttft_s": 15.577, "wall_s": 15.693, "prompt_tokens": 190000, "cached": 0, "first_ids": [197, 197, 8004, 2184]}`
  - `CAP_SMOKE correct=12/12 finish=['stop'] mean_completion_tokens=2209`

## 014-dev_b120cap4096_n6  [cancelled]
- 补丁：000 101 105 110 111 112 113 120
- 参数：``
- 环境：``
  - （无关键结果行）

## 014b-dev_b140_n6  [cancelled]
- 补丁：000 101 105 110 111 112 113 140
- 参数：``
- 环境：``
  - （无关键结果行）

## 015-dev_b113_n10  [cancelled]
- 补丁：000 101 105 110 111 112 113
- 参数：``
- 环境：``
  - （无关键结果行）

## 016-dev_b120on_n10  [cancelled]
- 补丁：000 101 105 110 111 112 113 120
- 参数：``
- 环境：``
  - （无关键结果行）

## 017-dev_b120cap4096_n10  [cancelled]
- 补丁：000 101 105 110 111 112 113 120
- 参数：``
- 环境：``
  - （无关键结果行）

## 018-dev_b113_n14  [cancelled]
- 补丁：000 101 105 110 111 112 113
- 参数：``
- 环境：``
  - （无关键结果行）

## 020-coldprobe_scat  [done]
- 补丁：000 101 105 110 111 112 113
- 参数：``
- 环境：``
  - `COLDPROBE {"len": 20000, "ttft_s": 2.788, "wall_s": 2.868, "prompt_tokens": 20000, "cached": 0, "first_ids": [3226, 309, 197, 197]}`
  - `COLDPROBE {"len": 60000, "ttft_s": 6.014, "wall_s": 6.098, "prompt_tokens": 60000, "cached": 0, "first_ids": [271, 197, 197, 1]}`
  - `COLDPROBE {"len": 190000, "ttft_s": 17.518, "wall_s": 17.612, "prompt_tokens": 190000, "cached": 0, "first_ids": [197, 197, 63, 1406]}`
  - `CAP_SMOKE correct=12/12 finish=['stop'] mean_completion_tokens=1997`

## 021-coldprobe_ll128  [done]
- 补丁：000 101 105 110 111 112 113
- 参数：``
- 环境：``
  - `COLDPROBE {"len": 20000, "ttft_s": 2.281, "wall_s": 2.362, "prompt_tokens": 20000, "cached": 0, "first_ids": [3226, 309, 197, 197]}`
  - `COLDPROBE {"len": 60000, "ttft_s": 6.18, "wall_s": 6.264, "prompt_tokens": 60000, "cached": 0, "first_ids": [271, 197, 197, 1]}`
  - `COLDPROBE {"len": 190000, "ttft_s": 19.057, "wall_s": 19.152, "prompt_tokens": 190000, "cached": 0, "first_ids": [197, 197, 63, 1406]}`

## 022-coldprobe_capacity  [done]
- 补丁：000 101 105 110 111 112 113
- 参数：``
- 环境：``
  - `COLDPROBE {"len": 20000, "ttft_s": 2.083, "wall_s": 2.163, "prompt_tokens": 20000, "cached": 0, "first_ids": [3226, 309, 197, 197]}`
  - `COLDPROBE {"len": 60000, "ttft_s": 6.122, "wall_s": 6.205, "prompt_tokens": 60000, "cached": 0, "first_ids": [271, 197, 197, 532]}`
  - `COLDPROBE {"len": 190000, "ttft_s": 19.174, "wall_s": 19.269, "prompt_tokens": 190000, "cached": 0, "first_ids": [197, 197, 63, 1406]}`

## 023-dev_b120v2_n6  [cancelled]
- 补丁：000 101 105 110 111 112 113 120
- 参数：`--chunked-prefill-size 16384`
- 环境：`SGLANG_AX_SCHED_COLD_CAP=8192 SGLANG_AX_SCHED_SHORT_TOKENS=8192`
  - （无关键结果行）

## 024-dev_best_n10  [cancelled]
- 补丁：000 101 105 110 111 112 113 120
- 参数：`--chunked-prefill-size 16384 --enable-attn-tp-input-scattered --cuda-graph-max-bs-decode 64 --max-mamba-cache-size 200`
- 环境：`SGLANG_AX_SCHED_COLD_CAP=8192 SGLANG_AX_SCHED_SHORT_TOKENS=8192`
  - （无关键结果行）

## 024-ladder_best  [failed]
- 补丁：000 101 105 110 111 112 113 120
- 参数：`--chunked-prefill-size 16384 --enable-attn-tp-input-scattered --cuda-graph-max-bs-decode 64 --max-mamba-cache-size 200`
- 环境：`SGLANG_AX_SCHED_COLD_CAP=8192 SGLANG_AX_SCHED_SHORT_TOKENS=8192`
  - `LADDER N=10 ENGINE_DEAD`

## 025-dev_best140_n10  [cancelled]
- 补丁：000 101 105 110 111 112 113 140 120
- 参数：`--chunked-prefill-size 16384 --enable-attn-tp-input-scattered --cuda-graph-max-bs-decode 64 --max-mamba-cache-size 200`
- 环境：`SGLANG_AX_SCHED_COLD_CAP=8192 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_KDA_DUAL_SNAPSHOT=1`
  - （无关键结果行）

## 025-ladder_dcp  [failed]
- 补丁：000 101 105 110 111 112 113 114 115 120
- 参数：`--chunked-prefill-size 16384 --mem-fraction-static 0.75 --dcp-size 8 --cuda-graph-max-bs-decode 64`
- 环境：`SGLANG_AX_SCHED_COLD_CAP=8192 SGLANG_AX_SCHED_SHORT_TOKENS=8192`
  - `LADDER N=10 ENGINE_DEAD`

## 025b-ladder_best106  [done]
- 补丁：000 101 105 106 110 111 112 113 120
- 参数：`--chunked-prefill-size 16384 --enable-attn-tp-input-scattered --cuda-graph-max-bs-decode 64 --max-mamba-cache-size 200`
- 环境：`SGLANG_AX_SCHED_COLD_CAP=8192 SGLANG_AX_SCHED_SHORT_TOKENS=8192`
  - `LADDER N=10 formal_est=False harness=False tpot=0.0472 | fast_i:2.11(7/23) overal:3.12(12/27) turn_s:3.49(0/3) chain_:26.38(12/22)`
  - `LADDER stop at N=10 (formal-est FAIL)`

## 026-ladder_best140  [failed]
- 补丁：000 101 105 106 110 111 112 113 140 120
- 参数：`--chunked-prefill-size 16384 --mem-fraction-static 0.75 --enable-attn-tp-input-scattered --cuda-graph-max-bs-decode 64 --max-mamba-cache-size 200`
- 环境：`SGLANG_AX_SCHED_COLD_CAP=8192 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_KDA_DUAL_SNAPSHOT=1`
  - `LADDER N=18 formal_est=False harness=False tpot=0.0829 | fast_i:2.44(10/23) overal:4.41(18/27) turn_s:11.66(0/3) chain_:25.43(12/22)`
  - `LADDER first level N=18 failed -> descending: 14 10`

## 026a-ifx_a_b113  [done]
- 补丁：000 101 105 106 110 111 112 113
- 参数：`--chunked-prefill-size 8192 --mem-fraction-static 0.75 --enable-attn-tp-input-scattered --cuda-graph-max-bs-decode 64 --max-mamba-cache-size 200`
- 环境：``
  - `INTERFERENCE {"decode_streams": 12, "cold_len": 190000, "cold_ttft": 18.96, "cold_wall": 21.09, "decode_gap_during_cold_ms": {"p50": 18.6, "p95": 252.7, "max": 20230.1}, "stream_tpot": {"mean": 0.0226, "max": 0.0226}, "short_hit_ttft": {"n": 21, "p50": 9.15, "p95": 16.97, "max": 17.96}}`

## 026b-ifx_b_v3cap2048  [done]
- 补丁：000 101 105 106 110 111 112 113 140 120
- 参数：`--chunked-prefill-size 16384 --mem-fraction-static 0.75 --enable-attn-tp-input-scattered --cuda-graph-max-bs-decode 64 --max-mamba-cache-size 200`
- 环境：`SGLANG_AX_SCHED_COLD_CAP=2048 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_KDA_DUAL_SNAPSHOT=1`
  - `INTERFERENCE {"decode_streams": 12, "cold_len": 190000, "cold_ttft": 29.18, "cold_wall": 30.1, "decode_gap_during_cold_ms": {"p50": 270.1, "p95": 411.7, "max": 538.9}, "stream_tpot": {"mean": 0.0254, "max": 0.0254}, "short_hit_ttft": {"n": 30, "p50": 0.54, "p95": 0.68, "max": 0.74}}`

## 026c-ifx_c_v3cap2048_i3  [done]
- 补丁：000 101 105 106 110 111 112 113 140 120
- 参数：`--chunked-prefill-size 16384 --mem-fraction-static 0.75 --enable-attn-tp-input-scattered --cuda-graph-max-bs-decode 64 --max-mamba-cache-size 200 --prefill-decode-interval 3`
- 环境：`SGLANG_AX_SCHED_COLD_CAP=2048 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_KDA_DUAL_SNAPSHOT=1`
  - `INTERFERENCE {"decode_streams": 12, "cold_len": 190000, "cold_ttft": 33.14, "cold_wall": 34.12, "decode_gap_before_cold_ms": {"p50": 16.9, "p95": 17.3}, "decode_gap_during_cold_ms": {"p50": 23.8, "p95": 375.2, "max": 413.7}, "stream_tpot": {"mean": 0.026, "max": 0.026}, "short_hit_ttft": {"n": 34, "p50": 0.53, "p95": 0.72, "max": 0.73}, "spec": {"spec_accept_length": 0.0, "spec_accept_rate": 0.0}}`

## 026d-ifx_d_v3cap4096_i2  [done]
- 补丁：000 101 105 106 110 111 112 113 140 120
- 参数：`--chunked-prefill-size 16384 --mem-fraction-static 0.75 --enable-attn-tp-input-scattered --cuda-graph-max-bs-decode 64 --max-mamba-cache-size 200 --prefill-decode-interval 2`
- 环境：`SGLANG_AX_SCHED_COLD_CAP=4096 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_KDA_DUAL_SNAPSHOT=1`
  - `INTERFERENCE {"decode_streams": 12, "cold_len": 190000, "cold_ttft": 23.46, "cold_wall": 24.08, "decode_gap_before_cold_ms": {"p50": 16.6, "p95": 17.1}, "decode_gap_during_cold_ms": {"p50": 25.8, "p95": 567.1, "max": 606.1}, "stream_tpot": {"mean": 0.0231, "max": 0.0231}, "short_hit_ttft": {"n": 24, "p50": 0.69, "p95": 0.99, "max": 1.0}, "spec": {"spec_accept_length": 0.0, "spec_accept_rate": 0.0}`

## 026e-ifx_e_v3cap1024_i4  [done]
- 补丁：000 101 105 106 110 111 112 113 140 120
- 参数：`--chunked-prefill-size 16384 --mem-fraction-static 0.75 --enable-attn-tp-input-scattered --cuda-graph-max-bs-decode 64 --max-mamba-cache-size 200 --prefill-decode-interval 4`
- 环境：`SGLANG_AX_SCHED_COLD_CAP=1024 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_KDA_DUAL_SNAPSHOT=1`
  - `INTERFERENCE {"decode_streams": 12, "cold_len": 190000, "cold_ttft": 54.78, "cold_wall": 55.19, "decode_gap_before_cold_ms": {"p50": 16.6, "p95": 17.1}, "decode_gap_during_cold_ms": {"p50": 19.5, "p95": 288.6, "max": 339.1}, "stream_tpot": {"mean": 0.0302, "max": 0.0302}, "short_hit_ttft": {"n": 55, "p50": 0.42, "p95": 0.55, "max": 0.6}, "spec": {"spec_accept_length": 0.0, "spec_accept_rate": 0.0}`

## 026f-ifx_f_mtp  [done]
- 补丁：000 101 105 106 110 111 112 113 140 120 130 150 160
- 参数：``
- 环境：``
  - `INTERFERENCE {"decode_streams": 12, "cold_len": 190000, "cold_ttft": 21.36, "cold_wall": 25.09, "decode_gap_before_cold_ms": {"p50": 0.0, "p95": 33.1}, "decode_gap_during_cold_ms": {"p50": 0.0, "p95": 72.0, "max": 22458.2}, "stream_tpot": {"mean": 0.0165, "max": 0.0178}, "short_hit_ttft": {"n": 25, "p50": 9.63, "p95": 19.36, "max": 20.36}, "spec": {"spec_accept_length": 3.0, "spec_accept_rate": 0.`
  - `CAP_SMOKE correct=12/12 finish=['stop'] mean_completion_tokens=2249`

## 026g-cc_a_v3_114  [done]
- 补丁：000 101 105 106 110 111 112 113 114 140 120
- 参数：`--chunked-prefill-size 16384 --mem-fraction-static 0.75 --enable-attn-tp-input-scattered --cuda-graph-max-bs-decode 64 --max-mamba-cache-size 200 --prefill-decode-interval 3`
- 环境：`SGLANG_AX_SCHED_COLD_CAP=2048 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_KDA_DUAL_SNAPSHOT=1 SGLANG_AX_INDEXER_ROW_SHARD=1`
  - `FIT 0 {'fixed_ms': 107.5, 'us_per_tok': 60.2}`
  - `FIT 32768 {'fixed_ms': 113.3, 'us_per_tok': 61.3}`
  - `FIT 98304 {'fixed_ms': 119.4, 'us_per_tok': 63.1}`
  - `FIT 180224 {'fixed_ms': 132.8, 'us_per_tok': 65.1}`

## 026h-cc_b_v3  [done]
- 补丁：000 101 105 106 110 111 112 113 140 120
- 参数：`--chunked-prefill-size 16384 --mem-fraction-static 0.75 --enable-attn-tp-input-scattered --cuda-graph-max-bs-decode 64 --max-mamba-cache-size 200 --prefill-decode-interval 3`
- 环境：`SGLANG_AX_SCHED_COLD_CAP=2048 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_KDA_DUAL_SNAPSHOT=1`
  - `FIT 0 {'fixed_ms': 105.0, 'us_per_tok': 62.0}`
  - `FIT 32768 {'fixed_ms': 107.4, 'us_per_tok': 68.7}`
  - `FIT 98304 {'fixed_ms': 112.1, 'us_per_tok': 81.7}`
  - `FIT 180224 {'fixed_ms': 124.9, 'us_per_tok': 97.9}`

## 026i-ifx_g_mtp_v3  [done]
- 补丁：000 101 105 106 110 111 112 113 140 120 130 150 160
- 参数：``
- 环境：``
  - `INTERFERENCE {"decode_streams": 12, "cold_len": 190000, "cold_ttft": 39.32, "cold_wall": 40.14, "decode_gap_before_cold_ms": {"p50": 0.0, "p95": 32.7}, "decode_gap_during_cold_ms": {"p50": 0.0, "p95": 300.4, "max": 461.3}, "stream_tpot": {"mean": 0.0184, "max": 0.0196}, "short_hit_ttft": {"n": 40, "p50": 0.63, "p95": 0.83, "max": 0.87}, "spec": {"spec_accept_length": 3.0, "spec_accept_rate": 0.666`
  - `CAP_SMOKE correct=12/12 finish=['stop'] mean_completion_tokens=1911`

## 026j-ab170_bcg_scat  [done]
- 补丁：000 101 105 106 110 111 112 113 140 120 170
- 参数：``
- 环境：``
  - `FIT 0 {'fixed_ms': 63.9, 'us_per_tok': 83.9}`
  - `FIT 32768 {'fixed_ms': 43.0, 'us_per_tok': 93.0}`
  - `FIT 98304 {'fixed_ms': 48.3, 'us_per_tok': 106.2}`
  - `FIT 180224 {'fixed_ms': 58.4, 'us_per_tok': 123.9}`
  - `INTERFERENCE {"decode_streams": 12, "cold_len": 190000, "cold_ttft": 34.3, "cold_wall": 35.12, "decode_gap_before_cold_ms": {"p50": 15.8, "p95": 16.3}, "decode_gap_during_cold_ms": {"p50": 22.8, "p95": 410.5, "max": 921.5}, "stream_tpot": {"mean": 0.0255, "max": 0.0255}, "short_hit_ttft": {"n": 35, "p50": 0.58, "p95": 0.96, "max": 1.1}, "spec": {"spec_accept_length": 0.0, "spec_accept_rate": 0.0}}`
  - `CAP_SMOKE correct=0/12 finish=['length', 'stop'] mean_completion_tokens=8356`

## 026k-ab170_eager_scat  [done]
- 补丁：000 101 105 106 110 111 112 113 140 120 170
- 参数：``
- 环境：``
  - `FIT 0 {'fixed_ms': 77.4, 'us_per_tok': 76.3}`
  - `FIT 32768 {'fixed_ms': 79.3, 'us_per_tok': 83.3}`
  - `FIT 98304 {'fixed_ms': 83.1, 'us_per_tok': 97.6}`
  - `FIT 180224 {'fixed_ms': 92.4, 'us_per_tok': 115.5}`
  - `INTERFERENCE {"decode_streams": 12, "cold_len": 190000, "cold_ttft": 33.71, "cold_wall": 34.11, "decode_gap_before_cold_ms": {"p50": 17.0, "p95": 17.4}, "decode_gap_during_cold_ms": {"p50": 24.8, "p95": 389.8, "max": 655.7}, "stream_tpot": {"mean": 0.0259, "max": 0.026}, "short_hit_ttft": {"n": 34, "p50": 0.54, "p95": 0.79, "max": 0.99}, "spec": {"spec_accept_length": 0.0, "spec_accept_rate": 0.0}`
  - `CAP_SMOKE correct=12/12 finish=['stop'] mean_completion_tokens=2852`

## 026l-ab170_bcg_noscat  [done]
- 补丁：000 101 105 106 110 111 112 113 140 120 170
- 参数：``
- 环境：``
  - `FIT 0 {'fixed_ms': 41.0, 'us_per_tok': 85.2}`
  - `FIT 32768 {'fixed_ms': 42.2, 'us_per_tok': 92.2}`
  - `FIT 98304 {'fixed_ms': 50.2, 'us_per_tok': 105.6}`
  - `FIT 180224 {'fixed_ms': 59.1, 'us_per_tok': 123.3}`
  - `INTERFERENCE {"decode_streams": 12, "cold_len": 190000, "cold_ttft": 33.26, "cold_wall": 34.12, "decode_gap_before_cold_ms": {"p50": 16.8, "p95": 17.3}, "decode_gap_during_cold_ms": {"p50": 23.9, "p95": 406.6, "max": 501.2}, "stream_tpot": {"mean": 0.0257, "max": 0.0257}, "short_hit_ttft": {"n": 34, "p50": 0.51, "p95": 0.77, "max": 0.79}, "spec": {"spec_accept_length": 0.0, "spec_accept_rate": 0.0`
  - `CAP_SMOKE correct=12/12 finish=['stop'] mean_completion_tokens=2576`

## 027-ladder_v3  [failed]
- 补丁：000 101 105 106 110 111 112 113 140 120
- 参数：`--chunked-prefill-size 16384 --mem-fraction-static 0.75 --enable-attn-tp-input-scattered --cuda-graph-max-bs-decode 64 --max-mamba-cache-size 200 --prefill-decode-interval 3`
- 环境：`SGLANG_AX_SCHED_COLD_CAP=2048 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_KDA_DUAL_SNAPSHOT=1`
  - `LADDER N=18 formal_est=False harness=False tpot=0.0823 | fast_i:14.74(27/23) overal:18.93(48/27) turn_s:173.14(6/3) chain_:107.03(66/22)`
  - `LADDER first level N=18 failed -> descending: 14 10`
  - `LADDER N=14 formal_est=False harness=False tpot=0.0678 | fast_i:1.48(15/23) overal:10.89(37/27) turn_s:59.78(3/3) chain_:91.97(47/22)`

## 028-ladder_mtp114  [failed]
- 补丁：000 101 105 106 110 111 112 113 114 140 120 130 150 160
- 参数：`--kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 32 --cuda-graph-max-bs 32 --prefill-decode-interval 2`
- 环境：`SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_COLD_CAP=4096 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1`
  - `PRECHECK kv_tokens=1036288 min_required=900000 patches=000-interface-compliance.patch,101-d1v12-on-base.patch,105-role-split-single-partial.patch,106-defer-chunk-on-no-kv.patch,110-sm80-dsa-indexer.patch,111-sm80-fp8-moe-marlin.patch,112-sm80-indexer-kernels.patch,113-sm80-prefill-indexer.patch,114-indexer-row-shard.patch,140-kda-dual-snapshot.patch,120-sched-protect-chain.patch,130-async-tokenize`
  - `LADDER N=18 formal_est=False harness=False tpot=0.0528 | fast_i:5.45(31/23) overal:10.96(44/27) turn_s:16.93(1/3) chain_:78.54(45/22)`
  - `LADDER first level N=18 failed -> stop (no descent; diagnose and fix)`

## 028b-ladder_bcg114_mtp  [failed]
- 补丁：000 101 105 106 110 111 112 113 114 140 120 130 150 160 170
- 参数：`--chunked-prefill-size 4096 --cuda-graph-backend-prefill breakable --kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 32 --cuda-graph-max-bs 32 --prefill-decode-interval 2`
- 环境：`SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_COLD_CAP=4096 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1`
  - `PRECHECK kv_tokens=1225728 min_required=900000 patches=000-interface-compliance.patch,101-d1v12-on-base.patch,105-role-split-single-partial.patch,106-defer-chunk-on-no-kv.patch,110-sm80-dsa-indexer.patch,111-sm80-fp8-moe-marlin.patch,112-sm80-indexer-kernels.patch,113-sm80-prefill-indexer.patch,114-indexer-row-shard.patch,140-kda-dual-snapshot.patch,120-sched-protect-chain.patch,130-async-tokenize`
  - `CAP_SMOKE correct=12/12 finish=['stop'] mean_completion_tokens=2166`
  - `LADDER N=18 formal_est=False harness=False tpot=0.0483 | fast_i:18.14(138/23) overal:18.14(117/27) turn_s:14.24(0/3) chain_:71.12(49/22)`
  - `LADDER first level N=18 failed -> stop (no descent; diagnose and fix)`

## 028c-ladder_bcg114_c2i3  [failed]
- 补丁：000 101 105 106 110 111 112 113 114 140 120 170
- 参数：`--chunked-prefill-size 4096 --mem-fraction-static 0.75 --cuda-graph-max-bs-decode 64 --max-mamba-cache-size 200 --cuda-graph-backend-prefill breakable --prefill-decode-interval 3`
- 环境：`SGLANG_AX_SCHED_COLD_CAP=2048 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_KDA_DUAL_SNAPSHOT=1 SGLANG_AX_INDEXER_ROW_SHARD=1`
  - `PRECHECK kv_tokens=1316416 min_required=900000 patches=000-interface-compliance.patch,101-d1v12-on-base.patch,105-role-split-single-partial.patch,106-defer-chunk-on-no-kv.patch,110-sm80-dsa-indexer.patch,111-sm80-fp8-moe-marlin.patch,112-sm80-indexer-kernels.patch,113-sm80-prefill-indexer.patch,114-indexer-row-shard.patch,140-kda-dual-snapshot.patch,120-sched-protect-chain.patch,170-glm-bcg-prefil`
  - `CAP_SMOKE correct=12/12 finish=['stop'] mean_completion_tokens=2286`
  - `LADDER N=18 formal_est=False harness=False tpot=0.0715 | fast_i:15.41(89/23) overal:18.14(99/27) turn_s:180.07(5/3) chain_:93.59(62/22)`
  - `LADDER first level N=18 failed -> stop (no descent; diagnose and fix)`

## 029a-num_eager_scat  [failed]
- 补丁：000 101 105 106 110 111 112 113 114 140 120 170
- 参数：``
- 环境：``
  - （无关键结果行）

## 029b-num_bcg_scat  [done]
- 补丁：000 101 105 106 110 111 112 113 114 140 120 170
- 参数：``
- 环境：``
  - `NUMCMP wrong=18/18`

## 029c-num_bcg_noscat  [done]
- 补丁：000 101 105 106 110 111 112 113 114 140 120 170
- 参数：``
- 环境：``
  - `NUMCMP wrong=5/18`

## 029d-num_eager_noscat  [done]
- 补丁：000 101 105 106 110 111 112 113 114 140 120 170
- 参数：``
- 环境：``
  - `NUMCMP wrong=6/18`

## 031-ladder_bcg114_c4i2  [cancelled]
- 补丁：000 101 105 106 110 111 112 113 114 140 120 170
- 参数：`--chunked-prefill-size 4096 --mem-fraction-static 0.75 --cuda-graph-max-bs-decode 64 --max-mamba-cache-size 200 --cuda-graph-backend-prefill breakable --prefill-decode-interval 2`
- 环境：`SGLANG_AX_SCHED_COLD_CAP=4096 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_KDA_DUAL_SNAPSHOT=1 SGLANG_AX_INDEXER_ROW_SHARD=1`
  - （无关键结果行）

## 033-ladder_B1_mtp_c16k  [failed]
- 补丁：000 101 105 106 110 111 112 113 114 140 120 130 150 160 170
- 参数：`--chunked-prefill-size 16384 --mem-fraction-static 0.74 --kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 32 --cuda-graph-max-bs 32`
- 环境：`SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_COLD_CAP=16384 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1`
  - `PRECHECK kv_tokens=821504 min_required=900000 patches=000-interface-compliance.patch,101-d1v12-on-base.patch,105-role-split-single-partial.patch,106-defer-chunk-on-no-kv.patch,110-sm80-dsa-indexer.patch,111-sm80-fp8-moe-marlin.patch,112-sm80-indexer-kernels.patch,113-sm80-prefill-indexer.patch,114-indexer-row-shard.patch,140-kda-dual-snapshot.patch,120-sched-protect-chain.patch,130-async-tokenize.`
  - `PRECHECK FAIL: KV tokens 821504 < 900000`
