# T52 evidence — patch 170 (GLM breakable prefill CUDA graph)
Dev box GPU1, TP1, 8-layer rank surrogate, dummy weights, full stack 000…160 + 170, SGLANG_AX_KDA_DUAL_SNAPSHOT=1.
Rerun: scripts/analysis/devbox_bcg_check.sh (client: bcg_correctness.py, compare: bcg_compare.py, test-only init: t52_test_dummy_init.patch).

| dir / file | what |
|---|---|
| attempt1_warmup_vocab/job.log | first run: stock server warmup prompt ids ~154k > surrogate vocab 19360 -> device assert in ALL arms (not a 170 bug); fixed by --skip-server-warmup |
| bcg170, eager170, base, negctl | stock ±1e-3 dummy init: everything bit-identical INCLUDING negative control -> insensitive, not evidence |
| *_s + cmp_*_s.json | context-sensitive test init (t52_test_dummy_init.patch): the real numerics comparison (see patches/170-*.md table) |
| bcg170_t, eager170_t | chunkcost with --enable-metrics (extend cost + decode step); bcg170/eager170 chunkcost has 0 extend times (no metrics) but valid decode curve |
| */server_excerpt.txt | capture lines, every Prefill batch line (cuda graph: True/False), errors |
| test140.jsonl | scripts/test_kda_snapshot_140.py on the s170 tree vs base_exact fla kernels: 8/8 pass |
| jobs/ | gjob logs (t52_bcg, t52_bcg2, t52_neg, t52_strong, t52_strong_e, t52_140) |
