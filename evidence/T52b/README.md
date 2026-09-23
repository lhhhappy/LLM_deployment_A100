# T52b evidence — patch 170 wrong outputs with --enable-attn-tp-input-scattered (8-card 026j 0/12)
Dev box, surrogate 8-layer model, context-sensitive [test-only init](../T52/t52_test_dummy_init.patch), full stack 000…160+170,
SGLANG_AX_KDA_DUAL_SNAPSHOT=1, chunk 4096. 22 requests: cold exact lengths 37/100/500/512/1000/1024/3000/4096/5000, prefix hits
P=20k/100k × c=100/1000/3000 (+warm), concurrent mixed pair. Greedy 32 tokens + top-5 logprobs.
Verdict file: SUMMARY_v2.txt (written by the historical one-off summary runner after job t52b_v2; that runner has been removed).

| dirs | what |
|---|---|
| *_s (eager170sc_s, bcg170sc_s), cmp_eager170sc_s__bcg170sc_s.json | v1 repro, TP2 + scatter: 10/22 first tokens same, top-5 diff up to 4.22 — wrong at bucket AND padded lengths |
| *_v2tp2 | v2, TP2 (GPU0,1): scatter and no-scatter, eager vs BCG |
| *_v2tp1 | v2, TP1 (GPU1): eager vs BCG |
| */server_excerpt.txt | "attn_tp_input_scattered is enabled", capture buckets, every Prefill batch line (graph: True/False) |
| t52b_*.log | gjob logs (rep = v1 repro, v2 = validation matrix, sum = verdicts, done = coordinator's DONE writer) |
