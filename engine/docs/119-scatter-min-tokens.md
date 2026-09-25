# 119 — scatter attention-TP inputs only for large extends (candidate)

## Why
The deployed MTP configuration runs without `--enable-attn-tp-input-scattered`, so every TP rank computes mHC
(hyper-connection pre/post) and the norms on all tokens of a prefill; in a TP8 profile of an 8192-token chunk the
mHC/norm kernels take ~12% of the kernel time (`evidence/L081p-tp8_prefill_profile_8k_16k/`). With the flag, the target
model's prefill runs them on 1/8 of the tokens per rank, replacing each layer's two all-reduces by two reduce-scatters
and two all-gathers (NCCL). Screen 085 (N30, 60 minutes, paired with 081 on the same requests) turned the flag on:
chain-start misses 23→21 and TPOT mean 33.2→32.3 ms, but fast misses 238→257. Extends of at most 1024 tokens
(4096-wide BF16, 8 MiB) use the fast custom all-reduce today; scattered they pay four NCCL collectives per layer,
which falls on the short cache-hit requests the fast gate counts.

## What it does
Off unless `SGLANG_AX_SCATTER_MIN_TOKENS` > 0, and then only with `--enable-attn-tp-input-scattered` (startup refuses
otherwise). `AttnTpContext.use_input_scattered` additionally requires the extend to have at least that many tokens.
The input padding (`ForwardBatch.prepare_attn_tp_scatter_input`) and the forward's scatter scope
(`maybe_input_scattered`) both decide through that method, and the token count is identical on every TP rank, so they
always agree. Verify, decode and the MTP draft are unchanged (never scattered). The startup line becomes
"attn_tp_input_scattered is enabled for extends of at least N tokens"; the mechanism line shows `119=on:N`.
Suggested N: 1025 (above the custom all-reduce cap, where the unscattered path already uses NCCL).

## Evidence
CPU harness: the mechanism line (`tests/test_sched_protect_chain.py`). The decision itself runs only on GPUs.
Not validated: TP8 (capability smoke, the fast/chain/TPOT effect against 081 and 085).
