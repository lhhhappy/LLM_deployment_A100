# R1: GLM-5.3-Flash model and engine internals (SGLang v0.5.20 / vLLM) on 8x A100

Tags: **[V]** = verified from the cited source (code, config, PR text). **[I]** = inferred or computed by me.
Local sources: SGLang v0.5.20 at `/workspace/Agentic_science_challenge/src/sglang` (tag v0.5.20, 2026-09-18). vLLM main was cloned 2026-09-22 into a scratch dir. Paths below written `S/...` mean `src/sglang/python/sglang/srt/...`.

---

## 0. Most decision-relevant findings

1. **The KV pool, not the KDA state, is what limits sessions on TP8 [I].** The MLA latent (512 bf16 per token per layer) is replicated on every TP rank. That is about 12.3 KB/token/GPU (11 DSA layers plus the MTP layer), plus the indexer keys. KDA state is only about 18.5 MB per request slot per GPU at TP8. DP-attention stops the replication, which gives about 5x the token capacity, at a cost of about 10 GB more weights per GPU (details in §3 and §4).
2. **The SGLang hybrid cache loses one turn of reuse in exactly our "tail rewrite" pattern [V, from code].** DSA forces page_size=64, so the mamba strategy must be `extra_buffer`. With that strategy each extend forward checkpoints **one** KDA state. When the prefix match found KV deeper than the deepest KDA state (a "branching point"), that one checkpoint goes to the branching point **instead of** the end of the prompt. Result: the next turn's hit falls back to the previous turn's branch point. The uncached work becomes roughly 2 turns of growth + the rewrite distance d, instead of 1 turn + d. §3.3 gives the worked analysis and a small patch (split the prefill chunk at the branch point and at about L-1536).
3. **Upstream SGLang v0.5.20 and vLLM main have no sm_80 DSA path.** DSA backends default to flashmla_sparse/fa3 (Hopper). The `triton` DSA backend is rejected on CUDA. The indexer calls `deep_gemm.fp8_(paged_)mqa_logits` unconditionally. So the organizer images (`arena-sglang-glm53:260918`, `vllm-backport:260918-sm80`) must carry private patches. Inspect them before tuning anything. Known Ampere sparse-MLA fallback performance is poor (vLLM #57971: the fallback kernel is 58.8% of prefill time and reads the latent once per head).
4. **HiCache is unsafe for this model in v0.5.20.** It does not restore the DSA indexer buffers, which produced wrong outputs (sglang PR #39156 and #38212, both open). `--enable-mixed-chunk` corrupts mamba checkpoints (#39526, open). MTP/EAGLE is not validated together with DP-attention (cookbook).
5. **Dev box: `/sjtu` has only 184 GB free**, so the full 328 GB checkpoint does not fit. Use a layer-truncated checkpoint (§1.3).

---

## 1. The model

### 1.1 Identity and release
- The GLM-5 family README lists GLM-5.3-Flash as 320B-A18B, FP8 (plus a separate `GLM-5.3-Flash-BF16` repo). It is the first GLM model with a hybrid of sparse and linear attention, and it uses mHC and a 30T-token multimodal pretraining corpus. [V] https://github.com/zai-org/GLM-5 (README). Blog: https://z.ai/blog/glm-5.3-flash (could not fetch through the proxy).
- The architecture is brand new; InferenceX gives a release date of 2026-08-26. [V] https://github.com/SemiAnalysisAI/InferenceX/issues/2794
- SGLang support landed in PR #36507 (merged 2026-09-06) and is first released in **v0.5.20**, whose release notes list #36507 and #38621. [V] https://github.com/sgl-project/sglang/pull/36507, https://github.com/sgl-project/sglang/releases/tag/v0.5.20
- vLLM support landed in PR #53906 (merged to main 2026-09-03). The PR text says "Please use docker image" and "only the first commit is fully verified". [V] https://github.com/vllm-project/vllm/pull/53906. vLLM main as of 2026-09-22 has `vllm/models/glm5next/`. I did not check whether the v0.29.0 wheel (2026-09-09) includes it; the coordinator reports it does not.
- The chat template's `clear_thinking` defaults to false, so reasoning is kept in the history, which helps prefix sharing across turns. [V] GLM-5 README.

### 1.2 Structure from the local config and index (`s1-dev/glm_tok/`) [V]
- There are 45 text layers. KDA layers are indices [0,1,2, 4,5,6, …, 44]. DSA/MLA layers are every 4th: {3,7,…,43}, which is 11 layers. The MTP layer is index 45; it is a DSA+MoE layer with `eh_proj`, `enorm`, `hnorm` and `shared_head`.
- **KDA projections (q/k/v/o, f_a/f_b, g_a/g_b, b) are BF16**: they have no `weight_scale_inv` and are listed in `modules_to_not_convert`. MLA, MoE and dense-MLP weights are FP8 block-128.
- Indexer tensors: `wq_b`, `wk`, `weights_proj`, `k_norm`, `index_kpool_compress_ape` and `index_kpool_compress_gate`.
- The vision tower has 24 layers, hidden 1024. The `/generate` text workload never runs it, but its weights are still loaded.
- Per-layer weight sizes [I]:
  - One MoE layer: 288 × 3 × 4096 × 2048 FP8 ≈ **7.25 GB**. There are 43 MoE layers (42 plus MTP), about 312 GB total.
  - One KDA layer: about 272 MB BF16, 9.25 GB for all 34.
  - One MLA+indexer layer: about 0.14 GB, 1.7 GB for all 12.
  - Embedding plus lm_head: 2.5 GB BF16.
- Shards are ordered by the **lexicographic** layer string: layers 0,1 are in shard 2, layer 10 in shard 3, layer 2 in shard 17, layer 3 in shards 31-32, layer 45 (MTP) in shards 1-2, and norm plus visual in shard 62.

### 1.3 Small models for local development
- `malaiwah/glm5-next-tiny-random-bf16` has the real structure (5 layers, KDA with `num_heads` 4 and `head_dim` 16, MLA with `kv_lora` 16, `index_topk` 8, `kpool` 4, `hc_mult` 4) at toy sizes. It works for functional and cache-logic tests only, not for kernel performance. `aday777/glm5_next_tiny_fixture` is 4 layers with no KDA or MLA fields. [V] https://huggingface.co/malaiwah/glm5-next-tiny-random-bf16, https://huggingface.co/aday777/glm5_next_tiny_fixture
- `moonshotai/Kimi-Linear-48B-A3B-Instruct` has the same KDA plus MLA family. SGLang handles it through `kimi_linear.py` and the same `KimiLinearCacheParams`. It has no DSA and no mHC. [V] HF search; `S/configs/glm5_next.py` reuses `KimiLinearStateShape`.
- **A layer-truncated real checkpoint is the best option for performance work [I].** Layers 0-3 (3 KDA dense + 1 DSA-MoE) plus MTP 45 need shards 1, 2, 3, 17, 31, 32 and 62: about 37 GB to download, about 12 GB of weights. Do not try to load partial shards in place, because leftover tensors (for example layers 10, 19, 20, 29, 30) would hit the loader. Instead:
  1. Write a new safetensors file that keeps layers {0..3, optionally 4..7}.
  2. Rename `layers.45.*` to `layers.{N}.*`.
  3. Edit `num_hidden_layers`, `layer_types`, `mlp_layer_types`, `indexer_types`, and `linear_attn_config.kda_layers` / `full_attn_layers`.

  With 8 layers (2 DSA, 5 MoE) the weights are about 40 GB, which fits TP2 on 2x A100. Per-layer timings then scale linearly to 45 layers.

---

## 2. How the engines implement the hybrid, and sm_80 status

### 2.1 SGLang v0.5.20
- **Model code:** `S/models/glm5_next.py` has `Glm5NextLinearAttention`, which builds `RadixLinearAttention` (the KDA backend) and fuses the BF16 projections as `fused_qkvbfg_a_proj`. MLA and MoE are reused from `deepseek_v2`. `glm5_next_nextn.py` (MTP) subclasses DeepSeek NextN. [V]
- **KDA kernels:** `--linear-attn-backend` accepts triton (default), cutedsl, flashinfer, flashkda, nvidia_kda, ptx_kda and helion. `triton` is the portable FLA path, and it is the only backend that allows `extra_buffer` (`supports_mamba_cache_extra_buffer`: KDA archs require `linear_attn_backend == "triton"`). [V] `S/arg_groups/choices.py:194`, `S/arg_groups/overrides.py:503`. My expectation that triton FLA runs on sm_80 is [I].
- **DSA backend selection:** `_dsa_split_backend_resolution` works as follows:
  - With a bf16 KV cache it defaults to prefill `flashmla_sparse` and decode `fa3` (on sm<10). Both are Hopper-only.
  - The `triton` DSA backend is rejected on CUDA with "only supported on ROCm/HIP".
  - `tilelang` is valid only with a bf16 KV cache.
  - The cookbook recipe for H100/H200 is `--dsa-prefill-backend tilelang --dsa-decode-backend tilelang --kv-cache-dtype bfloat16`.

  [V] `S/arg_groups/overrides.py` (~L580-760); https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx
- **Indexer:** `S/layers/attention/dsa/dsa_indexer_kpool.py` stores FP8 index keys (132 B/token: 128 fp8 + scale) and calls `deep_gemm.fp8_paged_mqa_logits` / `fp8_mqa_logits`. The tilelang alternative is used only when `arch_major == 9`. `assert page_size == 64` ("DeepGEMM paged-MQA requires 64-token pages"). **Upstream has no sm_80 path, so the organizer image must patch this.** [V] code
- FP8 KV cannot be used with `index_kpool > 1` on CUDA (it excludes flashmla_kv), which costs 1.75-2x of KV capacity versus GLM-5.2. [V] https://github.com/sgl-project/sglang/issues/36830
- The dense-attention threshold is set automatically, so prefill uses dense MHA for KV lengths up to `index_topk` (2048) (`SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD`). [V] `S/arg_groups/model_hook.py:195-208`
- **FP8 weights on A100:** linear layers automatically use **FP8 Marlin (W8A16)** when `80 <= sm < 89` (`can_auto_enable_marlin_fp8`; also `SGLANG_FORCE_FP8_MARLIN`). MoE runners are auto, triton, marlin and others; `--moe-runner-backend marlin` exists. [V] `S/layers/quantization/fp8.py:495-501`, `fp8_utils.py:2159`. Which MoE runner the organizer image uses on sm_80, and how fast it is, is unknown. Measure triton against marlin. [I]
- `SGLANG_OPT_USE_TOPK_V2` (default True) is the TileLang/TVM JIT top-k. On A100 it must be 0 (organizer task.md L424). [V]
- **MTP:** use `--speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6` (cookbook "Low Latency"). `mamba_track_interval` must be at least num_draft_tokens. Open bug: NextN crashes at TP8 on the support branch (#37548); check whether it reproduces on v0.5.20. [V] cookbook; https://github.com/sgl-project/sglang/issues/37548
- H100 measurement (TP8/EP8, deep_gemm MoE, BF16 KV, tilelang DSA, MTP with simulated accept length 3), 8k input / 1k output:
  - concurrency 1: 212.6 tok/s.
  - concurrency 16: 1176 tok/s.
  - With HiCache, concurrency 16: TPOT 10.35 ms, TTFT 2.99 s.

  [V] `docs/src/snippets/configs/zai-org/glm-5.3-flash-benchmarks.jsx`

### 2.2 vLLM
- Main has `vllm/models/glm5next/{common,nvidia,amd}`. FlashKDA prefill is used only on SM 9/10/12, otherwise the Triton `chunk_kda`. [V] `glm5next/common/kda.py:136-150`
- No main-line sparse-MLA backend accepts sm_8x. On sm_89 every backend fails. [V] https://github.com/vllm-project/vllm/issues/54059. GLM-5 DSA on A800 had the DeepGEMM hard dependency with no fallback. [V] https://github.com/vllm-project/vllm/issues/35021
- An SM8x fallback, `_sparse_mla_fwd_with_sink_kernel` in `sparse_mla_kernels.py`, exists somewhere outside main (probably in backports like the organizer's). Measured on 2x A100-40GB PCIe, TP2, with host-to-device copies visible (possibly offload):
  - The kernel took 58.8% of CUDA time.
  - TTFT was 28.7 s at 4k, 38.5 s at 8k and 56 s at 16k.
  - It reads the latent once per head (about 64x redundant traffic) and reaches about 6% of HBM bandwidth.
  - **A roughly 10x fix is available by restructuring the grid (BLOCK_H).**

  [V] https://github.com/vllm-project/vllm/issues/57971
- Community SM80 kernels (Triton INT8 MLA KV, sparse MLA, MoE align, TP4 indexer 20.6 ms → 6.6 ms) exist as vLLM patches for GLM-5.3-Flash W4A16. [V] https://github.com/HalfVulpes/GLM-5.3-Kernels. There is also vLLM PP8 on 8x CMP 170HX (SM80). [V] https://github.com/promisezackr/glm53-flash-170hx-pp8
- On A100, the organizer's reference submission uses the sm_80 backport with TP8, `--max-num-seqs 16`, `--max-num-batched-tokens 8192`, prefix caching, and MTP with 3 speculative tokens. It is "not tuned". [V] `llm-challenge-arena-v1/task.md` L438-462

---

## 3. KV and state memory, and prefix reuse granularity

### 3.1 Per-token and per-request footprint [I, computed from config and code]
- **MLA latent:** 512 × bf16 = 1024 B per layer per token. 11 layers give 11.3 KB/token, and 12.3 KB with the MTP draft layer's pool. **This is replicated on every attention-TP rank** (single latent KV head). [I, standard SGLang MLA behaviour]
- **Indexer K:** 132 B per token per layer (fp8 128 + fp32 scale; `head_dim_with_sf = 132`), about 1.45 KB/token. `index_kpool 4` compression may cut this toward 0.36 KB. [V] layout: `dsa_indexer_kpool.py`; compression effect [I]
- **Total ≈ 13-14 KB/token/GPU at TP8, identical on all 8 GPUs.**
- **KDA recurrent state:**
  - Size: 34 layers × 64 heads × 128 × 128 × fp32 = 142.6 MB. The conv state (3 × 8192 channels × 3 taps × bf16 × 34) adds 5 MB, for **about 148 MB per request slot**, divided by attn_tp_size (TP8: 18.5 MB per GPU).
  - Dtype: fp32 by default (`mamba2_state_dtype`, overridable with `--mamba-ssm-dtype` or `SGLANG_MAMBA_SSM_DTYPE`).
  - `--enable-int8-mamba-checkpoint` stores cached states as int8, for about 2x more cached states. It is incompatible with HiCache.

  [V] `S/configs/mamba_utils.py:47-125`, `S/arg_groups/fields/exec_.py:345-395`, `S/arg_groups/mamba_hook.py`
- **Budget at TP8, per GPU [I]:** weights are about 41-42 GB. With `mem_fraction_static` ≈ 0.88 that leaves about 28 GB, which is **about 2.0M tokens** of KV shared by all sessions (for example 18 sessions × about 110k). KDA slots cost roughly 1 active + 2 ping-pong + cached checkpoints ≈ 18.5 MB each, so they are small.
- The SGLang cookbook's sizing flags are:
  - `--mamba-full-memory-ratio` (default 0.9; state pool bytes / full KV pool bytes).
  - `--max-mamba-cache-size`.
  - Too small a state pool clamps `max_running_requests`.

  [V] cookbook "Size both memory pools"; `S/arg_groups/fields/schedule.py:199-207`

### 3.2 How the SGLang hybrid radix cache works (v0.5.20)
- **The default tree is `UnifiedRadixCache` with `MambaComponent`.** The legacy `MambaRadixCache` has the same semantics. [V] `S/mem_cache/registry.py:80-147`, `S/mem_cache/unified_cache/components/mamba.py:156-186`
- **Strategy:** `auto` picks `extra_buffer` when page_size > 1 or overlap scheduling is on. `no_buffer` asserts page_size == 1, and DSA forces 64, so **only extra_buffer or extra_buffer_lazy are possible for this model.** [V] `S/arg_groups/overrides.py:545-553`, `S/arg_groups/mamba_hook.py:128-133`
- **Grid:** `mamba_cache_chunk_size = max(FLA chunk 64, page_size 64) = 64`. Decode checkpoints happen at `seq_len % lcm(64, --mamba-track-interval=256) == 0`. [V] `S/arg_groups/overrides.py:1896-1921`, `S/managers/schedule_batch.py:3482-3514`
- **Match:** KV is truncated to the deepest node that holds a KDA state. If more KV matched beyond that, `mamba_branching_seqlen = floor64(full KV hit)` is returned. [V] `mamba_radix_cache.py:1105-1224`
- **Extend checkpoint: one per request per forward** (`_mamba_radix_cache_v2_req_prepare_for_extend`).
  - By default the checkpoint is at `prefix + floor64(extend_len)`.
  - If the branching point lies inside this extend, **it replaces** the end checkpoint (`mamba_track_seqlen = _force_track_h(req.mamba_branching_seqlen)`).
  - After prefill, `cache_unfinished_req` inserts KV only up to `mamba_last_track_seqlen`.
  - At finish, `cache_finished_req` inserts up to the last decode track point and frees any KV beyond it.
  - Each chunked-prefill chunk (default `chunked_prefill_size` = 8192 on 80 GB GPUs) produces its own checkpoint.

  [V] `S/managers/schedule_batch.py:2893-2990`, `mamba_radix_cache.py:546-822`, `batch_result_processor.py:372-383`, `S/arg_groups/memory_hook.py`
- Related upstream work:
  - #22326 (checkpoint proposal: interval checkpoints during prefill) was closed without the prefill interval being implemented.
  - #36935 (open): branch reuse collapses to a 0 hit when the number of chunk checkpoints exceeds `max_mamba_cache_size`, because only the last node is refreshed in the LRU.
  - #38000 (open): thin cached states by coverage instead of evicting the LRU tail.
  - #38018 (open): Prometheus counters `sglang:mamba_cache_miss_tokens_total`.
  - #37198 (open): an env var to store the first checkpoint at a hit-ratio fraction of the length.
  - #38625 (open): interior checkpoint for no_buffer.
  - #22935 (open): split-at-n-1 can produce a 0 hit.

  [V] https://github.com/sgl-project/sglang/issues/22326, /issues/36935, /pull/38000, /pull/38018, /pull/37198, /pull/38625, /issues/22935. **No merged flag adds role/message-boundary or "K tokens before the tail" checkpoints.** [V, from GitHub search 2026-09-22]

### 3.3 What our workload actually gets [I, derived from the code above; verify with the #38018 counters or `cached_tokens`]
Definitions: turn k has a prompt of length L_k. Turn k+1 shares its prefix with turn k up to L_k − d (d = 340-1400 in 57% of cases).
- Turn k itself usually matched KV beyond its KDA state, because the previous turn's tail was rewritten or because turn k−1 inserted KV up to its decode track point. So turn k checkpointed at its branch point b_k = floor64(L_{k−1} − d_{k−1}), **not** at L_k.
- Turn k's decode checkpoint lands in the generated output, which the replayed next prompt never contains, so it is useless.
- Turn k+1 matches KV up to about L_k − d. The deepest KDA state at or before that is **b_k**, so the prefix hit is b_k.
  - Uncached tokens ≈ (L_{k+1} − L_k) + (L_k − L_{k−1}) + d_{k−1}.
  - Ideal would be (L_{k+1} − L_k) + d_k.
  - **This is about one extra turn of growth per request**, and it also applies to the 43% of non-rewrite turns that follow a rewrite turn.
- An exception: if the extend exceeds 8192 tokens, chunk-end checkpoints also exist.
- Consequences: roughly 2x prefill tokens for mid-chain requests, and a "≤4096 uncached" request often becomes 5-10k. This hits the 3 s p95 bucket directly.
- **Cheap fix (patch the scheduler or PrefillAdder):**
  1. When `mamba_branching_seqlen` falls inside the extend, cut the first chunk exactly at the branching point. It is checkpointed and inserted through `cache_unfinished_req(chunked=True)`, and the remainder gets the end checkpoint.
  2. Additionally cut at floor64(L − 1536) for every prompt, which catches the system-reminder rewrite.

  Cost: 1-2 extra forward launches and about 18.5 MB of state per checkpoint per GPU. FLA already computes h at every 64-token chunk, so extracting the states is only a copy. An alternative is to generalise the track entry to 2-3 positions per extend (the ping-pong buffer already has 2 slots). Watch the eviction issue from #36935.
- vLLM equivalent: `mamba_cache_mode = "align"` (the default with prefix caching) caches the state at the last token of each scheduler step when it lands on a block boundary. `"all"` caches at every `i*block_size`, which gives full granularity at a large memory cost. `enable_mamba_shared_prefix_checkpoint` covers the EAGLE junction case only. [V] `vllm/config/cache.py:166-192`

### 3.4 Flags summary
The relevant knobs are:
- `--mamba-radix-cache-strategy {auto,extra_buffer,extra_buffer_lazy}`
- `--mamba-track-interval` (256)
- `--mamba-ssm-dtype`
- `--max-mamba-cache-size`
- `--mamba-full-memory-ratio` (0.9)
- `--mamba-max-states-per-path`
- `--enable-int8-mamba-checkpoint` / `--int8-mamba-ckpt-size`
- `--linear-attn-{prefill,decode,verify}-backend`
- `--enable-linear-replayssm`. It requires no_buffer, so it is unusable here, and it is usually slower for KDA.

Things to avoid in v0.5.20: `--enable-hierarchical-cache` (DSA index not restored, #39156) and `--enable-mixed-chunk` (#39526). [V] `S/arg_groups/fields/exec_.py:313-455`; https://github.com/sgl-project/sglang/pull/39156, /pull/39526

---

## 4. DP-attention and EP on 8x A100
- **Supported:** SGLang CI runs GLM-5.3-Flash **TP8/EP8/DP8 with DP attention** (DeepGEMM + DeepEP, H200) and passes GSM8K at 94%. [V] https://github.com/sgl-project/sglang/pull/39350
- The cookbook exposes DP-Attention 1-8 and EP 2-8. **"MTP speculative decoding with DP-Attention is not validated on this model"**, and DFlash is also disallowed with DP-attention. [V] cookbook jsx
- `--enable-dp-attention` is silently reset when `--dp-size 1`, so always pass `--dp-size`. [V] https://github.com/sgl-project/sglang/issues/36840
- Two-batch overlap (TBO) crashes on the mamba fields. [V] #36840
- Hybrid-SSM DP issues:
  - `max_running_requests // dp` must be at least 1.
  - An assertion fires when one rank decodes while a peer prefills.

  [V] https://github.com/sgl-project/sglang/pull/34535 (open)
- KDA fused projections are not fused under attn-TP ≠ TP before PR #39350 (open). [V]
- **Benefit estimate for DP8 [I]:**
  - The MLA/indexer KV is no longer replicated. Each rank stores only its own requests. About 18 GB free per rank (after about 10 GB of extra replicated KDA and MLA weights) holds about 1.3M tokens per rank, **about 10M in total, versus about 2M at TP8**.
  - KDA state per slot grows to 148 MB, but on fewer requests per rank.
  - The price: prefill of a single long request uses only one GPU's attention compute. The MoE is still all-gathered, or all-to-all with EP.
  - DeepEP/NVSHMEM-style all-to-all on A100 is untested here.
  - A middle ground is `--dp-size 2` or `4` (attn-TP 4 or 2), which cuts replication 2-4x.
- EP-MoE (`--ep-size 8`) is the H100 recipe (with `deep_gemm`, which is Hopper-only). On A100 the MoE kernel must be triton or marlin. Whether EP beats TP-MoE there is unmeasured. [V] recipe / [I]

---

## 5. Suggested next checks
1. Pull the organizer SGLang image on the GPU box (disk is tight; use docker's root on a different volume). Run `pip show sglang` and diff it against v0.5.20 to find the sm_80 DSA and indexer implementation. Check its prefill attention speed against vLLM #57971.
2. Enable the #38018-style logging, or compare `cached_tokens` with the prefix length expected from the trace, to confirm the §3.3 loss.
3. Try `--dp-size 2/4/8 --enable-dp-attention` against TP8 for KV capacity, without MTP first.
