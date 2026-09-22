# R2 — Serving techniques to push N@SLO beyond 18 (GLM-5.3-Flash, 8×A100-80GB)

Tags: **[V]** = checked against the source (the local code or the linked page). **[I]** = my inference or an estimate. I name local code as `SGL/…`, meaning `/workspace/Agentic_science_challenge/src/sglang/python/sglang/srt/…`. That checkout is SGLang **v0.5.20** (release branch, commit `94602c9`, 2026-09-18), which should match the `arena-sglang-glm53:260918` base image **[I: same date; confirm with `pip show sglang` inside the image]**. The upstream `main` I also checked is commit `56fee88` (2026-09-21).

## 0. What actually limits N (my working model)

- **KV capacity is the binding resource.** MLA KV under TP8 is *replicated* on every rank. GLM-5.3-Flash has 11 DSA/MLA layers with `kv_lora_rank=512` and `qk_rope_head_dim=0`, plus a 128-dim indexer key (from `s1-dev/glm_tok/config.json`) **[V]**. That works out to roughly 11–13 KB/token/GPU in BF16 KV **[I]**. FP8 KV + TRT-LLM DSA is Blackwell-only, and TileLang DSA + FP8 KV is "not a valid CUDA combination", so A100 has to use BF16 KV **[V: SGL docs `docs/cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx`]**. With about 20 GB/GPU left for pools, the device holds roughly 1.5M tokens **[I]**. At N=18 with contexts of 100–250k, the live working set is 2–3M tokens, so sessions evict each other during their 2–35 s think gaps. When that happens, a "≤4096-uncached" mid-chain request (the gate class is data-defined through `uncached_expected`, `s1-dev/harness/s1_score.py`) actually turns into a 100k+ recompute and fails the 3 s gate **[I]**.
- **The KDA state pool is a second capacity limit.** Each running request holds S = 3–5 state slots, and each cached prefix needs a Mamba checkpoint at its frontier. A radix prefix is only usable up to the last node that has a Mamba state (`SGL/mem_cache/unified_cache/components/mamba.py: create_match_validator`, `finalize_match_result_in_tree_core`) **[V]**. The per-slot state is 34 layers × (64/8 heads) × 128×128 × fp32 ≈ 17 MiB/GPU **[I]**.
- **Head-of-line blocking in chunked prefill.** In v0.5.20, a chunked request in progress takes the *entire* chunk budget every round (`SGL/managers/schedule_policy.py: add_chunked_req`, `_rem_tokens = min(rem_chunk_tokens, rem_total_tokens)`) **[V]**. So a cold 85k–250k prefill blocks every short mid-chain request for the whole prefill, which takes seconds to tens of seconds on A100. This fits the competitors' pattern: `overall_intra` p95 of 3.5–5.2 s is the binding gate **[I]**.

## 1. Scheduling

### 1a. SGLang v0.5.20 (what exists in the base image)
- `--schedule-policy {lpm,random,fcfs,dfs-weight,lof,priority,routing-key,hrrn}`, default `fcfs` (`SGL/arg_groups/fields/schedule.py`) **[V]**.
  - `hrrn` sorts the waiting queue by `waited_prefill_tokens / uncached_len`, and requests with zero uncached tokens go first. It falls back to FCFS when there are more than 128 waiting requests (`SGL/managers/schedule_policy.py: _sort_by_hrrn`, `_determine_active_policy`) **[V]**. It only reorders the *waiting* queue and does not pre-empt a running chunk **[V: see add_chunked_req above]**.
  - `lpm` sorts by longest cached prefix, which is close to "short uncached first" for this workload, also with the 128-request FCFS fallback **[V]**.
- Priority scheduling: `--enable-priority-scheduling`, `--schedule-low-priority-values-first`, `--priority-scheduling-preemption-threshold` (default 10), `--default-priority-value`, `--disable-priority-preemption`, and `--retraction-policy {length,priority}` **[V: schedule.py]**. It **requires `--schedule-policy fcfs` or `lof`** (`SGL/arg_groups/validation_hook.py:172`) **[V]**. The priority is a per-request field `priority` in `GenerateReqInput` (`SGL/managers/io_struct.py:311`) **[V]**. The harness never sends it (`s1-dev/harness/s1_loadgen.py: call_engine`) **[V]**, so we would need a server-side shim that assigns priority. Background: roadmap issue https://github.com/sgl-project/sglang/issues/13526 and the excessive-preemption bug https://github.com/sgl-project/sglang/issues/12493 **[V: titles]**.
- Other knobs: `--chunked-prefill-size`, `--max-prefill-tokens` (default 16384), `--prefill-max-requests`, `--prefill-decode-interval` ("number of decode rounds after a prefill batch before the next prefill"), `--enable-mixed-chunk`, `--num-continuous-decode-steps`, `--schedule-conservativeness`, and `--enable-dynamic-chunking` (PP only) **[V: schedule.py]**.
  - `--enable-mixed-chunk` is kept with EAGLE/EAGLE3 speculation (`supports_mixed_chunk`, `SGL/speculative/spec_info.py:139`) and forced off for UNO, frozen-KV MTP and ngram **[V]**.
  - `prefill_decode_interval` is a direct lever for TPOT vs TTFT **[V flag; effect I]**.
- Chunk size tradeoff: a GLM-5 study of long-context agent traffic (28–30k input) found `chunked-prefill-size=3072` beat 2048 and 4096 (TTFT −25.5%, throughput +10.7%) https://arxiv.org/abs/2607.02518 **[V: abstract page]**. For GLM-5.3-Flash with breakable prefill CUDA graphs, SGLang defaults the chunk to 4096 (`SGL/arg_groups/cuda_graph_hook.py: _apply_glm5_chunked_prefill_default`) **[V]**. Mamba extra_buffer warns if the chunk is smaller than `mamba_cache_chunk_size`, because states are then not checkpointed at chunk handoff (`SGL/arg_groups/mamba_hook.py`) **[V]**.

### 1b. Upstream fixes for exactly our head-of-line problem (not in v0.5.20)
- **PR #40024 "[Scheduler] Add shortest-prefill-first scheduling"**, merged 2026-09-18 (commit 65ef55e): https://github.com/sgl-project/sglang/pull/40024 **[V]**. It adds `--schedule-policy shortest-prefill-first`. The queue is sorted by remaining *uncached* tokens, and `shortest_prefill_chunk_limit()` **caps the active long chunk** so that waiting shorter prefills fit into the same pass (`sglang main: managers/schedule_policy.py`, `scheduler.py:3914 adder.chunked_req_limit`) **[V: main source]**. It is **absent from v0.5.20** **[V: grep]**, so it would need a backport. The PR gives no benchmark numbers **[V]**.
- **PR #39717 "prefill interleaving controls + HRRN"** (still open): https://github.com/sgl-project/sglang/pull/39717. It adds `--enable-prefill-interleaving` and `--prefill-interleaving-min-continuation-tokens`. Reported A/B: short-request TTFT 2.80→1.50 s (−46%). Reported canary: TTFT p50 14.29→2.29 s, p99 180→92 s **[V: PR page]**.
- Background: SRPT-style scheduling can starve long requests, see https://arxiv.org/pdf/2606.09061 **[V: search snippet]**. Here the chain-start gate is 30 s (with slack), so we can afford to starve cold prefills a little; the continuation reserve still keeps them progressing **[I]**.

### 1c. PD multiplexing
- `--enable-pdmux`, `--pdmux-config-path`, `--sm-group-num` (default 8) **[V: `SGL/arg_groups/fields/disagg.py`]**. It asserts `chunked_prefill_size == -1`, `disable_overlap_schedule`, and no PP/disaggregation, and it warns about degradation on torch ≥ 2.7 (`SGL/arg_groups/validation_hook.py:140`) **[V]**. It is also incompatible with UNO speculation **[V]**.
- The blog reports up to 3.06× goodput vs chunked prefill for Llama-3.1-70B on 8×A100. It lists MoE as preliminary and MTP as unsupported: https://www.lmsys.org/blog/2025-09-28-pdmux/ **[V]**.
- Verdict: running 250k-token prefills unchunked on a 30 GB/GPU budget, with MoE + MTP + KDA, is not viable **[I]**.

### 1d. vLLM equivalents
- vLLM main (`c723a83`) has `--scheduling-policy {fcfs,priority}` (lower value = earlier), `--long-prefill-token-threshold` (default 0 = off), `max_num_active_seqs`, and `max_num_queued_tokens` (TTFT QoS reject) **[V: `vllm/config/scheduler.py`]**.
- `max_num_partial_prefills` / `max_long_partial_prefills` **no longer exist** on main **[V: grep]**.
- Hybrid models need `mamba_cache_mode` = `align`/`all` for prefix caching **[V: `vllm/config/cache.py`]**.
- CPU offload uses `kv_offloading_size` + `kv_offloading_backend {native,lmcache}` **[V]**. The LMCache hybrid docs require align mode + `--separate-object-groups` for KDA: https://docs.lmcache.ai/mp/hybrid_models.html **[V: snippet]**. vLLM notes that hybrid external offload needs extra work: https://github.com/vllm-project/vllm/issues/38230 **[V: title]**.
- The organizer's vLLM backport ran with `--max-num-seqs 16 --max-num-batched-tokens 8192 --disable-custom-all-reduce`, MTP 3, and `cudagraph_capture_sizes [1..16]` (`llm-challenge-arena-v1/task.md`) **[V]**.

## 2. Keeping the prefix cache

### 2a. HiCache on v0.5.20 (L1 GPU + L2 host)
- Flags: `--enable-hierarchical-cache`, `--hicache-host-memory-mode {cache,buffer_only}`, `--hicache-ratio` (default 2.0), `--hicache-size` (GB, overrides the ratio), `--hicache-write-policy {write_back,write_through,write_through_selective}` (default write_through), `--hicache-io-backend {direct,kernel}` (default kernel), and `--hicache-mem-layout {layer_first,page_first,page_first_direct,page_first_kv_split,page_head}` (default page_first) **[V: `SGL/arg_groups/fields/memory.py`]**.
- Hybrid support: for this model, the default tree is `UnifiedRadixCache` with FULL + MAMBA components. `init_hicache` attaches host pools, and the MAMBA component has a host LRU and load-back (`SGL/mem_cache/registry.py: _create_unified_radix_cache`; `components/mamba.py: build_hicache_transfers`, `drive_host_eviction`) **[V]**. The cookbook offers "L1+L2: `--enable-hierarchical-cache --hicache-size 32`", marked Not Verified for GLM-5.3-Flash. On H100 low-latency with L1+L2 they measured 1206 vs 1176 tok/s at c16, TTFT 2.99 s, TPOT 10.35 ms (random 8k/1k, not multi-turn) **[V: `docs/src/snippets/configs/zai-org/glm-5.3-flash*.jsx`]**.
- Why it matters here: reloading 100k tokens × ~12 KB ≈ 1.2 GB/GPU over PCIe4 (~20 GB/s) takes about 60 ms. Recomputing the same 100k tokens takes several seconds **[I]**. The HiCache blog reports up to 6× throughput and −80% TTFT, with layer-wise load/compute overlap: https://www.lmsys.org/blog/2025-09-10-sglang-hicache/ **[V]**. Mooncake/Kimi report the same "trade storage for compute" result at scale (+115% requests on A800): https://arxiv.org/abs/2407.00079 **[V: snippet]**.
- **Risks:**
  1. Open bug https://github.com/sgl-project/sglang/issues/39830 (2026-09-16, main): on a GDN hybrid, **host-tier restores give wrong output 20/20**, while device hits are fine **[V]**. KDA is not named. Our replay uses greedy decoding with ignore_eos, so timing is unaffected, but the aime/gpqa capability gate (>90) could be hit if host restores occur during eval **[I]**. Test accuracy with HiCache on.
  2. https://github.com/sgl-project/sglang/issues/29034: `--hicache-size` over-allocates the Mamba host pool about 15×, so size host RAM carefully **[V: title/snippet]**.
  3. https://github.com/sgl-project/sglang/issues/24121: HiMambaRadixCache crash (older) **[V: title]**.
  4. The code notes that "incremental persistence of a new branching state is currently write-through only; write-back eviction may discard the device-only state" (`components/mamba.py:166`) **[V]**. **Use write_through.**
- **flush_cache semantics:** `Scheduler.flush_cache` → `tree_cache.reset()` → `UnifiedRadixCache._reset_full()` calls `cache_controller.reset()` **and `mem_pool_host.clear()`**. It only runs when the scheduler is fully idle, otherwise it returns success=False (`SGL/managers/scheduler.py:4981`, `SGL/mem_cache/unified_radix_cache.py:373`) **[V]**. So the L2 host tier really is cleared. An **L3** storage backend is *not* cleared by flush; that needs `clear_hicache_storage` (`SGL/managers/tokenizer_control_mixin.py`) **[V]**. So stay L2-only, or clear L3 in a `/flush_cache` wrapper.

### 2b. Mamba-state retention knobs (v0.5.20)
- `--mamba-radix-cache-strategy {auto,no_buffer,extra_buffer,extra_buffer_lazy}`. Auto selects extra_buffer for Glm5Next when overlap or paging is on (`SGL/arg_groups/overrides.py:545`, `_MAMBA_EXTRA_BUFFER_ARCHS`) **[V]**.
- `--mamba-track-interval` (default 256), `--mamba-full-memory-ratio` (fallback 0.9), `--max-mamba-cache-size`, and `--mamba-ssm-dtype {float32,bfloat16,float16}` **[V: `SGL/arg_groups/fields/exec_.py`, `schedule.py`]**.
- **`--mamba-max-states-per-path N`**: the shallowest interior states beyond N are dropped while their full KV stays. Default −1 = unlimited **[V]**. In multi-turn traffic, each turn leaves an interior checkpoint that will never be reused, so a cap of 1–2 should free state slots for more sessions **[I]**.
- `--enable-int8-mamba-checkpoint` gives about 2× cached-state capacity, but it is **incompatible with `--enable-hierarchical-cache`** (`SGL/arg_groups/mamba_hook.py`) **[V]**. It is also a precision risk **[I]**.
- `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1` lowers the per-request slot count by one (S: 5→4 on extra_buffer with overlap; 4→3 on lazy) (`SGL/environ.py:1375`, `mem_cache/kv_cache_configurator.py:_calculate_mamba_ratio`) **[V]**.
- Formula from the repo skill: `r* = (S+D)·token_equiv/L`, with token_equiv = state_bytes/kv_bytes_per_token read from the boot log (`src/sglang/.claude/skills/compute-mamba-ratio/SKILL.md`) **[V]**. Also on the ratio, the cookbook says: "too low starves the KDA state pool and clamps max_running_requests; too high shrinks the KV pool" **[V]**.

### 2c. Session-aware eviction
- `--enable-session-radix-cache`: "Track per-session references on UnifiedRadixCache KV: eviction consumes unreferenced entries before referenced ones" (`SGL/arg_groups/fields/memory.py:90`) **[V]**. It needs a `session_id` in the /generate body; the field is documented as "does not alter or reconstruct the prompt" (`io_struct.py:180`) **[V]**. The harness sends only an `X-S1-Session-ID` *header* **[V]**, and SGLang's http_server does not map it **[V: grep]**. So we would need a small middleware that copies the header into `session_id`. The design is described at https://www.lmsys.org/blog/2026-08-11-unified-radix-cache/ (Kimi-K3 = FULL+MAMBA, `/close_session`, session-ref-ordered eviction) **[V]**.
- `--radix-eviction-policy {lru,lfu,slru,priority}` **[V: memory.py; evict_policy.py]**.
- Related prior work: Continuum pins KV with a TTL across tool calls and reports large JCT gains on SWE-bench/BFCL: https://arxiv.org/abs/2511.02230 **[V: snippet]**. TRT-LLM priority/duration retention gives about +20% hit rate: https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/ **[V: snippet]**.

## 3. Parallelism on one node
- **TP8+EP8** is the only verified Hopper recipe (`--tp-size 8 --ep-size 8 --moe-runner-backend deep_gemm`, mem-frac 0.70–0.75, BF16 KV + TileLang DSA) **[V: glm-5.3-flash.jsx]**. DeepGEMM is SM90+, so A100 needs a Triton/Marlin FP8-W8A16 MoE path **[I]**.
- **DP-attention** (`--enable-dp-attention`, `--load-balance-method {auto,round_robin,follow_bootstrap_room,total_requests,total_tokens}`) **[V: `SGL/arg_groups/fields/parallel.py`]**. The upside is that it removes the 8× MLA KV replication, which would multiply KV capacity **[I]**. The downsides:
  - The cookbook says "MTP with DP-Attention is not validated", and DFLASH rejects DP-attention **[V]**.
  - There is no session-affinity load balancer; `routed_dp_rank` exists per request (`io_struct.py:286`) **[V]**, so a proxy would have to pin each session to a rank.
  - KDA state goes from TP-sharded to a full per-rank copy (8× more state bytes per request per GPU) **[I]**.
  - A cold 250k prefill would run attention on a single GPU **[I]**.
  Overall this is a high-effort, high-variance bet.
- **DeepEP**: the current README lists "Hopper (SM90) … or other architectures with SM90 PTX ISA" (https://github.com/deepseek-ai/DeepEP/blob/main/README.md) **[V]**, so it is not an option on A100. That also rules out `--enable-two-batch-overlap`, which depends on an a2a backend **[I]**.
- **Single-node PD disaggregation** needs 2 copies of about 328 GB of weights, which does not fit in 640 GB with any KV left, and the model will not fit on 4×80 GB at all **[I]**. The cookbook's PD path is "dummy weights only", without speculation **[V]**. Not an option.
- **DCP** (`--dcp-size 4`) is validated only on GB300 **[V]**. In principle it shards KV to fix the replication problem, which makes it worth a probe as `--dcp-size 8` if it runs on sm80 **[I]**.

## 4. Decode speed (TPOT, the secondary rank)
- The cookbook's low-latency recipe is MTP via `--speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6` **[V]**. With speculation on, SGLang forces `max_running_requests` to 48 unless you set it (`SGL/arg_groups/speculative_hook.py`) **[V]**. Each draft token adds to the per-request state slots (D in the formula) **[V: skill]**, which trades against session capacity. Try steps 3/draft 4, as in the encoder-disagg recipe **[V]**, to balance state memory **[I]**.
- CUDA graphs: `--cuda-graph-max-bs-decode` (the recipes use 32), `--cuda-graph-bs-decode`, and `--cuda-graph-backend-prefill breakable` (PR #38522) **[V]**. N≈22–26 needs a decode graph of at least N **[I]**.
- All-reduce:
  - The custom all-reduce is on by default (`--disable-custom-all-reduce` default False) **[V]**.
  - `--enable-torch-symm-mem` is SM90+ only **[V: exec_.py:574]**.
  - `--enable-symm-mem` (NCCL symmetric memory) is available **[V]**.
  - The organizer's vLLM run disabled custom all-reduce on A100 **[V]**, so A/B test it on SGLang.
- `--enable-torch-compile` is experimental **[V]**. `--linear-attn-decode-backend` and `--enable-linear-replayssm` are "typically slower for KDA" **[V: exec_.py help]**.
- Required on A100: `SGLANG_OPT_USE_TOPK_V2=0` (task.md) **[V]**.

## 5. Case studies
- LMSYS HiCache agent/coding case: Qwen3-Coder-480B coding agent with about 25k tokens × 8 turns; TTFT −56%, hit rate 40→80% (link above) **[V]**.
- Mooncake + SGLang HiCache, −84% TTFT (Chinese write-up): https://zhuanlan.zhihu.com/p/1959366095443064318 **[V: title]**.
- FlexKV × SGLang multi-agent cache: https://zhuanlan.zhihu.com/p/2047357501037864148 **[V: title]**.
- GLM-5 OpenClaw tuning (3072 chunk / max-running 24): https://arxiv.org/abs/2607.02518 **[V]**.

## Ranked candidate experiments
1. **Backport PR #40024 (shortest-prefill-first + chunk cap)** onto v0.5.20, or write an equivalent patch to `add_chunked_req`/`calc_priority`. Run with `--schedule-policy shortest-prefill-first --chunked-prefill-size 4096`. Target: cut fast_intra/overall_intra p95. This is the most direct fix for head-of-line blocking.
2. **HiCache L2**: `--enable-hierarchical-cache --hicache-size <≈100–200 per rank, after checking the Mamba host-pool over-allocation> --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first`. Verify accuracy (#39830), and verify `cached_tokens` goes to 0 after `/flush_cache`. Target: keep sessions cached through think gaps at N≥22.
3. **Mamba pool tuning**: `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1`, `--mamba-radix-cache-strategy extra_buffer_lazy`, `--mamba-max-states-per-path 2`, and a `--mamba-full-memory-ratio` computed from boot-log bytes (L≈150k). Raise `--mem-fraction-static` as far as graph capture allows.
4. **Session-aware eviction**: middleware that copies `X-S1-Session-ID` into the body `session_id`, plus `--enable-session-radix-cache`.
5. **Native fallback without a patch**: `--schedule-policy hrrn` (or `lpm`) plus `--chunked-prefill-size 2048–3072`, with `--prefill-decode-interval` swept 0–2 to trade TPOT against TTFT.
6. **Priority shim**: a proxy estimates uncached tokens per session (current prompt length minus the previous prompt length when the prefix matches) and sets `priority`, used with `--enable-priority-scheduling --schedule-policy fcfs --disable-priority-preemption`.
7. **Speculation depth sweep**: MTP 3/1/4 vs 5/1/6 vs off, with `--cuda-graph-max-bs-decode 32`. Also A/B the custom all-reduce vs `--enable-symm-mem`.
8. **Long shots**: DCP8 on sm80 (KV sharding), and DP-attention with session pinning through `routed_dp_rank`.
9. **Do not pursue**: PDMux, single-node PD disaggregation, DeepEP/TBO on A100, L3 storage (flush does not clear it).
