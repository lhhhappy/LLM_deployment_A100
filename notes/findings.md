# Findings log (arena / GLM-5.3-Flash on 8×A100)

> **当前事实基线（2026-09-22 清理后）**
> - 底包 SGLang = GitHub 公开提交 `fe236ea6c3` + 两处多模态修复（F53）；完整逐字节副本 `build/base_exact/`（4686 文件，指纹全对）。**不是 v0.5.20**：v0.5.20 线（001/002/004、SPF）不适用于底包。
> - 提交线：000（接口合规）+ 101（D1 v1.2 底包版）；镜像 0922e/0922f，正式 45734/45735 待出分。
> - 仍成立的核心：F3（reminder 分叉）、F4（长冷预填充队头阻塞）、F9（flush 须 JSON 且全 worker）、F10、F23、F31（正式集 808/9023/65）、F40/F35（前排卡 TTFT 而非 TPOT；N=22 已有）、F45（数值门未过）、F46（chain_start 统计余量：808 条允许约 51 条超 30s）。
> - 以下 F36/F37/F42/F43/F47 为 v0.5.20 替身上的 L1 实测，只证明机制方向，不代表底包数值。
> - 已归档（v0.5.20 线 / 模拟器 / 已被取代）：`notes/archive/findings-superseded.md`。

## F1 — Model is a hybrid, not plain MLA (from glm_tok/config.json)
- 45 layers: 34 KDA linear-attention + 11 MLA/DSA (kv_lora 512, rope dim 0, indexer 32h×128, topk 2048); MoE 288e top-8; FP8 block weights ≈328 GB; 1 MTP layer.
- MLA latent KV ≈ 11 layers × 512 × bf16 ≈ 11 KB/token (+ indexer keys) — replicated per GPU under TP8 → ~1.5M tokens capacity/GPU (R2 estimate).
- KDA state per request ≈ 34×64×128×128×fp32 ≈ 142 MB total ≈ 17.8 MB/GPU under TP8.

## F2 — Workload shape (dev set; hidden-split chain metadata is in chains.jsonl)
- Hidden-split chains (178): 4030 requests, 94% intra; chains up to 240 requests; prompts up to ~250k tokens.
- Loadgen is closed-loop: N worker threads, each replays chains sequentially, sleeps replay_gap, sends next only after previous finishes.
- Gates: fast_intra uses FROZEN uncached_expected ≤ 4096 (not measured misses) → any real cache miss on those requests hits the 3 s gate hard.

## F3 — The "reminder" divergence (KEY)
- 57% of intra requests are `append-only / reminder-replaced`: each prompt ends with
  `<|user|><system-reminder>…<runtime-injection-bundle>…` which the next turn REMOVES and replaces with
  `<|assistant|>…tool calls…<|observation|>…` + a new reminder.
- Token LCP ends exactly at the `<|user|>` that opens the trailing reminder; 340–1400 tokens before prev prompt end (p50 ≈ 340).
- Verified on 235 pairs: re-tokenized LCP == frozen glm_lcp_with_prev in 230/235.
- Implication for hybrid (KDA) prefix cache: a hit needs a saved KDA state AT the LCP. SGLang v0.5.20
  extra_buffer tracks one state per extend at the (64-aligned) extend end, plus a branching point.
  The end-of-prompt state is useless for the next turn in these cases → fallback to an older checkpoint → actual uncached ≫ expected.
- Candidate fix (general, legitimate): also checkpoint KDA state at the last user/observation role boundary of each prompt (split the extend at that position so the chunk end is tracked). Must keep end-of-prompt checkpoint too (strict append-only edges need it).
- TO MEASURE: report field `uncached_actual_vs_expected_rel_dev` in run_dev output.

## F4 — Scheduler: chunked long prefill starves short requests (R2)
- v0.5.20 add_chunked_req takes the whole chunk budget; short intra requests wait. Upstream PR #40024 (shortest-prefill-first, merged 2026-09-18, after v0.5.20) / #39717 (open, interleaving).

## F6 — R1 confirmations (research/claude/R1_model_and_engines.md)
- DSA forces page_size=64 → only `extra_buffer` mamba strategy usable; one KDA state saved per extend (branch point takes priority over prompt end) — consistent with F3 (V, code).
- No upstream sm_80 path for DSA sparse attention / indexer → organizer images must carry private patches (V/I). Inspect base image first.
- HiCache unsafe in v0.5.20 for DSA (#39156); `--enable-mixed-chunk` corrupts mamba checkpoints (#39526).
- vLLM main registers Glm5Next (PR #53906, merged 2026-09-03) — supersedes my earlier claim that vLLM main lacks it.

## F8 — Codex: exact baseline pool accounting (refines F1 / R1 / R2 estimates)
- VERIFIED dimension calculation for fixed v0.5.20 / frozen config, no MTP or tail/scratch: BF16 DSA pool = 11×512×2 + 11×132 = **12,716 B/token/attention rank**. 128 Ki tokens ≈1.552 GiB/rank.
- `index_kpool=4` does not divide the current physical indexer allocation by four: `DSATokenToKVPool.index_buf_size` defaults to token pool size (`memory_pool.py:4849`); `IndexKeyCache._buffer_shape` allocates 132 B per physical token slot per DSA layer.
- FP32 recurrent + BF16 conv, one KDA slot across 34 layers: TP8/DP1 =18,452,480 B/rank (17.598 MiB). TP8/DP2/4/8 multiplies per-slot per-rank cost by 2/4/8, while requests are distributed across DP groups. Full-request slot count and all pool budgets must be included separately.
- Existing estimates of 1.5M/2M/10M cached tokens are conditional memory budgets, not measured capacities. Artifact: `research/codex/audit.json`; source locations in its R1 report §2. Author: Codex.

## F9 — Codex: native flush needs JSON and all-worker success
- VERIFIED source: `entrypoints/http_server.py:990` returns text for `/flush_cache`, whereas task.md requires JSON `{"success": true}`.
- `managers/tokenizer_control_mixin.py:301` awaits the DP fan-out responses but returns only `[0]`; this is not an all-worker success reduction and can mask another worker's failure.
- Scheduler reset requires fully idle; UnifiedRadixCache reset includes host L2, not external L3 storage. Exact source references and un-deployed adaptation notes: Codex R1 §6. No successful GPU endpoint validation claimed. Author: Codex.

## F10 — Codex: dev PASS is not formal PASS
- VERIFIED: task.md requires eleven gates, including `tpot_p95 <= 0.10s/token`. `s1-dev/harness/s1_common.py:40` lists ten; `s1_score.py:448` does not include TPOT in SERVICE_GATES_PASS and compares TTFT p95 directly.
- Keep harness unchanged. Any future result must report the additional formal criterion and statistical differences separately. Author: Codex.

## F11 — Codex cross-check of Claude F6 / upstream correctness reports
- VERIFIED upstream evidence: SGLang PR #39156 describes the hybrid DSA indexer restore gap independently located in the local assembler and reports GLM-5.3-Flash/B300 controls; it is open at lookup. This is not our A100 reproduction. https://github.com/sgl-project/sglang/pull/39156
- VERIFIED: #39526 addresses mixed-chunk Mamba tracking loss; its reported GPU controls are GDN/L40, not KDA/A100. https://github.com/sgl-project/sglang/pull/39526
- Refinement to F6's "only extra_buffer": local `arg_groups/mamba_hook.py:110` explicitly also accepts `extra_buffer_lazy`; page=64 excludes no_buffer, not the whole lazy variant.
- VERIFIED vLLM PR #53906 merged 2026-09-03, commit `98ed0856f31fa3aaf5e27464e2b4ef5a8ee6b2f5`; main support does not prove release wheel or sm80 backport contents. https://github.com/vllm-project/vllm/pull/53906
- Author: Codex. Cross-review: Codex R1 §11. No experiments executed for this review.

## F12 — Codex: D1 checkpoint lifecycle / scheduler constraints (source only)
- VERIFIED: UnifiedRadixCache's `MambaComponent.prepare_for_caching_req` (`components/mamba.py:525`) uses `mamba_last_track_seqlen`. For unfinished extra-buffer requests it allocates a replacement slot and donates the written ping-pong slot to the cache; retained checkpoints therefore consume separate slots across forwards.
- VERIFIED: `schedule_policy.py:1135,1459,1543` separates continuation, admission and commit. `scheduler.py:3986` asserts that `adder.new_chunked_req` is accepted only when `self.chunked_req is None`. An artificial role-boundary split must preserve this scheduling invariant.
- Dimension calculation using F8: 22 extra checkpoints at TP8/DP1 cost ≈406 MB per rank, ≈3.25 GB machine-wide. This is one added checkpoint per logical session, not an upper bound on retained history or a measured allocation.
- Author: Codex. Design interpretation and acceptance conditions: `research/codex/R2_review_shared_directions.md` §2. No experiment run.

## F16 — Codex: static hybrid pool split and DP request-flooring
- VERIFIED source (`src/sglang/python/sglang/srt/mem_cache/kv_cache_configurator.py:2148,2433,2480`; `arg_groups/fields/schedule.py:203`): with static pools, no MTP and resolved fallback ratio r=0.9, KDA gets approximately r/(1+r)=47.37% of the post-weight/runtime-reserve candidate cache budget; the remainder is KV. Slot/padding/page rounding applies. This is not a measured GPU budget.
- VERIFIED: explicit `max_mamba_cache_size` is divided by attention DP size; explicit `max_running_requests=R` is also floored to R//D per worker before other caps (`kv_cache_configurator.py:2263`). R=22, DP4 gives at most 5 per worker / 20 total running slots, not 22. Logical closed-loop N and engine running slots are distinct.
- Dimension-only DP weight replication and conditional DP1/2/4 tables are in `research/codex/R5_dp_memory_accounting.md`; they are not loaded VRAM measurements. Author: Codex. No model or service run.

## F18 — Codex: Mamba per-path cap is soft, not role/end protection
- VERIFIED: `src/sglang/python/sglang/srt/mem_cache/unified_cache/components/mamba.py:248–307` evicts shallow eligible checkpoints but preserves tail, forks, locked nodes and device leaves. It is explicitly a best-effort soft cap and does not identify semantic role boundaries.
- Thus `mamba_max_states_per_path=2` cannot guarantee exactly two states or that they are role-boundary plus prompt-end; later deeper checkpoints may change the retained set. Author: Codex; R4 §5 / D1 design review. Source inspection only.

## F20 — Codex: lazy supports native prefill donation, with distinct allocation-failure semantics
- VERIFIED: lazy initially allocates one track slot and leaves the other as -1 (`memory_pool.py:1585–1615`); prefill tracking still runs but skips the index swap (`schedule_batch.py:2921–2986`). The normal chunk stash invokes unfinished-cache insertion, which allocates a replacement then donates the recorded slot and updates device mapping (`scheduler.py:3456,3587–3598`; `components/mamba.py:580–596`; `memory_pool.py:1628–1649`).
- VERIFIED: prefill replacement allocation evicts/retries then asserts on failure (`components/mamba.py:492–500`); decode temporary-slot allocation instead does not evict/retry and can track in place (`schedule_batch.py:3331–3360`). These are not interchangeable failure policies.
- Local lazy configuration disallows PD disaggregation and requires overlap in the Mamba capacity path (`arg_groups/mamba_hook.py:110–116`; `kv_cache_configurator.py:2220–2224`). D1 lazy compatibility is conditionally plausible from source, not runtime-validated. Author: Codex; D1 design §8.4. No experiments.

## F21 — Claude: organizer base images in the Trisol catalog (read-only lookup, 2026-09-22)
- `bohr trisol image list --query …` (read-only) lists deployable images: SGLang base **id 160721** `…/prod-3732438/20675/arena-sglang-glm53:260918`; vLLM official 160722; vLLM backport 160685 (`260918-sm80`) and 160627 (`260917-sm80`); TokenSpeed 160726.
- `bohr image get 160721` / `bohr image dockerfile 160721` → **403 PERMISSION_DENIED** for this account (V). ~~Contents remain unknown (D6).~~ 〔更正：底包 = 公开提交 fe236ea6c3 + 两处多模态修复，见 F53〕
- The catalog also lists many other contestants' public images; by policy we do NOT inspect competitors' images.

## F22 — Codex: merged role-boundary checkpoint prior art, not a local performance result
- VERIFIED / SOURCE: llama.cpp PR #22929 merged on 2026-05-25. It extracts message spans, splits prompt batching before the last user input, and saves a context checkpoint there, explicitly targeting agentic coding responsiveness. https://github.com/ggml-org/llama.cpp/pull/22929
- Current source at `c550d2f60bde72df19fcef1fef627895095b8ba8`, `tools/server/server-context.cpp:3520–3634`, still stops at the last-user boundary and has additional near-tail checkpoint splits. This is direct implementation prior art for D1 A, not proof of SGLang/KDA/A100 compatibility or speedup.
- VERIFIED / SOURCE: TensorRT-LLM merged #18724 preserves explicit Mamba snapshot placement instead of unconditionally adding prompt end; its diff documents that a later recurrent checkpoint in the same tree block can replace the intended one. https://github.com/NVIDIA/TensorRT-LLM/pull/18724
- Author: Codex. Details, Chinese/GitHub coverage and evidence limitations: `research/codex/R6_prior_art_cn_github.md`. Research only; no experiments.

## F23 — Codex: public dev phases reuse rid; successful process exit is not a flush guarantee
- VERIFIED / SOURCE: `s1-dev/harness/s1_common.py:164` derives rid only from pack/view/logical_call_id; `s1_loadgen.py:108,112,347` sends it unchanged. Overlapping rows in preflight, warmup and measure reuse the same rid. Distinct cache namespaces are headers, not rid prefixes (`run_dev.py:187–243`; loadgen adds `-warmup` at `:544`).
- VERIFIED: stages run via synchronous subprocesses and client threads are joined. However request errors/timeouts become result records; warmup still returns 0 (`s1_loadgen.py:145–151,552–559`). Client completion does not by itself establish server-side cancellation/cleanup after failures.
- VERIFIED: `run_dev.py:46–56` does not parse the flush JSON body, and its boolean result is ignored at `:232`. Thus entering measure does not establish successful KV flush. No harness changes made.
- Scope: `run_dev.py:266–267` explicitly scores lane=dev. These facts establish the public dev flow, not the unpublished formal runner. Author: Codex; R6 §6 contains source hashes and D0 implications.

## F26 — Codex: vLLM internal-checkpoint precedent is merged, with dtype and hardware caveats
- VERIFIED / SOURCE: #50587 links #52789 (merged 2026-08-22, `9eb9d9d3953959695108600c8ed33d36bc6a1e5f`) and #53614 (merged 2026-09-06, `144e79c8106da23141ac010394b782f730cc7fe8`). The former's merged KDA code calls FlashKDA once with checkpoint state/offsets; its opening diagram still shows two KDA calls. The latter unifies export validity and re-keys partial/spec checkpoints to their true state depth. https://github.com/vllm-project/vllm/issues/50587 https://github.com/vllm-project/vllm/pull/52789 https://github.com/vllm-project/vllm/pull/53614
- VERIFIED / SOURCE: pinned FlashKDA `ee0be888cd0e972f9409bf53756f8c38c6652173` exports one offset per sequence to an FP32 tensor, but its recurrence's resident/shared state is BF16 and is converted on export. This is not the same precision contract as F25's FP32 accumulator snapshot. https://github.com/vllm-project/FlashKDA/blob/ee0be888cd0e972f9409bf53756f8c38c6652173/csrc/smxx/fwd_kernel2.cuh
- VERIFIED: the corresponding vLLM FlashKDA CMake target does not include sm80. The PR's reported TTFT improvements are Kimi-K3 measurements, not a reproduced GLM/A100 agentic result; measured revision vs final merge was not established here.
- Author: Codex; R7 §2 includes pinned source links and report scope. Plan A remains the agreed first option; no implementation or experiment was performed.

## F33 — Claude: arena Trisol queue policy (renumbered from a duplicate F25) (read-only, 2026-09-22)
- `trisol team scheduling get arena`: policy `shared`; **member quota A100-SXM4-80GB = 8 GPUs per member** (PPU810E = 0). Team quota 400 A100 on w1; many members at 8/8 running, others queued. Kueue priorities 1–8 are automatic; pinning needs a team admin.
- Implications: we can run at most ONE 8-GPU service at a time (or several smaller ones summing to 8); any inspection pod counts against the 8. Admission wait depends on the shared pool. Plan runs so that each 8-GPU slot does a full ladder, and delete promptly.
- SGLang v0.5.20 has `--prefill-decode-interval` (`arg_groups/fields/schedule.py:63`): decode rounds after each prefill batch, synchronized across DP ranks (AgentX reports +141% output throughput / −97.3% latency at the cost of TTFT, R4).

## F27 — Codex W1: AgentX decode interval is local, but the +141% headline is fixed-length evidence
- VERIFIED / SOURCE: SGLang PR #35017 merged 2026-08-19; local `src/sglang/python/sglang/srt/managers/scheduler.py:1343,1350,3652,3696` defers prefill and synchronizes the interval from the DP-global extend signal. `--prefill-decode-interval` is declared None (`arg_groups/fields/schedule.py:63`) and resolves to **0 for GLM** (`arg_groups/validation_hook.py:452`); no backport needed. https://github.com/sgl-project/sglang/pull/35017
- VERIFIED / UPSTREAM REPORT, not our measurement: +141.1% output throughput / −97.3% p99 ITL are the PR's **fixed-length GB300** interval 0→4 comparison (192 requests, C64, ISL 131072 / OSL 1024), not the separate AgentX prototype runs on B200/interval16. TTFT p50 rose 36.46→58.98s and p95 95.34→101.49s. This corrects the attribution in Claude R4 and the final decode-interval bullet under Claude's duplicate F25 queue-policy entry; the queue-policy facts are unaffected.
- INFERRED: a 0/1/2/4 sweep may help TPOT, but can hurt our 3s/5s intra TTFT gates. D1-generated extend boundaries also arm the interval. No change to the agreed priority and no experiment performed.
- Author: Codex W1 / T11. Full 66-row, 130-PR checklist: `research/codex/R8_agentx_checklist.md`; PR inventory: `research/codex/R8_agentx_pr_inventory.json`. Read-only research; no service or experiment run.

## F28 — Codex W1: AgentX DP affinity proposals are not equivalent to shipped internal routing
- VERIFIED / SOURCE/API: #26091 is closed without merge; local SGLang has **no `--dp-cache-affinity`** / `routing_key_to_dp_rank`. `data_parallel_controller.py:752` supports explicit routed_dp_rank, followed by load-based methods. The local gateway has DP-aware request routing (`sgl-model-gateway/src/routers/http/pd_router.rs:240,310,330`), with gateway `--dp-aware=False`, `--policy=cache_aware` and balance thresholds 64/1.5 by default (`bindings/python/src/sglang_router/router_args.py:54`). https://github.com/sgl-project/sglang/pull/26091 https://github.com/sgl-project/sglang/pull/26245
- VERIFIED: proposed `--cache-balance-weight` (#26293) is absent. UnifiedRadix placement events exist (`mem_cache/unified_cache/unified_tree_core.py:447,1429,2395`; #26387 behavior), but proposed SWA-validity metadata (#26579) is absent. `--kv-events-config` defaults None. https://github.com/sgl-project/sglang/pull/26293 https://github.com/sgl-project/sglang/pull/26387 https://github.com/sgl-project/sglang/pull/26579
- INFERRED: DP2/4 needs validated prefix/session affinity and per-rank cache metrics; enabling DP alone does not preserve conversation history. FULL-KV events do not establish KDA checkpoint reachability.
- Author: Codex W1 / T11. Full 66-row, 130-PR checklist: `research/codex/R8_agentx_checklist.md`; PR inventory: `research/codex/R8_agentx_pr_inventory.json`. Read-only research; no service or experiment run.

## F29 — Codex W1: AgentX architecture filters and the hybrid DSA offload blocker
- VERIFIED / SOURCE: runtime-scalar #30255 is present at `src/sglang/python/sglang/kernels/ops/attention/dsv4/metadata_kernel.py:8` and `.../dsv4/sparse_prefill_kernels.py:39`; it optimizes DSv4 C128/sparse-combiner compilation, not our GLM DSA path. FlashInfer GDN checkpoints #29735 are present (`srt/layers/attention/linear/kernels/gdn_flashinfer.py:66`) but are GDN, not our 34 KDA layers. https://github.com/sgl-project/sglang/pull/30255 https://github.com/sgl-project/sglang/pull/29735
- VERIFIED / SOURCE/API: #39156 remains open; local hybrid-Mamba HiCache assembly still contains only KV + MAMBA entries (`srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py:886–908`), without the proposed target DSA-indexer sidecar. Existing flat-DSA or draft sidecars do not repair this path. `--enable-hierarchical-cache=False`; default host ratio resolves 2.0, write policy write_through, IO kernel, layout page_first, storage None (`arg_groups/fields/memory.py:102`; hicache_hook.py:62). https://github.com/sgl-project/sglang/pull/39156
- INFERRED: HiCache capacity savings cannot count until all KDA/conv/MLA/DSA state survives eviction/reload and actual flush. SWA-tail recomputation cannot substitute for recurrent state. This reconfirms the prior correctness hold rather than changing direction.
- Author: Codex W1 / T11. Full 66-row, 130-PR checklist: `research/codex/R8_agentx_checklist.md`; PR inventory: `research/codex/R8_agentx_pr_inventory.json`. Read-only research; no service or experiment run.

## F30 — Codex W1: Transferable frontend work exists; external/mocker speedups are not local engine gains
- VERIFIED / SOURCE: local `SGLANG_USE_PICKLE_IPC=True` selects pickle; False uses msgpack (`srt/environ.py:367`, `srt/managers/io_struct.py:2485`). `--incremental-streaming-output=False`, stream interval 1, tokenizer workers 1, dynamic tokenizer disabled (batch size32 / wait0.002s if enabled; `arg_groups/fields/serving.py:57,253,265,288`). Cross-turn boundary-aware input tokenization like TRT-LLM #17462 is not established by the local full/batch encode path (`managers/tokenizer_manager.py:997,1011`); incremental output is a different feature. https://github.com/NVIDIA/TensorRT-LLM/pull/17462
- VERIFIED / PRIMARY-SOURCE SCOPE: Dynamo #11095/#12329 describe mocker KV bookkeeping/request leases, including a SGLang simulator backend. Their reported gains are not a demonstrated production SGLang optimization. https://github.com/ai-dynamo/dynamo/pull/11095 https://github.com/ai-dynamo/dynamo/pull/12329
- VERIFIED: native LMCache adapter extends RadixCache and passes K/V buffers (`mem_cache/storage/lmcache/lmc_radix_cache.py:86,120`), allocates the full uncached load at :349, delegates MP retrieve at :407 and resets local state at :173. LMCache #3382's vLLM chunk-load cap and #4524's daemon hybrid-holder lock fix are not proved present by this connector. External LMCache version and hybrid/all-tier-flush support remain unverified. https://github.com/LMCache/LMCache/pull/3382 https://github.com/LMCache/LMCache/pull/4524
- INFERRED: profile exact input tokenization, serialization and delta SSE as lower-cost directions; do not treat LMCache as a ready bypass around the hybrid HiCache gap.
- Author: Codex W1 / T11. Full 66-row, 130-PR checklist: `research/codex/R8_agentx_checklist.md`; PR inventory: `research/codex/R8_agentx_pr_inventory.json`. Read-only research; no service or experiment run.

## F31 — Claude: formal-set composition is recorded in the dev cohort file (read-only)
- `s1-dev/harness/g0a/samples_v3/cohort_dev-combined-v1.json` → `composition_alignment_measured.formal_set_reference`: formal set "v3 sample-50" (cohort de7cd762) has **chain_start n=808, intra n=9023, turn_start n=65** (≈9.9k requests; **91% intra**). The dev cohort is 722 requests: chain_start 314 (43%), intra 388 (54%), turn_start 20.
- Full-population p95 of frozen `uncached_expected` per gate (`gate_p95_alignment.full`): chain_start 113,977; **intra 11,740**; turn_start 42,015. fast stratum (uncached ≤ 4096) = **85.2%** of intra requests in the population (84.5% in dev).
- Implications (I): (1) the formal run is intra-dominated → the binding gates are fast_intra/overall_intra and D1 (which targets intra misses) acts on ~9k requests; (2) dev over-represents cold chain starts (43% vs ~8%), so dev puts relatively more prefill pressure on the engine and fewer intra samples per gate than formal — dev critical N is not the formal N (as task.md says), but dev is pessimistic on cold-prefill load; (3) formal chains are longer on average (≈12 req/chain vs 2.3 in dev), so per-session KV/KDA residency pressure over time is higher than dev suggests.
- Dev reference runtime: 35 min @ N=6 on 8×A100 (`reference_runtime`).

## F32 — Claude: chat template vs SGLang parsers (source read, 2026-09-22)
- `s1-dev/glm_tok/chat_template.jinja:256`: the generation prompt ends with `<|assistant|><think>` — thinking is opened **by the prompt**, so model output starts inside reasoning and closes with `</think>`; prior turns render `<think>…</think>` (or `<think></think>`) (:145-152). `clear_thinking` defaults false (:4).
- Tool-call format (:50, :162-163): `<tool_call>{name}<arg_key>k</arg_key><arg_value>v</arg_value>…</tool_call>` = the GLM-4.7 style.
- SGLang v0.5.20: `--reasoning-parser glm45` → `Glm45Detector` (`parser/reasoning_parser.py:825,2170`), format `(<think>)*(.*)</think>` (start tag optional → compatible with the prompt-opened `<think>`), and `<tool_call>` switches out of reasoning. Tool parsers: `glm45`→Glm4MoeDetector, `glm47`→Glm47MoeDetector (`function_call/function_call_parser.py:81-82`); the arg_key/arg_value format matches **glm47**.
- Decision input: keep `--reasoning-parser glm45` (final answer lands in `message.content`, which is what capability scoring reads); add `--tool-call-parser glm47` (same as the organizer's verified example) — aime/gpqa don't use tools, but it keeps /chat/completions behaviour consistent. Runtime check still required: a real aime-style answer must appear in `choices[0].message.content`, not only in `reasoning_content`.

## F35 — Claude: public formal results contradict the uncalibrated simulator's "TPOT binds first" (data/all_att.json)
- Median over competitors' passing levels (formal set, real deployments): N=6 (44): tpot_mean 0.015, **tpot_p95 0.023**, fast_intra 2.91s, overall 3.84s, turn 5.7s, chain 10.4s; N=10 (26): tpot_p95 0.051, fast 3.15, overall 3.97; N=14 (33): tpot_p95 0.054, fast 2.60, overall 4.08, chain 37.8; **N=18 (7): tpot_p95 0.055, fast 2.55, overall 4.66, turn 9.8, chain 46.3**.
- ⇒ In real deployments tpot_p95 sits at ~half the 0.10 gate while fast/overall_intra sit at ~85–95% of 3s/5s: **TTFT gates bind, not TPOT** (consistent with F5). R9's default decode (30–50 ms base + slope) is too pessimistic; the simulator must be calibrated before its "first binding gate" is trusted.
- These are other teams' configs on the formal set, so they are only a coarse prior (envelope), not our stock baseline.

## F36 — Codex main: E1 stock KDA+MLA stand-in works; F13 misses resident FULL-KV lifetime
- VERIFIED / local A100: unmodified SGLang `94602c9` with Python UnifiedTreeCore, page64/extra_buffer/chunk8192, random 3-KDA+1-MLA Kimi-Linear (`q_lora_rank=null`, no DSA) served 20 selected dev chains / 72 requests without errors. All prompt counts equal the unchanged Renderer, all outputs have 4 tokens with ignore_eos; 70/72 cached-token predictions match exactly, two actual hits are lower by 1088/3712. Fast-intra actual and ideal-stock uncached p95 both equal 4750 over 42 requests; this aggregate does not erase the mismatches. Evidence: `evidence/E1_stock/`, `notes/experiments.md` E1; remote full records `/sjtu/linhang/arena/runs/E1_20260922/stock20/`.
- VERIFIED / IF-07 L2: identical 18678-token prompt has cached_tokens 0 → 18624 → 0 after flush. Native flush still returns text (not D0 JSON compliance); only one worker was used. All GPU work stopped afterward; no Trisol, full GLM, 8-card quota or submission. The final qfull weights are recorded in `evidence/E1_stock/model_manifest.json`; earlier q_lora=128 configuration failed before replay because the pinned Kimi wrapper did not set AttentionInputs for that shared MLA branch.
- VERIFIED / source distinction: F13's stock predictor derives a branch from previous-full-prompt LCP. Actual `src/sglang/python/sglang/srt/mem_cache/unified_cache/components/mamba.py:154` derives it from resident `full_kv_hit_length`; its `prepare_for_caching_req` at line525 caps finished extra_buffer caching to the last tracked state. `unified_radix_cache.py:987` truncates both FULL-KV and radix key to the minimum component length. `managers/schedule_batch.py:2957` substitutes a branch only inside the current extend. These are missing lifetimes/placement details, not just page rounding.
- INFERRED / not runtime-instrumented: when branch replaces end, the discarded FULL-KV tail cannot support the next prompt's abstract-LCP branch; the two observed lower hits are consistent with that mechanism. Do not call this a confirmed engine bug or silently rewrite F13/F24 results. Use same-config real stock/D1 A/B as primary evidence; refine the predictor separately. This functional stand-in cannot establish GLM performance, numerical correctness or N@SLO.

## F37 — Claude (from Codex E1): real stock stand-in confirms the D1 premise, and stock is slightly WORSE than modelled
- E1 (random-weight 3 KDA + 1 MLA Kimi-Linear stand-in, stock SGLang 94602c9, extra_buffer, page 64): 72 requests / 20 chains, 0 errors; server prompt_tokens == Renderer counts for all 72; per-request stock cached_tokens matched the F13 stock prediction **exactly in 70/72**; fast_intra real-uncached p95 4750 (predicted 4750).
- The 2 mismatches are both **below** prediction (−1088, −3712 tokens): a branch checkpoint pre-empts the end checkpoint, so later turns fall back further (details: experiments E1 final section). ⇒ stock loses reusable KDA state in exactly the way D1 targets; the simulator is, if anything, optimistic for stock.
- IF-07 passes on L2 (cached 0 → 18624 → 0 after flush); IF-08 fails on stock as expected (text body) → needs D0.

## F39 — Claude: technical routes visible in the public Trisol image catalog (names/tags only; contents NOT inspected)
- Source: `bohr trisol image list --query arena|glm53|sglang|vllm` (330 distinct names; saved `data/image_names.txt`). Names carry numeric user-ID prefixes, not leaderboard nicknames → **cannot be mapped to specific top players**; we did not try to deanonymize, pull, or read any image content.
- Route clusters (I, inferred from names only):
  - **Scheduling (most frequent arena theme)**: `tail-srpt-mtp` / `tailspec` (SRPT = shortest-remaining-first ≈ our D2 SPF, plus MTP / tail speculation), `edf` (earliest-deadline-first), `aged-fifo`, `priority`, `custom-scheduler`, `scheduler-bypass`, `prefix64`, `hwlimit-prefill-cache`.
  - **KDA state caching / lifecycle**: `rowchunk-kda-cpu-fresh` with tags `marker`, `lifecycle`, `workspace` → someone is working on KDA recurrent-state checkpoint placement/offload, the same family as D1.
  - **Capacity / parallelism**: `vllm-dp2-affinity`, `vllm-dp2-balanced`, `glm53-tp4x2` (two TP4 replicas) ≈ our D3; `nextn-hicache` (MTP + HiCache).
  - **sm80 kernels**: `kpool` (DSA index_kpool on sm80, many iterations), `native-mqa-marlin`, `actquant`, `pagedmqa`, `ragged`, `bm8-prefill`, `graphsafe`, `deterministic`, `ragged-logits-chunk`.
  - **Interface**: `flush` variants (others also had to fix JSON flush → consistent with D0).
- Take-aways: (1) SRPT/SPF-style scheduling is already in use by others → D2 is table stakes, not an edge; (2) **EDF with per-bucket SLO deadlines** (3/5/15/30 s) is a natural next step beyond SPF → new direction I6; (3) someone else is pursuing KDA checkpoint lifecycle → D1 is contested, so its execution quality matters; (4) TP4×2 / DP2 with affinity is an explored capacity route.

## F40 — Claude: first N@SLO=22 on the leaderboard, and its gate profile points to SLO-aware prioritisation (data/all_att.json refreshed, 494 attempts)
- LewyM, attempt 45417 (2026-09-21): **n_at_slo=22**, tpot_mean 0.0273, tpot_p95 0.065; fast_intra p95 **1.79 s** (limit 3), overall_intra **3.00 s** (limit 5), turn_start 10.9 s (15), **chain_start 61.0 s** (nominal 30 — passes only via the formal statistical allowance / small bucket).
- Contrast with the seven N=18 runs: fast 1.9–3.0, overall 2.9–5.3, chain ~45 s. The N=22 run holds intra far below its limits while letting chain_start drift far above 30 s ⇒ strongly suggests **deliberate priority for intra (tight-SLO) requests over cold chain starts**, i.e. SLO-aware / EDF-style scheduling (I6), not raw throughput. (I: inferred from public scores only.)
- Implication: the formal chain_start gate has large effective slack; a scheduler that sacrifices chain_start (within the allowance) to protect fast/overall_intra is the demonstrated path to 22. Our D2 (SPF) should be extended to I6 (deadline/slack-aware with per-gate deadlines), and D1 (fewer intra misses) compounds with it.


## F42 — E2实测D1 v1.1局部增命中，但reminder-heavy尾部目标未过（T13 / Codex main）

2026-09-22，**VERIFIED / L2 GPU**：原stock 94602c9 / D0+D1关闭 / D0+D1开启，固定E1 qfull随机Kimi-Linear权重、GPU0、page64/chunk8192/extra_buffer/path cap=-1、context131072；无DSA/MTP/HiCache/002/003。四个内部样本smoke6、reminder_heavy70、strict_append67、stock20共215请求（集合间可能重叠，不声称215唯一请求），每组顺序N=1、每链flush、输出4 tokens ignore_eos；所有prompt完整，非正式评分。关闭组与原stock的215项缓存完全相等；E2 stock20与E1的72条ID/prompt哈希/缓存也完全相等。

**VERIFIED**：reminder_heavy开启后31改善、39相同、0退步；未命中总tokens 700622→657678，fast_intra n=54的未命中p95仍8135，理想role预测2439。与role_conservative预测误差≤64的只有45/70（64.3%），所以既定D1-02验收**FAIL**，不能称F24收益已经落地。strict_append 67条全部相同，D1-03在本次限定配置**PASS**。smoke也无改善（fast p95=6315，role理想值562）。证据：`notes/e2_d1/comparison.json`、`scripts/compare_e2.py`，完整原始记录在GPU机`runs/E2_20260922/`。

**VERIFIED / source + CPU**：D1 §10确认长冷请求跳过钩子且续跑无钩子，而F24理想模型仍为它们加入role；另有HiCache host-miss重新选择partial绕过guard的CPU witness。因此只批准无HiCache的诊断，不批准部署。数值检查另记E2后续结果；不得从缓存改善推导数值正确性或临界N提升。

## F43 — Claude (verifying Codex main E2, notes/e2_d1/comparison.json): D1 v1.1 has no cache regression and helps (numerics not yet safe, see F45), but misses the chunked-cold-head case
- VERIFIED on L2 stand-in (same qfull weights/cases, 215 requests): **D1 off == stock on every request** (D1-01 pass); **D1 on is never worse** than stock (0 worse; D1-03 strict_append 67/67 equal → pass).
- Gains: reminder_heavy 31/70 requests better, total uncached −6.1% (700,622 → 657,678) but fast_intra real-uncached p95 unchanged (8135; role-boundary prediction 2439); stock20 fast p95 **4750 → 3962** (prediction 3196); smoke no change.
- Cause (matches Codex §10.3): the request before the first intra turns is usually a cold chain head longer than the chunk size (8192 here; 16k–100k+ in the real workload) → admission is chunked → v1.1 skips the role split, so the next turn falls back to a chunk-end checkpoint (e.g. cached 8192 of 9330). ⇒ Next step D1 v1.2: also split the FINAL chunk of a chunked request at the last role boundary.

## F44 — Claude: official submission quota — **user-confirmed 2 per day** (older platform changelogs said 3; user confirmed on 2026-09-22 that the doc/limit is 2 — follow 2)
- VERIFIED: 49 attempts in data/all_att.json carry the platform changelog "【未进入评分 · 每日提交额度已满】Submission quota exceeded: at most 3 submissions per user per day on this challenge … (UTC+08:00)". Refused-for-quota attempts are not scored. R7 (top-players analysis) also reports deploy failures are not charged (from attempt changelogs; not independently re-verified).
- Scoring takes ~18 h median (R7), so plan 2–3 variants per day submitted together rather than sequentially. 〔更正：配额以用户确认的每天 2 次为准，守护进程按 2/天运行（决策 21）。〕

## F45 — E2 raw-logits验收失败；“缓存无退步”不等于数值安全（Codex main，T13）

2026-09-22，**VERIFIED / L2**，收窄F43标题中的“safe”：它只验证缓存计数无退步，不构成数值安全。三个真实相邻请求对，先按原顺序重放完整链前缀，再调用目标prompt；D1开启时通过实际split trace和响应cached_tokens确认分别恢复到90624、71488、74176，而不是普通末尾命中。导出`LogitsProcessor.forward`返回、sampler之前的全154880词表FP32 logits（模型BF16），每prompt另做两次flush后冷算。脚本`e2_serve_trace.py`/`e2_raw_logits.py`，数据`notes/e2_d1/`，raw tensors留在远端E2 trace目录。

- 三对的cold/cold max abs均0；按既定2×冷噪声门，容差为0。D1 on冷/暖max abs依次0.00390625、0.0087890625、0.00390625；greedy32依次相同/不同/相同，第二对首次不同在零起点index19。因此**D1-04 FAIL**，不修改阈值绕过。
- 同配置D1 off、相同完整链前缀的普通缓存对照：max abs依次0、0.00390625、0.009765625；greedy32依次相同/不同/不同。即普通缓存也能触发零阈值门/greedy分歧。**不能据on失败直接判定D1快照损坏，也不能据off失败豁免D1**；需要重新预先定义有判别力的精度基线/状态验证，当前不批准数值安全或真实能力结论。
- 数值run独立于无trace回放，导出会同步GPU；不使用其TTFT作性能证据。当前只验证随机小模型、TP1、extra_buffer、无DSA/HiCache/MTP；不是完整GLM能力验证。全部自有服务已停，两卡4MiB/0%，其他tmux不动。

## F46 — W12 / T25: chain-start统计条数余量≠秒数放宽；固定R10模型里EDF未胜SPF

2026-09-22，Codex W12。**VERIFIED / 数学与CPU**：task.md:565定义超标率单侧95%下界>5%才失败，但未指定置信区间算法；现有score_formal.py的exact Clopper–Pearson继续标 **estimated**。独立60位Decimal CDF验证：n=65/314/808/9023时allowed_over=6/22/51/485，第k+1条均失败。808时L(51)=.0496342，L(52)=.0507405；758×20s+50×61s有p95=61s、L=.0485294，chain gate可过。该检验只计超过30s的条数，不能由p95推出统计判定，也没有单独的61s上限；F31构成不是公开attempt实际窗口桶数。**Refines F40's inference**：61s与PASS相容，但公开分数不能识别EDF、证明故意牺牲链首或证明D1叠加必然有效；不推翻F40原始公开观测。

**MODEL OUTPUT / WHAT-IF**：固定R10 C6/C17/C18/C19/C31及C42/C43，dev/formal-mix、FCFS/SPF-upstream/SPF+D1/EDF/EDF+D1/least-slack/EDF-weight2/weighted+D1，完整输出共896次模拟；主候选formal-mix严格首败SPF均18，EDF=10/14/10/10/10，EDF+D1=10/14/14/14/14，绑定fast/overall。weight2和least-slack均未稳定超过SPF。strict/estimated分开展示，后者可改变过档而非时间。观测代理：stock模型2558/7627个冻结fast formal-intra被实际cache work分到5s类；无法从session/cache独立判断turn_start，保守用3/5s。不能把任何模型N当硬件容量。

**VERIFIED / 003草稿**：`patches/003-slo-aware-scheduling.patch`从ready 002独立副本起草，3可选策略/独立session+namespace+可信arrival透传；deadline冻结、aligned continuation reservation复用002单partial guard。001→002→003及加000均fuzz=0，16新CPU tests（含450随机策略对照、真实flush方法忙/闲路径）、47模拟、29评分、21加权工具、16 D2、15 D1，共144项通过。默认未启用；live SLO-08/09 todo。报告`research/codex/R12_slo_aware_scheduling.md`，证据`evidence/T25_slo/`；CPU only，无GPU/服务/镜像/提交，只读目录未改。

## F47 — T29：v1.1的flush/池回收与N4单partial取得L2实测；统计只覆盖新准入

2026-09-22，Codex main，**VERIFIED / 随机qfull替身TP1**。同E2的000+001 v1.1、page64/chunk8192/extra_buffer，不含002/003/004、HiCache、MTP。

- IF-08：原context/KV131072，W6工具真实4096-token流在途，flush timeout=0得到400/false；timeout=180等待54.409s后200/true，重打cached=0。D1-10：完整原链warm后，上一请求split trace深度90624，目标91819-token prompt确实命中90624；flush后同prompt cached=0。
- cold_heavy最大256733 tokens，原配置在发送前拒绝。新mixed-only配置context262144/KV524288（模型本身max_position_embeddings=262144），完整cold35+reminder70=27链105条、客户端N=4：0请求错误、306个调度轮次、每轮partial最大1、最大prefill batch3。trace所有实际batch RID和commit RID逐轮一致，无未记录准入。
- 新准入105、准入尝试215、续跑229、总commit334；ROLE_BOUNDARY_STATS和为105（taken45、already_chunked34、branch20、active5、short1），**等于新请求准入，不等于尝试或含续跑的总准入**。单partial子项通过；总门保留未通过，等待协调方明确统计分母，不把统计缺项写成崩溃或第二partial。
- D1-08：混合run完，flush前KV/Mamba空闲8320/378，flush后524288/512，request槽8，全部等于启动基线。仅此配置和负载，不代表其他缓存策略无泄漏。

证据`evidence/T29/{first,large}/`，远端`runs/E2_T29_20260922*`；完整配置、第一次回放拒绝、初版池检查空跑误判的撤销及反例测试见experiments T29/README。没有修改原raw收据来掩盖错误。D1-02/04既有失败不变；本结果不是数值安全、完整GLM、DP/HiCache或N@SLO证明。

## F48 — T33：Trisol CLI 真实返回格式与守护进程的模拟不一致（已修）
- `bohr trisol inference get <id> --spec --output json` 把字段嵌套在 `runtime`/`resources` 下；
  我们的会话启动命令是 `python3 -m http.server 8000 --bind 0.0.0.0`。守护进程原先按平铺字段、
  无 `--bind` 校验，会拒绝接管自己的会话（adoption-spec-mismatch）。
- 排队中的行是 `status=deploying, stage=launching, substage=queueing, reason=WaitingForAdmission`；
  原 `phase()` 把 deploying 当 unknown → 视为已占卡开始计时，排队几小时就会耗尽时间盒并触发清理。
- 修复后用真实返回核对：接管成功，phase=waitingforadmission；回归测试 `RealSpecAdoptionTests`。
  VERIFIED 2026-09-22 11:45Z（logs/trisol_test.events：ADOPT + PHASE waitingforadmission）。

## F49 — T34：AgentX快照保留三阶段与本地prefill节拍核对（仅调研）

2026-09-22，Codex main，**VERIFIED / 公开PR API与本地源码，非性能复现**。vLLM #43447（06-04）只先做SWA选择性保留；#45845（06-23）扩展至Mamba/线性注意力；#47782（07-13）把共享前缀边界贯通cache manager、request、align切块和保留mask。三个merged状态均由REST确认；后者核心diff已读。不能将第一个PR单独当作KDA完整实现，亦不能据旧hybrid设计文的WIP文字否定已合入代码。

本地`94602c9`已有`--prefill-decode-interval`（fields/schedule.py:63；scheduler.py:1343–1363、:3650）；extend后计数器暂停新prefill，DP用同步extend标志，未设置最终归零。这不是SSE流式间隔，也不与vLLM每N步准入逐项等价。`enable_unified_memory`已有但默认关闭且有后端/HiCache/graph/spec限制；`mamba_max_states_per_path`为保护tail/fork/locked的软上限。详见[R13](../research/codex/R13_agentx_mlperf_reading.md)，包含两篇文章、三个PR、源码位置和待测清单。**INFERRED**：D1/D2、容量/亲和性与小batch优化应分项验收，不从外部数字推导本赛收益。无引擎改动、GPU、服务或提交，既有数值门保持。

## F50 — T35：GLM内部快照新PR、ReplaySSM接线与DP亲和路由；容量统计不等于容量增加

2026-09-22，Codex main，**VERIFIED / GitHub API、PR正文、两个关键diff及本地源码，非性能复现**。vLLM #56960（open）为GLM KDA增加单forward内部checkpoint及配套conv导出，沿用FlashKDA；测试后端涉及sm90，非A100开箱证明。SGLang #40517（09-21 merged）增加`hybrid_kda_config`，把GLM纳入ReplaySSM spec池/预算判断；本地仍Kimi专属。普通decode ReplaySSM与spec模式不同，前者本地明确KDA可能更慢且要求no_buffer，不能混称D1替代品。

SGLang #31170（open）是单实例内部DP rank的routing_key亲和路由，不仅是网关；本地无prefix_affinity，PR的OpenAI header透传不能假定适用于赛题/generate。#40680（open）补write_back内部节点Mamba状态备份，和仍open的DSA indexer恢复#39156是两个问题。vLLM #57261正文明确只改报告容量，11.88GiB池及分配不变；不能将+14.7%写成真实显存收益。作者sm80测试使用未包含在PR中的out-of-tree稀疏后端，未解除D6来源不明。

详见[R14](../research/codex/R14_recent_pr_watchlist.md)及[状态/head收据](../evidence/T35/github_status.json)，另含PTX sigmoid修复、长decode退化报告及GB300吞吐/TPOT取舍。**INFERRED**：可借鉴实现/测试，但作者benchmark非本赛结果；不据此放行HiCache、数值门、完整GLM或N@SLO。仅研究/文档，无实验、安装、服务、补丁或队列变更。

## F51 — T33：底包源码（diag04，407 文件，sha256 025ee541…）里两条路线的现状

编号勘误（Codex T36）：此条原误标 F49，与前面的 T34 重号；这里只改编号，原研究结论不变，用户本轮所指“底包 F49”即本条。
- **KDA 记状态**：`schedule_batch.py:_mamba_radix_cache_v2_req_prepare_for_extend`（2766–2860）每次 extend
  只记一个状态：默认=提示末尾（按 checkpoint grid 下取整）；若 `mamba_branching_seqlen` 落在本次 extend 内且
  与 cache_chunk_size 对齐，则改记分叉点，经 `_force_track_h` 从中间状态 h 取，**不需要额外 forward**。
  → 把"对话边界"作为第三种候选放进这里，即可替代 101 的拆分预填充（补丁 102）。
- **routing key**：`/generate`（http_server.py:908）不读任何 routing 头；只有 OpenAI 端点读 `x-smg-routing-key`
  （serving_base.py:259）。压测发的是 `X-S1-Routing-Key`，所以底包的 `routing-key` 调度策略对我们是空转。
- **DP 路由**：`LoadBalanceMethod` 只有 ROUND_ROBIN/FOLLOW_BOOTSTRAP_ROOM/TOTAL_REQUESTS/TOTAL_TOKENS，无亲和；
  但 `maybe_external_dp_rank_routing` 支持按 `req.routed_dp_rank` 指定分片（http 头覆盖需
  `SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES`）。KDA/GLM 代码里未见禁用 dp-attention 的断言（能否跑通待 L2）。

## F52 — T36：实际底包的单点快照不等价101；S1 routing族键不等于session

2026-09-22，Codex main。**VERIFIED / 静态源码与数据，无实验**：从diag04日志在内存重组归档，SHA256 `025ee541…` /1343324 bytes，407文件与base_src_full全部一致。refs固定head `7565389`/`ec6e4e8`。见[R15](../research/codex/R15_base_source_exploration.md)、[收据](../evidence/T36/source_receipt.json)。原底包F49仅编号更正为F51，正文保留。

- `schedule_batch.py:2766–2860`每extend一个track，backend的b+1是选h编码、实际入树深度仍b；`mamba_radix_cache.py:686–735`只捐赠该点，ping-pong两个槽不代表自动role+end。**INFERRED / 收窄F51“即可替代101”**：单点role替换会失去end机会，branch-first不保证strict_append无退步，不能继承101/E2b结论。
- 实际底包从普通h复制（hybrid backend:872–874），没有参考src树的FP32 h_track_buf路径；本次407文件未含实际`sglang.kernels`，不能确认h dtype或用参考kernel假定底包无损。FlashKDA tracked路径回退Triton、NVIDIA内部点离开fast path、CuteDSL拒绝track均有wrapper证据；少一次forward不是速度保证。
- harness:110–120/347–350明确routing_key来自sys_tools_hash，session另传。722请求/136session/156族，18session跨族，最大族79请求跨19session；只是静态公开数据，不是并发分布。`/generate`不映射S1头，但body routing_key已能经tokenizer传递；现有routing-key policy仅排序某个scheduler的队列，不做DP分发。
- PR31170核心diff涉及5生产文件，含arg_groups和load_snapshot ownership/refresh；指定routed_dp_rank绕过负载策略，不等于该PR的软亲和。Namespace放路由key不等于真实radix隔离；103/104应先明确族、session、salt三种契约。

以上仅收窄设计推论，不更改101、不创建102/103/104、不修改harness或底包、不启动GPU/服务/构建/提交；既有数值与完整模型验收保持。R15列待验证问题，非新执行授权。

## F53 — 底包 SGLang = GitHub 公开提交 fe236ea6c3 + 一处与我们无关的多模态修复；中途状态 h 是 bf16
- 取回的底包文件逐字节对比公开提交 `sgl-project/sglang@fe236ea6c3`（2026-09-01）：srt 407 个中 406 个相同，
  kernels 210 个全部相同。唯一差异 `srt/speculative/eagle_worker_v2.py` +13 行，只处理多模态嵌入
  （`mm_input_embeds`）在投机草稿中的对齐，纯文本负载不经过。→ 以后以 `refs/sglang-fe236ea6c3/` 为底包源码，
  不再需要诊断构建取文件（诊断构建日志另有约 2.5 MB/次 的总量上限）。
- 102 精度：`kernels/ops/attention/fla/chunk_delta_h.py:349` `h = k.new_empty(...)`（与 k 同为 bf16）；
  最终状态 `kda.py:85` 用 `initial_state.dtype`（fp32 状态池）。底包在中途位置记状态走
  `hybrid_linear_attn_backend.py:872` 从 h 取再转 fp32 —— 即 **bf16 截断过的状态**。
  101（拆分）在边界处取的是最终状态（:876，fp32，精确）。所以"单点 102 走 h"有精度代价，需要改 kernel
  输出 fp32 快照（v0.5.20 的 h_track_buf / vLLM #56960 的做法）或保留拆分。底包自己的分叉点记状态也走这条 bf16 路径。

## F54 — L3 实际运行的 SGLang 代码 = build/base_exact + 我们的补丁（证据链已闭合）
1. 提交 A/B 的镜像 0922e/0922f（构建 163117/163118）：平台记录的 Dockerfile 与本地 build/image/A|B.Dockerfile 逐字相同；
   FROM 底包 digest `sha256:f24781f0…`；内嵌补丁指纹 = 本地 000（e0d79249…）与 101（93dbc0e0…）。
2. 同一 digest 上的诊断构建：`import sglang` → `/sgl-workspace/sglang/python/sglang`，版本 `0.0.0.dev1+gfe236ea6c3.mmfix1`；
   该目录 4686 个文件的 sha256 清单（diag08）与 `build/base_exact/` 全部吻合。
3. 提交命令 `python3 -m sglang.launch_server` 加载的就是该目录。
→ 正式评测跑的源码：A = `build/l3_0922e/`（base + 000，改 2 文件），B = `build/l3_0922f/`（base + 000 + 101，改 3 文件）。
不在此范围：torch、CUDA、sgl-kernel、flashinfer 等编译好的二进制（镜像里原样继承，不改）。

## F55 — 平台拒收 `name:tag@sha256:digest` 形式的 image；45734/45735 因此部署失败（不计额度）
- 两次提交的 changelog：`trisol inference create` 422，`image_ref ... could not be verified against image center`，
  `business code 210009 imageName format is not right`；"未取得完整 benchmark 分数，本次不计入每日额度"。
- 主办方可用示例是纯 `name:tag`（task.md:442）。task.md 虽"建议钉 digest"，但平台不接受 tag+digest 组合。
- 修正：`scripts/check_submission.py` 改为禁止 `@sha256`、要求 tag；tag 一律唯一且不重建。已重交 A=45766、B=45767（仅 image 字段改动）。
- 同一 changelog 写"每日 3 次额度"；以用户确认的 2 次为准执行。

## F56 — 原版底包在真实 8×A100 上启动即崩（DeepGEMM 不支持 sm80）；A/B 提交不可能运行
- 2 卡替身（m0，底包依赖版本对齐）与 L2 pod（真实模型、8 卡、提交命令原样）一致：CUDA graph 捕获阶段，
  DSA indexer 调 `deep_gemm.get_paged_mqa_logits_metadata` → `Unsupported architecture`，8 个 TP rank 全部失败。
- 源码：`dsa_indexer_kpool._should_use_tilelang_paged_mqa_logits` 只在 sm9x 走 tilelang；预填充 `fp8_mqa_logits` 只有 DeepGEMM。
  另有 `index_kpool>1` 时预填充后端必须是 fa3/tilelang/trtllm（默认 flashmla_sparse 会 NotImplementedError）。
- 对策：补丁 110（sm80 上把三个 indexer 入口换成 torch 实现，单测相对误差 <0.5%、可 CUDA graph 捕获）
  + 启动参数固定 `--dsa-prefill-backend fa3 --dsa-decode-backend fa3`。真实 8 卡验证中。
- 公开镜像名中的 kpool / native-mqa / pagedmqa 系列即为同一问题的他人修复。

## F57 — A100 上 DSA 注意力后端必须用 tilelang，不能用 fa3（supersedes F56 中的 fa3 参数）
- 任务 005（B+110，fa3/fa3）：indexer 已过（110 生效），decode CUDA graph 捕获时 `sgl_kernel.fwd` 报
  `Only Hopper supports different V headdim`（`dsa_backend.py:_forward_fa3`，qv=512 形态只有 FA3/Hopper 实现）。
- 真实 config：64 头（TP8 每卡 8）、`kv_lora_rank=512`、`qk_rope_head_dim=0`、index_topk 2048、kpool 4、45 层、1 个 MTP 层。
  rope=0 → `tilelang_sparse_fwd` 走 `sparse_attention_fwd_kernel_v1`（关 TMA、关 warp specialization，sm80 可编译）；
  v2（set_max_nreg/mbarrier，Hopper 专用）只在 tail_dim>0 时用，本模型不触发。kpool 检查允许 tilelang。
- sm80 上 MHA_ONE_SHOT 预填充只在 sm90/sm100 启用（`dsa_backend.py:4461`），所以预填充也走 tilelang 稀疏路径。
- VERIFIED（pod 单卡 A100）：对 torch 参考相对误差 ≤3.6e-3，CUDA graph 可捕获；decode（≤64 token）单层 ≈0.32 ms，4096 token 预填充单层 6.4 ms。
  证据 `evidence/F57/`。启动参数改为 `--dsa-prefill-backend tilelang --dsa-decode-backend tilelang`；8 卡启动验证 = 任务 006。

## F58 — A100 上 FP8 MoE 专家必须走 Marlin W8A16（补丁 111）
- 任务 006（B+110，tilelang）：DSA 已过，MoE `fused_moe_kernel` 编译报 `type fp8e4nv not supported in this architecture`（Triton sm80 只有 fp8e4b15/fp8e5）。
- 底包已有 vLLM 移植的 `prepare_moe_fp8_layer_for_marlin` 与支持 kFE4M3fn 的 Marlin MoE JIT 核，但 Fp8MoEMethod 未接线；`fused_marlin_moe.get_scalar_type` 在 8 bit 只返回 uint8b128。
- VERIFIED（pod 单卡 A100）：补丁 111 后相对误差 ≈6e-3，CUDA graph 可捕获；证据 `evidence/F58/`，说明 `patches/111-sm80-fp8-moe-marlin.md`。

## F59 — B+110+111（tilelang）在真实 8×A100 启动并通过功能探测；首次形状触发服务期 Triton 编译
- pod 任务 008（000+101+110+111，`--dsa-prefill-backend tilelang --dsa-decode-backend tilelang`，env `SGLANG_OPT_DEEPGEMM_HC_PRENORM=0`、`SGLANG_OPT_USE_TOPK_V2=0`）：
  ENGINE_READY 165 s（页缓存热）；decode CUDA graph 捕获 34 s。
- 容量（每卡）：KV bf16 943,360 token（11.17 GB）；KDA 状态槽 584（ssm 9.71 GB + conv 0.34 GB，每请求 5 槽）→ max_running_requests=116；chunked_prefill 8192。
- 探测：500/1500/3000/9000/20000 token 全 200，前缀命中正常（448/1472/2944/8960），`/flush_cache` 返回 JSON 成功。
- **TTFT 异常**：500 token 58.9 s、9000 token 67.8 s；其余 0.34–1.6 s。原因：KDA Triton 核（`chunk_kda_fwd_kernel_inter_solve_fused`、`_recompute_w_u_fwd_kernel`）
  按形状在服务期编译，每个变体 1–7 s × 多个（日志自带 "pre-compile it during engine init" 警告）。评测计时阶段若遇到新形状会直接吃掉 TTFT → 需要启动期预热/编译缓存（优化项）。

## F60 — T42 / 130：完整分词线程池逐 token 等价，真实长输入显著降低 HTTP loop 阻塞（CPU）
- **VERIFIED（M3-01…05）**：base_exact 副本按 000→101→110→111→130、fuzz=0；14/14 单测。真实 glm_tok、完整 dev 722 条/34,416,777 tokens，原版/线程/关闭 IDs 逐项相同且冻结 glm_tokens 全同；另有7边界、21并发、3batch/pair对照。证据 `evidence/T42/`，复现 `scripts/test_async_tokenize.py`。
- **VERIFIED（本地 aarch64 CPU）**：transformers 5.12.1 / tokenizers 0.22.2，原样100,214 /256,733 token 对话各3组交替；最大 loop lag 的中位数同步68.20/194.57ms→线程2.48/8.59ms；分词耗时中位数69.15/195.52ms→54.77/176.63ms。直接 Rust encode_batch 204.4ms 内有102次 loop 心跳（最大 lag 4.89ms），证明此版本编码会释放 GIL；不代表零 GIL 阻塞或所有后端。
- **VERIFIED（源码/CPU）**：routing key 按 body非None→Routing-Key头→Session-ID头，batch继续透传；不写 native session/cache_salt。取消后实际工作完成才释放槽位；不加入前缀缓存。GLM `encode('hello')=[14978]`，`encode('hel')+encode('lo')=[48808,385]`，逐字节前缀相同不足以安全拼接 token IDs。
- **INFERRED**：SSE/TTFT/TPOT 可受益，但本地 CPU 数字不代表8卡吞吐或N@SLO；M3-06仍todo，完整服务导入、取消/flush/RSS和开发集A/B待Claude安排。没有GPU、Trisol、镜像或提交动作。

## F61 — T41/120 调度保护 CPU 验证与 101 既有双 partial 反例（2026-09-22，Codex W15）

- **VERIFIED（真实调度方法 + CPU mock，非 GPU 性能）**：`tests/test_sched_protect_chain.py` 27/27；000→101→110→111 副本加 120，实际 `get_next_batch_to_run` / admission / PrefillAdder 执行冷长与短命中混排、prefill/decode 交替、101 role split、Mamba 拒绝清理、KV/页/请求槽位门。`validation.json` 确认只改 scheduler/schedule_policy，4684 文件逐字节一致，3617 Python 文件 py_compile 全过。
- **VERIFIED（有条件轮次上界）**：100000 token、cap=2048、持续短请求到达：无 role split 用 49 次 prefill/97 总轮，101 尾切分用 50/99；数据 `evidence/T41/starvation_bound.json`。前提是已准入、每轮资源够 C、无 retract/abort；不是 chain_start 秒数保证。原 LPM 在无限热流下的未准入冷请求饥饿没有被修复，完整论证见补丁文档。
- **VERIFIED（off 对照）**：24 组成对决策序列字节一致；22 组各 40 轮正常，2 组基线与 off 在同一原生断言处失败。定向反例：已有 chunk 1536 token、101 在 1024 切尾，余预算允许新长请求变成第二个 partial，`scheduler.py` 原 `assert self.chunked_req is None` 失败。120 on 的已有/新 partial 守卫拒绝第二个请求；不是回退行为变化。证据 `off_parity.json` / `cpu_tests.log`。
- **INFERRED**：较少连续 prefill 可缓解 intra TTFT/TPOT；更小 chunk 增加轮数可能伤害 chain_start/吞吐。仅 CPU 不能确认数值、真实缓存回收、NEXTN 或 N@SLO，未进行 GPU/Trisol/bohr/镜像/提交操作。

## F62 — 101 在并发下触发"双 partial"断言使引擎崩溃；补丁 105 修复（真实 8 卡）
- pod 任务 011（b111 = 000+101+110+111，dev N6）：warmup 阶段 18:05:26 全部 TP rank `scheduler.py:3789 assert self.chunked_req is None` → 引擎退出；
  正式测量 722/722 请求 `Connection refused`，但任务仍记为 done（harness 不检查引擎存活）。012（N10）同样会崩，已中止。
- 根因（源码）：101 的 tail split 把本应是最后一块的续算请求再次切分（仍为 chunked，`has_chunked_req=True`），
  省下的预算又允许一个新请求走截断分支成为 `new_chunked_req`；101 的保护只看 `new_chunked_req`，漏了 `has_chunked_req`。W15 在 CPU 上也复现过（T41）。
- 修复：`patches/105-role-split-single-partial.patch`（角色切分开启且有续算分块时，拒绝再截断新请求）；与 120 上下文不冲突，000→101→105→110→111→120→130 全栈 fuzz=0 可打。
- 影响：已提交的 B（45767，含 101 无 105）即使能在 A100 启动也会在并发下崩。dev 模板已加"跑完检查引擎存活"。

## F63 — A100 上的主瓶颈是补丁 110 的 torch 版 DSA indexer（占约 60% GPU 时间）
- b112（000+101+105+110+111）dev N6 运行中：引擎全程存活（105 生效）；纯 decode 每步约 18ms（bs=1）→ 69ms（bs=6），每请求约 +10ms。
- 12 步 profile（TP0，`evidence/T43/profile_decode_b112_n6_TP0.txt`）：at::native elementwise/unrolled/reduce/gather 约 60%，外加 bf16 bmm 132 次（= 11 个 DSA 层 × 12 步）——全部来自 110 shim；
  Marlin MoE 8%、tilelang 稀疏注意力 3.5%、KDA 约 1%、allreduce 约 2%。预填充 fp8_mqa_logits 按 32 头循环 mm，同样是 TTFT 主因（推断，待 112 前后对比）。
- 对策：补丁 112 sm80 融合 kernel（T43，W17）。在此之前 120/130 的 A/B 只能看相对趋势。

## F64 — T43/112：sm80融合indexer保持110数值/边界/graph语义，开发机算子加速（W17）

- **VERIFIED（A100算子，P112-01…06）**：`patches/112-sm80-indexer-kernels.patch` + `scripts/make_112.py`；uint8软件e4m3→bf16解码，逐头MMA结果保留110的bf16舍入，再融合relu/权重/归约/scale。decode页64/4warps；prefill H32时2query×64key/4warps，clean=True跳过无交集tile，clean=False按110全宽计算。负页仍映射0；context [B]/[B,N]、N>1及尾部0保持。
- **VERIFIED（最终同源）**：`evidence/T43/final_all.log` 88组数值（B1/6/32、N1/2、ctx1024/32000/190000、边界/strides/大幅值、8192query大矩阵）全过；最大逐行相对L∞1.1185e-5，topk集合最低99.951171875%。254有限fp8编码逐bit同torch，2NaN类别同；4种graph×3次动态输入重放与eager逐bit同。真实模型形状的随机激活，不是真实模型q/K/weights。
- **VERIFIED（单卡微基准，非TP8/SLO）**：B6/N1 decode CUDA graph，32k key：0.4859→0.1023ms（4.75×），190k：2.6867→0.5917ms（4.54×）；8192query prefill causal：32k 207.93→96.38ms（2.16×），190k 1248.86→622.45ms（2.01×）；ragged：32k 207.71→76.11ms（2.73×），190k 1249.07→423.39ms（2.95×）。CUDA event中位数、排除JIT与数据生成，完整样本与eager结果见 `summary.json` / `final_all.log`。
- **VERIFIED（交付栈）**：000→101→105→110→111→112→120→130全fuzz0、3623文件py_compile、确定生成、全栈反向逐字节还原、base_exact未改；实际PTX为sm80 bf16 MMA、代表形状0 spills。`summary.json`绑定源码/oracle/测试/补丁/compiler SHA。
- **INFERRED / 开放**：应降低F63中的110算子成本，但没有8卡TTFT/TPOT或N@SLO结果；服务导入、真实激活/能力、完整NEXTN集成待Claude。未改RELEASE/队列/服务、未操作bohr/Trisol/pod/镜像/提交；不能直接用池化后key长度的算子耗时推导原prompt耗时。

## F65 — T44/113：预填充131–186等效TFLOPS，decode保持112（W18）

- **VERIFIED（P113-01…04）**：独立 `113-sm80-prefill-indexer` 叠加112。H32/nq≥32/nk≥1024先各解码q/K一次到bf16；query-major 2query×128key、每program复用query计算4个key tile、GROUP32/4warps/stages1。保留逐头bf16舍入、fp32输出、clean=False全宽及clean=True完整-inf写回；小形状/其他head数回退原112。
- **VERIFIED / 完整入口微基准**：8192query，32k causal/ragged分别96.854→15.378ms（139.6等效TFLOPS）/76.112→11.662ms（184.1）；95k为314.727→47.073ms（135.4）/227.133→34.326ms（185.7）；190k为622.164→97.047ms（131.4）/425.674→68.814ms（185.3）。相对112加速6.19–6.69×，包含每次预解码/分配；全宽等效口径2*nq*nk*32*128，剪枝计入收益，不当作实际MMA利用率。CUDA event七轮交替中位数、随机激活；证据 `evidence/T44/final_all.log`、`performance_table.md`。
- **VERIFIED / 数值**：沿用112全部用例并分别对未改110与112，加新边界及全部六个大矩阵逐行对照；222组全过，最大逐行相对L∞/L2 3.956824e-5，topk最小99.951171875%。30动态graph重放（含q/K/scale变化）与eager逐bit一致。decode函数源码、代表形状PTX SHA与T43/112完全相同；本轮32k/190k graph约0.1024/0.5925ms，F64历史数据不改。
- **VERIFIED / profile与交付**：190k causal CUDA profiler，112为624.397ms，113主kernel96.102ms、两次预解码合计0.104790ms（0.11%GPU时间）；113主kernel162寄存器/0spill/48KB shared，sm80 bf16 MMA。9补丁000→101→105→110→111→112→113→120→130 fuzz0、3623编译、确定生成、整栈反向逐字节还原/base_exact未改通过。`summary.json`绑定所有SHA；`make_113.py`可复现。
- **INFERRED / 边界**：有利于单卡prefill算子，但没有TP8/服务/能力/NEXTN/SLO证明；额外scratch最大测试形状110.39MiB/调用，完整fp32输出6.226GB仍存在，需Claude实际服务验证显存和JIT预热。未操作bohr/Trisol/pod/镜像/提交，未入RELEASE；回滚反向113恢复112。

## F66 — T45/140：KDA双点fp32快照、数值前提与离线命中估算（W19）

- **VERIFIED（P140-01/02）**：底包chunk_delta_h的fp32累加器可在一次extend直接导出角色边界及对齐末尾；卷积历史从原位conv之前的raw QKV最后3行保存。最终8组GPU算子（64×128为主，含8×128分片、65至8192长度、变长/stride/初态/全部NT_BUCKET），7组有效边界的SSM、conv、对齐末尾、续算输出/最终状态均逐元素相同，最大误差0。实际KDA forward_extend/dispatcher/Triton方法；证据 `evidence/T45/numeric_final_v3.log`。
- **VERIFIED / 数值前提**：直接保留底包按`B*NT*H<=256`选择融合intra，会在one-shot/截断前缀跨阈值时产生8.535385e-5最大SSM差（numeric_01.log）。140开启统一非融合路径后达上述零误差；关闭保持原选择，并调用源字节原样保留的原kernel。独立helper冷autotune曾有一次off比较失败、暖重跑通过；最终对照共享未改helper及其autotune决定，不据此声称所有独立冷启动天然bitexact。
- **VERIFIED（P140-03…05）**：21/21真实cache/controller/tree/tracking/pool/flush方法CPU测试，含checked allocator与原sanity_check；额外role槽请求→树移交/重复释放/abort清理、锁、tail优先淘汰、busy/idle真flush通过。off缓存三种branch轨迹JSON字节相同；off调度32组×30轮=960轮同原栈，on8组无双partial；role示例2→1次extend。CPU不等于服务/overlap长跑验证。
- **VERIFIED（P140-06 / MODEL OUTPUT）**：原Renderer+glm_tok、722请求冻结token数全匹配，311独立链prompt-only无限缓存串行模型，无未来oracle。chunk8192，101+105 off/on命中16,886,400→16,903,104（+16,704），extend3284→2616；chunk2048为16,806,912→16,905,152（+98,240），9430→8949；分别6/73请求改善、均0退步。没有模拟decode状态、淘汰/容量压力、实际并发准入/retraction，不是实测cached_tokens/SLO。
- **VERIFIED（P140-07）**：140补丁/说明/make_140已交付；000→101→105→110→111→112→140→120→130 fuzz0、3622源码+10工具编译、确定生成、全栈反向字节还原；开关默认0。未入RELEASE/构建/队列。
- **INFERRED / 开放项**：减少调度轮次和保留最新角色状态可能改善链中间延迟，但固定非融合intra的短请求成本及额外17.6MiB/rank状态驻留可能抵消收益；实际TP8/权重/overlap/能力/SLO未测。当前限制普通TP extra_buffer，NEXTN/HiCache/lazy/int8/unified/DP/CP/PP/PD/mixed/TBO等拒绝开启；8卡A/B交Claude。全程仅指定开发机arena内GPU1算子，未操作bohr/Trisol/pod、8卡、镜像或提交。

## F67 — T46 / 150：请求预热的覆盖边界与真实清理（W20）

- **VERIFIED（源码）**：`http_server.py` custom warmup在lifespan yield之前；但scheduler在更早的init阶段调用`triton_load_watch.mark_serving_started()`，该模块明确说明request-driven warmup仍受“after serving started”告警约束。不能把这条日志字符串直接等同于HTTP ready后编译；需时间分界。证据：底包`sglang/srt/utils/triton_load_watch.py:19`、`managers/scheduler.py:1789`。
- **VERIFIED（源码，覆盖不完整的反例）**：112 `_ragged`与113 `_prefill`的NQ/NK、113 `_unpack_prefill`的R均为constexpr；autotune key有限不等于JIT specialization有限。600→601可能新编译；发并发6…32不能保证实际eager B6…32，graph可能padding。完整清单`evidence/T46/jit_inventory.json`（667显式Triton函数），重点63函数/12autotune及取值/缺口见补丁150说明。未修改112/113 kernel。
- **VERIFIED（P150-01…05，CPU/源码）**：150 `@warmup("ax_shapes")`；48组564个合成请求另加真清探针，直接input_ids覆盖指定链/长冷/角色/边缘/ragged/并发。21项CPU测试通过：真实flush wrapper/mixin/scheduler方法+mock池、请求序列与drain、取消/异常、全worker汇总、TP非主rank失败归并、空池断言、零命中探针、log_metrics与exporter守卫。全11补丁fuzz0、3623源码+8工具编译、确定再生成、反向逐字节还原/base不变通过；证据`evidence/T46/`。
- **INFERRED / 待L2**：应把已实际执行的冷形状编译移到HTTP ready前，但未证明任意服务形状无编译、实际模型启动/缓存恢复或SLO改善；NT_BUCKET2、graph外形状与其它路径见说明缺口。没有8卡/Trisol/pod/镜像/提交操作。开发机仅算子验证单列收据，不替代全栈。
- **VERIFIED（P150-06，A100算子）**：最终新cache运行`operators_cold5.log`的12组首次调用共113个JIT miss/113实际编译/102次autotuner._bench，编译磁盘命中0；同tensor/shape再跑12组，JIT miss/编译/autotune/新增或变化cache文件全部0。113 query长度600→601新增2次编译（16文件），证明150不能用有限请求覆盖所有新长度。开发机A100-SXM4-80GB、Torch2.13.0+cu130/Triton3.7.1；随机激活，未加载模型，取证秒数不是SLO。30个远端测试/算子源码SHA与交付匹配。早期夹具/JSON序列化失败原始日志保留。
- **VERIFIED（P150-06，跨进程缓存）**：同A100环境新进程复用cache，首次12组57 JIT内存miss/57编译磁盘命中/0实际编译，仍42次autotune benchmark；12组重复再次全0。`operators_persistent.log`说明FLA部分未传cache_results的装饰器仍会调优，预置cubin不能取代启动期请求预热。测试已退出，两卡4MiB/0%，见`gpu_final_idle.log`。

## F68 — T47/W21：112/113 v2 新长度不再产生精确形状JIT，原性能门通过（VERIFIED）

- `scripts/kernels/sm80_indexer_{112,113}.py`四kernel的N/NQ/NK/R/P/S与全部stride去掉显式constexpr，仅模型/tile/CLEAN常量保留；默认值1和16整除特化保留。`evidence/T47/kernel_keys.md`与`cache.log`实际286个编译键核对，可变参数进入常量表的值仅1，没有其它精确长度或stride。
- P47-01：A100独立cache有限类别预热634调用，286 JIT miss＝153实际编译＋133磁盘命中（两版共用代码）；随后50个不同NQ/NK/P/batch×两版×prefill/decode＝200调用，另600→601×两CLEAN×两版8调用，JIT miss/实际编译/磁盘命中均0。修复F67/T46的112/113 v1长度反例；不推翻F67对其它kernel/整服务预热范围的限制。
- P47-02：原112 88数值/12动态graph，113 222数值/30graph全部通过；对110最大逐行相对L∞1.11846e-5/3.95682e-5，最低topk99.95117%。同机v1/v2 16行配对性能，最大退步3.38%<5%；原tile保留。数据为随机激活，完整表见`evidence/T47/performance_table.md`。
- P47-03：完整11补丁fuzz0，3623源码+6工具py_compile，确定再生成/反向字节还原/base未改通过；12远端输入SHA与本地一致。生产补丁原名更新，v1在`patches/drafts/*-v1.*`。最终汇总`evidence/T47/summary.json`，仅开发机arena GPU0算子，未操作8卡/服务/镜像/提交；固定模型/dtype/布局类别未热时仍可发生有限首次编译，TP8/SLO交Claude。

## F69 — T48/W22：NEXTN sm80完整路径、兼容补丁160与KDA回滚算子（VERIFIED / 源码及L1）

- **源码**：NEXTN解析为EAGLE，GLM必须显式`--speculative-draft-model-path /mnt/models`；48仅在MR未指定时写入（`base_exact/sglang/srt/arg_groups/speculative_hook.py:654`）。draft构造单层Deepseek NextN DSA/MoE，没有KDA和target mHC；target verify仍经过34 KDA+11 DSA。共享索引seed宽2051，3个proposal来自draft-extend首个及2次draft-decode。全文件/行号、不兼容入口和权重ignore映射见`research/codex/R17_nextn_sm80.md`。
- **修复/交互**：110已修spec多pool FP8写，111覆盖draft专家，112/113代理覆盖verify logits。160只补sm80 GLM EAGLE兼容配置（两DSA tilelang、BF16 KV、KDA Triton、禁DeepGEMM HC/TOPK计划V2），复用这些kernel。140角色槽与spec scratch没有已证实物理别名；140显式拒绝spec，组合生命周期未验证，160启动时关闭140并清101角色IDs，保留原extra_buffer和verify tracking/rollback。120首轮off，150对MTP跳过。
- **P160-01/02/03**：10CPU+真实ServerArgs.resolve_once/draft ModelConfig/quant mapper通过。A100 KDA 5形状、20个接受长度，fused对unfused输出最大绝对误差7.45e-9、SSM3.73e-9；verify不提前提交SSM，接受后active/tracking/conv回写逐元素相同，含graph。kpool T2/4/6量化误差0；TileLang DSA4形状relL2≤0.001980；112/113两个indexer路径max_abs1.431e-6；共享seed选择/carry/finally清理通过。
- **P160-04/05**：EH norm、argmax、greedy/单热点target-only采样、accept prologue、普通2048与pool512 topk、mHC pre/post、FP8 MoE clip10数值/graph通过；MoE缩小专家数33，真实288+1未完整加载。12补丁fuzz0、3624源码+8工具编译、确定再生成/整栈反向还原/base未改，4689远端候选文件与本地hash一致。全部最终覆盖/数值以`evidence/T48/README.md`和`summary.json`为准，不作全kernel/服务证明。
- **统计/INFERRED资源**：160原日志新增同窗口spec tokens/rounds，采集器按Σtokens/Σrounds求每request-step接受长度（包含target保证token，不冒充HTTP输出数）。D4/MR32 scratch约2.268GiB/rank，MR48约3.368GiB；auto-fit可能使持久KDA槽约为普通的55.6%，并非保持原容量。稳态TPOT倍率=r/A，r1.4且A1.6–2.0时约0.875–0.700；这是敏感性假设，不是SLO实测。
- **边界/纠正**：仅开发机随机算子，没有TP8/完整权重/服务/能力/flush/SLO证据；未触碰bohr/Trisol/pod、镜像或提交。早期C++ JIT实际读取SGLANG_JIT_CACHE_DIR，7个T48 build误写根盘；发现后停自有编译，按源码路径确认归属后迁入arena，最终runner已修，迁移证据保留。准备的8卡脚本仅交Claude审阅。
- **P160-04补充 / 最终收据**：原生dense FP8 Marlin 9形状与27次动态graph均通过，最大relL2约0.002843；clip10 MoE四形状+graph≤0.00589。KDA追加tracking卷积与masked槽直接断言后重跑通过。`scripts/summarize_160.py`校验全部最终PASS、候选/脚本/config/runner SHA与两卡空闲，输出`evidence/T48/summary.json`；完整服务资格仍保持P160-06 todo。

## F70 — INT8 W8A8 MoE 在 A100 上不值得（开发机实测）；MoE 预填充只有 bf16 峰值约 27–30%
- 形状：GLM TP8 每卡 E=288、K=4096、N=256、top-8、FP8 块 128×128；随机权重（块内幅度抖动）；参考 = FP8 反量化权重的 fp32 精确计算。
- M=8192：Marlin FP8 W8A16 4.99ms/82.5TF/误差 6.1e-3；Triton INT8 分块 4.88ms/84.6TF/**4.6e-2**；Triton INT8 逐通道 4.32ms/95.6TF/**2.6e-2**；Triton bf16 4.70ms/87.7TF/4.9e-3。M=16384 结论相同。
- 结论：现有 Triton INT8 只快 0–16%，而精度差 4–8 倍（FP8→INT8 重量化损失块内小值），能力门风险大 → R8 A1 降级。
  所有路径都停在约 85–95 TFLOPS（每专家约 230 token 的小 GEMM + 镜像无 A100/E=288/N=256 调优配置）→ 新方向"A100 MoE 形状调优"。
- 开发机环境（torch 2.13/Triton 3.7.1），**须在 pod 复测**：`scripts/pod/verify/bench_moe_int8.py` 已进验证套件。证据 `evidence/INT8/`。

## F71 — GLM-5.3 单卡（TP8 份额）逐组件成本表：DSA 稀疏注意力 + mHC ≈ 37%（两者在 8 卡上重复），MoE 33%，KDA 仅 5%
- 方法：SGLang 真实模型代码 + 随机权重，8 层缩小版（[KDA×3, DSA]×2，全 MoE），形状 = TP8 单卡份额（注意力/KDA 8 头、MoE N=256×288 专家、indexer 32 头与低秩投影全量、mHC hc_mult=4），`sglang.benchmark.one_batch` + torch profiler；
  脚本 `scripts/analysis/{make_rank_model.py,rank_profile.sh,component_table.py}`（pod 可原样重跑）。开发机 A100，全补丁树（000–160）。
- 预填充 8192 token（冷，上下文 8k），8 层 GPU 111.4ms：MoE 36.4（32.7%）、DSA 稀疏注意力 24.1（21.6%，每 DSA 层约 12ms）、稠密 GEMM 17.3（15.5%）、mHC 14.8（13.3%）、逐元素 9.1、KDA 5.2（4.7%）、indexer 3.7（3.3%，随上下文增长）。
- 外推到 45 层每卡每个 8192 块约 580ms（不含 allreduce；indexer 在 10 万上下文约 +110ms）→ 19.5 万 token 冷启动单独约 15s。
- decode（bs=1，上下文 8k）每步 GPU 2.5ms：稠密 GEMM 32%、MoE 17%、mHC 14%、稀疏注意力 13%。
- 含义：稀疏注意力（每卡重读同一潜在 KV）与 mHC（每卡算全量 token）是结构性 8× 冗余，按 token 切分可去掉约 1/3 预填充时间 → 支持 R8 §7 G1/G3/G4；MoE 为 F2。开发机结果，须 pod 复测。

## F72 — 补丁 114（indexer 按查询行切分）逐位等价；TP2 6.5 万 token 预填充 −3.8%
- 见 `patches/114-indexer-row-shard.md`。开发机结果，8 卡待测（TP8、真实长 prompt）。
- 附带：开发机双卡 NCCL 在 SGLang 进程内需 `--disable-custom-all-reduce`、`NCCL_CUMEM_ENABLE=0`、降低 mem-fraction（纯 NCCL 冒烟正常）。

## F73 — 8 卡 b113 启动探测通过；能力冒烟 12/12（长思维链正常）
- lh-arena-sess-b，任务 001（b113 = 000+101+105+110+111+112v2+113v2，tilelang）：500–20000 token 探测全 200、前缀命中正常、flush 正常；首次新形状仍有约 60s KDA Triton 编译（补丁 150 待测）。
- 任务 002 能力冒烟（`scripts/pod/jobs/cap_smoke.sh`，12 道 Python 校验答案的数学题，/v1/chat/completions，不限长度）：**12/12 正确**，全部 finish=stop，平均 2209 输出 token，最长 13964 token（163s，12 路并发）。
- 含义：长 decode 路径（112 decode indexer、Marlin、tilelang decode）数值未坏；不代表官方 AIME/GPQA >90 门槛。

## F74 — 真机（L2 pod，比赛镜像）复核：开发机 kernel 结论 11/11 成立
- 任务 003（`scripts/pod/jobs/verify_kernels.sh`，全补丁树 000–160，单卡）：F57 tilelang 稀疏注意力、F58 Marlin MoE、112/113 数值+graph、T47 换长度不重编译、140 KDA 快照、160 MTP 四组算子、INT8/MoE 基准 —— **全部 PASS**。
- 环境：8×A100-SXM4-80GB，驱动 580.105.08，torch 2.13.0+cu130（与开发机同版本 torch）。
- INT8/MoE 基准与开发机一致（M=8192：Marlin 4.99ms/82.7TF；INT8 逐通道 95.4TF 但误差 2.6e-2）→ findings 中 INT8 结论在真机确认。

## F75 — 8 卡 b113 实时日志（dev N6 进行中）：decode 已修好（bs6 16.4ms/步）；冷预填充 746ms/8192 块；N6 时 KV 峰值已占 50%
- 工具：`scripts/pod/verify/logstat.py <server.log> [port]`（解析 TP0 调度行 + /metrics）。
- decode（连续 decode 日志行）：bs1 10.8ms/步、bs6 16.4ms/步（110 torch 版为 18/69ms）→ 每增一请求约 +1.1ms（原 +10ms）。
- 冷预填充续块（cached=0，n=291）：约 11k tok/s ≈ 746ms/8192 块（成本表估算 580ms + allreduce 等，吻合）→ 19 万 token 冷启动预填充约 18s。
- 池占用：KV 峰值 0.50（N6！整机约 94 万 token）、KDA 状态 0.04。→ 推断约 N14 起 KV 满、开始驱逐；**容量（DCP/cuda-graph bs/FP8 KV）优先级上调**。
- 注意：日志 "input throughput" 以日志间隔计，首块含等待/decode 时间，不可直接用；/metrics cache_hit_rate=0 疑似统一缓存未更新 → 以 harness 逐请求 cached_tokens 为准。

## F76 — 8 卡 b113 开发集 N6 基线：卡在 intra 两门（排队所致），chain_start 有余量；冷预填充约 1 万 tok/s；真机 profile
- 结果（`evidence/N6_b113/analysis.txt`，`scripts/pod/verify/analyze_run.py`，允许超标数按 95% 下界规则自算）：722/722 成功；
  fast_intra p95 7.51s（超标 55 / 允许约 23，FAIL）、overall_intra p95 7.52s（44/27，FAIL）、turn_start 4.01s（PASS）、chain_start 19.86s（5/22，PASS）。TPOT 均值 0.0304，p95 0.073。
- **intra 超标主因是排队**：intra 排队 p95 6.4s，exec→首 token p95 1.0–2.8s；最差 10 个请求排队 8–15s、未命中仅数百到数千 token → 冷预填充连续占用 GPU（补丁 120 的目标）。
- 缓存：总实际命中 2485 万 ≥ 冻结期望 2244 万（×1.107，跨会话共享带来额外命中），但 19 个 intra 请求相对同链前一请求丢失 >4096 token：
  (a) 长空闲（244–300s）后几乎全丢（原因待查，当时 KV 峰值仅 50%）；(b) 短间隔丢 5–8k（KDA 状态停在上一轮角色边界而非末尾 → 补丁 140 的目标）。
- 冷预填充单请求（012b）：2 万 1.98s、6 万 6.37s、19 万 19.3s（≈1 万 tok/s）。8 卡真实 profile（6 万请求 3 块，≈645ms/块）：MoE 31.8%、稀疏注意力 17.0%、稠密 GEMM 13.2%、mHC 11.8%、**allreduce 11.1%**、逐元素 6.1%、indexer 4.3%、KDA 4.1%（与单卡成本表吻合）。

## F77 — 补丁 120 在 8 卡 N6 上消除了 intra 排队：四门全过（自估规则）；chain_start 尾部变差
- b113+120（cap 2048、short 4096），dev N6（`evidence/N6_b120on/analysis.txt`）对比 F76：fast_intra p95 7.51→**1.01s**（超标 55→9）、overall_intra 7.52→**5.29s**（44→21，允许约 27，余量仅 +6）、turn_start 4.01→5.69s、chain_start 19.86→21.56s（max 73.5→**120.7s**，超标 5→12）。
- intra 排队 p95 6.41→**0.36s**；chain_start 排队 p95 6.8→10.7s、exec→首 token p95 10.3→14.2s（冷续块被切成 2048 → Fable 指出 cap 在无等待时也生效）。
- 缓存丢失模式不变（lost 4.68M），极端例（25 万 prompt cached≈0）仍在 → T49 分析中。

> 更正（09-23）：上条"补丁 120 … 四门全过"为**正式规则估算**（超标率 95% 下界 ≤5%）。开发集 harness 用**硬性 p95** 判定：overall_intra p95 5.29s > 5s → harness **FAIL**（report_*.md）。今后同时报告两种判定（analyze_run.py 已更新）。

## F78 — 8 卡探针：mHC 输入分散 −6~9%；DCP 8（+114+115）KV 逻辑容量 ×7.8、19 万冷预填充 −19%，能力冒烟均 12/12
- 冷预填充首字（秒，20k/60k/190k）：b113 1.98/6.37/19.29；`--enable-attn-tp-input-scattered` 2.79¹/6.01/17.52；`--dcp-size 8`+114+115 2.04/5.89/**15.58**。¹新引擎首请求含编译。
- DCP：每卡 914,688 token 槽，调度器按 ×attn_dcp_size 计（`scheduler.py:2249,2423`、`tp_worker.py:430`）→ 逻辑约 **732 万 token**（原 94 万）；max_running_requests 113。原版 DCP 因 tilelang 64 头 256KB 共享内存失败，补丁 115 修复（F 前述）。
- 与 Fable 预测（DCP 预填充注意力约 3× 慢）相反：实测更快 —— 64 头一次读 KV 的效率收益大于每卡多算的头。decode TPOT 与梯子表现待测（ladder_dcp）。
## F79 — T49：b113 N6缓存/容量口径纠正与逐请求源码归因（VERIFIED；候选根因单列INFERRED）

- **supersedes F75/F76的容量口径，纠正F77超长缓存例归因**：Unified统计`pool_stats_observer.py:249`从capacity同时减free和evictable，50%/4%是不可淘汰占用，不能证明两个物理池未满。原方法CPU夹具证明两池free=0仍输出0.50/0.0411（T49-01）。Claude回传012完整measurement日志后，峰值为**863296 KV token/0.92**，状态非evictable最大24；50%是中途观察，不是全程峰值。`evidence/T49/remote/012_server.log:5540`。
- **启动账已闭合**：载权增量39.15GiB/rank；KV943360×12716B=11.17193GiB；KDA584+padding共10.05335GiB（SSM9.71191+conv0.34143）；真实decode graph捕获bs≤116、增量**1.31GiB**，不是6.1。f=0.77805含VLM折减，运行余量约17.25GiB；capture后最小free15.78GiB不是prefill峰值空余。`012_server.log:14/:95/:128/:135/:156/:199`，完整账见R18§7、`boot_memory_audit.json`。
- **T49-03/08，真实输入核对**：25万prompt/cached64及最大五条“lost”均为实际cohort链首，没有本次同链前驱。严格idx>0、冻结LCP−cached>4096筛出**18 intra+1 turn_start**；重渲染36个prompt、长度全匹配，raw374真实LCP52400而冻结93013，“丢43797”收窄为3184；raw721实际差5053而非6536。逐项raw行号、LCP与rid在R18§4.3/`remote_analysis.json`。
- **短回退证据与INFERRED根因**：5条命中恰落在前请求cached+k×8192较早chunk末尾，下一LCP早于后一个状态；例如raw255从23040计算到31232/39424再剩50，角色在38720，101只在最后chunk扫描，漏掉角色；raw257 LCP38781只能命中31232。源码与批日志`:2900/:2901`支持，仍缺逐节点事件。另branch优先覆盖end/105准入保护可产生回退（原方法CPU T49-02）。140在角色所在extend双点导出直接覆盖这类限制，且已经修改FULL/MAMBA/path-cap淘汰；额外角色槽只取真free，池满可跳过。
- **长idle证据与INFERRED根因**：raw476/626的当前实际LCP均大于上次已使用深度，但cached分别77952→33600、33600→1216；raw695短idle+14.77s等待也17664→576。measurement无flush/restart/retract日志，path cap=-1；LRU/分配压力解释强，尚不能区分FULL先满或状态槽淘汰连带删KV。不得写成已确认5分钟TTL或已确认KV单池原因。
- **容量算术，不是配置实测**：N26按请求均值约124万token/14.68GiB，按完整N6峰值线性压力约374万/44.30GiB。固定总池r0.9→0.5只增KV约1.27×；graph512→64含VLM自动预算只约+3.8%；再固定状态200约157万/1.66×，不能称零风险2×。FP8 MLA+既有index布局约1.75×；DCP+115仍须核验DSA逻辑/物理地址、indexer和高虚拟slot，不用启动成功代替正确性。
- 报告`research/codex/R18_cache_loss_and_capacity.md`含文件:行号、修复方案、T49-04…07待验证方案和过时文档清单。本会话只读源码/回传日志、执行本地CPU夹具/重渲染；未修改引擎/补丁、操作GPU/8卡/pod、构建或提交。

## F80 — 容量零风险杠杆实测：KV 94.3 万 → 156.9 万 token（+66%）
- `--cuda-graph-max-bs-decode 64 --max-mamba-cache-size 200`（探针 022）：KV 每卡 1,569,152 token（18.58GB，原 943,360/11.17GB），KDA 槽 200（ssm 3.34GB，原 9.71GB），max_running_requests 40；冷预填充不变（2.08/6.12/19.17s）。
- 仍有约 16GB 可用显存（available_gpu_mem 16.08GB）→ mem-fraction 可再加。与 DCP（逻辑 ×7.8）互为替代/叠加，待梯子对比。

> 更正（09-23，Codex R18 核验）：① F75/F76 的 "KV 峰值 50%、KDA 4%" 是**扣除可淘汰缓存后的占用**（`pool_stats_observer.py:249,271`），不是物理驻留；完整 N6 测量期 KV 不可淘汰峰值 **863,296 token / 92%**。② 012 的 decode CUDA graph 实际增量 **1.31GiB**（非 6.1GB）；容量探针的 +66% 主要来自 KDA 池 10.05→3.46GiB。
> ③ "25 万 prompt cached=64" 等最大损失例均为**本次 cohort 链首**（无同链前驱），不是运行中丢缓存；"19 个 intra" 实为 18 intra + 1 turn_start，其中 5 条短回退符合"101 只在最后 chunk 拆角色点"机制，140 可覆盖（但压力下额外角色槽可能被跳过）。详见 research/codex/R18_cache_loss_and_capacity.md。

## F81 — chunked_prefill 16384 使自动 mem_fraction_static 从 0.7885 降到 0.646 → KV 从 157 万掉到 63 万，运行时 27GB 闲置
- 024（120v2 + 输入分散 + 扩容 + chunk 16384）：max_total_num_tokens 633,536（7.5GB），available_gpu_mem 27.22GB；对照 022（chunk 8192）1,569,152 / 16.08GB。
- 对策：025/026 显式 `--mem-fraction-static 0.75`（估计 KV ≈ 130 万、激活余量约 20GB）。024 保持原样作参照。
