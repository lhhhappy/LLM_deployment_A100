# 04 — Model compute, kernels, MTP and DP-attention (base image, sm80)

> **2026-09-28 适用范围更正（Codex）**：9月24日“已开启 attention 输入 scatter”的证据是旧 S1/042，不是现行 chain-max。081 未开启，085 曾单独开启；47043 FINAL、eznb/eznc、124m ON/OFF 均未传 `--enable-attn-tp-input-scattered`，运行日志为 False。114 的 indexer 行分片仍开启，两者独立。当前 117 已用 Humming；以下旧 profile 百分比不作为当前瓶颈排序。复核路径、旧实验取舍和开发方向见 [当前 scatter 核对](../../../notes/reports/attention-input-scatter-scope-0928.md)。
>
> **历史测量（2026-09-23，8 卡 + Fable 审阅）**：当时 MoE 走 Marlin W8A16（111），DSA 为 tilelang；冷预填充约1万 tok/s（36k约3.6秒），MoE31.8%、稀疏注意力17.0%、稠密GEMM13.2%、mHC11.8%、allreduce11.1%、indexer4.3%、KDA4.1%。这些比例与 F70 的 MoE 27–30% 利用率属于当时形状，不能直接套给当前配置。

> 旧的 09-23/09-24 逐项复核与路线筛选仍见 [成本复核](../../../notes/codex-分析-2026-09-24.md#10-对prefill效率是根的逐项复核与可改代码) 及 [更新的路线筛选](../R9_upstream_since_base.md)；其中“S1 已启用 attention 输入 scatter”一句只对 042 时代成立。

2026-09-22, Claude subagent, read-only source map. Paths below are relative to `build/base_exact/sglang/` (the image replica; upstream fe236ea6c3). `S/` = `srt/`, `K/` = `kernels/`. Model facts come from `s1-dev/glm_tok/config.json`. **[V]** means verified in the Python source. **[D]** means derived arithmetic. **[U]** means unverified: it depends on compiled binaries (sgl_kernel, deep_gemm, triton, tilelang) or on runtime. No GPU was used.

## 1. Model structure (`S/models/glm5_next.py`)

- **Layer order [V].** `is_kda_layer` uses `linear_attn_config.kda_layers` (`S/configs/glm5_next.py:244`). The config pattern is `KKKF` ×11 plus a final `K`: full MLA/DSA at layers 3,7,…,43 and KDA at the other 34. MTP is layer 45. The dispatch is at `glm5_next.py:632-669`: KDA → `Glm5NextLinearAttention` (`:327`), otherwise `DeepseekV2AttentionMLA(skip_rope=True)` (`:648`). `qk_rope_head_dim=0`, so the MLA latent is 512 with no rope part.
- **FFN [V].** Layers 0-2 are dense (`first_k_dense_replace=3`, `:823-828`), intermediate size 12288. The other layers use `Glm5NextMoE = DeepseekV2MoE` (`:110`, `:688`): 288 experts, top-8, sigmoid router with fp32 scoring, one shared expert. The shared expert is fused as an extra expert on sm≥80 when EP=1 (`:1339-1358`).
- **mHC [V].** `hc_mult=4`, so the residual stream is 4×4096. Each layer calls `hc_pre` twice (`:775-812`). These calls go to `K/ops/layernorm/mhc.py:1813`: the TileLang `mhc_pre` by default, plus the DeepGEMM `tf32_hc_prenorm_gemm` (`mhc.py:1009-1019,1621`), controlled by env defaults at `S/environ.py:1361-1362`. The model hook turns these off only for sm120/HIP (`S/arg_groups/model_hook.py:333-345`), not for sm80. **[U]** Upstream DeepGEMM is sm90+, so whether this works depends on the image's binary.
- **KDA projections are BF16 [V].** `quantization_config.modules_to_not_convert` lists every KDA projection (`q/k/v/b/f/g/o`), `kv_b_proj`, the indexer weights and the router. The fused qkvbfg path requires `quant_config is None` (`:370`), so the checkpoint takes the unfused path (`:401-456`). Conv weights are fp32 params (`:467-475`), and `gate_lower_bound=-5` is set at `:519`.
- **Dense FP8 GEMMs on sm80 [V].** `Fp8LinearMethod` auto-enables Marlin when 80≤sm<89 (`S/layers/quantization/fp8_utils.py:2069-2075`, `fp8.py:465-471`). `apply` then calls `apply_fp8_marlin_linear` (`fp8.py:972`). This is weight-only W8A16 with BF16 math, and no block-FP8 GEMM runs on A100.
- **MoE backend on sm80 [V].** The backend resolves to `MoeRunnerBackend.TRITON` (`fp8.py:2372-2382`; DeepGEMM is only used with a DeepEP-family a2a, `:1113-1133`). Marlin is used for MoE only with FP4 experts (`fp8.py:398,1701`), so our FP8-block experts run the Triton `fused_moe_kernel` with `use_fp8_w8a8` and block [128,128] (`K/ops/moe/fused_moe_triton_kernels.py:506-596`). Activations are quantized to FP8 per token group, then fed to `tl.dot(fp8,fp8)`. **[U]** Whether this is legal and fast on sm80 depends on the Triton version, because Ampere has no FP8 MMA. **[V]** The image has no tuned A100 config for E=288 or 289 FP8 block: the only E=288 file is `E=288,N=64,device_name=NVIDIA_A800…` in `S/layers/moe/moe_runner/triton_utils/configs/`. Lookup is at `fused_moe_triton_config.py:36-110`, so the kernel falls back to the default config.

## 2. KDA

- **Prefill [V]: Triton `chunk_kda`.** `--linear-attn-backend` defaults to `triton` (`S/server_args.py:2654-2661`), and prefill and decode inherit it (`S/layers/attention/linear/utils.py:76-101`). `KDAAttnBackend` is chosen for `glm5_next_config` (`S/layers/attention/attention_registry.py:508`). The call path is `TritonKDAKernel.extend` → `K/ops/attention/fla/kda.py` `chunk_kda_fwd`, with chunk 64 (`kda.py:1098`), the state recurrence `chunk_gated_delta_rule_fwd_h` and the output kernel `chunk_gla_fwd_o_gk` (`:1162-1186`).
- **Other prefill backends do not help on A100.** cutedsl and nvidia_kda need SM100 (`kernels/kda_cutedsl.py:36`, `kda_nvidia.py:83`), ptx_kda needs SM103 (`kda_ptx.py:54`), and flashinfer KDA needs sm≥10 (`kda_flashinfer.py:49`). FlashKDA needs the optional `flash_kda` module (`kda_flashkda.py:17-27`), whose CMake has no sm80 target (findings F-series). The safe gate (`lower_bound` is set for GLM) forces any kernel without `supports_safe_gate` back to Triton (`kda_backend.py:346-350`). Changing the *base* `--linear-attn-backend` also disables `extra_buffer`, which requires `linear_attn_backend == "triton"` (`S/arg_groups/overrides.py:547-552`).
- **Decode [V]: Triton `fused_sigmoid_gating_delta_rule_update(is_kda=True)`** (`kernels/kda_triton.py:124-156`). GLM sets `lower_bound`, so the packed-decode fast path is skipped: it requires `lower_bound is None` (`kda_backend.py:652-655`). Decode goes through `kernel_dispatcher.decode` (`:688`) instead. The K3 fully fused decode (`:590`) runs only for kimi_k3.
- **conv1d [V].** Prefill uses Triton `causal_conv1d_fn` in one packed call over the qkv width (`kda_backend.py:756`). Decode uses Triton `causal_conv1d_update` (`:642`) (`K/ops/mamba/causal_conv1d_triton.py`).
- **State dtypes [V].** The SSM state is fp32 by default and the conv state is bf16 (`S/configs/mamba_utils.py:47-104`). They can be overridden with `SGLANG_MAMBA_SSM_DTYPE` / `SGLANG_MAMBA_CONV_DTYPE` (`environ.py:1268-1269`). The per-chunk `h` is allocated in `k`'s dtype, which is bf16 (`fla/chunk_delta_h.py:349`). An extend-time track snapshot is copied from `h` and then `.to(fp32)` (`S/layers/attention/hybrid_linear_attn_backend.py:874`). That means mid-prompt checkpoints are BF16-rounded, while the end-of-prompt final state stays fp32.
- **Per-token prefill cost [D].** KDA projections are about 136 M MACs per token per layer, or 34 MFLOP/token per rank at TP8. The chunked recurrence is about 64 heads × ~160 kFLOP ≈ 10 MFLOP/token per layer, or about 1.3 MFLOP/rank. KDA prefill is therefore GEMM-dominated (BF16 cuBLAS). The Triton chunk kernels mainly add launch and memory overhead, including writing `h` at 256 KB per 64-token chunk per layer per rank. The mamba state per slot is 34×(64/8)×128×128×4 B = 17.8 MiB/rank at TP8.

## 3. MLA/DSA on sm80

- **Backend [V].** `attention_backend="dsa"` and `page_size=64` are forced (`S/arg_groups/model_overrides/deepseek_v2.py:46,102`). The KV dtype defaults to `bfloat16` when sm<10 (`overrides.py:633-635`). The split backends then resolve to **prefill=`flashmla_sparse`, decode=`fa3`** (`overrides.py:740-744`), logged as `Set DSA backends for …`. The MHA one-shot shortcut only fires on SM90/SM100 (`dsa_backend.py:4461-4467`), so every A100 prefill goes through the absorbed-MLA sparse path.
- **Possible hard failure on A100 [V in Python, U at runtime].** GLM has `index_kpool=4`. `_resolve_kpool_tail_backend` maps `flashmla_sparse` → fa3 only on sm9x and → trtllm on sm10+; on sm80 it stays `flashmla_sparse` (`dsa_backend.py:858-873`). `_check_kpool_tail_backend` then raises `NotImplementedError` for anything other than fa3/tilelang/trtllm (`:840-856`, called at `:2947-2948`). The indexer skips top-k only when `max_kv_len ≤ index_topk` (2048) (`S/layers/attention/dsa/dsa_indexer_kpool.py`, around `:1600`). So on this reading, any prompt longer than 2048 tokens would fail with default flags on sm80. In addition, the `sgl_kernel.flash_mla` sparse kernel is Hopper-class. Before trusting the defaults, confirm the boot log line and a long prompt on a real A100.
- **fa3 on sm80:** the base Python/AOT capability checks originally made this appear plausible, but the GLM-specific A100 path failed at runtime (F57). S0 uses tilelang for DSA prefill and decode; do not use `fa3` as an A100 fallback for this model.
- **Indexer [V].** `IndexerKPool` is used when kpool>1 (`S/models/deepseek_v2.py:1859`). Logits come from `deep_gemm.fp8_mqa_logits` / `fp8_paged_mqa_logits` (`dsa_indexer_kpool.py:910,998,1112,1346`). The TileLang paged variant is used only when `arch_major == 9` (`:831-836`), and there is no torch fallback in this path. **[U]** DeepGEMM on sm80 depends on the binary; this matches the D6 open item.
- **Top-k [V].** `--dsa-topk-backend sgl-kernel` is the default (`server_args.py:1816-1823`). `SGLANG_OPT_USE_TOPK_V2=0` turns off the v2 JIT kernel, which uses thread-block clusters and hangs graph capture on A100 (`dsa_topk_backend.py:52-53,123-151`). With it off, the legacy `fast_topk_transform_fused` / `…_ragged_fused` kernels run (`:156-185`).

## 4. MTP / speculative decoding

- **Support [V].** For the draft, `ModelConfig` rewrites the architecture to `Glm5NextForConditionalGenerationNextN` and sets `linear_attn_config=None` (`S/configs/model_config.py:743-748`). The draft class (`S/models/glm5_next_nextn.py:24-80`) is a single DeepseekV2 decoder layer, with no KDA and no mHC. It uses MoE and DSA MLA. The draft backend is full attention on layer [0] (`attention_registry.py:529-531`). The organizer's change to `deepseek_nextn.py:224-246` affects only the multimodal embedding path.
- **Flags [V].** Use `--speculative-algorithm NEXTN` (an alias for EAGLE, `S/arg_groups/speculative_hook.py:69-76`). You must also pass **`--speculative-draft-model-path /mnt/models`**. Glm5Next is missing from the auto-fill list (`:681-700`), and the draft worker loads `speculative_draft_model_path` (`S/managers/tp_worker.py:465-469`). Without the flag the steps default to (3,1,4) (`:966-997`), with `num_draft_tokens=steps+1` when topk=1 (`:790-800`). `max_running_requests` is reset to **48** unless set explicitly (`:653-660`). Mixed chunked prefill is turned off if unsupported (`:672-684`).
- **Mamba interplay [V].**
  - During TARGET_VERIFY the KDA layers use the Triton verify kernel with `disable_state_update=True`. It writes per-draft states into `intermediate_ssm` (`kda_triton.py:158-217`).
  - `commit_mamba_states_after_verify` scatters the last accepted step and the track-interval crossing state (`S/speculative/spec_utils.py:843+`, step math at `:797-840`).
  - `prepare_mamba_track_for_verify` rebuilds the track indices and clears the mask (`:770-795`).
  - The lazy strategy is supported through `mamba_lazy_spec_prepare` (`S/managers/schedule_batch.py:3228-3270`, window check at `:1987-1997`), called from `spec_prepare_for_decode` (`spec_utils.py:1096-1104`).
  - The `extra_buffer` validator requires `mamba_track_interval ≥ num_draft_tokens` and `% page_size == 0` (`S/arg_groups/mamba_hook.py:119-122`). The default is 256, which is fine.
  - ReplaySSM-spec is incompatible with extra_buffer (`spec_utils.py:951-956`).
- **Memory [D].** `intermediate_ssm` is `[34, S+1, D, 8, 128, 128]` fp32 (`S/mem_cache/memory_pool.py:737-749`), about 17.8 MiB × D per spec slot per rank. With D=4 and 48 slots that is about 3.4 GiB/rank.
- **TPOT effect [D/U].** Decode is bound by reading expert weights; at batch ≥ ~32 nearly all 289 experts are touched, about 38 GB/rank per step. Verifying 4 tokens costs roughly one step, so with acceptance of 2-2.5 TPOT could fall by 1.5-2×. The task hard gate is tpot_p95 ≤0.10 without statistical slack (task.md:519,577). S0 measured 0.219 at dev N18 and 0.296 at dev N22, both failing it (F96). The risks are extra verify work competing with prefill (TTFT), the 48-request cap, the scratch memory, and no A100 validation. There is also a reported NextN crash at TP8 (#37548, R1).

## 5. DP attention (`--enable-dp-attention --dp-size 8`)

- **Asserts [V].** Nothing in the KDA, hybrid backend or Glm5Next code rejects DP. KDA heads shard on `attn_tp_size`, which is 1 at DP8 (`glm5_next.py:345-365`), and the verify scratch is padded for DP (`linear/utils.py:106-137`). The enforced constraints are elsewhere:
  - `tp % dp == 0`;
  - `chunked_prefill_size` is **divided by dp** (8192 → 1024 per rank, `S/arg_groups/parallel_hook.py:191-208`) and `schedule_conservativeness` is multiplied by 0.3;
  - the tc-piecewise graph is disabled under DP (`cuda_graph_hook.py:181`);
  - `--dp-size 1` silently drops DP (`overrides.py:1406-1409`);
  - NGRAM, DFLASH and STANDALONE spec reject DP (`speculative_hook.py:195,646,903`), but EAGLE/NEXTN does not.
  - Whether it actually runs end to end is **[U]**.
- **MoE [V].** With `moe_a2a_backend=none`, MoE stays TP8 and all ranks gather tokens through `LayerCommunicator` scatter modes. DeepEP, pplx and flashinfer-a2a need sm90 or DeepGEMM (`moe_hook.py:255-335`).
- **Memory [D].** Attention weights are replicated on every rank. KDA BF16 projections are about 273 MB/layer ×34 ≈ 9.3 GB, and MLA+indexer about 1.6 GB. That is **about +9.5 GB/rank** compared with TP8. KDA state per slot becomes about **148 MiB/rank** (8× more). MLA KV per token is no longer replicated.
- **KDA caching per rank [V].** Each DP rank has its own scheduler, radix tree and `MambaPool`. The controller offers round_robin, total_requests or total_tokens routing, with no prefix affinity (`S/managers/data_parallel_controller.py:85-99`). Under DP8 our heavily prefix-cached chains would therefore lose hits (R15 §6).

## 6. CUDA graphs [V]

- **Decode.** On an 80 GB GPU with TP≥4 the defaults are `max_bs=512` and `chunked_prefill_size=8192` (`S/arg_groups/memory_hook.py:103-115`). `max_bs` is clamped by `max_running_requests / attn_dp` (`moe_hook.py:405-409`).
  - Capture list without spec: `[1,2,4,8,12] + 16..256 step 8 + 272..511 step 16 + 512..` (`cuda_graph_hook.py:519-551`).
  - Capture list with spec: 1..8, 10..32 step 2, 40..64 step 4, and so on.
  - Reserved memory grows with max_bs, ×dp×3 under DP (`memory_hook.py:300-311`).
- **Prefill.** The default breakable graph is **disabled for KDA** (`cuda_graph_hook.py:262-278`, `uses_kda_attention`), so prefill runs eagerly.

## 7. Where time goes

- **[D] Prefill.** Active weights are about 16.5 B parameters (MoE ≈ 9.5 B, KDA projections ≈ 4.6 B, MLA ≈ 1.3 B, dense ≈ 0.45 B, head 0.63 B), or about 33 GFLOP/token. Sparse MLA adds about 3 GFLOP/token (11 layers × topk 2048 × 64 heads). The indexer grows with context: about 0.8 GFLOP/token at 36k and about 6 at 257k. This sentence describes a mixed base-image estimate; the working S0 instead uses 111 Marlin W8A16 for FP8 MoE, 110 tilelang DSA indexer, and BF16 KDA. The older 40–60% MFU estimate was not observed: F70 measured MoE at only 27–30% of peak, and F76 measured roughly 10k token/s whole-service cold prefill (36k tokens about 3.6 s). mHC adds 90 small calls per forward.
- **[D] Decode.** Decode is bound by expert weight bandwidth, plus about 17.8 MiB×2 of KDA state traffic per request per step and about 90 Triton KDA/conv launches per step (captured in graphs).
- **Hooks [V].**
  - `/start_profile` accepts `num_steps`, `profile_by_stage`, `profile_stages`, `activities` and `merge_profiles` (`S/entrypoints/http_server.py:1171`, `S/managers/io_struct.py:2110-2135`), with output to `SGLANG_TORCH_PROFILER_DIR` (`environ.py:448`).
  - `--enable-layerwise-nvtx-marker` (`server_args.py:1933`), `--enable-metrics`, `--enable-request-time-stats-logging` and `--expert-distribution-recorder-mode` (`:2474`).
  - The one-time log lines `Linear attention kernel backend:` (`linear/utils.py:97`), `KDA kernel dispatcher:` (`kda_backend.py:204`) and `Set DSA backends` confirm which kernels were actually picked.

## Current stack and validation boundary

The sections above explain the base source. S0 runs with 110/111 and tilelang DSA backends; fa3 is Hopper-only for this model's path on A100 (F57). The earlier suggestion to select fa3 or tune an unworkable FP8 Triton MoE path is obsolete. `SGLANG_OPT_USE_TOPK_V2=0` remains required on A100 (task.md:424).

The [patch inventory](../../../engine/README.md) records current mechanisms and their 8-card evidence. Candidate 114 divides indexer prefill rows; 115 adds DCP on sm80; 160 brings NEXTN; 170 adds opt-in breakable prefill CUDA graph. The candidate labels are not claims of scoring benefit. Compare complete dev levels with the harness scorer and check output/ability before promotion.
