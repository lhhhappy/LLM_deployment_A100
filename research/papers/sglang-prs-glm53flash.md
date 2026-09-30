# SGLang GitHub PR/Issue 快照（GLM-5.3-Flash 与混合缓存相关）

抓取日期：2026-09-24（UTC），经 GitHub REST API 未认证读取。只保存标题、状态与正文；评论和 diff 未抓取。

## #36507 GLM-5.3-Flash support

- URL: https://github.com/sgl-project/sglang/pull/36507
- 类型: PR；状态: closed；merged_at: 2026-09-06T09:28:00Z；创建: 2026-08-26T13:58:08Z；更新: 2026-09-07T02:56:21Z

Support GLM-5.3-Flash.



## #40134 Store index cache per page independently and enable hicache offload for GLM-5.3-Flash

- URL: https://github.com/sgl-project/sglang/pull/40134
- 类型: PR；状态: open；merged_at: None；创建: 2026-09-18T07:44:59Z；更新: 2026-09-18T08:10:53Z

## Motivation
Currently indexer cache is missing for GLM-5.3-Flash
exiting PR https://github.com/sgl-project/sglang/pull/39156, https://github.com/sgl-project/sglang/pull/38212 try to fix this, but the require model-specific adaptations.
The root cause is GLM pack indexer cache of 256 token into it's first 64 token.

## Modifications
Make index cache each page (64 token) stored into it corresponding page, and add indexer sidecar for L2 L3 offload.
As a result, we do not need complicated and model-specific adaptations in hicache, compatible with both L2 and L3 storage.

## Accuracy Tests
test script:
```
# clear L1 L2 kv cache
curl -s -X POST http://localhost:12121/flush_cache
# clear L3 kv cache
curl -s -X POST http://localhost:12121/hicache/storage-backend/clear

# first round bench without L3

# short,medium,long
evalscope eval \
--model GLM-5.3-Flash \
--api-url http://localhost:12121/v1 \
--api-key EMPTY \
--eval-type openai_api \
--generation-config timeout=1800 \
--datasets longbench_v2 \
--dataset-args '{"longbench_v2": {"subset_list": ["short"], "local_path": "/data1/datasets/llm_dataset/llm_accuracy_bench/LongBench-v2/"}}' \
--generation-config '{"max_tokens":24576}' \
--eval-batch-size 20 \
--limit 100

# clear L1 L2 kv cache
curl -s -X POST http://localhost:12121/flush_cache

# second bench with all loading from L3 cache

# short,medium,long
evalscope eval \
--model GLM-5.3-Flash \
--api-url http://localhost:12121/v1 \
--api-key EMPTY \
--eval-type openai_api \
--generation-config timeout=1800 \
--datasets longbench_v2 \
--dataset-args '{"longbench_v2": {"subset_list": ["short"], "local_path": "/data1/datasets/llm_dataset/llm_accuracy_bench/LongBench-v2/"}}' \
--generation-config '{"max_tokens":24576}' \
--eval-batch-size 20 \
--limit 100
```
before fix without hicache l2 l3
```
┌───────────────┬──────────────┬──────────┬──────────┬───────┬─────────┬─────────┐
│ Model         │ Dataset      │ Metric   │ Subset   │   Num │   Score │ Cat.0   │
├───────────────┼──────────────┼──────────┼──────────┼───────┼─────────┼─────────┤
│ GLM-5.3-Flash │ longbench_v2 │ mean_acc │ short    │   100 │    0.66 │ default │
└───────────────┴──────────────┴──────────┴──────────┴───────┴─────────┴─────────┘

```
after fix:
```
first round: writing L3
longbench_v2 report table:
┌───────────────┬──────────────┬──────────┬──────────┬───────┬─────────┬─────────┐
│ Model         │ Dataset      │ Metric   │ Subset   │   Num │   Score │ Cat.0   │
├───────────────┼──────────────┼──────────┼──────────┼───────┼─────────┼─────────┤
│ GLM-5.3-Flash │ longbench_v2 │ mean_acc │ short    │   100 │    0.67 │ default │
└───────────────┴──────────────┴──────────┴──────────┴───────┴─────────┴─────────┘


second round: clear L1 L2, loading from L3

2026-09-18 13:28:58 - evalscope - INFO: Overall report table:
┌───────────────┬──────────────┬──────────┬──────────┬───────┬─────────┬─────────┐
│ Model         │ Dataset      │ Metric   │ Subset   │   Num │   Score │ Cat.0   │
├───────────────┼──────────────┼──────────┼──────────┼───────┼─────────┼─────────┤
│ GLM-5.3-Flash │ longbench_v2 │ mean_acc │ short    │   100 │    0.66 │ default │
└───────────────┴──────────────┴──────────┴──────────┴───────┴─────────┴─────────┘
```


## Speed Tests and Profiling
to do



## #40915 Declare hybrid Mamba indexer host pool

- URL: https://github.com/sgl-project/sglang/pull/40915
- 类型: PR；状态: open；merged_at: None；创建: 2026-09-23T13:04:23Z；更新: 2026-09-23T13:27:08Z

## Motivation

Part 3 of 5 of https://github.com/sgl-project/sglang/pull/40608. Hybrid Mamba plus DSA models (GLM-5.3-Flash) restore KV from host without the indexer on main, so a host hit runs attention on stale block selection. Host restore KL against warm is 0.06 to 0.71 on main. build_hybrid_mamba_stack now assembles the full attention pool's declarations next to the Mamba host pool, and _MambaStrategy raises when a declared pool is missing. HybridLinearKVPool.host_pool_decls() declares only the buffers of its full attention sub pool. Mamba state stays with its owner and the existing Mamba host path.


Stack: https://github.com/sgl-project/sglang/pull/40913 (declarations and DSA target), https://github.com/sgl-project/sglang/pull/40914 (separate draft), https://github.com/sgl-project/sglang/pull/40915 (hybrid Mamba), https://github.com/sgl-project/sglang/pull/40916 (QSA), https://github.com/sgl-project/sglang/pull/40917 (plain KV and cleanup). Each PR is based on the previous one.
## Accuracy

GLM-5.3-Flash tp8, page 64, hicache size 20 GB, extra_buffer strategy: stack KV + INDEXER + MAMBA, indexer host 1.76 GB, host restore KL against warm 1.9e-3, 1.6e-3, 3.5e-3 at the finished sequence length. Main on the same runs: 0.06 to 0.71.



## #38212 Preserve DSA indexes and recurrent checkpoints in HiCache

- URL: https://github.com/sgl-project/sglang/pull/38212
- 类型: PR；状态: open；merged_at: None；创建: 2026-09-06T11:19:54Z；更新: 2026-09-21T19:27:42Z

## Motivation

GLM-5.3-Flash can resume cached prompts with incorrect attention state: HiCache omits DSA index buffers, divergent suffixes can share compressed index rows, and chunked prefill can save recurrent state beyond the prefix that owns it. For example, with a 256-token ownership grid, two 320-token prefill chunks could attach the checkpoint at 576 to the key ending at 512.

Addresses these paths in #38031. Replaces #38161 after its base was deleted and includes the adapted packed-row dependency from #37534.

## Modifications

- Restore target and draft DSA index buffers, skipping empty shared-topk layers and preserving physical transfer pages.
- Align compressed-prefix ownership and recurrent checkpoints. Skip unavailable snapshots and validate the final key limits before donating state, including SWA limits.
- Match host rows to device storage. Use a separate sidecar for one incompatible MLA draft; retain compatible packed drafts.
- Unregister the legacy DSA host index buffer on destruction. Consolidating the GPU cases exposed that inherited cleanup missed this separately stored buffer.

Compressed DSA supports L2 HiCache only. FP4 MLA KV storage, incompatible packed index widths, and mismatched multi-runner MTP are rejected. Explicit hybrid host-size budgets include target indexes; ratio mode, standalone DSA indexes and draft sidecars can allocate beyond the KV host budget.

## Accuracy Tests

Author validation on main `833bce9df5`: CPU checks ran at final head `0ee51670a7`. The GPU sanitizer run used `5ea9a412d3`; radix checks and initial fault injections preceded formatting and test-file relocation, with matching runtime and test logic verified. Subsequent changes only move the GPU test into the regular runtime PR suite and strengthen a CPU checkpoint-lifetime assertion:

| Check | Result |
|---|---|
| Eight CPU test files | 124 tests and 111 subtests passed; 15 skipped |
| Changed strip-thinking and unaligned-tail radix cases on GPU | 26 passed; 12 skipped |
| Registered DSA GPU transfer suite, RTX PRO 6000 Blackwell Max-Q | 20 passed; Compute Sanitizer reported zero errors |
| Deliberate fault injections | 20 detected by relevant assertions |

The GPU matrix combines four layout/backend pairs with target-only, packed drafts, live/empty draft-index sidecars and legacy uncompressed DSA. It checks host contents after each incremental backup and all destination bytes, including untouched pages. It uses production assembly, transfer discovery and preparation; controller initialization/background scheduling are bypassed and destination allocation is controlled.

The checkpoint tests use real scheduler/backend index selection, tree operations and pools with symbolic states distinguished by prefix content and layer. They establish state ownership, not KDA numerical accuracy. Fault checks cover wrong checkpoints, missing convolution copies, shortened state owners, omitted index transfers, wrong pages/layers, incorrect paired sorting, dropped drafts, neighboring writes and premature reuse of donated checkpoint slots.

No new full-model run is claimed for this head. [Earlier author serving results](https://github.com/sgl-project/sglang/pull/38212#issuecomment-5574636314) and [reporter H100 validation](https://github.com/sgl-project/sglang/pull/38212#issuecomment-5574703242) retain their stated revision/configuration scopes. The retired numerical smoke passed both checkpoint implementations and was not a discriminating regression.

Run the GPU regression with:

```bash
PYTHONPATH=python compute-sanitizer --tool memcheck --error-exitcode 99 \
  python test/registered/e2e/mem_cache/test_hicache_dsa.py
```

## Speed Tests and Profiling

No isolated speed benchmark or performance claim.

## Checklist

- [x] Run repository pre-commit checks, including CI registration.
- [x] Add and run CPU and GPU regressions.
- [x] Separate current source validation from historical full-model results.

Developed with AI assistance.



## #38474 test: add GLM-5.3-Flash HiCache KL coverage

- URL: https://github.com/sgl-project/sglang/pull/38474
- 类型: PR；状态: open；merged_at: None；创建: 2026-09-08T09:30:40Z；更新: 2026-09-08T10:00:00Z

## Summary

**Draft: GPU calibration found a flaky cold-repeat gate. This test is not ready to enable in CI.**

- Add a standalone GLM-5.3-Flash L2 HiCache KL test on `extra-b / 4-gpu-b200`, without including the implementation changes from #38212.
- Exercise shared prefixes at 64/128/192 tokens inside a 256-token compressed-index group, plus a 256-token-aligned control. Cover cold-repeat, device-hit, device-fork, fork+host-loadback, and concurrent branching/eviction/loadback.
- Use four reproducible prompt groups, 512 greedy continuation tokens per group, and three fixed repetitions. Reuse each continuation for all comparisons; no external dataset is required. Greedy selection removes token-sampling randomness, not nondeterministic model execution.
- Require actual shared-prefix hits and nonzero host-loadback hits. Scoring may recompute at most the final 255 tokens of A, so it cannot silently recompute the entire branch and heal its index state.

## Provisional Numerical Gate

The test gates the **median of 12 per-sequence k3 KL estimates per scenario at `< 0.5`**. It does not require bit-exact equality or clip individual estimates. Every observation, maximum/mean absolute logprob delta, and cache-hit count is printed.

The greedy-continuation statistic is a fixed-token k3 precision probe, not full-vocabulary KL or an unbiased estimate sampled from the reference distribution.

This is intentionally a loose initial guard: GLM DSA deterministic inference is not available, and #38212 has not merged. Calibration also exposed large outliers in ordinary device-hit comparisons; these are not assumed to be harmless rounding noise. A mean-only gate was unstable, so the provisional median gate prevents a small number of extreme exponential estimates from deciding the whole run. It can miss regressions confined to a minority of cases and is **not** a claim that every boundary case is already correct or that this threshold detects #38212's known bug.

An owned TODO tracks tightening the gate after #38212 and deterministic GLM DSA support. Missing cache hits, token mismatches, nonfinite logprobs, incomplete generations, and request failures remain hard failures regardless of the KL threshold.

The test uses `unittest.TestCase`, not `CustomTestCase`, to avoid the latter's implicit CI retries. All three repetitions are mandatory, not retries until success.

## Validation

- Full pre-commit checks, including registered-test taxonomy and CI registration: passed.
- GitHub lint on commit `adaba9df9d`: passed.
- Full-model GPU validation on 4 x H200 with BF16 KV / TileLang: two greedy-continuation runs completed, one passed and one failed. The second ran the exact pushed file (SHA256 verified); the first had the same test semantics with an unused optional temperature argument in the request helper.
- Each run completed all 60 comparisons and verified 24 positive host-loadback scores. No cache-hit or token-integrity assertions failed.
- Registered B200 CI configuration still needs CI validation.

| Scenario | Greedy run 1 median | Exact-file run 2 median |
| --- | ---: | ---: |
| Cold repeat | 0.203724 | **0.733031 (FAIL)** |
| Device hit | 0.041463 | 0.368843 |
| Device fork | 0.098061 | 0.279101 |
| Fork + host loadback | 0.236689 | 0.146775 |
| Concurrent host loadback | 0.256495 | 0.241899 |

The second run fails even without cache reuse in its cold-repeat comparison. This establishes that `< 0.5` is not a stable gate for this configuration; it does not isolate the responsible operator or prove the variation harmless. The threshold has not been increased in response to this failure. Numerical calibration remains unresolved, so this PR stays draft.

Earlier stochastic-continuation calibration included a failed mean-gated run and two passing median-gated runs. Those results and all individual observations are retained; they do not override the failed exact-file validation above. Server shutdown logs also contain `CancelledError` during shared-helper teardown, so the runs are not claimed to have pristine logs.

## Scope

Test-only change. No production code, deterministic-inference implementation, MTP, L3, or streaming-session changes.



## #39830 [Bug] Hierarchical cache returns wrong output on a hybrid (GDN/Mamba) model: a full-length host-tier hit answers in a different pass's voice, 20/20, on main

- URL: https://github.com/sgl-project/sglang/issues/39830
- 类型: Issue；状态: open；merged_at: None；创建: 2026-09-16T15:54:38Z；更新: 2026-09-17T04:42:39Z

### Checklist

- [x] I searched related issues but found no solution.
- [x] The bug persists in the latest version.
- [x] Issues without environment info and a minimal reproducible demo are hard to resolve and may receive no feedback.
- [x] If this is not a bug report but a general question, please start a discussion at https://github.com/sgl-project/sglang/discussions. Otherwise, it will be closed.
- [x] Please use English. Otherwise, it will be closed.

### Describe the bug

On a hybrid linear-attention (GDN) model with `--enable-hierarchical-cache`, a
prefix that has been evicted from the device pool and restored from the host
tier produces **different and wrong output** for a byte-identical request. There
is no crash, no warning and no accounting anomaly: the serve reports the hit at
the **full** `cached_tokens`, the request finishes with `finish_reason=stop`,
and the answer is fluent — it simply is not an answer to the prompt that was
sent.

Measured on **main HEAD `2929a39927a3943cee03e498f4e5f651185f1b1f`**
(2026-09-15): **20 of 20** host-tier hits wrong, against **10 of 10** correct
cold and **10 of 10** correct device-pool hits on the same prompts in the same
process.

This is the same defect class already reported on the **device** path —
#37836 (merged 2026-09-04), #39342, #31833 — but on the **host** path, where we
can find no report.

### Result, 10 prompts

| leg | n | answers correct | mean reported `cached_tokens` |
|---|---|---|---|
| cold | 10 | **10/10** | 0 |
| hit (device) | 10 | **10/10** | 27,628 |
| **host** | 10 | **0/10** | 27,628 |
| **host2** | 10 | **0/10** | 27,628 |

Every host leg reported a hit; none reported a miss; none errored.

### What the wrong answers look like

The prompt asks for three imperative one-line suggestions. Verbatim, same
prompt, cold then out of the host tier:

```
cold   Ask Orrin what your father was hiding in the glasshouse.
       Go to the glasshouse and look for what your father hid.
       Ask Orrin why your father stopped feeding the bees.
hit    (byte-identical to cold)
host   act:local_action
host2  act:speech local_action implied - -
```

```
cold   Let Jada go now.
       Keep Jada here until the auditor finishes.
       Ask the auditor to read the logbook aloud.
host   act horizon
host2  act horizon
```

```
host   act=speech act=normal act=act=act=act=act=act=act=act=act=act=act=act=
       act=act=act=act=act=act=act=act=... (hits the 96-token bound)
```

```
host   The Stranger releases you, stepping back with a sudden, jerky motion, and
       turns his shoulder deliberately toward the riders gathered near the front,
       his eyes fixed on the narrow door of the back office. "The writ is in the
       back office."
```

Read by eye, all 20 host legs:

| shape | n of 20 |
|---|---:|
| a line from a **different pass's wire format** (`act:local_action`, `act horizon`, `act=speech route=implied_npc target=implied_npc`, `cast_third intervene target:cast_second`) | 9 |
| a **paragraph of narration or two-speaker dialogue** — a different pass's output shape entirely | 6 |
| **degenerate repetition** to the token bound | 1 |
| fewer than the requested three lines, right voice | 2 |
| right shape, wrong voice (not an imperative the user could issue) | 2 |
| **an answer a user could be shown** | **0** |

Two properties that narrow it:

- **It is deterministic.** `host` and `host2` return the same wrong text
  repeatedly. Temperature is 0 and does not explain it.
- **It is not truncation. Measured on this build.** Appending a marker
  instruction to the END of the prompt ("Answer in the usual three lines, but
  begin every line with the word ZEBRA") is obeyed **3/3 lines on the host leg
  of both payloads tested**, while the content is wrong — the host leg answers
  `ZEBRA The player asks the Abbot whether the box has been moved.` where the
  cold leg answers `ZEBRA Ask the Abbot to show you the box.` So the tail of the
  restored prefix is present and attended to; the damage is in the body.
- **Waiting does not help.** 150 s of complete idle between the churn and the
  host leg (75 s after warming, 75 s after the churn) changes nothing, so an
  unfinished device->host copy does not explain it. (This one was measured on
  the older commit `4ccff141d`, not on main.)
- **Three other write/eviction settings do not help either**, all on this build:
  `--hicache-write-policy write_through_selective` (8 genuine restores, 0
  correct), `--mamba-max-states-per-path 4` (12 genuine restores, 0 correct),
  and `--max-mamba-cache-size 96` (6 genuine restores, 1 correct).

### Why this looks like the recurrent state, not the KV

The prefix match itself requires a Mamba checkpoint — in
`unified_cache/components/mamba_component.py` the MAMBA validator accepts a node
when `component_data[ct].value is not None or ... .host_value is not None` — so a
full-length `cached_input_len` means *a* checkpoint was present at the frontier.
The answers are fluent, in-world and about the right conversation; they are the
wrong PASS, not the wrong story. That is what a restored recurrent state taken
at a different position looks like, and it is exactly the symptom #37836 / #39342
describe on the device path.

### Two observations that may help whoever picks this up

1. **`--hicache-write-policy write_back` changes the outcome, and partly by
   caching less.** On the same build, 6 prompts: 4 of the 6 `host` legs reported
   `cached_tokens = 0` — a genuine miss, recomputed, correct by construction —
   and only **2** were genuine host restores. One of those two was usable and one
   was degraded (it named a person the campaign's cast does not contain). So
   `write_back` is not a fix; it is a smaller sample of the same event.
2. **`host2` is wrong too, and differently wrong.** By the time `host2` runs, the
   `host` leg has already pulled the prefix back into the device pool, so `host2`
   should be an ordinary device hit — and an ordinary device hit is correct
   10/10 in leg 2. It is not correct here. Whatever the host tier restores, it
   restores INTO the device pool in a state that stays wrong.

### Suggested starting points

`mem_cache/unified_cache/components/mamba_component.py`
(`build_hicache_transfers`, phases `BACKUP_HOST` / `LOAD_BACK`;
`commit_hicache_transfer` assigns `cd.host_value` and `cd.value` from cloned
index tensors with no visible synchronisation before use), and
`finalize_match_result_in_tree_core`, whose own comment is *"Full KV may extend
beyond the latest reusable Mamba state"* — the clip that has to hold on the host
path. The per-pool eviction metric and host-coverage boot line proposed in
**#39436** would say immediately whether a checkpoint was evicted under the
prefix whose KV rows survived.

### What would make this easy to confirm

A boot-time or per-request assertion that the Mamba checkpoint restored for a
host hit was taken at the same token position as the last KV page restored with
it. Today nothing checks it, and the failure is silent all the way to the user.

### A related omission in the host pool, with a proposed patch (not a fix for the above)

Reading the source for the cause, we found that the specialized HiCache Mamba host pool (`mem_cache/pool_host/mamba.py`) saves the GDN convolution and recurrent state of a checkpoint but never saves the model's registered PLE side state (the BF16 `[10240, 9]` PLE convolution window and its two int64 token-history entries, ~184 KB per checkpoint, registered as slot siblings in `memory_pool.py`). The ordinary device-side copy and the generic CPU offload path do preserve those siblings. The omission is present on the pin above, on main `2929a399` and on later main `63845a1`.

A proposal patch that carries the PLE siblings through the host pool (separate host arrays with the original per-slot shape and dtype, included in capacity accounting and buffer registration, restored at the first local Mamba layer with an event wait before the N-gram gather; whole-page serialization appends typed sections with size validated before any host tensor changes) is here: https://gist.github.com/JarJarBeatyourattitude/5bfe8db0b9f04378fc60962fb0622fb0

Two honest caveats about that patch:

- It was verified on CPU only (an AST-loaded round trip of the real classes with leaf DMA substituted); no GPU execution, no CUDA event timing, no performance measurement.
- We built and ran it on the same GPU and model. **The host restores were still wrong with the patch applied** (0 of 2 genuine host restores correct, same failure shapes as above), with the patched module demonstrably loaded. So the omission is real and worth fixing, but it is not the whole cause of the wrong output. We also checked for an adapter-id collision in the cache keys; main already namespaces them.

### Reproduction

Serve command (paths elided):

```
sglang serve --model-path <nvfp4-checkpoint> --load-format safetensors \
  --tp 1 --dtype bfloat16 --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.92 --context-length 65536 --page-size 64 \
  --max-running-requests 4 --chunked-prefill-size 4096 \
  --cuda-graph-max-bs-decode 4 --cuda-graph-max-bs-prefill 4 \
  --mamba-ssm-dtype bfloat16 --max-mamba-cache-size 48 \
  --mamba-radix-cache-strategy extra_buffer --mamba-track-interval 64 \
  --linear-attn-decode-backend flashinfer --linear-attn-prefill-backend flashinfer \
  --enable-cache-report --ple-offload-embedding \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --enable-lora --max-lora-rank 32 --max-loras-per-batch 7 --lora-paths <7 adapters> \
  --enable-hierarchical-cache --hicache-size 96 \
  --hicache-write-policy write_through --hicache-mem-layout page_first \
  --hicache-io-backend direct
```

Boot lines, for the pool shapes:

```
Mamba Cache is allocated. max_mamba_cache_size: 48, conv_state size: 0.10GB,
  ssm_state size: 2.58GB intermediate_ssm_state_cache size: 1.05GB
  intermediate_conv_window_cache size: 0.02GB
KV Cache is allocated. dtype: torch.float8_e4m3fn, #tokens: 240128, K 1.37 GB, V 1.37 GB
max_total_num_tokens=240128, ... available_gpu_mem=8.87 GB
Allocating kv hierarchical KV host pool: 3035904 tokens, 40.41 GB host memory,
  packed MTP KV layers: target_layers=12, draft_layers=1, total_layers=13.
Allocating 55.60 GB host memory for hierarchical Mamba cache (layout=page_first_direct).
Tree cache initialized: source=default impl=UnifiedRadixCache hybrid_swa=False
  hybrid_ssm=True hicache_attached=True streaming_wrapped=False
```

One request at a time, `temperature 0`, `max_tokens 96`, no grammar:

1. **cold** — send prompt P (10k-49k tokens). Prepend a unique random prefix so
   no radix node can match it.
2. **hit** — send P again immediately. Device-pool hit.
3. **churn** — send other large, mutually distinct prompts until the *uncached*
   token count pushed through exceeds 1.35x `max_total_num_tokens` (here
   ~324,000), so P is certainly evicted from the device pool.
4. **host** — send P again. The serve reports the full `cached_tokens`; the KV
   comes back from the host tier.
5. **host2** — send P again.

### Environment

| | |
|---|---|
| sglang | `2929a39927a3943cee03e498f4e5f651185f1b1f` (main, 2026-09-15), `0.5.20.dev728` |
| torch | `2.13.0+cu130`, CUDA 13.0 |
| GPU | 1x RTX PRO 6000 Blackwell WS, 97,887 MiB, driver 595.71.05, sm120 |
| model | `RadixArk/Qwen3.8-Flash-Next-NVFP4` @ `7b719225242aacd3dbd3f9407468c2ee9a9d2594` (`Qwen4ExpForConditionalGeneration`, `qwen4_exp`), 48 layers, `full_attention_interval 4` -> 36 GDN + 12 full-attention |
| local delta | 134 lines in 4 files, LoRA enablement only (`lora/lora.py`, `lora/utils.py`, `models/qwen3_5.py`, `models/qwen4_exp.py`): `in_proj_ba` in `supported_lora_modules` / `_KNOWN_LORA_TARGET_MODULES`, and a nested-`text_config` `get_hidden_dim` on `Qwen4ExpForConditionalGeneration`. **Nothing in `mem_cache/`, `attention/` or `speculative/` is patched.** |

`python3 -m sglang.check_env` output. Note: this was captured from the same GPU model and image with the venv we serve production from, which is on the older pin `4ccff141d` with the LoRA delta noted above; the measurements in this report were taken on main `2929a39927a3943cee03e498f4e5f651185f1b1f` built the same way on a since-released machine (driver 595.71.05 there).

```
venv python: /root/models/sglang-official/.venv/bin/python
Python: 3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]
CUDA available: True
GPU 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
GPU 0 Compute Capability: 12.0
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 13.0, V13.0.88
CUDA Driver Version: 590.44.01
PyTorch: 2.13.0+cu130
sglang: 0.0.0.dev1+g4ccff141d
sglang-kernel: 0.4.6.post1+cu130
flashinfer_python: 0.6.17
flashinfer_cubin: Module Not Found
flashinfer_jit_cache: Module Not Found
triton: 3.7.1
transformers: 5.12.1
torchao: Module Not Found
numpy: 2.3.5
aiohttp: 3.14.3
fastapi: 0.141.1
huggingface_hub: 1.31.0
interegular: 0.3.3
modelscope: 1.40.0
orjson: 3.12.0
outlines: 0.1.11
packaging: 26.3
psutil: 7.2.2
pydantic: 2.14.0b2
python-multipart: 0.0.32
pyzmq: 27.2.0
uvicorn: 0.52.4
uvloop: 0.22.1
vllm: Module Not Found
xgrammar: 0.2.1
openai: 2.6.1
tiktoken: 0.14.0
anthropic: 1.5.0
litellm: Module Not Found
torchcodec: 0.15.0
```

## #40865 [RFC] Explicit, budgeted tail-replay for Mamba/GDN state in UnifiedRadixCache

- URL: https://github.com/sgl-project/sglang/issues/40865
- 类型: Issue；状态: open；merged_at: None；创建: 2026-09-23T06:36:48Z；更新: 2026-09-24T01:52:02Z

**Update 2026-09-23**: experimental results (real-server A/B on a Qwen3.8-27B hybrid, Ascend 910B3) posted in the comments — see [the results comment](https://github.com/sgl-project/sglang/issues/40865#issuecomment-5805944907).---

**Area**: `python/sglang/srt/mem_cache/` (UnifiedRadixCache, MambaComponent, schedule_batch) **Related**: HiCache buffer-mode (#37424), eviction-policy parameterization (#37795), external linker series (#37091–#37914), HiMambaRadixTree removal (#33468), Mamba checkpoint depth fix for off-page prefixes (#39115), ReplaySSM decode/spec-verify state reconstruction (#28511)

## Summary

Hybrid models (FULL + SWA + MAMBA/GDN components) reuse a prefix only up to the latest reusable Mamba state checkpoint. When that checkpoint is missing, evicted, or simply far behind the full-KV hit, the engine retreats the *entire* hit to the checkpoint boundary and re-prefills everything after it — even though only the GDN recurrent state needs rebuilding, and the full-attention KV for the gap is already cached.

We propose making the existing *implicit* tail recompute an **explicit, budgeted "tail-replay" path**: accept the full-KV hit up to `H`, seed the GDN state from the checkpoint at `C`, and replay only `(H - C)` tokens through the GDN layers, gated by a configurable budget and admission-time cost accounting.

## Motivation

1. **Checkpoint density vs. hit rate.** Mamba state is leaf-only checkpoint data (node split clears the parent's state). Checkpoints are sparse relative to full-KV pages, so on hybrid models the reusable prefix is effectively quantized to checkpoint positions. Short prompts and agentic workloads (many shared prefixes appended in small turns) are hit hardest: the nearest checkpoint often sits one full grid behind, and the hit collapses. The checkpoint-depth semantics here are live upstream surface — see #39115 (merged 2026-09-17), which fixed exactly the depth computation for prefixes ending off the radix page; our proposal composes with that fix and extends it from "place the checkpoint right" to "tolerate the checkpoint being behind".
2. **Eviction can strand the state.** Incremental persistence of a branching state is currently write-through only; write-back eviction may discard device-only state (comment in `MambaComponent.finalize_match_result_in_tree_core`). The next match then retreats to a much older checkpoint — or the root — with no policy deciding between "replay a longish tail" and "give up the hit".
3. **Two hard fences are landing soon.** `init_hicache` raises for Mamba components under `hicache-host-memory-mode=buffer_only` (TODO(Jialin) referencing #34798/#35769), and `MambaComponent.build_external_linker_transfer` asserts "will support soon". Both follow-ups will need a replay/charge semantics for the gap between a restored KV prefix and a missing state. Defining that semantics now lets the linker work build on it instead of inventing its own. (The linker-mode interface is being negotiated right now in #38652; a POC pair — #40759 and LMCache/LMCache#5304 — was posted and withdrawn by the author on 2026-09-22, so the interface is not yet locked. This RFC is timed to feed that discussion.)

## Current state (verified on main @ 224a247d5b, 2026-09-23)

- `MambaComponent.create_match_validator` (`mem_cache/unified_cache/components/mamba.py:143`; note the 10/N refactor renamed `mamba_component.py` → `mamba.py`, symbols unchanged): a node is a valid match boundary if it has a device mamba value (device-only mode) or any value (HiCache mode). Missing checkpoint ⇒ walk up to nearest checkpointed ancestor. No error path — silent retreat.
- `MambaComponent.finalize_match_result_in_tree_core` (`mamba.py:156`): computes `mamba_branching_seqlen = align_down(full_kv_hit_length, mamba_checkpoint_grid)` when the full hit extends past the mamba boundary; the retreated tail is then re-prefilled through **all** layers by the extend pass (gap KV recomputed, deduplicated at insert), and a new branching checkpoint is physically captured via `_force_track_h` (`managers/schedule_batch.py:2986`; `mamba_branching_seqlen` field at :1161).
- SWA precedent for an explicit short-tail re-prefill: `UnifiedRadixCache.swa_reprefill_tail_tokens` (`unified_radix_cache.py:3506`) caps the match by one sliding window when the SWA ring is not content-stable, with scheduler-side cap call sites at `schedule_batch.py:1595` and `:1613`. This is the pipeline we intend to mirror for Mamba.
- `init_hicache` at `unified_radix_cache.py:410`; the buffer_only Mamba fence and TODO(Jialin) at :414.
- No metrics expose how much recompute the mamba boundary retreat causes.

### Relation to ReplaySSM (#28511)

Not to be confused: ReplaySSM (`--enable-linear-replayssm` / `--enable-linear-replayssm-spec`; #28511, parts #28451/#28695) already reconstructs GDN/KDA recurrent state on the *decode* and *spec-verify* paths — from a frozen checkpoint plus a ring of recent `(d, k, g)` records, output-only, with pointer rollback for rejected drafts. It optimizes decode state-traffic bandwidth and never touches the match/finalize/admission path: a prefix-cache hit still retreats to the checkpoint. (The radix *track* does consume ReplaySSM force-flushes to materialize checkpoints — #35544 — but that is the write side.) The two share the algebraic primitive — a linear recurrence's state is reconstructable from a checkpoint plus suffix records — and compose; they do not overlap.

## Proposal

1. **`mamba_replay_tail_tokens()` on `UnifiedRadixCache`** (sibling of `swa_reprefill_tail_tokens`): returns the tail length the scheduler should treat as "hit for FULL, replay for GDN" for the current match, instead of collapsing the whole hit to the checkpoint boundary.
2. **Budget policy**: accept the extended hit iff `(H - C) <= mamba_replay_budget`, where the budget is `max(min_tail, replay_ratio * prefix_len)`; defaults chosen so current behavior is unchanged unless the policy is enabled (`SGLANG_MAMBA_REPLAY_TAIL_RATIO`, default off or a conservative value — open to maintainer preference).
3. **Admission-time charge**: the replayed tail consumes prefill compute; charge it against `req.mamba_host_hit_length`-style accounting so the scheduler's token budget reflects the real cost (this is also the hook the buffer_only and linker follow-ups will need).
4. **Metrics**: `mamba_replay_tokens_total`, histogram of `(H - C)` at admission, and a counter of "hit collapsed to checkpoint" events so operators can tune the budget.
5. **Out of scope** (coordination, not overlap): buffer_only state handoff and external-linker transfer for Mamba — owned by the TODO(Jialin) follow-up and the linker series (#37914+, and the #38652 LMCache discussion). This RFC only defines the replay/charge semantics they should consume.
6. **Execution-path note (the real engineering delta)**: today the extend pass re-prefills the retreated tail through *all* layers (gap KV recomputed, deduplicated at insert). The proposed win therefore needs a split-prefix extend — full-attention layers start at `H` (their KV is already cached) while the GDN layers consume `[C, seq_end)` seeded from the checkpoint at `C`. The forward path currently carries a single `extend_prefix_len` per request (`hybrid_linear_attn_backend` consumes `extend_prefix_lens` uniformly), so this needs forward-batch/backend plumbing on top of the cache-layer hooks above; no new kernels are required (the existing FLA chunked prefill runs the replay tokens), but the metadata layer must represent both prefix lengths. Once a ReplaySSM ring covers the gap, its reconstruction primitive (#28511) is a candidate accelerator for the replay segment; we scope plain chunked prefill as the baseline. Happy to split the execution path into a follow-up PR if maintainers prefer to land the policy/charge/metrics layer first.

## Cost-model evidence (offline measurements, our Draft-OPD/Tail-Replay setup)

On Qwen3-class GDN hybrids, replaying the trailing 5–10% of a prefix reconstructs the GDN recurrent state with 93–99.9% fidelity (measured against full prefill states), while a full re-prefill of the gap costs the entire gap's attention compute. The replay path turns "checkpoint miss ⇒ full gap re-prefill" into "checkpoint miss ⇒ GDN-only tail replay", which is strictly cheaper whenever the full KV is cached. We will publish the measurement harness and curves in the RFC thread.

## Testing plan

- Unit: synthetic unified trees with FULL+MAMBA components; assert match/finalize behavior for (a) checkpoint present, (b) checkpoint missing within budget, (c) missing beyond budget (hit collapses), (d) budget=0 ⇒ byte-identical to current behavior.
- Integration: HiCache L2 offload on a Qwen3-Next-class hybrid; measure hit-rate recovery on an agentic multi-turn trace.
- Regression: existing `test/` unified-cache and hicache suites must pass with the policy disabled (default).

## Asks

1. Budget-policy shape: env var vs. server arg vs. per-model default in the pool configurator?
2. Should the replay charge live in the admission path (`init_next_round_input`) or in the retract/eviction accounting?
3. Any objection to landing this ahead of the Mamba linker support, given (5) explicitly defers the handoff/transfer semantics to that work — and would the #38652 linker-mode participants rather consume this as an interface input now?

## #36830 [Bug] GLM-5.3-Flash cannot use FP8 KV cache: `index_kpool > 1` excludes `flashmla_kv`, and no CUDA DSA backend supports bf16-query x fp8-KV

- URL: https://github.com/sgl-project/sglang/issues/36830
- 类型: Issue；状态: open；merged_at: None；创建: 2026-08-28T08:46:05Z；更新: 2026-09-23T08:52:43Z

## Summary

On 8xH20 (SM90), `--kv-cache-dtype fp8_e4m3` cannot be used with **GLM-5.3-Flash**, while it works fine
with **GLM-5.2** on the *same hardware, same cluster, same image* (in production for months).

The difference is one model-config field: **GLM-5.3-Flash sets `index_kpool: 4`; GLM-5.2 does not set it
(defaults to 1)**. That single value excludes `flashmla_kv` — the backend GLM-5.2 uses for FP8 KV — and the
three replacement backends (`fa3` / `tilelang` / `trtllm`) have no bf16-query x fp8-KV path on CUDA.

Net effect: GLM-5.3-Flash is forced onto bf16 KV and loses **~1.75x-2.0x** of KV cache capacity.

## Positive control: GLM-5.2 with FP8 KV works on the same hardware

Production node, running continuously for months:

```
--model-path <path>/GLM-5.2-W4AFP8
--quantization w4afp8
--kv-cache-dtype fp8_e4m3
```
Engine-resolved:
```
kv_cache_dtype='fp8_e4m3'
dsa_prefill_backend='flashmla_kv'
dsa_decode_backend='flashmla_kv'
```
```
KV Cache is allocated. dtype: torch.float8_e4m3fn, #tokens: 958208
```

So FP8 KV itself is healthy on SM90. The problem is specific to GLM-5.3-Flash.

## Root cause chain

**1. The only relevant config difference**

```
GLM-5.3-Flash : "index_kpool": 4   "index_topk": 2048  "index_n_heads": 32  "index_head_dim": 128
GLM-5.2       : (absent)           "index_topk": 2048  "index_n_heads": 32  "index_head_dim": 128
```
```python
# srt/configs/model_config.py:280
def get_dsa_index_kpool(config) -> int:
    return getattr(config, "index_kpool", 1)
```

**2. The guard excludes `flashmla_kv`**

```python
# srt/layers/attention/dsa_backend.py  _check_kpool_tail_backend
if (topk_indices is None
    or self.dsa_index_kpool <= 1                      # GLM-5.2 exits here
    or dsa_impl in ("fa3", "tilelang", "trtllm")):    # GLM-5.3 is limited to these
    return
raise NotImplementedError(
    "index_kpool > 1 appends tail tokens to topk_indices and is "
    "currently only supported by the FA3/TileLang/TRTLLM DSA {phase} backend.")
```

**3. The existing escape hatch only covers the bf16 path**

```python
# _resolve_kpool_tail_backend
if (topk_indices is None or self.dsa_index_kpool <= 1
    or dsa_impl != "flashmla_sparse"):     # does not cover flashmla_kv
    return dsa_impl
if self.device_sm_major == 9: return "fa3"
```

This is why the **bf16** configuration works: the engine auto-selects `flashmla_sparse` + `fa3`, and the
resolver hands the KPool tail to `fa3`. FP8 requires `flashmla_kv`, which this branch does not handle.

**4. None of the three allowed backends can consume FP8 KV on CUDA**

```python
# dsa_backend.py:594-601 - the only mixed q/kv dtype path
if (self.dsa_prefill_impl == "aiter" or self.dsa_decode_impl == "aiter") \
   and model_runner.kv_cache_dtype == fp8_dtype:
        self._ensure_aiter_dsa_decode_metadata_buffer(
            q_dtype=torch.bfloat16, kv_dtype=fp8_dtype)
```

It exists only for `aiter` (ROCm). `flashmla_sparse_q8` is native FP8 but **prefill-only** (its own error
message: *"flashmla_sparse_q8 is a prefill-only backend. For FP8, use --dsa-prefill-backend
flashmla_sparse_q8 together with --dsa-decode-backend flashmla_kv"*) - and `flashmla_kv` is exactly what
the KPool guard rejects.

## Reproduction

8xH20, image `lmsysorg/sglang:glm-5.3-flash`, two instances of `--tp-size 4 --expert-parallel-size 4`
(GPU 0-3 and 4-7), `--kv-cache-dtype fp8_e4m3`, `--context-length 200000`:

| # | prefill / decode backend | MTP | Failure |
|---|---|---|---|
| 1 | `flashmla_sparse_q8` / `flashmla_kv` | on | `NotImplementedError: index_kpool > 1 ...` (decode cuda-graph capture) |
| 2 | `flashmla_sparse_q8` / `fa3` | on | `RuntimeError: query and key must have the same dtype` (target-verify graph) |
| 3 | as #2 + `--speculative-attention-mode decode` | on | same dtype error |
| 4 | `flashmla_sparse_q8` / `fa3` | **off** | **same dtype error** |

Case #4 matters: **the failure is not specific to speculative decoding.**

The KV pool itself allocates fine - the blocker is on the consumer side:

| config | KV tokens / instance |
|---|---|
| bf16 + MTP (the only working one) | 2,280,000 |
| fp8_e4m3 + MTP | 3,993,472 (1.75x) |
| fp8_e4m3, no MTP | 4,610,560 (2.02x) |

Launch command for case #2:
```bash
python3 -m sglang.launch_server \
  --model-path <path>/GLM-5.3-Flash --trust-remote-code \
  --tp-size 4 --expert-parallel-size 4 \
  --quantization fp8 --disable-shared-experts-fusion \
  --moe-runner-backend deep_gemm \
  --kv-cache-dtype fp8_e4m3 \
  --dsa-prefill-backend flashmla_sparse_q8 --dsa-decode-backend fa3 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --context-length 200000 --mem-fraction-static 0.85
```

## A config-level workaround does not exist

`--json-model-override-args '{"index_kpool":1}'` is accepted, but leaves the engine in an inconsistent
half-KPool state:

| attempt | assertion |
|---|---|
| kpool=1 + `flashmla_kv` + MTP | `AssertionError: DSA kpool target_verify requires kpool_write_plan` |
| kpool=1 + `flashmla_kv`, no MTP | `AssertionError: kpool_decode_update_index_cache called when kpool compress is disabled` |

KPool has a dedicated indexer (`srt/layers/attention/dsa/dsa_indexer_kpool.py`) reading three fields
(`index_kpool`, `index_kpool_always_select_tail`, `index_kpool_compress`); overriding one is not enough.
We deliberately did **not** override the rest: KPool changes which tokens sparse attention selects, and the
indexer weights were exported for `kpool=4`, so such a run would silently produce degraded output.

**Consequently we could not verify whether KPool is the *only* blocker for FP8 KV on this model** - that
needs a code change, not a config.

## Suggested fixes (independent)

1. **Teach `flashmla_kv` to handle the KPool tail.** GLM-5.2 already proves `flashmla_kv` + FP8 KV is sound;
   only the tail handling is missing. This looks like the smaller change.
2. **Give the fa3 DSA path a bf16-query x fp8-KV mode**, mirroring what `aiter` already does.

(Extending `_resolve_kpool_tail_backend` to also cover `flashmla_kv` is *not* a third option on its own -
the tail would land on `fa3`, which still faces fp8 KV and hits the same dtype error. It depends on #2.)

## Environment

- 8x NVIDIA H20 (SM90), CUDA, driver/runtime per `lmsysorg/sglang:glm-5.3-flash`
- sglang `0.0.0.dev1+g033446bb05` (image `lmsysorg/sglang:glm-5.3-flash`)
- Models: `ZhipuAI/GLM-5.3-Flash` (native FP8, e4m3) and GLM-5.2-W4AFP8
- Related: #36802 (same model, `--enable-dp-attention` incompatibility)

## #37712 [Bug] GLM-5.3-Flash: CUDA OOM in fp8_mqa_logits during long-context prefill kills all TP ranks

- URL: https://github.com/sgl-project/sglang/issues/37712
- 类型: Issue；状态: open；merged_at: None；创建: 2026-09-03T04:35:47Z；更新: 2026-09-20T03:02:14Z

### Checklist

- [x] I searched related issues but found no solution.
- [x] The bug persists in the latest version.
- [ ] Issues without environment info and a minimal reproducible demo are hard to resolve and may receive no feedback.
- [ ] If this is not a bug report but a general question, please start a discussion at https://github.com/sgl-project/sglang/discussions. Otherwise, it will be closed.
- [x] Please use English. Otherwise, it will be closed.

### Describe the bug

Serving GLM-5.3-Flash on 4 B300s with ​official image: `lmsysorg/sglang:glm-5.3-flash`​, the server occasionally die during prefill with a huge single allocation inside the DSA indexer, and every TP rank hits it in the same step:
```
dsa_indexer_kpool.py, line 998, in _get_topk_ragged_kpool_plan
    logits = deep_gemm.fp8_mqa_logits(
tvm.error.InternalError: CUDA out of memory. Tried to allocate 73.65 GiB. GPU 1 ... 50.40 GiB is free.
```
The KV pool itself is far from full at that point (token usage is around 0.35). It seems related to the long-context traffic with prefix-cache hits: single long requests up to 1M tokens work fine, the crash shows up when several long-context requests are being prefilled around the same time. 

### Reproduction

Reproduce it on current `lmsysorg/sglang:glm-5.3-flash` image, here're the serving parameters:
```
--tp-size 4 
--ep-size 4 
--dsa-prefill-backend trtllm 
--dsa-decode-backend trtllm 
--kv-cache-dtype fp8_e4m3 
--chunked-prefill-size 16384 
--max-prefill-tokens 16384 
--mem-fraction-static 0.78 
--enable-hierarchical-cache 
--hicache-size 208 
--speculative-algorithm DFLASH
```
Sending agent-style traffic with shared long contexts (300K to 1M tokens) will reproduce it.

### Environment

`lmsysorg/sglang:glm-5.3-flash` image, GLM-5.3-Flash FP8 (also reproduced with the RadixArk NVFP4 checkpoint)

## #37524 GLM-5.3-Flash bug tracking

- URL: https://github.com/sgl-project/sglang/issues/37524
- 类型: Issue；状态: open；merged_at: None；创建: 2026-09-02T03:20:25Z；更新: 2026-09-16T11:12:06Z

- [x] #36550
- [ ] #36653
- [ ] #36669
- [ ] #36711
- [ ] #36830
- [ ] #36886
- [ ] #36906
- [ ] #37548
- [ ] #37712
- [x] #37745
- [ ] #38031
- [x] #39692

## #41057 [Perf] Elide redundant DSA index-K storage with HiCache L2

- URL: https://github.com/sgl-project/sglang/pull/41057
- 类型: PR；状态: open；merged_at: None；创建: 2026-09-24T06:32:16Z；更新: 2026-09-24T07:38:58Z

Depends on #38426 (HiCache producer-layer host sidecar). This PR adds the matching device-side allocation and KV capacity budget. Please review the final commit for this PR; the current diff also includes its unmerged dependency.

# [Perf] Elide redundant DSA index-K storage with HiCache L2

## Motivation

GLM-5.3 has 78 DSA layers. Of these, 21 produce index-K entries and 57 reuse a preceding layer's top-k result. Without HiCache, SGLang can leave zero-row index-K placeholders for the reuse layers. Enabling HiCache L2 currently disables that device-side elision, allocating and budgeting index-K for all 78 layers even though the 57 reuse layers never produce it.

With the official GLM-5.3 FP8 checkpoint on 8×H200, removing this redundant allocation increases reported GPU KV capacity per DP pool from **350,016 to 398,144 tokens (+48,128; +13.75%)** under an otherwise identical configuration.

## Modifications

- Enable the existing `skip_topk_layers` device allocation path for **non-DCP HiCache L2** and use the same producer-layer selection when computing bytes per KV token. Zero-row placeholders keep layer IDs stable.
- Reuse the producer-layer packing in the HiCache indexer sidecar from [#38426](https://github.com/sgl-project/sglang/pull/38426). Host allocation and transfers therefore also cover only layers with nonempty device index-K buffers; this is not an isolated GPU-only change.
- Keep the EAGLE draft layer allocated. Leave DCP, HiCache L3, PD disaggregation, HiSparse, and external-linker paths unchanged.

## Accuracy Tests

Both variants started successfully (`/health` returned HTTP 200) and answered `4` to “What is 2+2?” with the same official GLM-5.3 FP8 checkpoint and EAGLE configuration.

We then tested the **changed variant** through HiCache L2 eviction and load-back, routing every request to the same DP pool. Two distinct ~215K-token prompts (C and D) each ended with the same simple arithmetic question. Both cold requests completed normally and answered `4`. Across C and D, `evicted_tokens_total` rose from 264,960 to 1,122,304 and `hicache_backup_tokens_total{pool="kv"}` rose from 264,960 to 989,952. Replaying C after D reported 214,912 cached prompt tokens, including **34,752 from host L2**; the L2 load-back counter increased from 132,224 to 201,728 across the two TP ranks, consistent with 34,752 logical host-hit tokens per rank. The answer remained `4`, the finish reason was `stop`, and `/health` remained 200.

For a small concurrent check, two distinct ~100K-token prompts (E and F) were completed, followed by a separate ~340K-token prompt to evict them from GPU cache. Replaying E and F **concurrently** returned HTTP 200 and `4` for both, with host-hit counts of **99,904** and **42,624** tokens respectively. The load-back counter increased from 201,728 to 486,784 across two TP ranks, matching `(99,904 + 42,624) × 2`. `hicache_dropped_tokens_total{reason="host_pressure"}` stayed at 0 and `/health` remained 200. These counter totals include both TP ranks; the response host-hit counts are per logical request.

This verifies the modified L2 path for these requests, not output equivalence over a larger test set, a 1M request, or prolonged/high-concurrency stability. The baseline variant was used for the capacity and short-generation comparison; its L2 eviction/load-back was not rerun in this round.

PR unit coverage should verify the 78→21 device allocation and matching capacity budget, the retained EAGLE draft layer, producer-layer sidecar backup/load-back, and guards for unchanged paths. The submitted source passed targeted H200 GPU tests: 60 passed, 1 skipped in the main test files; three related HiCache regression files passed 39 tests with 2 skips. Pre-commit passed. The skips are environment/backend exclusions; CI is pending.

## Speed Tests and Profiling

Not applicable: this PR targets **KV capacity**, not latency or throughput. The measurements below are capacity results, not speed benchmarks.

We started the baseline and changed server sequentially, waiting for all eight GPUs to release memory between runs. Both used the same local copy of the official `zai-org/GLM-5.3` FP8 checkpoint (`glm_moe_dsa`, 78 layers, FP8 E4M3 block size 128×128), SGLang source stack, and launch options. The only source difference between the two runtime overlays was the device index-K elision policy in `kv_cache_configurator.py`.

| Setting | Value in both runs |
|---|---|
| Hardware / parallelism | 8×H200; TP8, DP4, DCP1 |
| GPU KV | `--mem-fraction-static 0.85 --kv-cache-dtype fp8_e4m3 --page-size 64` |
| Speculation | EAGLE, 5 steps, top-k 1, 6 draft tokens |
| HiCache | L2, `--hicache-size 128` (GB per rank), `write_back`, `direct`, `page_first_direct`; no L3 backend |
| Prefill | Requested chunk size 32768; DP attention resolved it to 8192 per DP rank in both runs |

| Observation | Baseline | Device index-K elision |
|---|---:|---:|
| GPU KV capacity, `sglang:max_total_num_tokens`, per DP pool | 350,016 | **398,144 (+13.75%)** |
| GPU KV pool memory, `sglang:kv_cache_memory_usage_gb`, per rank | 20.04 GB | 20.00 GB |
| HiCache host indexer allocation, per rank | 25.76 GB (78 target + 1 draft layer) | 7.17 GB (21 target + 1 draft layer) |
| HiCache layer layout | 78 target + 1 draft | 21 target + 1 draft |

The startup `max_total_num_tokens` logs agree with the metrics on every rank. Each pair of TP ranks belongs to **one** DP pool, so their reported token capacities must not be added together. The KV pool memory metric includes its index-K buffers, but does not represent total process HBM. Nearly the same pool size holds more tokens because index-K consumes fewer bytes per token.

## Checklist before marking ready for review

- [x] Run the relevant unit tests and pre-commit checks on the exact PR diff.
- [ ] Run CI and add its results to the PR description.
- [x] Keep the capacity claim separate from latency/throughput claims; no speed benchmark is required for this capacity-focused PR.


GPU test details: before rebasing onto the latest #38426, 99 passed and 3 skipped across five related files on one H200. After rebasing, the submitted source was retested on one H200: 91 passed and 2 skipped across the four files changed by this PR and its #38426 dependency. All tests ran against an isolated copy of the submitted source on one H200, without changing the running service source. Skipped tests: CUDA 13 page-first-direct batch-copy (`cudaErrorInvalidValue` also with the old dense layout); a pre-existing Mamba write-back staging TODO; and CUDA 13 MiniMax direct+page-first-direct transfer (`cudaErrorInvalidValue`). Pre-commit passed on the rebased commit. Separate 8×H200 service measurements and L2 load-back checks are described above. CI pending.



## #40434 [DCP] Add GLM-5.3 Flash EAGLE/MTP support

- URL: https://github.com/sgl-project/sglang/pull/40434
- 类型: PR；状态: open；merged_at: None；创建: 2026-09-20T08:16:58Z；更新: 2026-09-20T08:17:50Z

Split from #39117, which remains open for reference. **Stacked on #40433**; review against that branch. The base PR intentionally rejects DSA + DCP + speculative decoding.

Add GLM-5.3-Flash EAGLE/MTP support on top of target-only DCP:

- Replicate draft KV/indexers through DSA pool constructor flags and budget their full global capacity. At DCP4/base 64, draft pool pages are 256; indexer kernels retain 64-entry blocks.
- Initialize effective DCP size/rank from the worker role: targets use the global group, replicated drafts use `(1, 0)`. Shared MLA forward and backend metadata use that geometry instead of `is_nextn`.
- Translate target-verify KV indices, return LSE for the DCP merge, and keep draft execution out of target collectives. Restore the speculative e2e variant.
- Replace the base's blanket rejection with a DSA-scoped EAGLE-family `topk > 1` rejection pending relocation work. Generic MLA/GQA/SWA draft-accounting changes are excluded; existing K3 + DSpark accounting remains unchanged.

HiCache + DCP + EAGLE/MTP stays guarded. General dense-MLA draft pool replication and heterogeneous-DCP target KV transfer remain outside this change.

Validation: 68 existing CPU tests passed, one CUDA-only test skipped; 32 speculative guard checks, 540 backend role/metadata checks, and 248 pool/restore/boundary checks passed. GPU/transport operations used CPU substitutes. Pre-commit passed. GPU/e2e and GSM8K/AIME reruns remain pending; evaluation is paused.



## #40433 [DCP] Restore GLM-5.3 Flash target-only decode CP

- URL: https://github.com/sgl-project/sglang/pull/40433
- 类型: PR；状态: open；merged_at: None；创建: 2026-09-20T08:14:58Z；更新: 2026-09-20T08:17:54Z

Split from #39117, which remains open for reference. This is the base of the two-PR stack; speculative decoding is added by stacked PR #40434.

Restore GLM-5.3-Flash DCP prefill/decode and fix global KV locations addressing rank-local buffers (#36886).

- Shard target KV, replicate the target indexer, and combine sparse decode outputs with base-2 LSE. Prefill gathers KV and remaps RAGGED indices.
- At DCP4/base 64, target KV pages are 64 and indexer/allocator pages are 256; indexer kernels retain packed 64-entry blocks. Keep indexer capacity, accounting, CPU copies, and host geometry consistent.
- Reject **DSA + DCP + any speculative algorithm** at startup. Existing K3 + DSpark remains allowed. Normal GLM-5.3/RoPE DSA remains unsupported.

Indexer PD descriptors retain their base-page wire layout; heterogeneous-DCP target KV transfer remains unresolved.

Validation on this split: 55 existing pool/DCP CPU tests passed; 110 argument-guard checks and 126 target pool/restore/boundary checks passed, with CPU substitutes for GPU/transport operations. Pre-commit passed. GPU/e2e and accuracy reruns remain pending; evaluation is paused.



## #39350 [GLM-5.3] Fuse unquantized KDA projections with attention TP

- URL: https://github.com/sgl-project/sglang/pull/39350
- 类型: PR；状态: open；merged_at: None；创建: 2026-09-14T03:29:11Z；更新: 2026-09-21T18:31:53Z

## Motivation

Enable the existing BF16 KDA projection fusion for GLM-5.3-Flash with DP attention. The checkpoint uses FP8 elsewhere but keeps these projections in BF16; attention TP can also differ from global TP.

This adapts the per-projection eligibility design from linkedlist771's #38254 and adds attention-TP sharding. The overlapping eligibility change should be reconciled if #38254 lands first.

## Modifications

- Check the six original projection prefixes using native FP8 exclusion rules.
- Construct unquantized fused projections with the attention TP rank and size.
- Keep the original projections for LoRA and unsupported quantization configurations.

## Accuracy Tests

### H200 end-to-end CI

The existing `TestGLM53FlashH200HighThroughput.test_gsm8k` passed on commit `8c7eaa6925` on 2026-09-20: **470/500 correct (94.0%)**, above the **93.0%** threshold, with 20-shot prompts. The test ran on **8 H200 GPUs**, using native GLM-5.3-Flash, **TP8/EP8/DP8 with DP attention**, **DeepGEMM + DeepEP**, BF16 KV cache and full decode CUDA Graph, in a **combined prefill/decode deployment**. [CI test log](https://github.com/sgl-project/sglang/actions/runs/35490442229/job/106024347864).

The loaded model snapshot (`eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`) excludes the relevant projections from FP8 quantization in all 34 KDA layers. Together with attention TP=1 and LoRA disabled, this configuration selects the updated fusion path; the accuracy test does not separately assert branch execution. This is a candidate-only accuracy check; the paired PD comparison below is a separate result.

### PD accuracy comparison

Short and 64K PD serving smokes pass.

| Dataset | Main | Candidate |
|---|---:|---:|
| Short GSM8K | 63/64 | 63/64 |
| Full GSM8K | 1265/1319 | 1262/1319 |
| 64K-input GSM8K | 60/64 | 63/64 |

Full GSM8K has 8 paired wins and 11 losses; the conservative 95% interval is [-1.361, +0.911] pp. The 1 pp noninferiority gate is **INCONCLUSIVE**, so quality acceptance remains **NOT_APPROVED** and this PR stays draft. The long set has no new losses; one baseline response reached the 2,048-token cap.

## Speed Tests and Profiling

**2 × 8 H20:** prefill TP8/EP8, decode TP8/EP8/DPA8; Triton MoE, Mooncake PD, BF16 KV, no MTP. Prefill graphs disabled; decode uses full CUDA Graph. Requests have **65,536 input / 1,536 output tokens**.

Original A→B measurements:

| Point | Main | Candidate | Change |
|---|---:|---:|---:|
| Cold C1, N5: median TTFT (ms) | 4,888.82 | 4,890.34 | +0.03% |
| Cold C32, N32: output tok/s | 271.24 | 273.64 | +0.88% |
| Warm C32, N128: output tok/s | 1,376.89 | 1,572.55 | +14.21% |
| Warm C128, N512: output tok/s | 4,235.50 | 4,383.37 | +3.49% |

Warm points had asymmetric first-batch TTFT. A separate B→A confirmation used an identical C128/N128 primer on the same GPUs; service age differed between arms:

| Point | Main | Candidate | Change |
|---|---:|---:|---:|
| Warm C32, N128: output tok/s | 1,620.02 | 1,663.79 | +2.70% |
| Warm C128, N512: output tok/s | 4,292.62 | 4,360.92 | +1.59% |

Confirmation latency:

| Confirmation point | Main mean TTFT | Candidate mean TTFT | TTFT change | Main mean TPOT | Candidate mean TPOT | TPOT change |
|---|---:|---:|---:|---:|---:|---:|
| d-lat | 2,003.92 ms | 2,028.99 ms | +1.25% | 18.08 ms | 17.52 ms | -3.14% |
| d-load | 6,028.90 ms | 6,353.34 ms | +5.38% | 22.65 ms | 22.17 ms | -2.12% |

First-batch TTFT:

| Run | Point | Main first-batch TTFT | Candidate first-batch TTFT |
|---|---|---:|---:|
| Original A→B | d-lat | 15,765.88 ms | 5,637.41 ms |
| Original A→B | d-load | 21,273.59 ms | 17,961.70 ms |
| Confirmation B→A | d-lat | 4,782.88 ms | 4,797.57 ms |
| Confirmation B→A | d-load | 17,840.07 ms | 18,742.10 ms |

All performance requests succeeded without retries. These are whole-patch measurements for this setup, not an isolated fusion speedup.

### KDA operator profile

Captured during **real GLM-5.3-Flash HTTP requests** through router → prefill → Mooncake → decode, using the full checkpoint and live hidden states on the same 2 × 8 H20 setup. Both arms use identical 49K-input / 64-output requests, long-request warmup, and a KV-cache flush before each phase.

The comparison selects **KDA layer 0** on the same physical ranks in both arms. Means cover five complete forward steps:

| Phase | Tokens/rank | Attention TP | Compute kernels before → after | Mean GPU kernel sum before → after |
|---|---:|---:|---:|---:|
| Decode, TP7/DP7, full CUDA Graph | 1 | 1 | 9 → 2 | 86.05 → 66.92 µs |
| Prefill, TP0, eager | 8192 | 8 | 7 → 2 | 1717.03 → 1697.10 µs |

Six separate matrix multiplications become one merged GEMM and one batched GEMM; the original decode path also has three split-K reduction kernels.

The blue area in each original-trace screenshot selects **all projection kernels**, from the first kernel's start to the last kernel's end. Perfetto's bracket displays the **whole group's elapsed time, including inter-kernel gaps**. These are layer 0 in the third of five steps; the table above reports five-step kernel-duration sums.

**Decode before: 9 kernels, selected interval 85.983 µs**

![Decode before: all projection kernels selected in original Perfetto trace](https://raw.githubusercontent.com/bytedance-iaas/sglang/a8e0aa2d6b051bb98ee1ddf37546de4968ced392/real-request-profile/area-screenshots/decode-baseline.png)

**Decode after: 2 kernels, selected interval 64.000 µs**

![Decode after: all projection kernels selected in original Perfetto trace](https://raw.githubusercontent.com/bytedance-iaas/sglang/a8e0aa2d6b051bb98ee1ddf37546de4968ced392/real-request-profile/area-screenshots/decode-candidate.png)

**Prefill before: 7 kernels, selected interval 1732.741 µs**

![Prefill before: all projection kernels selected in original Perfetto trace](https://raw.githubusercontent.com/bytedance-iaas/sglang/a8e0aa2d6b051bb98ee1ddf37546de4968ced392/real-request-profile/area-screenshots/prefill-baseline.png)

**Prefill after: 2 kernels, selected interval 1702.566 µs**

![Prefill after: all projection kernels selected in original Perfetto trace](https://raw.githubusercontent.com/bytedance-iaas/sglang/a8e0aa2d6b051bb98ee1ddf37546de4968ced392/real-request-profile/area-screenshots/prefill-candidate.png)

Each image is cropped directly to the native time ruler and GPU track. The original trace, kernel names, step annotations and time axis are unchanged; before/after use equal zoom within each phase. Adjacent operations outside the blue selection are excluded. All selected kernels and source hashes were verified against the original files. [Exact selection ranges, kernel-only sums and verification](https://github.com/bytedance-iaas/sglang/tree/a8e0aa2d6b051bb98ee1ddf37546de4968ced392/real-request-profile/area-screenshots).

Full-model tables precede the equivalent predicate simplification; these request profiles use the current implementation.

### End-to-end launch and validation commands

Run the workers in GPU Pods and validation in a client Pod. Use the same model, dependencies and arguments for both arms. The measured setup used 2 × 8 H20, PyTorch 2.13/CUDA 13, sglang-kernel 0.4.7, DeepGEMM 0.2.0, NCCL 2.29.7 and EvalScope 1.11.1.

The commands below use `/work/models/GLM-5.3-Flash` for the model. Replace the explicitly marked `<prefill-pod-ip>`, `<decode-pod-ip>` and `<router-pod-ip>` with your Pod IPs. Adjust the model path and RDMA interface names to your environment. Startup commands list non-default settings for the measured H20 setup at SGLang `8c7eaa6925` (the baseline has the same relevant defaults); `LD_LIBRARY_PATH` preserves the image's existing library search path.

#### Prefill worker

```bash
env NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 \
  NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4 \
  NCCL_IB_GID_INDEX=3 MC_GID_INDEX=3 \
  LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}" \
  python3 -m sglang.launch_server \
  --model-path /work/models/GLM-5.3-Flash --served-model-name GLM-5.3-Flash \
  --trust-remote-code --host 0.0.0.0 --tp-size 8 --ep-size 8 \
  --disaggregation-bootstrap-port 8997 --disaggregation-ib-device mlx5_1,mlx5_2,mlx5_3,mlx5_4 \
  --context-length 69632 --max-prefill-tokens 8192 \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --disable-shared-experts-fusion --moe-runner-backend triton \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --skip-server-warmup --enable-metrics --enable-metrics-for-all-schedulers \
  --random-seed 2201 \
  --port 31231 --disaggregation-mode prefill \
  --mem-fraction-static 0.70 --max-running-requests 32 \
  --cuda-graph-backend-decode disabled
```

#### Decode worker

```bash
env NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 \
  NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4 \
  NCCL_IB_GID_INDEX=3 MC_GID_INDEX=3 \
  LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}" \
  SGLANG_DEEPGEMM_STANDARD_LAYOUT=compact \
  python3 -m sglang.launch_server \
  --model-path /work/models/GLM-5.3-Flash --served-model-name GLM-5.3-Flash \
  --trust-remote-code --host 0.0.0.0 --tp-size 8 --ep-size 8 \
  --disaggregation-bootstrap-port 8997 --disaggregation-ib-device mlx5_1,mlx5_2,mlx5_3,mlx5_4 \
  --context-length 69632 --max-prefill-tokens 8192 \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --disable-shared-experts-fusion --moe-runner-backend triton \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --skip-server-warmup --enable-metrics --enable-metrics-for-all-schedulers \
  --random-seed 2201 \
  --port 31232 --disaggregation-mode decode \
  --mem-fraction-static 0.73 --max-running-requests 128 \
  --dp-size 8 --enable-dp-attention --enable-dp-attention-local-control-broadcast \
  --cuda-graph-max-bs-decode 128
```

#### Router

```bash
python3 -m sglang_router.launch_router --pd-disaggregation \
  --prefill "http://<prefill-pod-ip>:31231" 8997 --decode "http://<decode-pod-ip>:31232" \
  --port 30030 --prometheus-port 29030 \
  --worker-startup-timeout-secs 3600 --disable-retries
```

#### Serving correctness smoke

```bash
curl --fail-with-body "http://<router-pod-ip>:30030/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash","messages":[{"role":"user","content":"What is 17 plus 25? Answer with only the number."}],"temperature":0,"max_tokens":256,"chat_template_kwargs":{"reasoning_effort":"low"}}'
```

#### Full standard GSM8K validation

```bash
evalscope eval --model GLM-5.3-Flash --eval-type openai_api \
  --api-url "http://<router-pod-ip>:30030/v1" --api-key EMPTY --datasets gsm8k \
  --dataset-args '{"gsm8k":{"few_shot_num":4,"subset_list":["main"]}}' \
  --model-args '{"max_retries":0,"timeout":600}' \
  --generation-config '{"temperature":0,"max_tokens":2048,"retries":0,"timeout":600,"stream":false,"seed":42,"reasoning_effort":"low","extra_body":{"chat_template_kwargs":{"reasoning_effort":"low"}}}' \
  --eval-batch-size 8 --limit 1319 --seed 42 --work-dir results/gsm8k
```

The reported quality subsets use fixed replay prompts; the standard GSM8K command above is a separate reproducible validation entry point.

## Checklist

- [x] Format and static checks pass.
- [x] Provide correctness, quality and performance results with limitations.
- [x] Reuse existing Linear and quantization interfaces.

## Co-author

@linkedlist771 contributed the per-projection quantization eligibility design in #38254.

Co-authored-by: LLinkedlist <72634327+linkedlist771@users.noreply.github.com>



## #40854 [DSA] Chunk the kpool indexer MQA logits by query rows under a free-memory budget

- URL: https://github.com/sgl-project/sglang/pull/40854
- 类型: PR；状态: open；merged_at: None；创建: 2026-09-23T05:08:33Z；更新: 2026-09-23T22:15:16Z

## Summary

Chunk the kpool indexer's MQA logits by query rows under the #39095 free-memory budget, which did not cover `dsa_indexer_kpool.py`.

GLM-5.3-Flash, 4x GB300, TP4, chunked prefill 16384; 16 requests of 1,015,808 in / 256 out, concurrency 8.

| | main | this PR |
|---|---|---|
| Allocator OOM retries | 104 | **0** |
| Largest failed allocation | 15.72 GiB (10.26 GiB free) | none |
| Peak GPU memory | 283,210 MiB (99.65%) | 271,382 MiB (95.49%) |
| Mean TTFT | 132.3 s | 134.2 s |
| GSM8K | 96.97% | 97.27% |

<details><summary>Commands</summary>

```bash
sglang serve --model-path zai-org/GLM-5.3-Flash --tp-size 4 \
  --reasoning-parser auto --tool-call-parser auto \
  --speculative-algorithm EAGLE --speculative-num-steps 5 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 6

python -m sglang.benchmark.serving --backend sglang --dataset-name random \
  --random-input-len 1015808 --random-output-len 256 --random-range-ratio 1.0 \
  --num-prompts 16 --max-concurrency 8 --warmup-requests 0 --flush-cache

sgl-eval run gsm8k --base-url http://127.0.0.1:30000/v1 \
  --temperature 1.0 --top-p 0.95 --max-tokens 16384
```

</details>



## #36590 fix(hicache): accept direct DSA MTP draft pools

- URL: https://github.com/sgl-project/sglang/pull/36590
- 类型: PR；状态: open；merged_at: None；创建: 2026-08-27T02:07:31Z；更新: 2026-08-28T10:55:27Z

## Problem

`build_hybrid_mamba_stack` assumes every MTP draft pool is a `HybridLinearKVPool` wrapper and unconditionally reads `pool.full_kv_pool`.

GLM-5.3-Flash uses DSA for its MTP draft layer and exposes a direct `DSATokenToKVPool`. Starting the official NEXTN recipe with HiCache fails after CUDA graph capture:

```text
AttributeError: 'DSATokenToKVPool' object has no attribute 'full_kv_pool'
```

## Fix

Resolve the wrapped `full_kv_pool` when present, otherwise preserve the direct KV pool. Add unit coverage for both layouts.

## Validation

- `ruff check` passes for both touched files.
- Reproduced on the dedicated `lmsysorg/sglang:glm-5.3-flash` image, GLM-5.3-Flash TP8/EP8, NEXTN adaptive MTP 5/1/6, FP8 KV + TRT-LLM DSA, HiCache dynamic backend.
- Before patch: scheduler aborts in `build_hybrid_mamba_stack`.
- After patch: server reaches HTTP 200 and serves requests.
- MTP benchmark (8K input / 1K output, 64 prompts, concurrency 16): 223.16 output tok/s, 1850.53 total tok/s, accept length 2.58.
- The same no-MTP configuration measured 147.62 output tok/s and 1284.56 total tok/s.



## #39436 [Feature] hicache: --hicache-mamba-size-gb for the hybrid host tier, plus per-pool host occupancy and eviction metrics

- URL: https://github.com/sgl-project/sglang/pull/39436
- 类型: PR；状态: open；merged_at: None；创建: 2026-09-14T15:59:42Z；更新: 2026-09-14T18:42:16Z

## Motivation

Hybrid (attention + Mamba/KDA) models back up one Mamba state checkpoint per prefill chunk plus one per finished request next to the KV rows of every cached token. `build_hybrid_mamba_stack` splits the fixed `--hicache-size` budget between the KV and Mamba host pools in proportion to their *device* bytes, and the device Mamba pool is small (sized for running requests), so the host Mamba pool ends up with far fewer checkpoint slots than the KV host tier needs anchors. Once a prefix's last checkpoint is evicted its KV rows on host cannot be restored and sit there as dead memory.

Reproducible on GLM-5.3-Flash TP4 (4x RTX PRO 6000 Blackwell, `--enable-hierarchical-cache --hicache-size 32 --hicache-write-policy write_through`, chunked prefill 4096): the default split gives 565 host Mamba slots against a 2,680,896-token KV host tier (coverage 82%). Fill 200 prompts of 8,448 tokens (3 checkpoints each), then replay 8 evicted ones: 0/8 host hits, replay TTFT 0.82 s (cold), with the Mamba pool at 565/565 and the KV pool 35% empty. The same cell with `--hicache-mamba-size-gb 14` (734 slots): 8/8 host hits, replay 0.15 s, `cached_tokens` 8,192 from host. With `auto` the split is KV 19.12 GB / Mamba 12.88 GB, 676 slots, coverage 109%, and a 36 x 65,536-token fill + 8 replays gives 8/8 host hits at 0.42 s vs 5.98 s cold.

Nothing on `/metrics` showed this: the anchor gauges `sglang:hicache_host_used_tokens` / `_total_tokens` read one host pool, and the Mamba pool's occupancy and its evictions were invisible. Related: #29037 proposes scaling the Mamba host pool proportionally when `--hicache-size` is set; this PR keeps the default split and adds an explicit knob plus the metrics to see when it is needed.

## Modifications

Commit 1, the knob (default path byte-identical):

- `--hicache-mamba-size-gb <GB | auto>` (`arg_groups/fields/memory.py`, `Optional[str]`, default `None`): a number reserves that many GB for Mamba checkpoints and gives the rest to KV; `auto` solves `slots = ceil(kv_host_tokens / chunked_prefill_size) + 4 x max_running_requests` as a two-pass fixed point with the KV share reduced accordingly. Unset keeps today's byte-proportional split.
- `hicache_hook.validate_hicache_mamba_size`: needs `--hicache-size > 0`; an explicit figure must be below it; `auto` needs chunked prefill.
- Both mamba stacks (plain and SWA) in `hybrid_pool_assembler.py` apply the knob; the KV pools stay proportional among themselves. One boot line per hybrid host tier: `host mamba slots N cover T tokens at one checkpoint per C-token chunk; KV host tier K tokens (coverage x%)`, WARNING below 100%. The line is skipped unless every input is a real size, so pool assembly under mocked params never trips on it.
- The arithmetic lives in the torch-free `hybrid_cache/hicache_mamba_sizing.py`.

Commit 2, the metrics (additive, existing gauges untouched):

- `sglang:hicache_host_pool_used_tokens{pool}` and `sglang:hicache_host_pool_total_tokens{pool}` (SchedulerMetricsCollector), fed from `SchedulerStats.hicache_host_pool_{used,total}_tokens` through the torch-free `observability/hicache_pool_stats.py::collect_host_pool_stats` when the tree cache has a `host_pool_group`.
- `sglang:hicache_host_pool_evicted_tokens_total{pool}` (RadixCacheMetricsCollector): `UnifiedRadixCache.evict_host` counts the host slots a `drive_host_eviction` step frees per pool before `_free_values` consumes them. KV counted under mamba pressure is a host prefix lost with its checkpoints. Restores per pool already exist as `sglang:load_back_tokens_total{pool}`.

Commit 3: `server_arguments.mdx` row.

## Accuracy Tests

No numerics touched (host pool sizing and metrics only). On the GLM-5.3-Flash TP4 cell above: greedy restore == cold output, EAGLE accept length 3.08 to 3.23 vs 3.09 unpatched (within noise), chat and tool-call smoke identical, no skipped-checkpoint lines. Boot split lines and the `Allocating X GB host memory` lines agree (mamba 10.78 / 14.00 / 12.90 GB for unset / 14 / auto); gauges present from boot with `pool=kv|mamba|indexer`, totals equal to the pools, `used` tracks fills, `evicted` appears for kv and mamba after the first host eviction; host hit = replay `cached_tokens` 8,192 / 61,440 with `cached_tokens_total{cache_source="host"}` moving.

Unit tests, all run here on CPU: `test_hicache_mamba_sizing.py` 28 (torch-free: unset knob byte-identical to `_split_hicache_size`, explicit and auto rules, parser, validation, the coverage-line input guard), `test_hicache_pool_stats.py` 8 (torch-free), `test_hybrid_pool_assembler.py` 12 (incl. `TestMambaHostCoverageLine`, torch CPU), `test_server_args.py` HiCache cases 9 (torch CPU).

## Speed Tests and Profiling

Sizing runs once at pool construction. The gauges add one dict walk per metrics interval; the eviction counter adds one `len()` per freed tensor on the host-eviction path. Decode TPOT unchanged on the cell above.

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [x] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed).
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).



## #39856 perf(hicache, hybrid): Full-KV prefetch anchor, eager Mamba L3 write, independent host Mamba pool ratio

- URL: https://github.com/sgl-project/sglang/pull/39856
- 类型: PR；状态: open；merged_at: None；创建: 2026-09-16T21:36:44Z；更新: 2026-09-17T04:30:55Z

> Stacked on #39845 (exclusive L2->L3 tiering). The first two commits here are that PR; review the last three. The three changes are independent knobs and can be split into separate PRs if preferred.

## Motivation

Three HiCache changes for hybrid GDN/Mamba models with a storage (L3) backend, each behind its own env knob and off by default. Together they took the L2 host hit rate on Qwen3.5-397B-A17B-NVFP4 (GB300, PD-disaggregated, AgentX closed-loop trace) from 89.5% to 91.9% and, with the tiering mode of #39845, made the L3 store worth 15-25K total TPS/GPU at concurrency 1000 that it otherwise does not deliver.

1. **Prefetch anchor** (`SGLANG_HICACHE_PREFETCH_ANCHOR_FULL_KV`). Hybrid models keep a Mamba state only at the last leaves of a path. Once a chain's tail has been tiered to L3 and evicted from host, the all-components `match_prefix` anchor falls back to an early node, and the L3 prefetch re-requests pages that L2 already holds. The prefetch now starts from the deepest host-backed node of the Full-KV walk instead. This is the cache-mode analogue of the buffer-mode re-anchoring added by #39283. Requires the Python tree core (`node_by_id` is not ported to the Rust core); otherwise the knob is ignored with a warning.
2. **Eager Mamba L3 write** (`SGLANG_HICACHE_L3_MAMBA_EAGER_WRITE`). Under exclusive tiering the KV pages of a node reach L3 at host eviction, but its Mamba state lives in a much smaller host pool and is usually gone by then, so the L3 copy is unusable for a hybrid prefetch. Persist only the Mamba state when its host backup completes; the KV pages follow at eviction as usual. Only meaningful with `SGLANG_HICACHE_L3_WRITE_ON_HOST_EVICT`; otherwise ignored with a warning.
3. **Host Mamba pool ratio** (`SGLANG_HICACHE_MAMBA_HOST_RATIO_SCALE`). The host Mamba-state pool is sized as `hicache_ratio x device Mamba pool`, the same ratio as KV. The state pool is tiny relative to KV, so on a node with spare host DRAM it is the first thing to thrash. The scale lets it grow independently (1.15-1.25 in our runs) without over-provisioning host KV. On the `--hicache-ratio` path the Mamba pool simply grows; on the `--hicache-size` path the byte budget is kept and the extra state bytes are taken out of the KV share.

## Modifications

- `base_prefix_cache.py`: `MatchResult.full_kv_last_node` (deepest node of the Full-KV walk regardless of other components) and a `storage_prefetch_anchor(req, anchor, matched_len)` hook with a pass-through default; `Req` carries `full_kv_last_node`. `scheduler._prefetch_kvcache` delegates to the hook; `model_runner.py` is untouched.
- `unified_radix_cache.py`: `storage_prefetch_anchor` walks up from `full_kv_last_node` to the first `backuped` node; `_write_backup_storage_mamba_only` builds the node's backup spec and issues `write_storage` with an empty KV part and the Mamba transfer as `extra_pools`. Mamba beliefs are recorded on the storage ack from the operation's pool transfers, the same point as KV beliefs, and healed at the same prefetch-cut invalidation (Mamba keys are the node's last KV page hash, so the KV chain cut applies). All knobs are resolved in `init_hicache`. Counters: `anchor_advanced`, `anchor_advanced_tokens`, `mamba_eager_writes`.
- `unified_tree_core.py`: `full_kv_last_node` in the match walk.
- `hybrid_pool_assembler.py`: `_mamba_host_ratio` scales the ratio path; `_rebalance_mamba_host_size` moves bytes from the KV share to the Mamba share inside the `--hicache-size` budget.
- `test_unified_radix_cache_unittest.py`: `TestStoragePrefetchAnchorFullKV` checks the upward walk (head node host-backed, leaf device-only: anchor moves to the head node with its prefix length; never shallower than the matched anchor; pass-through when off).

## Accuracy Tests

Not applicable (cache placement only).

## Speed Tests and Profiling

Qwen3.5-397B-A17B-NVFP4, GB300, 3P:2D (PP4 prefill, TP4/DP4/EP4 decode), Mooncake store with #39845, AgentX 3600 s:

| change | effect |
|---|---|
| host Mamba ratio 1.0 -> 1.25 | host hit 89.5% -> 91.9% |
| eager Mamba L3 write | C1000 +15-25K total TPS/GPU (L3 prefetches become usable for hybrid) |
| Full-KV anchor | removes repeated L3 requests for L2-resident pages after tail eviction (correctness of the prefetch span; throughput-neutral at C950) |

Unit tests: `test/registered/unit/mem_cache/{test_unified_radix_cache_unittest,test_storage_prefetch_lifecycle,test_hicache_staged_write_back_dispatch}.py`, Python tree-core backend, 1294 passed / 0 failed in the 2026-09-13 nightly container.

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [ ] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed).
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).



## #37198 [Hicache] Support Mamba branching setting by env

- URL: https://github.com/sgl-project/sglang/pull/37198
- 类型: PR；状态: open；merged_at: None；创建: 2026-08-31T04:58:27Z；更新: 2026-08-31T05:02:03Z

## Motivation
This PR optimizes cache hit ratio for single‑turn inference scenarios of hybrid‑Mamba models such as Kimi‑K3.

For multi‑turn conversations, storing the Mamba state at the end of each request is sufficient. However, real‑world deployments also feature single‑turn workloads, where different requests may share reusable prefixes. Previous PR https://github.com/sgl-project/sglang/pull/31181 enables mamba state creation at full‑KV branching points. This works well when single‑turn requests share **exactly identical** prefixes.

We have encountered a distinct real‑world case: single‑turn requests do share common prefixes, but the shared prefix lengths fluctuate around a threshold rather than being perfectly identical.

For example: Request 1 has length 6000; Request 2 shares a prefix of length 5120 with Request 1; Request 3 shares a prefix of length 4992 with Request 1. Given a page_size of 64, neither Request 2 nor Request 3 achieves a cache hit under the existing implementation. This happens because Request 1 stores its Mamba state at offset 5952 (near its end), while Request 2 store at length of 5120.

Our solution pre‑computes a stable prefix threshold and uses an environment variable to specify the storage location for the first incoming request, instead of requiring multiple iterations to obtain a stable prefix. In the scenario above, with a hit‑ratio coefficient set to 0.8, Request 1 will persist its Mamba state at offset 4800. Consequently, both Request 2 and Request 3 can hit the prefix up to length 4800. Therefore, the hit ratio is improved compared with the existing approach.

Note that requests do not universally reuse one single prefix. Instead, requests form small groups (e.g., dozens per group): one group shares prefix A, another group shares prefix B, with minimal overlap between A and B.

## Modifications
Introduce an environment variable to control where to store the Mamba state when the Mamba state length matches the full sequence length. When this environment variable is not set, or the full‑KV hit length is greater than the Mamba length, or the Mamba hit length exceeds the configured hit‑rate threshold, the original branch policy remains unaffected.

## Speed Tests and Profiling
On our production workloads, built upon PRs https://github.com/sgl-project/sglang/pull/31181 and https://github.com/sgl-project/sglang/pull/33639, this optimization further improves the KV‑cache hit rate by approximately 7%.


