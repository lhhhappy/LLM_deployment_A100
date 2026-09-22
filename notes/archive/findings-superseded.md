# Archived findings (superseded 2026-09-22)

原编号保留，不再复用。

> 归档于 2026-09-22：已被 F40（N=22 已出现） 取代。

## F5 — Competitor landscape (API scrape, 479 attempts)
- Max n_at_slo = 18 (7 attempts), best tpot_mean 0.0242. Nobody at 22. Binding gates at 18: fast_intra 2.2–3.0 s, overall_intra 3.5–5.2 s.


> 归档于 2026-09-22：已被 预言机模拟，被 F7/F13 及实测取代 取代。

### F3 quantified (scripts/sim_checkpoints.py, dev requests, no-eviction model, scorer gate assignment)
| KDA checkpoint policy | fast_intra real-uncached p50 / p90 / p95 / max | overall_intra p95 |
|---|---|---|
| end-of-prompt only | 2504 / 6468 / 7519 / 10676 | 14601 |
| + branch point (≈ v0.5.20 extra_buffer) | 2281 / 5474 / 6918 / 10066 | 14071 |
| + role-boundary (proposed) | 1348 / 2972 / 3326 / 5832 | 14071 |
| frozen `uncached_expected` (what the gate assumes) | p95 3144 | 14033 |
→ On the binding fast_intra gate, stock checkpointing makes the p95 request prefill ~2.2× more tokens than needed; a boundary checkpoint removes that. (Model only; eviction under memory pressure would make stock worse.)


> 归档于 2026-09-22：已被 E1/E2/E2b 实测（F36/F42–F47） 取代。

## F7 — Codex review: F3 simulation evidence boundary (supersedes F3 quantified interpretation)
- VERIFIED by source inspection, not a new experiment: `scripts/sim_checkpoints.py:29` reads the NEXT request's frozen LCP to select the `boundary` checkpoint. Thus 6918 → 3326 is an oracle-policy model result, not a measured or causally implemented saving.
- The simulator stores checkpoint lengths without token-path identity or eviction; it adds both end and branch. Actual `schedule_batch.py:2913–2986` has one extend track position, with a selected branch replacing end. The simulator is not an exact stock-engine model or a rigorous bound.
- Hybrid reuse needs the latest valid same-prefix KDA checkpoint at or before LCP, not necessarily AT LCP (`components/mamba.py:166`). The role-boundary idea remains a candidate; no GPU/TTFT benefit has been established.
- Author: Codex. Detailed review: `research/codex/R1_directions_and_review.md` §3.


> 归档于 2026-09-22：已被 E1/E2/E2b 实测 取代。

## F13 — Claude: non-oracle, path-aware D1 re-simulation (answers F7)
- Script: `scripts/sim_role_boundary.py` (tokenizer only, GPU-box CPU, no serving). Checkpoint candidates are computed from the CURRENT prompt only: last `<|user|>` / `<|observation|>` token inside the extend range, floor-64. Hits are path-aware (checkpoint valid only if token prefix identical). Stock model = ONE track per extend (branch point replaces end, per F7/`schedule_batch.py:2913-2986`) + 8192 chunk ends.
- Reachability over 411 consecutive pairs: next LCP ≥ last `<|user|>` of prev prompt in **394/411**; exactly equal in 241; next LCP == prev full length (strict append) in 146.
- fast_intra real-uncached (n=328): stock p50/p90/p95 = 2376 / 5807 / **7238**; role-boundary = 1348 / 2982 / **3326** (frozen expected p95 = 3144). overall_intra p95: 15744 → 15082. Max 46728 unchanged in both (outlier, not addressed by D1).
- Still a model: no eviction, no timing. Supersedes the oracle row of "F3 quantified"; F7's objection to the oracle is resolved, its other caveats stand.


> 归档于 2026-09-22：已被 底包源码已完整取回（F53） 取代。

## F14 — Codex: D6 external image access route, not access verification
- VERIFIED from official docs: external Docker access uses `registry.bohrium.dp.tech`, while platform refs use `registry.dp.tech`. The documented external route supports public and own custom images, not other users' shared images. https://bohrium-doc.dp.tech/docs/userguide/image/
- VERIFIED local CLI help: bohr 2.7.7 (`f4868539d66dffeb6fb2ef217c36342155e97eda`) exposes catalog and Dockerfile/build-log reads; image pull rewrites the registry front and uses Bohrium AK. A catalog ID/build ID cannot be inferred from the numeric segments of the image ref.
- No account-backed catalog request, credential inspection, registry login, pull, build, or pod was executed. Permission/digest/source/backend of the competition refs remain unknown. Author: Codex; `research/codex/R3_base_images.md`.


> 归档于 2026-09-22：已被 底包无 SPF；002 线退役（决策 29） 取代。

## F15 — Codex: #39717 now extends #40024, not an independent alternative
- VERIFIED on 2026-09-22: #40024 merged at `65ef55e2a8c51be0523723fcb98d7138b5839ebb`; open #39717 head is `742ddf60bb860ef19fabc61a566ca436e76279d5`. The latter refactors existing SPF interleaving and adds controls/HRRN. https://github.com/sgl-project/sglang/pull/40024 https://github.com/sgl-project/sglang/pull/39717
- VERIFIED diff/source: only one unfinished chunked prefill remains supported. Interleaved waiters must finish in that pass. A new D1 artificial split requires checking both the existing continuation and `adder.new_chunked_req` after final admission shaping.
- Historical 2.801s → 1.502s was an earlier fork on K3 DCP8 1P1D, not this exact patch or GLM/A100. No local apply-check/tests/performance run. Author: Codex; `research/codex/R4_prefill_scheduling.md`.


> 归档于 2026-09-22：已被 模拟器结论，被实测取代 取代。

## F17 — Codex: F13 fixes oracle/path identity, but its policy differs from D1 design v1
- VERIFIED by reading, not rerunning `scripts/sim_role_boundary.py:28–60`: role candidates use the current prompt and hits compare the same token path; those F7 issues are resolved.
- The role arm retains stock branch/chunk states, then always adds end, then role. Its improvement therefore includes restored end checkpoints and does not establish that replacing a branch checkpoint with role is lossless. This refines the interpretation of F13, not its reported output numbers.
- The 394/411 reach statistic checks the previous prompt's last USER token, not the full role policy's last USER/OBS candidate or its runtime admission/retention. No finite-slot, queueing, decode, or eviction model. Author: Codex; R4 §5 and appended review in `patches/001-role-boundary-mamba-ckpt.md`.


> 归档于 2026-09-22：已被 v0.5.20 专有函数，底包无 取代。

## F19 — Codex: artificial split does not itself stop the admission loop
- VERIFIED local source: `_commit_prefill_admission` assigns `new_chunked_req` and charges budgets; `add_one_req` then returns `budget_state()`, which does not inspect that field (`schedule_policy.py:933–955,1453–1457,1541–1564`).
- `scheduler.py:3930–3966` stops on non-CONTINUE, not on a newly set chunk pointer. A role split with budget remaining can therefore continue admitting requests. A second partial would overwrite the pointer; the final `self.chunked_req is None` assertion does not detect two new partials in one pass.
- Design inference: after an accepted artificial split, returning OTHER is a possible conservative stop; retain the already committed request and allocator-group cleanup. Alternatively enforce complete-only subsequent admissions. No patch or test executed. Author: Codex; D1 design §8.3.


> 归档于 2026-09-22：已被 模拟器结论 取代。

## F24 — Claude: D1 runtime-policy variants (answers Codex D1 §8.1/§8.5), `scripts/sim_role_boundary.py` v2
- Same non-oracle, path-aware model as F13 (tokenizer-only, no serving), now with explicitly named runtime policies:
  | policy | taken / skipped | fast_intra real-uncached p50/p90/p95 | overall_intra p95 |
  |---|---|---|---|
  | stock (1 track per extend, branch replaces end) | – | 2376 / 5807 / 7238 | 15744 |
  | **role_conservative** (skip D1 whenever a branch track competes; keep stock then) | 710 taken / **2 skipped_branch_conflict** | 1348 / 2989 / **3332** | 15082 |
  | role_over_branch (role replaces branch; end kept) | 712 | 1348 / 2982 / 3326 | 15082 |
  | role_all (branch + role + end) | 712 | 1348 / 2982 / 3326 | 15082 |
- Conclusion: the conservative variant Codex proposed keeps essentially all of the modelled gain; branch conflicts are rare once D1 is active (the next-turn fork lands on the previous boundary checkpoint). → D1 v1 = role_conservative. Still a model: no eviction, no timing.


> 归档于 2026-09-22：已被 v0.5.20 专有 FP32 快照缓冲；底包从 bf16 的 h 取（F53） 取代。

## F25 — Codex: SGLang already exports one internal FP32 KDA accumulator state per extend
- VERIFIED / SOURCE at local commit `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`: `linear/kda_backend.py:864–931` allocates an FP32 track buffer, checks backend capability, and passes `track_state` / `track_chunk_idx`. `kernels/ops/attention/fla/chunk_delta_h.py:137–207` snapshots the FP32 accumulator before the selected 64-token chunk; `:330–348` still writes the complete extend's final running state. This does not split model forward.
- VERIFIED: the scheduler/metadata currently carry one selected cache point per request, not a list (`schedule_batch.py:2893–2987`). Ordinary prefill result handling invokes unfinished-cache insertion (`batch_result_processor.py:383`), so a selected interior point need not wait for decode completion to be registered.
- VERIFIED: ordinary intermediate `h` inherits activation dtype (`chunk_delta_h.py:396`); casting it to FP32 does not recover the dedicated snapshot precision. Existing kernel/dtype regression tests were read, not run. The kernel test registers B200 CI, not evidence of an A100 run.
- INFERRED design consequence: role-only replacement can reuse the single-point kernel but is not F24's role+end policy. Dual-track requires multi-point metadata, conv windows, independent slot ownership and tree insertion. If prompt end is unaligned, the reusable end-ish point is itself interior: the live final state cannot substitute for it.
- Author: Codex; `research/codex/R7_kda_internal_checkpoints.md` §§3–6. Source-only; no experiments or source edits.


> 归档于 2026-09-22：已被 模拟器已失去可信度 取代。

## F34 — Codex W3: offline replay/scoring model is implemented; capacity remains uncalibrated
- VERIFIED / CPU: T15 delivers `scripts/sim_closed_loop.py` and 30 passing tests in `scripts/test_sim_closed_loop.py`; the simulator imports original load_index/build_gap_plan/phase_gate/in_ttft_gate/q/evaluate with harness bytecode writes disabled. Dev inputs are 311 chains / 722 requests, with 314/388/20 chain_start/intra/turn_start and 328 fast-intra; the 3600-second chain cap compresses no chains in this dev cohort. Input/harness hashes match; evidence `evidence/T15_simulator/tests.log` and `validation.json`.
- VERIFIED / artifact scope: 576 main sensitivity runs, 144 source-output-length sensitivity runs and 4 complete N6/N10 per-request examples are on disk. Exact ten dev gates and request-average TPOT p95 <=0.10 are reported separately; no formal statistical allowance or measurement contract is claimed.
- MODEL OUTPUT / INFERRED only: with 40k uncached tok/s, 50ms base decode +0.02 batch slope, 2ms forward overhead, FCFS, full output budgets and no eviction, dev stock/D1 both pass through tested N=10 and first fail TPOT at N=14. Synthetic 808/9023/65 formal composition changes those modeled ceilings to stock22/D1 26, illustrating load-composition sensitivity rather than predicting formal capacity. Default cache counts use frozen D1 plus F24 stock tail-ratio proxies, not exact F24 per-request token-path results; per-request overrides are supported.
- Handoff: `research/codex/R9_offline_simulator.md` documents assumptions, scheduling/cache/output sensitivities and a stock N6 fit / N10 validation measurement plan. Stock runs cannot validate D1 snapshot correctness/split overhead or formal residency. No GPU, Trisol, service, tokenizer or submission operation.


> 归档于 2026-09-22：已被 模拟器已失去可信度 取代。

## F38 — Codex W5: public-prior simulator calibration makes TTFT bind, but does not identify a D1 ceiling gain
- VERIFIED / CPU: T18 adds `sim_closed_loop.py --envelope-fit`, explicit public-scheduler assumption, complete median residuals and an effective decode-speedup parameter; original 30 + new 11 tests pass (41 total). Public passing counts/medians reproduce F35; original harness/data hashes match. Evidence: `research/codex/R10_sim_calibration.md`, `evidence/T18_calibration/tests.log`, `validation.json`, all fit/region JSON. W5 used CPU only; no GPU, service, Trisol, image or submission action.
- MODEL OUTPUT / coarse prior, not deployment truth: five sampled SPF points (prefill 38–42k tok/s, effective decode12ms, slope0.04, overhead2ms, chunk16384, length exponent1.2, decode interval24–32) fit all 24 N6/10/14/18 median cells within factor2 and first fail fast/overall at N18, before TPOT. No recorded candidate fits all cells within factor1.5. The full band is deliberately loose, not a confidence interval; these are different teams' maximum-passing submissions, not one engine's scaling trace.
- MODEL OUTPUT: main-region FCFS stock/D1 ceilings are both6; SPF stock14, D1 either10 or14. Four qualifying one-parameter neighbors also show no gain. Nearby full-fit C42 gives stock18/D1 18, both first fail at22 (stock fast+chain, D1 fast); C43 gives stock18/D1 14. Two other template seeds, source-output lengths, F24-tail-inflated D1 and extra-overhead sensitivities produce no robust D1 increase. Exact first-failure gates/residuals are in R10.
- INFERRED / limits: this fixes R9's default TPOT-first interpretation under conditional model assumptions, not real GLM performance. Decode/base/speedup are degenerate; interleaving/chunk/length penalty and cache misses are weakly identified. Formal statistical slack (public chain p95 above30s can PASS), measurement windows, mixed/overlapped prefill/decode, actual D1 splitting and FULL-KV/KDA residency/eviction remain structural gaps. F36/F37 stock functional evidence is not superseded. These results do not justify an 18→22 claim or abandoning D1; same-engine stock/D1 timing remains decisive.


> 归档于 2026-09-22：已被 SPF/002 线，底包无（决策 29） 取代。

## F41 — D2 SPF port composes with D1; stock HRRN/LPM do not reproduce its modeled benefit（T20 / W7）

2026-09-22，Codex W7。**VERIFIED / CPU + source**：已移植 [SGLang #40024](https://github.com/sgl-project/sglang/pull/40024) merge `65ef55e2` 至只读base `94602c9` 的 D1 v1.1 副本，交付 `patches/002-spf-scheduling.patch` 与同名说明。`001→002 --fuzz=0` 和最终3文件字节一致性通过；**002单独应用到clean base失败3个D1上下文hunk**，须先001。单一 `_can_start_partial_prefill` 同时约束普通截断、host miss重选和D1切分；默认fcfs关闭D1时56组paired/112次执行的范围/预算/verdict快照字节一致。KV/commit/Mamba gap/tile/SWA关键方法AST与base一致；较大DSA对齐时续跑保留page/truncation的LCM。16项生产CPU测试、47项模拟器测试、15项原D1回归通过，含500随机准入轮与实际生产代码差分。证据 `evidence/T20_d2/`，测试D2-01…10；live D2-11/12仍todo。

**VERIFIED / simulator semantics**：R9/R10原 `spf` 已预留完整短waiter预算，**并非纯排序**；但它先准入waiter且预留时考虑slot。#40024先预留（不看slot）、先续跑、后准入；budget4096/page64/continuation16384/waiter512/max_running1时原模型续跑4096，而上游语义续跑3584并空置512。新增 `spf-upstream` 建模后一流程；模型仍无真实cache/host miss/内存容量/duplicate prefix。新增stock `hrrn`按processed-prefill-token aging与rid tie、`lpm`按absolute matched prefix；二者无续跑预算预留，保留>128 fallback。12个完整原fcfs/spf结果及trace与改动前脚本字节一致。

**MODEL OUTPUT / WHAT-IF，非实测容量**：固定R10参数/构成/完整输出，C6/C17/C18/C19/C31/C42/C43 ×5策略×2缓存×8档=560次模拟；14组原/新SPF ceiling均一致。主5候选stock SPF=14，D1 proxy=10/14/14/10/14；C42=18/18、C43=18/14。全部FCFS与HRRN=6；LPM=6，唯C6/stock=2。C42/stock/N14两种SPF有小量非零timing差异但不改过门。结果支持D2继续实测，不支持将stock HRRN/LPM视作这里SPF收益的替代，也不证明D1能提高临界档。完整数据 `evidence/T20_d2/calibration/comparison.json`，复现 `scripts/check_d2_sim_calibration.py`；新选项可经 `scripts/sim_closed_loop.py --schedulers fcfs,spf-upstream,hrrn,lpm` 使用。

