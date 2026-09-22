# R9 — Offline closed-loop contest replay simulator

Author: Codex W3 / T15, 2026-09-22 UTC. **All simulated latencies, runtimes, gate verdicts and ladder levels below are MODEL OUTPUTS, not serving measurements. The formal-mix results are WHAT-IF ONLY, not formal predictions.** No GPU, Trisol, service, submission, model download or tokenizer execution was used. `s1-dev/` remains read-only.

## 1. Deliverable and intended use

`scripts/sim_closed_loop.py` is a standard-library-only discrete-event simulator. It reuses the public harness's cohort loading, gap planning, gate classification, quantiles and dev scorer. It accepts replaceable per-request cache/output profiles and a parameterized engine model. Use it to choose measurements and reject implausible configurations before requesting a scarce 8-GPU slot; it is not yet calibrated enough to assert which real ladder level passes.

`scripts/test_sim_closed_loop.py` contains 30 CPU tests. Evidence: `evidence/T15_simulator/tests.log`. The complete grid, provenance hashes and every per-level gate detail are in `evidence/T15_simulator/sensitivity/sweep.json`. Representative per-request records for stock/D1 at N=6 and N=10 are in `evidence/T15_simulator/reference/`. A separate output-length sensitivity is in `evidence/T15_simulator/source_output/sweep.json`.

**INFERRED from the model outputs:** cache improvements do not necessarily increase the passing N. In this grid the TPOT extension usually binds before the dev TTFT gates. SPF can remove a fast-intra failure without increasing the passing level if TPOT already fails. This is a reason to calibrate decode and prefill interference, not evidence against D1.

## 2. What is exact and what is assumed

**VERIFIED / local source and data:**

- `s1-dev/harness/s1_common.py:153` filters canonical serving rows and assigns `_idx_in_chain` by the original chain order. The simulator preserves those indices and the cohort's `chains` / `req_ids` arrays, rather than sorting the cohort anew. `chains.jsonl` validates chain membership; its unavailable original tails are not reconstructed.
- The provided cohort contains **311 chains / 722 requests**, with **314 chain_start / 388 intra / 20 turn_start** and **328 fast_intra**. Context resets explain why the chain_start gate has more requests than there are chains. Sources: `s1-dev/harness/g0a/samples_v3/cohort_dev-combined-v1.json`, F31, and the real-input integration test.
- `s1-dev/harness/s1_loadgen.py:192` supplies `build_gap_plan` directly. The default cap is **3,600 s per chain**, including the first request's gap. It rounds each scaled integer-millisecond gap and assigns the remainder to the last nonzero gap. All requests remain present. This dev cohort has **3,027,842 ms** total gap and **zero capped chains**; the cap still matters for alternate inputs. These are input facts, not model timing measurements.
- Each free worker claims the next chain from the single stored-order queue, retains it across requests and gaps, then claims another chain. For request i>0, `dispatch_i = finish_(i-1) + effective_gap_i`; the first request of a newly claimed chain follows the same rule from the worker's release time. This matches `drive` and the measurement worker queue in `s1_loadgen.py:354` and `:597`.
- No absolute source dispatch timestamp becomes an open-loop arrival time. Faster or slower completions change future arrivals and therefore batch sizes and queue pressure.
- Imports suppress bytecode writes into the harness. The CLI refuses output paths inside `s1-dev/`, `src/sglang/` and the read-only task directory. Unit tests check unchanged harness source and bytecode hashes across an actual CLI replay. Every sweep records SHA256 hashes of requests, chains, cohort, three harness modules, simulator and supplied profile files.

**ASSUMED / engine model:** one aggregate engine represents the whole proposed 8-GPU configuration. Throughput is aggregate tok/s, not per-GPU throughput. N is the number of closed-loop agent slots. `--max-running` independently caps admitted engine requests; zero resolves to N. Sleeping workers occupy agent slots but do not occupy engine running slots.

The engine is a deterministic, non-preemptible sequence of prefill and decode forwards. Requests arriving during a forward wait until its boundary. Initial workers start together; exact ties use deterministic creation order. This is an idealized version of real thread ordering. There is no JIT warmup, GPU/CPU pipeline overlap, mixed prefill/decode forward, multi-rank imbalance, cancellation, eviction, OOM, transport jitter, or failure model. All service/error gates assume successful requests; the simulation cannot certify those properties.

## 3. Engine equations and scheduling

For a prefill forward containing request chunks with `tokens_j` real work:

```text
prefill_seconds = forward_overhead_ms / 1000
                + sum(tokens_j / prefill_tps
                      * max(1, prompt_length_j / 32768)^prefill_length_alpha)
```

`--prefill-length-alpha=0` gives length-independent throughput. Positive alpha penalizes full prompt length, including cached context. It is a surrogate for length-dependent kernels, not an attention FLOP model. A fully cached request still needs at least one model-work token to produce its first output; the reported cache count itself is not changed by this floor. Token budgets are rounded up to pages for admission, but durations charge actual modeled work tokens. `--chunk-tokens` must be a positive multiple of `--page-tokens` (defaults **8192 / 64**, model settings).

FCFS prioritizes an existing partial prefill, then admits waiting requests by arrival. A continuation can consume the entire chunk budget. With `--schedulers spf`, waiting requests are ordered by **actual modeled uncached work**, never by the gate's frozen classification. While a continuation exists, complete shorter requests can share its forward if the total budget still leaves the continuation at least one page. At most one unfinished prefill remains after each forward. This implements the central budgeting idea described for SGLang #40024 in `research/codex/R4_prefill_scheduling.md` §3; it does **not** reproduce every SGLang priority, allocator, host-load, request-limit or de-prioritization branch.

Completing prefill produces the first output token at the forward's end. All admitted requests already decoding then advance together by one token per decode forward:

```text
decode_step_seconds(B) = (decode_ms * (1 + decode_batch_slope * (B - 1)) / decode_speedup
                          + forward_overhead_ms) / 1000
```

Alternatively, `--decode-curve 1:30,4:34,8:40,16:55,32:85` supplies a piecewise-linear batch/time curve in milliseconds, excluding fixed overhead. Values outside its batch range clamp to the nearest endpoint; **do not extrapolate capacity with an underspecified curve**. A curve overrides the base decode time; combining it with a multi-value decode-time sweep is rejected.

T18 addition: `--decode-speedup` defaults to one and divides the decode compute term (also for curves), leaving fixed overhead unchanged. It is an effective speedup sensitivity, not an MTP acceptance/verification simulation; multiplying base time and speedup together is exactly degenerate. The original results below retain their original defaults.

T18 public-prior calibration is documented in [R10](R10_sim_calibration.md). Use `--mix formal-mix --envelope-fit data/all_att.json`; `--fit-scheduler fcfs|spf` conditions the unknown deployment scheduler, and `--fit-candidates` supplies reproducible Engine override arrays. The coarse fitted region is still WHAT-IF, does not replace a stock baseline, and does not change this report's original sweep conclusions or the simulator defaults.

Prefill has priority by default. `--prefill-decode-interval K` forces K decode rounds after each prefill forward whenever decoding work exists. Decode requests receive no token while a separate prefill forward runs, so its blocking time contributes to their request-average TPOT. Interval zero is the uncalibrated baseline consistent with the local GLM default discussed in F27. Useful calibration may instead find an effective interleaving pattern that differs from this abstraction.

Consecutive unchanged decode forwards are coalesced until the earliest arrival, completion or scheduling boundary. This is an event-simulation optimization: the unit tests compare it against literal one-forward-at-a-time execution for both scheduling policies and multiple decode intervals.

Timing definitions are:

```text
ready = dispatch + frontend_ms / 1000
TTFT = first_token_time - dispatch
TPOT = (last_token_time - first_token_time) / (output_tokens - 1)
```

Single-token TPOT is null. The last token is also client completion in this model; transport, terminal-event lag and SSE grouping are absent. `frontend_ms` is an additive per-request time before engine admission, included in TTFT. Fixed forward overhead is charged once per shared forward, not once per request.

## 4. Cache-hit and output-length inputs

The default `role_conservative` arm uses **frozen `uncached_expected`**. This is the D1 proxy explicitly allowed by T15, not an exact reproduction of F24's role-conservative token-path simulation. The default stock arm uses:

```text
frozen fast_intra:       ceil(uncached_expected * 7238 / 3144)
remaining frozen intra:  ceil(uncached_expected * 15744 / 14033)
chain_start/turn_start:  uncached_expected
all cases:              clamp to prompt length
```

The ratios use the F24 stock tails and F13's frozen reference tails; floating-point integer boundaries are handled before rounding. The factors are configurable via `--stock-fast-factor` / `--stock-slow-factor`.

**MODEL OUTPUT / proxy diagnostics on this cohort:** stock fast/overall uncached p95 = **7238 / 15744**; D1 = **3144 / 14033**. Actual F24 role_conservative model p95 was **3332 / 15082**, with **2 branch-conflict skips** (source: F24, not a result of this simulator). Therefore the default D1 proxy is optimistic relative to F24 itself. Inflating each fast request by a tail ratio does not recover F24's real per-request distribution, misses on turn_start, cross-request correlations or concurrency-dependent cache behavior. The two intra-stratum factors are discontinuous at the frozen 4096 boundary; they are conditional sensitivity proxies, not a universal physical miss function.

For real per-request model/cache work, supply JSONL:

```json
{"req_id":"biomaster:canon:example:llm:1","stock":7238,"role_conservative":3332}
```

Use the original `pack:view:logical_call_id` ID. The input must cover every replayed source request and contain integer counts within `[0, glm_tokens]`; missing or invalid rows fail instead of silently falling back. Export `r['_req_id']` plus each policy's uncached count from a token-path run of F24's algorithm, or convert measured stock `prompt_tokens - cached_tokens`. A mixed provenance file may use measured stock and a separately modeled D1 count, but its provenance must be recorded externally. The simulator does not import `sim_role_boundary.py` directly because that script executes tokenizer/body work at import time and currently emits aggregate statistics only.

Cache profiles are fixed input work quantities. Varying chunk size, scheduling or N does **not** recompute checkpoint placement or eviction. This assumption isolates timing sensitivity; it is especially incomplete when testing D1's artificial boundary splits. F25 shows why a single internal KDA snapshot is not equivalent to preserving role+end. `--d1-extra-forward-equivalents` can charge extra fixed overhead on cache-hit D1 requests, but it **does not** insert a real role boundary or reproduce D1's changed admission/decode behavior. Its default is zero. N=6/N=10 stock runs cannot identify D1 split overhead or retention correctness; those remain independent uncertainties until a D1 run exists.

Default output length is each request's **`max_output_i` budget**, yielding **215,582 modeled output tokens** on dev. This is an explicit upper-budget workload assumption, not a claim that GLM always generates to its cap. `--output-mode source` uses source-model completion counts clipped to the cap, falling back to the budget when absent; source-model output lengths are also not measurements of GLM. `--output-scale` perturbs these counts, clipped to the cap. Best calibration uses `--output-input` JSONL with original `req_id` and measured integer `output_tokens`, covering every request. Output length changes slot occupancy, batch sizes, arrivals and per-request TPOT dilution; it cannot be treated as an isolated throughput multiplier.

## 5. Exact dev gates and the separately reported TPOT extension

`score_records` calls the unmodified `s1_score.evaluate(..., lane='dev')`. `phase_gate` / `in_ttft_gate` and `q` remain the original functions from `s1_common.py:98–147`.

| Gate / classification | Exact public dev rule |
|---|---|
| chain_start | index 0, or context_reset; first-index priority over phase; p95 <= 30 s |
| turn_start | non-first turn_start; p95 <= 15 s |
| overall_intra | all remaining intra, including fast; p95 <= 5 s |
| fast_intra | intra with **frozen** uncached_expected <= 4096; p95 <= 3 s |
| quantile | sort; select index `min(n-1, int(.95*n))`; no interpolation |
| empty gate | not evaluable, fails that gate and gated_phases_have_samples |
| other dev gates | exact coverage, harness data/render and engine/infra error tests from the scorer |

Equality at a TTFT threshold passes. The fast label never changes because the modeled engine missed more tokens. No `sampling_weight` is applied to TTFT p95. The summary retains the exact ten `dev_gates` and `dev_all_pass` verdict. It separately reports request-unweighted `tpot_p95_s`, `tpot_mean_s`, `tpot_p95_le_0_10`, and `model_pass_dev_plus_tpot`.

F10 distinguishes the task's additional **0.10 s/token TPOT** criterion from the public ten-gate dev scorer. **The ladder tables below use `model_pass_dev_plus_tpot`.** They do not implement an unpublished formal scorer or reinterpret TTFT p95 through statistical confidence allowances. Simulated records are labeled `ttft_source="model"`, never represented as measured server timestamps or accepted formal-lane evidence. No measurement-contract or correctness PASS is claimed.

Every requested N is evaluated, even after failure. The sweep reports passing levels, first failing N, all gates failing there, first failure by gate, maximum passing tested N, contiguous passing prefix, nonmonotonicity and right-censoring. Finite-cohort scheduling can produce nonmonotone p95, so a binary-search assumption is not embedded in the simulator.

## 6. Formal composition WHAT-IF

F31 exposes the reference composition **808 chain_start / 9023 intra / 65 turn_start**, totaling **9896**. `--mix formal-mix` creates exactly that many synthetic replay instances; `--mix both` runs dev and the what-if under the same engine grid.

Within each category, the simulator cycles through shuffled dev templates until reaching the target count, with a fixed seed. Each of 808 synthetic chains gets a sampled chain_start template. Sampled intra/turn_start continuations are shuffled and distributed round-robin, creating longer template chains. Each replay instance receives a unique synthetic request ID; `source_req_id` preserves the original cache/output-profile lookup. Gaps are inherited from templates and the original gap function is run again on these synthetic chains.

This changes **traffic, closed-loop occupancy and contention**, not just the weights used to summarize already completed requests. Simply reweighting three disjoint gate populations would not change their conditional p95 or the original queue trace.

**WHAT-IF limitations:** these are not semantically coherent conversations. Cached work is inherited, not derived from fictitious adjacent prompts; no long-history KV/KDA residency is simulated. Conditional prompt-length, output-length and gap distributions still come from dev, including dev's fast-intra fraction rather than forcing the formal 85.2% figure. Formal order, correlations and longer-tail workloads are unknown. The composition experiment can isolate cold-load dilution but can overestimate real formal capacity when residency dominates. It must not be substituted for the official cohort or sent to a service as evaluation data.

## 7. Sensitivity results — MODEL OUTPUTS ONLY

Common assumed settings: aggregate prefill **20k/40k/80k tok/s**; base batch-one decode **30/50/80 ms**; batch slope **0.02**; fixed overhead **2 ms/forward**; no frontend delay; 8192-token chunks / 64-token pages; no length penalty; interval zero; engine running cap=N; D1 extra-forward equivalents zero; full output budgets. Ladder = **2, 6, 10, 14, 18, 22, 26, 30**. These inputs are illustrative parameters, not A100 calibration.

<!-- BEGIN MODEL TABLES -->
**MODEL OUTPUT table: maximum passing tested N (dev ten gates plus TPOT).** Each cell is **stock / D1**. `≥30` means all tested levels through 30 pass and the ceiling is unobserved; `none` means no tested level passes. This is neither a measured capacity nor a confidence bound.

| Prefill tok/s | Base decode ms | FCFS dev stock / D1 | SPF dev stock / D1 | FCFS formal-mix WHAT-IF stock / D1 | SPF formal-mix WHAT-IF stock / D1 |
|---:|---:|---:|---:|---:|---:|
| 20,000 | 30 | 6 / 6 | 6 / 6 | 22 / 26 | 22 / 26 |
| 20,000 | 50 | 2 / 2 | 2 / 2 | 14 / 14 | 10 / 14 |
| 20,000 | 80 | 2 / 2 | 2 / 2 | 2 / 2 | 2 / 2 |
| 40,000 | 30 | 14 / 14 | 14 / 14 | ≥30 / ≥30 | ≥30 / ≥30 |
| 40,000 | 50 | 10 / 10 | 10 / 10 | 22 / 26 | 22 / 22 |
| 40,000 | 80 | 2 / 2 | 2 / 2 | 6 / 6 | 6 / 6 |
| 80,000 | 30 | 26 / 26 | 26 / 26 | ≥30 / ≥30 | ≥30 / ≥30 |
| 80,000 | 50 | 18 / 18 | 18 / 18 | ≥30 / ≥30 | ≥30 / ≥30 |
| 80,000 | 80 | 2 / 2 | 2 / 2 | 6 / 6 | 6 / 6 |

**MODEL OUTPUT table: which gate binds first.** All tests here use the same full-budget workload. “No TTFT fail ≤30” means none on the tested ladder, not a continuous-N guarantee.

| Mix / scheduler / cache | Prefill / decode | First failing N | Gates failing at that N | First TTFT-only failing N and gate |
|---|---|---:|---|---|
| dev / fcfs / stock | 20,000 / 30 ms | 10 | fast_intra + TPOT | 10: fast_intra |
| dev / spf / stock | 20,000 / 30 ms | 10 | TPOT | No TTFT fail ≤30 |
| dev / fcfs / D1 | 20,000 / 30 ms | 10 | fast_intra + TPOT | 10: fast_intra |
| WHAT-IF formal-mix / fcfs / stock | 20,000 / 30 ms | 26 | TPOT | No TTFT fail ≤30 |
| WHAT-IF formal-mix / fcfs / D1 | 20,000 / 30 ms | 30 | TPOT | No TTFT fail ≤30 |
| dev / fcfs / stock | 40,000 / 50 ms | 14 | TPOT | No TTFT fail ≤30 |
| WHAT-IF formal-mix / fcfs / stock | 40,000 / 50 ms | 26 | TPOT | No TTFT fail ≤30 |
| WHAT-IF formal-mix / fcfs / D1 | 40,000 / 50 ms | 30 | TPOT | No TTFT fail ≤30 |

At 20k/30 ms with FCFS, the dev stock and D1 proxies first fail fast_intra and TPOT together at N=10. Replacing the load composition with the synthetic formal mix moves the first stock failure to TPOT at N=26 (D1: N=30). This isolates a possible cold-prefill-load effect; it does not establish formal behavior. At 40k/50 ms, dev first fails TPOT at N=14, while the formal-mix stock/D1 first failures are N=26/N=30. SPF is not uniformly better: the formal-mix 20k/50 ms stock ceiling falls from 14 to 10, and the 40k/50 ms D1 ceiling falls from 26 to 22. The ordering changes feedback and decode stalls.

**MODEL OUTPUT table: per-request aggregate diagnostics for the representative 40k/50 ms FCFS runs.** Times are simulated seconds except wall minutes. All four runs pass dev plus TPOT.

| Cache | N | fast TTFT p95 | overall TTFT p95 | turn TTFT p95 | chain TTFT p95 | TPOT p95 | Wall min |
|---|---:|---:|---:|---:|---:|---:|---:|
| stock proxy | 6 | 0.2819 | 1.0586 | 2.8839 | 3.1086 | 0.0811 | 47.96 |
| stock proxy | 10 | 0.8394 | 1.1085 | 4.4080 | 3.2232 | 0.0973 | 33.64 |
| D1 proxy | 6 | 0.1321 | 0.9244 | 1.1004 | 3.0735 | 0.0787 | 47.70 |
| D1 proxy | 10 | 0.7458 | 1.0800 | 1.1029 | 3.2163 | 0.0978 | 33.36 |

**MODEL OUTPUT table: output-length sensitivity on dev, FCFS; maximum passing tested N.** The source-length variant contains 167,606 modeled output tokens after budget clipping and 12 missing-count budget fallbacks. All other settings are unchanged.

| Prefill tok/s | Decode ms | Full-budget stock / D1 | Source-length proxy stock / D1 |
|---:|---:|---:|---:|
| 20,000 | 30 | 6 / 6 | 2 / 6 |
| 20,000 | 50 | 2 / 2 | 2 / 2 |
| 20,000 | 80 | 2 / 2 | none / none |
| 40,000 | 30 | 14 / 14 | 10 / 10 |
| 40,000 | 50 | 10 / 10 | 6 / 6 |
| 40,000 | 80 | 2 / 2 | 2 / 2 |
| 80,000 | 30 | 26 / 26 | 22 / 22 |
| 80,000 | 50 | 18 / 18 | 14 / 14 |
| 80,000 | 80 | 2 / 2 | 2 / 2 |

Shorter modeled outputs can reduce the passing N: workers return to prefill sooner and each prefill stall is diluted over fewer output tokens in request-average TPOT. This is why measured output lengths are required. All 90 completed grid groups were monotone in their pass/fail sequence across this tested ladder; the simulator still checks and reports nonmonotonic cases rather than assuming they cannot occur.
<!-- END MODEL TABLES -->

The default 40k/50 ms stock model at N=6 takes about **48 model minutes**, versus the cohort's declared **35-minute N=6 reference**. The latter explicitly is not a runtime commitment, and neither value is a real stock measurement from our deployment. Do not force a fit to this single manifest number. D1 can change the replay alignment enough that a particular percentile worsens despite less prefill work; the per-request and per-gate outputs should be examined rather than assuming monotonic D1 gains.

## 8. Calibration plan for the first authorized stock 8-GPU slot

Run the unmodified measured cohort at **N=6 then N=10 on the same stock service configuration**, with successful real cache flush between measurement stages and appropriate JIT warmup before each. Reuse the existing ladder/flush tooling when that real run is authorized; this T15 task does not start it. Save exact image digest, engine commit/flags, GPU topology, allocator limits, request order, gap/workload hash, warmup/flush evidence and completed raw records. One 8-GPU service can collect both levels sequentially within the per-member quota.

| Measurement to retain | Parameter / uncertainty it sets | N=6 vs N=10 role |
|---|---|---|
| Each request's prompt_tokens, cached_tokens, output_tokens, source req_id, phase/index | Replace stock F24 inflation and output-budget assumptions with supplied row profiles; retain frozen gate labels | Fit counts at N=6; compare N=10 by phase and prompt length for concurrency-dependent misses. Disagreement means fixed cache profiles are insufficient |
| Request receive, API admission/dispatch finish, first model-forward entry, prefill-finished/first-token timestamps, first/last SSE and completion | Separate frontend, admission queue, prefill execution and output delivery; check TTFT/TPOT basis | N=6 offers more lightly queued requests; N=10 tests whether predicted queue growth and client feedback are correct |
| For every prefill forward: start/end or GPU-event duration, request IDs, per-request actual extend length, total prompt/context length, continuation flag, decode batch waiting during the forward | Fit aggregate `prefill_tps`, `prefill_length_alpha`, and per-forward fixed overhead; establish actual chunk lengths | At N=6 fit lightly queued execution across short/long shapes; use N=10 to validate shared-batch cost and blocking duration |
| For every decode forward: batch size B, duration, graph/eager mode, active request IDs, speculation acceptance if enabled | Fit `decode_curve(B)` or base `decode_ms` plus slope, excluding the separately fitted fixed overhead | N=6 anchors low B; N=10 extends observed B. Record the covered range; behavior above it remains an extrapolation |
| Forward order and admission decisions, partial-prefill count, waiting/running counts | Verify FCFS continuation priority, effective decode interval and `max_running`; test whether separate-forward scheduling is adequate | Fit/control settings at N=6; N=10 reveals queue bursts and prefill/decode interference. Counts alone cannot identify every scheduling cause |
| KV/KDA pool occupancy, evictions, prefix state depth, allocator pressure, host reloads, rank utilization | Determine whether the no-eviction, one-engine abstraction is usable at all | Compare occupancy/miss changes across N=6/N=10. A real memory limit must constrain projections to higher N |

Fit execution times from **forward-level measurements**, not `uncached / TTFT`: TTFT includes frontend, queue and other requests' work. For prefill, regress duration against actual token work and prompt-length features plus a nonnegative intercept. Estimate fixed overhead from many small and large batches or measured CPU/GPU boundaries; per-request timestamps alone cannot separate overhead from kernel time. For decode, use a table over actual observed B and avoid double-counting the same intercept in both overhead and the curve. If graph/eager regimes differ, add separate curves/model branches before claiming accurate forecasts.

Use **N=6 for the initial fit and N=10 as a held-out replay check**. Compare per-request TTFT/TPOT residuals, p50/p95 by all four TTFT gates, completion/arrival timelines, batch-size distribution, queue depth, prefill/decode time fractions and total wall time. If the N=10 model fails these diagnostics, revise the missing mechanism, then collect an independent validation level before interpreting an extrapolated pass. Fitting both endpoints does not create an independent validation set.

Carry plausible low/base/high parameter fits through every N and report gates that disagree between fits. Requests near a threshold need the real ladder check, not a binary decision from the midpoint model. Bootstrap/resample **whole source chains** for calibration uncertainty where useful; do not treat correlated intra requests as independent observations. Repeat what-if seeds when composition/order uncertainty matters. No uncertainty interval has been fitted in this report.

The stock run can calibrate timing and stock cache counts. It **cannot** establish D1 snapshot correctness, branch-skip frequency under the real scheduler, dual-state retention, artificial-split overhead, a new SPF implementation's timing, or formal long-chain residency. Those remain explicit sensitivity dimensions or require a later controlled run.

## 9. Reproduction and review

From the repository root:

```bash
python3 -B -m unittest discover -s scripts -p test_sim_closed_loop.py -v

# Full model grid; retain summaries for all 576 runs without a large raw archive.
python3 -B scripts/sim_closed_loop.py \
  --prefill-rates 20000,40000,80000 --decode-ms 30,50,80 \
  --mix both --schedulers fcfs,spf --summary-only \
  --out-dir evidence/T15_simulator/sensitivity

# Four complete dev per-request model traces at the calibration levels.
python3 -B scripts/sim_closed_loop.py --levels 6,10 \
  --out-dir evidence/T15_simulator/reference

# Output-length sensitivity, using source lengths (also not GLM measurements).
python3 -B scripts/sim_closed_loop.py \
  --prefill-rates 20000,40000,80000 --decode-ms 30,50,80 \
  --output-mode source --summary-only --out-dir evidence/T15_simulator/source_output

# Example interface for later offline calibration inputs; paths are placeholders.
python3 -B scripts/sim_closed_loop.py --levels 6,10,14,18,22 \
  --cache-input /path/to/cache_counts.jsonl --output-input /path/to/output_counts.jsonl \
  --decode-curve 1:30,4:34,8:40,16:55,32:85 --trace \
  --out-dir /path/to/offline_calibrated_model
```

The last command's curve is an **illustrative model input**, not a measured fit. Every run writes `sweep.json`; default mode additionally writes one `.requests.jsonl` per level and policy. `--trace` writes forward-event JSONL (decode bursts include their round count). Simulated times are seconds relative to replay start, not epoch timestamps. CLI output is explicitly prefixed MODEL OUTPUT; failing modeled gates still produce a successful report/exit zero, while invalid inputs exit nonzero.

The tests cover exact gap rounding, worker release/chain claims, first-token timing, output-token counts, frontend delay, decode batch transitions, prefill stalls in TPOT, fixed overhead sharing, length dependence, decode interpolation, FCFS/SPF contention, decode intervals, admission caps, token-budget conservation, event coalescing, frozen classification, supplied cache validation, output profiles, scorer parity, equality/missing-gate rules, nonmonotone ladders, exact formal composition/labels, real cohort order and read-only CLI behavior.

**Handoff:** the offline tool and evidence are complete. Before using a modeled passing N to allocate the next scarce slot, replace stock cache/output counts and fit/validate the engine timing at N=6/N=10. Keep the exact dev gate verdict, TPOT extension and formal-composition what-if separate.


## T20 / D2 scheduling semantics addendum

2026-09-22, Codex W7 / T20. The existing `spf` option already reserves complete short waiters. It admits waiters before the continuation and considers free request slots during reservation. New `spf-upstream` reproduces #40024 reservation-first/continuation-first admission under the same ideal cache/memory assumptions; a request-slot counterexample and 500-round production differential are documented in [patches/002](../../patches/002-spf-scheduling.md). New `hrrn`/`lpm` options model stock v0.5.20 ordering without reservation. Legacy defaults and `spf` remain available; the original results above are historical model outputs.
