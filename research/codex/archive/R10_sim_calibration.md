# R10 — Public-envelope calibration of the offline simulator

Author: Codex W5 / T18. **All simulated engine parameters, timings, residuals, gate outcomes and ceilings below are MODEL OUTPUT — formal-mix WHAT-IF, not measured deployment performance.** Public data and test evidence are explicitly labeled as such. Other teams' configurations remain unknown; no service, GPU, Trisol, image build or submission was used by W5.

## Decision input

**MODEL OUTPUT / INFERRED: this calibration does not support a reliable D1 ceiling increase.** The main accepted region makes fast/overall TTFT bind before TPOT, but D1 either preserves or lowers the strict modeled ceiling. Nearby fits that pass through N=18 also do not show an 18→22 increase. This is not evidence to abandon D1: cache-state lifetime, prefill/decode scheduling, the formal observation window and statistical slack remain uncalibrated. Keep the same-engine stock/D1 measurement as the deciding evidence; do not choose a deployment or reject the patch on these synthetic ceilings.

## Public prior and fitting contract

VERIFIED / PUBLIC PRIOR INPUT: the parser reads one canonical `scorecard.scorewheel_stress` row per attempt, keeps `passed is true`, groups by `n`, and falls back to nested `scorewheel_raw_result.stress` only if the canonical row is absent. It never counts both representations. Repeated authors/submissions remain present to reproduce F35. The task describes `n` as the recorded maximum passing level, so these groups are survivor-selected samples from different deployments, not successive measurements of a common stock engine. Every table column headed “prior” below is an observed public input, not a simulation.

**PUBLIC PRIOR INPUT table — seconds; exact medians from `data/all_att.json`, displayed rounded.**

| N | Passing attempts / authors | Fast prior | Overall prior | Turn prior | Chain prior | TPOT mean prior | TPOT p95 prior |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 44 / 30 | 2.9094 | 3.8363 | 5.6753 | 10.3702 | 0.014877 | 0.023421 |
| 10 | 26 / 23 | 3.1484 | 3.9659 | 5.8694 | 13.6986 | 0.036777 | 0.051119 |
| 14 | 33 / 23 | 2.6028 | 4.0792 | 8.0657 | 37.7538 | 0.027461 | 0.054100 |
| 18 | 7 / 5 | 2.5507 | 4.6622 | 9.7821 | 46.2874 | 0.026529 | 0.055000 |

MODEL OUTPUT / fit definition: use all 24 cells (six metrics at four anchors), equal weight in log space: `log_RMSE = sqrt(mean(log(model/prior)^2))`. A coarse accepted cell is between half and twice its median. This factor-2 band is deliberately loose for heterogeneous configurations and template workload mismatch; it is a diagnostic tolerance, not a confidence interval. A separate “core” diagnostic retains fast/overall and both TPOT statistics, but core-only candidates are never silently promoted to a full fit. No recorded candidate fits all cells within a tighter factor-1.5 band.

MODEL OUTPUT / binding condition: all sampled lower anchors must pass, first strict failure must be N=18 or N=22 with only fast/overall TTFT failures, and TPOT must pass through that first failure. Full ladders then check N=2,6,10,14,18,22,26,30. The verdict remains the unchanged ten dev gates plus request-average TPOT p95≤0.10; **it is not the formal statistical-slack verdict**. Chain co-binding examples are reported separately. Passing through N=18 means the first failure is N=22, not N=18.

MODEL OUTPUT / shared workload assumptions: formal-mix uses 808 chain starts, 9023 intra and 65 turn starts, seeded from the public dev templates; full output budgets; fixed cache proxies; page size 64; max-running=N; no frontend delay or eviction. `stock` means the simulator's F24 tail-inflated cache-work proxy. `role_conservative` is its frozen-uncached D1 proxy, not an exact F24 per-request export. All arms keep identical output profiles. The assumed public scheduler is either FCFS or SPF; fitting stock cache with SPF does not establish that our unmodified deployment has these parameters.

## Search and accepted region

MODEL OUTPUT / search scope: the broad search varies aggregate prefill 6k–100k tok/s, decode base 1–30 ms, batch slope 0–0.15, overhead 0–4 ms, chunks 1024–32768, length exponent 0–1.5 and decode interval 0/1/4. Targeted and SPF searches expand interleaving through 128 decode rounds and exponent through 2; exact sampled values are saved in candidate JSON, not inferred from these marginal ranges. The final local grid fixes a 16384-token chunk, 12 ms decode, 0.04 slope and 2 ms overhead while varying prefill, exponent and interleaving. The default simulator settings were not replaced.

**MODEL OUTPUT table — recorded candidate evaluations (including deliberate repeated points).**

| Search artifact | Candidates | All cells in factor-2 band | Full band + requested first-binding condition |
|---|---:|---:|---:|
| `broad/envelope_fit.json` | 128 | 0 | 0 |
| `targeted/envelope_fit.json` | 320 | 0 | 0 |
| `local/envelope_fit.json` | 128 | 0 | 0 |
| `spf/envelope_fit.json` | 192 | 0 | 0 |
| `spf_trials.json` | 8 | 2 | 0 |
| `refined/envelope_fit.json` | 60 | 21 | 5 |

MODEL OUTPUT: the main accepted region consists of these five sampled tuples, not every point in their bounding rectangle. Candidate C<n> refers to the zero-based ID in `refined/envelope_fit.json`. All use SPF, length exponent 1.2, decode base 12 ms, slope 0.04, overhead 2 ms, chunk 16384 and speedup 1. They therefore require an effective scheduling model quite different from the default interval-zero FCFS engine. The first-binding requirement is an imposed prior condition, not an independent validation result.

**MODEL OUTPUT table — main region. “K” is forced decode rounds after a prefill forward.**

| Candidate | Prefill tok/s | K | Full log-RMSE | Worst multiplicative error | Core log-RMSE |
|---|---:|---:|---:|---:|---:|
| C6 | 38000 | 28 | 0.3564 | 1.8608 | 0.2600 |
| C17 | 40000 | 24 | 0.3952 | 1.9620 | 0.2700 |
| C18 | 40000 | 28 | 0.3617 | 1.8619 | 0.2509 |
| C19 | 40000 | 32 | 0.3067 | 1.7433 | 0.2381 |
| C31 | 42000 | 32 | 0.3154 | 1.8675 | 0.2267 |

MODEL OUTPUT: additional one-parameter neighbors of C19 also satisfy the full band and requested binding condition: prefill 45k, decode 9 ms, slope zero, or overhead zero (change one at a time). These broaden the conditional region; they do not identify a unique engine. Supplemental full fits C42/C43 use prefill 46k and K=28/32, with fast-intra and chain-start co-binding at N=22.

## Residuals

**MODEL OUTPUT table — representative C19 at the fit anchors. Each cell is modeled seconds (signed relative residual versus PUBLIC PRIOR INPUT). Full signed second residuals for every candidate are in `fit.cells` in the JSON artifacts.**

| N | Fast TTFT p95 | Overall TTFT p95 | Turn TTFT p95 | Chain TTFT p95 | TPOT mean | TPOT p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 1.8881 (-35.1%) | 3.1522 (-17.8%) | 3.2555 (-42.6%) | 16.4654 (+58.8%) | 0.0194 (+30.3%) | 0.0336 (+43.3%) |
| 10 | 2.5144 (-20.1%) | 3.8249 (-3.6%) | 4.9199 (-16.2%) | 20.0598 (+46.4%) | 0.0242 (-34.2%) | 0.0505 (-1.3%) |
| 14 | 2.9865 (+14.7%) | 4.3379 (+6.3%) | 11.3988 (+41.3%) | 25.1719 (-33.3%) | 0.0296 (+7.6%) | 0.0612 (+13.2%) |
| 18 | 3.3889 (+32.9%) | 5.0445 (+8.2%) | 6.7330 (-31.2%) | 28.7123 (-38.0%) | 0.0351 (+32.4%) | 0.0680 (+23.7%) |

MODEL OUTPUT: C19's largest absolute timing residual is chain-start at N=18: −17.5751 s. Its fast-intra residual changes from −1.0213 s at N=6 to +0.8382 s at N=18. The fitted region spans only a coarse envelope; it does not reproduce the nearly flat cross-team fast-intra medians or the abrupt public cold-start growth. For C19 at N=18, modeled TPOT p95 is 0.0680 while fast/overall p95 are 3.3889/5.0445 s: this fixes the earlier TPOT-first interpretation under the stated model assumptions.

## Stock/D1 and FCFS/SPF decisions

**MODEL OUTPUT table — first failing N, then every failed gate at that N. F=fast-intra, O=overall-intra, T=turn-start, C=chain-start. Each arm is evaluated through the full ladder.**

| Candidate | FCFS stock | FCFS D1 | SPF stock | SPF D1 |
|---|---|---|---|---|
| C6 | 10 F+O+T | 10 F+O | 18 F+O | 14 F |
| C17 | 10 F+O | 10 F+O | 18 F | 18 F |
| C18 | 10 F+O | 10 F+O | 18 F | 18 F |
| C19 | 10 F+O | 10 F+O | 18 F+O | 14 F |
| C31 | 10 F+O | 10 F+O | 18 F | 18 F |
| prefill_high | 10 F+O | 10 F+O | 18 F | 18 F |
| decode_low | 10 F+O | 10 F+O | 18 F | 18 F |
| slope_zero | 10 F+O | 10 F+O | 18 F | 18 F |
| overhead_zero | 10 F+O | 10 F+O | 18 F | 14 F |
| co_binding_42 | 10 F+O | 10 F+O | 22 F+C | 22 F |
| co_binding_43 | 10 F+O | 10 F+O | 22 F+C | 18 F |

MODEL OUTPUT: primary FCFS stock/D1 ceilings are both 6. SPF stock passes through 14; SPF D1 passes through 10 for C6/C19 and through 14 for C17/C18/C31. There are no pass-after-fail reversals in these main ladders. C42 gives the direct requested comparison at a modeled stock ceiling of 18: D1 also passes through 18 and first fails fast-intra at 22. C43 moves from stock18 to D1 14. None of these fitted examples supports 18→22.

MODEL OUTPUT / scheduling diagnostic, C19 at N=14: D1 reduces modeled prefill work from 78,059,008 to 61,016,356 tokens and prefill engine time from 4818.5867 to 4000.4539 s. Nevertheless, fast-intra p95 changes from 2.9865 to 3.0412 s; aggregate queue-time p95 falls from 1.9204 to 1.7248 s. Prefill forward counts are 8465/8420 and decode forward counts 385455/397208. Thus the small TTFT regression is not a simple extra-prefill-work cost: closed-loop arrivals, batches, SPF order and the forced-decode schedule change. These aggregates do not identify one causal request path, and the model cannot represent D1's actual boundary split/retention mechanics. A threshold crossing this small is not evidence of a real regression.

**MODEL OUTPUT table — SPF ceiling sensitivity across the five main candidates; paired D1 improvement counts compare the same candidate and workload.**

| Workload variant | Stock ceilings | D1 ceilings | D1 ceiling increases |
|---|---|---|---:|
| nominal | 14,14,14,14,14 | 10,14,14,10,14 | 0/5 |
| template_seed_20260923 | 10,14,14,10,14 | 10,14,14,10,10 | 0/5 |
| template_seed_20260924 | 10,14,14,10,14 | 10,14,10,10,14 | 0/5 |
| source_output_lengths | 10,14,10,10,14 | 10,14,10,10,10 | 0/5 |

MODEL OUTPUT: inflating the frozen D1 intra counts by the F24 tail ratios 3332/3144 and 15082/14033 gives ceilings 10,14,10,10,14 for C6/C17/C18/C19/C31. This is another distribution proxy, not exact F24 placement. Charging two additional fixed overhead equivalents gives 10,14,14,10,14. These checks show sensitivity; neither substitutes for the real artificial split. The source-output test changes only the offline assumed lengths, never a live output budget.

## Parameter identifiability and structural limits

**MODEL OUTPUT table — one-at-a-time perturbations about C19; all remain only conditional model tests.**

| Change | In-band cells / 24 | Full log-RMSE | Requested TTFT-first condition |
|---|---:|---:|---|
| `prefill_low` | 24 | 0.3062 | no |
| `prefill_high` | 24 | 0.3273 | yes |
| `decode_low` | 24 | 0.3357 | yes |
| `decode_high` | 24 | 0.3263 | no |
| `slope_zero` | 24 | 0.2981 | yes |
| `slope_double` | 24 | 0.3148 | no |
| `overhead_zero` | 24 | 0.3159 | yes |
| `overhead_double` | 24 | 0.3298 | no |
| `chunk_half` | 19 | 0.5160 | no |
| `chunk_double` | 21 | 0.4139 | no |
| `alpha_low` | 24 | 0.3488 | no |
| `alpha_high` | 24 | 0.3537 | no |
| `interval_half` | 17 | 0.4778 | no |
| `interval_high` | 24 | 0.2805 | no |
| `decode_speedup_equivalent` | 24 | 0.3067 | yes |

- **Prefill rate: weakly identifiable.** It trades against prompt-length penalty, actual cache misses and chunk service time. MODEL OUTPUT: conditional neighbors span 38k–45k tok/s with the requested first-binding pattern; a 35k point still fits every cell but fails earlier. These are sampled rates, not a hardware throughput estimate.
- **Decode base, slope and overhead: weakly identifiable individually.** Only their effective time over the realized batch distribution is constrained. MODEL OUTPUT: base 9–12 ms, slope 0–0.04 and overhead 0–2 ms each have accepted one-at-a-time examples; 15 ms, slope 0.08 or overhead 4 ms can still fit the medians but bind too early. They are not a Cartesian feasible box. The public rows provide no per-forward times or batch histogram.
- **MTP-like factor: exactly non-identifiable against decode base in this surrogate.** The equation is `(base_ms * (1 + slope*(B-1)) / decode_speedup + overhead_ms)/1000`. MODEL OUTPUT: base 12/speedup1 and base24/speedup2 produce identical timings; the test also verifies this on a complete replay. No speculative acceptance, rejected token verification or multi-token timestamp grouping is simulated. A speedup factor was not needed beyond an effective small decode base.
- **Chunk/interleaving: important structurally, poorly identified physically.** MODEL OUTPUT: the accepted main region uses chunk16384 and K24–32; chunk8192/32768 perturbations miss the full band. Halving K to16 makes TPOT fail first at22; increasing K to48 improves aggregate fit yet causes TTFT failure at14. Consequently the observed medians do not uniquely identify the policy or the binding level. K is an effective scheduling assumption, not an inferred real launch flag.
- **Length exponent: correlated nuisance parameter.** MODEL OUTPUT: the main region uses 1.2; exponent1 and1.4 both fit the loose band, but respectively create a chain-start-first or too-early fast-intra failure. It mimics length-dependent work without modeling the hybrid attention kernels. Frontend delay was fixed to zero and is not identifiable here; it was not used to manufacture a low-N TTFT floor.

VERIFIED / source limits: `task.md` describes a one-sided exceedance-rate test with statistical slack. A public chain-start p95 above its target can therefore coexist with PASS; p95 alone cannot recover exceedance counts, sample count, confidence bound or the exact formal verdict. F35's cold tail cannot be “fixed” by raising the simulator threshold or forcing a PASS flag. The same issue applies to public fast-intra values above their strict target. The simulator models an entire synthetic cohort, not the unpublished measurement-window selection underlying each public score.

INFERRED / structural gaps: the synthetic mix preserves dev conditional prompt/output/gap distributions, not real long-chain correlations, formal order or real fast-intra share. A deterministic serialized prefill/decode engine lacks overlap, mixed forwards, rank imbalance, kernel-dependent interference, admission/allocator behavior and transport/SSE batching. Fixed cache counts cannot model D1 boundary placement, branch/end donation, FULL-KV residency, KDA capacity, eviction, OOM or concurrent cache conflicts. F36/F37, posted during this task, add real stand-in evidence that resident FULL-KV lifetime can make the old stock predictor optimistic; that functional observation is not a GLM timing calibration. The new fit does not supersede that evidence or silently modify the F24 cache distributions.

## Reproduction and verification

The entry point is `scripts/sim_closed_loop.py --envelope-fit`; `scripts/sim_envelope_fit.py` implements prior extraction, deterministic search, residuals and held-out policy comparisons. `--fit-scheduler` makes the public scheduler assumption explicit. `--fit-candidates` accepts saved Engine override arrays; `--fit-samples 0` disables additional random draws. Each report stores inputs, hashes, all residuals, counts and full selected ladders. Failed fits remain in the output and are not described as accepted. The previous simulator sweep mode and its defaults remain available.

```bash
python3 -B scripts/sim_closed_loop.py \
  --mix formal-mix --envelope-fit data/all_att.json \
  --fit-scheduler spf --fit-samples 0 \
  --fit-candidates evidence/T18_calibration/refined_candidates.json \
  --fit-compare-top 3 --out-dir evidence/T18_calibration/reproduce
python3 -B scripts/sim_calibration_checks.py \
  --fit-report evidence/T18_calibration/reproduce/envelope_fit.json \
  --out-dir evidence/T18_calibration/reproduce_checks
python3 -B scripts/sim_calibration_checks.py \
  --fit-report evidence/T18_calibration/reproduce/envelope_fit.json \
  --out-dir evidence/T18_calibration/reproduce_checks --supplemental-only
python3 -B -m unittest discover -s scripts -p 'test_sim*.py' -v
python3 scripts/check_records.py
```

VERIFIED / CPU test evidence: the original 30 tests and 11 new tests pass (41 total), including exact F35 extraction, duplicate-source handling, missing/invalid data, log residual symmetry and tolerance boundaries, first-gate classification, deterministic sampling, decode-speedup equivalence, actual formal-mix CLI execution and unchanged harness hashes. Evidence: `evidence/T18_calibration/tests.log`; independent region input hashes also match. Final record-check evidence is in `evidence/T18_calibration/records.log`. Numerical artifacts are under `evidence/T18_calibration/`; R9 remains the source for the original model assumptions.


## T20 / D2 fixed-candidate cross-check

2026-09-22, Codex W7 / T20. W7 re-evaluated C6/C17/C18/C19/C31 and C42/C43 with FCFS, legacy SPF, #40024 `spf-upstream`, stock HRRN and LPM, both cache proxies, N=2…30 (560 MODEL OUTPUT runs). All 14 legacy/upstream SPF ceiling pairs agree. HRRN stays at 6 in these models; LPM stays at 6 except C6/stock=2. The old SPF was already budget-sharing, but differs in request-slot reservation and within-forward ordering. Full assumptions, limits, tables and test receipts: [patches/002](../../patches/002-spf-scheduling.md), `evidence/T20_d2/calibration/comparison.json`. No timing refit, real serving evidence, or change to the original conclusions above.
