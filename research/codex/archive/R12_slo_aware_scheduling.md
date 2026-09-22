# R12 — SLO-aware scheduling: statistical allowance, models and draft 003

Codex W12 / T25, 2026-09-22. CPU only. **VERIFIED** means source/data/mathematical/CPU evidence; every synthetic timing and ladder is **MODEL OUTPUT**, never measured capacity. References: task.md, F31/F40, decisions #20, R10 and W7's completed 002.

## Finding and recommendation

**MODEL OUTPUT:** the requested EDF/least-slack policies do not beat SPF on the fixed R10 candidate sets. The formal-mix main candidates all first fail strict SPF at N=18; EDF first fails at N=10 except C17 at N=14. EDF+D1 first fails at N=14 except C6 at N=10. Thus D1 sometimes improves this EDF model relative to EDF alone, but the combination still falls short of SPF. A chain-start weight of two does not produce a stable improvement. Keep 003 opt-in and measure it against SPF before selecting it for deployment; this evidence does not support a claim of N=22.

**VERIFIED:** 61-second chain-start p95 can coexist with the specified statistical PASS rule. **INFERRED:** the public score profile is consistent with prioritizing intra, but it does not identify EDF, prove intentional cold-request deprioritization, establish causality or demonstrate that combining D1 will improve it. This refines F40's interpretation without disputing its observed score. Other scheduling, cache, service-time and workload-window differences remain possible.

## Exact estimated allowance

`llm-challenge-arena-v1/task.md:558–569` specifies target times 3/5/15/30 seconds and failure only when a **one-sided 95% lower confidence bound on the exceedance rate exceeds 5%**. It does not name the interval construction. Existing `scripts/score_formal.py` already implements exact one-sided **Clopper–Pearson**, explicitly labeled **estimated**; it was retained. This task adds validation of negative sample sizes and bounded memoization of repeated binomial calculations, without changing valid scores.

For `k = count(TTFT > limit)` out of `n`:

```
L(0,n) = 0
L(k,n) = Beta_inverse(0.05; k, n-k+1), k > 0
estimated FAIL iff L(k,n) > 0.05
allowed_over(n) = max {k: Pr[Binomial(n,0.05) >= k] >= 0.05}
```

An observation equal to 30 seconds is not an exceedance. Empty buckets remain unevaluable/failing; weights do not enter this binomial calculation. A p95 point comparison and an exceedance fraction can differ at a discrete quantile boundary; the statistical check uses integer exceedance counts directly. The public dev scorer's quantile convention is `sorted_values[floor(.95*n)]`.

**VERIFIED arithmetic / estimated formal rule**, independently checked by 60-digit Decimal CDF summation (SLO-01):

| Sample count n | Maximum accepted k | Accepted rate | L(k,n) | L(k+1,n) |
|---:|---:|---:|---:|---:|
| 20 | 3 | 15.000% | 0.042169 | 0.071354 |
| 65 | 6 | 9.231% | 0.040967 | 0.051646 |
| 100 | 9 | 9.000% | 0.047757 | 0.055263 |
| 314 | 22 | 7.006% | 0.047905 | 0.050576 |
| 388 | 27 | 6.959% | 0.049569 | 0.051772 |
| 808 | 51 | 6.312% | 0.049634 | 0.050740 |
| 9023 | 485 | 5.375% | 0.049898 | 0.050005 |

F31 gives a reference composition of **808 chain_start / 9023 intra / 65 turn_start = 9896 requests**, not the observed sample counts of attempt 45417's measurement window. At n=808, up to **51** requests may exceed 30s; 52 fail this estimated method. This is 6.312%, versus the nominal 5%; it is **not** a fixed relaxation of 30s to 61s. At n=65 the allowance is 6 (9.231%); at n=9023 it is 485 (5.375%). Use the actual bucket n from each measured run.

A concrete compatible distribution is **758 requests at 20s + 50 at 61s**. Then p95=61s, k/n=6.188%, L=0.0485294≤0.05, so this gate passes the estimate. At n=808, p95 can exceed 30s once k≥41; counts 41…51 can all pass. The exceedance test imposes **no finite upper bound on those tail durations**: replacing 61 by a larger finite value leaves k and its lower bound unchanged. Therefore p95 alone cannot recover PASS/FAIL or the exact public run's exceedance counts. All other gates, including TPOT, errors and coverage, still apply; this example proves only compatibility of the chain-start gate. It is not a proposal to delay requests until a quota is used.

The existing scorer calculates the inverse through stable log-space binomial tails and bisection. Official hidden implementation equivalence and independent Bernoulli sampling are not established; correlated replay requests and unknown observation-window selection remain limitations. All added estimated tables retain that label.

## Models and observable scheduling information

`sim_closed_loop.py` adds `edf`, `least-slack`, `edf-chain-weighted`, and `--chain-start-weight` (default 2 for the weighted policy). Scheduling uses a request's **observed session ordinal**, dispatch time and actual modeled cache work. It never reads `phase`, `uncached_expected`, future requests or future output lengths. Model cache work itself remains R10's documented frozen-field/F24 proxy: it represents an observed cache match in the simulation, not an exact live cache predictor.

First observed session request gets a 30s target; later requests with uncached≤4096 get 3s, others 5s. First-token deadline is arrival+target. Least-slack subtracts estimated remaining prefill service; weighted EDF extends only the chain-start scheduling deadline to arrival+2×30s. The scorer's target stays 30s. There is no trustworthy turn-start classification from these inputs, so later turn requests receive conservative 3/5s targets. Reconstructed context under an existing session is another unavoidable proxy mismatch. The draft freezes the bucket before partial progress, preserves original arrival through requeue, and guarantees aligned continuation progress while reserving capacity only for more urgent complete waiters.

**MODEL INPUT diagnostic:** dev has 311 observed session heads but 314 scored chain starts; three reconstructions are therefore classified as continuations. Of 328 frozen-fast dev requests, the stock cache proxy assigns 110 to the slower 5s class. Formal-mix stock similarly misclassifies 2558/7627 frozen-fast requests; the D1 cache proxy assigns all 7627 to fast by construction. All 20 dev / 65 formal-mix turn-starts get 3/5s rather than 15s. These confusion counts are saved in `workloads[].proxy_confusion`, not concealed by supplying frozen gate labels to scheduling.

**MODEL OUTPUT scope:** reuse C6/C17/C18/C19/C31 and supplemental C42/C43 from the unchanged R10 refined fit, **without refitting** to F40. Main tuples `(prefill tok/s, forced decode rounds)` are `(38000,28), (40000,24), (40000,28), (40000,32), (42000,32)`; C42/C43 use `(46000,28/32)`. All use chunk16384/page64, exponent1.2, decode12ms, slope.04, overhead2ms; full engine dictionaries are saved. Same seed20260922, full output budgets, cache proxies, max-running=N, gaps, no frontend delay/eviction. Both dev and formal-mix are evaluated at every N=2,6,10,14,18,22,26,30: **7 candidates × 2 mixes × 8 arms × 8 levels = 896 simulations**.

SPF here means W7's `spf-upstream`, including reservation-before-admission and continuation-first execution. EDF/least-slack replace both the queue key and the qualification for the same reservation machinery, not just queue sorting. The active continuation keeps at least a page; no extra prefill capacity or hidden output reduction is added. A nonfitting head stops reservation and can block later waiters. Least-slack can favor a long cold request because subtracting a large service estimate makes its slack small; EDF aging likewise allows overdue cold requests to regain priority. These are policy mechanics, not proof of a specific causal path in every modeled regression.

Both verdicts are reported: **strict** = unchanged dev gates + TPOT p95≤0.10; **estimated** replaces only TTFT checks with the chosen Clopper–Pearson rule. Real frozen labels still determine scoring buckets. The synthetic formal-mix does not reproduce hidden long-chain correlations, cache lifetimes, kernel overlap, memory pressure or the formal window. It inherits template sizes/gaps and an 84.5% fast-intra share, not a reconstructed hidden dataset. Its confidence bounds are arithmetic on generated data, not confidence in hardware capacity.

## Complete first-failure tables

**All cells below are MODEL OUTPUT**: `first failing N` followed by **every failed gate at that N**. F=fast-intra, O=overall-intra, T=turn-start, C=chain-start, P=TPOT. D1 denotes the `role_conservative` cache proxy. `w2` changes only scheduling priority. Full per-N timing distributions, exceedance counts/bounds, contiguous ceilings and pass-after-fail flags are in `evidence/T25_slo/calibration/comparison.json`. No failure is silently replaced with the highest isolated passing rung.

### MODEL OUTPUT — dev, strict

| Candidate | FCFS | SPF | SPF+D1 | EDF | EDF+D1 | Least-slack | EDF w2 | EDF w2+D1 |
|---|---|---|---|---|---|---|---|---|
| C6 | 6 F+O+T+C | 6 F+C | 6 F+C | 6 F+O+C | 6 F+O+C | 6 F+O+C | 6 F+O+C | 6 F+O+C |
| C17 | 6 F+O+T+C | 6 C | 6 C | 6 F+O+C | 6 F+C | 6 F+O+C | 6 F+C | 6 F+C |
| C18 | 6 F+O+T+C | 6 F+C | 6 F+O+C | 6 F+O+C | 6 F+O+C | 6 F+O+C | 6 F+C | 6 F+O+C |
| C19 | 6 F+O+C | 6 F+C | 6 F+O+C | 6 F+O+C | 6 F+O+C | 6 F+O+C | 6 F+O+C | 6 F+O+C |
| C31 | 6 F+O+T+C | 6 F+C | 6 F+O+C | 6 F+O+C | 6 F+O+C | 6 F+O+C | 6 F+O+C | 6 F+O+C |
| C42 | 6 F+O+T | 6 F+C | 6 F+C | 6 F+O+C | 6 F+O+C | 6 F+O+C | 6 F+O+C | 6 F+C |
| C43 | 6 F+O+T | 6 C | 6 F+C | 6 F+O+C | 6 F+O+C | 6 F+O+C | 6 F+C | 6 F+C |

### MODEL OUTPUT — dev, estimated

| Candidate | FCFS | SPF | SPF+D1 | EDF | EDF+D1 | Least-slack | EDF w2 | EDF w2+D1 |
|---|---|---|---|---|---|---|---|---|
| C6 | 6 F+O | 10 P+F+O | 10 P+F+O+C | 6 F | 6 F | 6 F+O | 6 F | 6 F+C |
| C17 | 6 F+O | 10 P+F+C | 10 P+F+C | 6 F | 10 F+O+C | 6 F+O | 6 C | 10 F+O+C |
| C18 | 6 F+O | 10 P+F+O+C | 10 P+F+O+C | 10 F+O+C | 6 F | 6 F+O | 6 C | 6 C |
| C19 | 6 F+O | 10 F+O+C | 10 F+O+C | 6 F+O | 10 F+O+C | 6 F+O | 6 F+C | 10 F+O+C |
| C31 | 6 F+O | 10 F+O+C | 10 F+C | 6 F | 6 C | 6 O | 6 C | 6 C |
| C42 | 6 F+O | 10 F | 10 F+C | 10 F+O+C | 6 F | 6 O | 10 F+O+C | 10 F+C |
| C43 | 6 F+O | 10 F+C | 10 F+C | 10 F+O+C | 10 F+O+C | 6 F+O | 6 C | 6 C |

### MODEL OUTPUT — formal-mix, strict

| Candidate | FCFS | SPF | SPF+D1 | EDF | EDF+D1 | Least-slack | EDF w2 | EDF w2+D1 |
|---|---|---|---|---|---|---|---|---|
| C6 | 10 F+O+T | 18 F+O | 14 F | 10 F | 10 F | 10 F+O | 10 F | 10 F |
| C17 | 10 F+O | 18 F | 18 F | 14 F+O | 14 F+O | 10 F+O | 10 F | 14 F |
| C18 | 10 F+O | 18 F | 18 F | 10 F | 14 F+O | 10 F+O | 10 F | 14 F+O |
| C19 | 10 F+O | 18 F+O | 14 F | 10 F | 14 F+O | 10 F+O | 10 F+O | 14 F+O+T |
| C31 | 10 F+O | 18 F | 18 F | 10 F+O | 14 F+O | 10 F+O | 10 F | 14 F+O |
| C42 | 10 F+O | 22 F+C | 22 F | 14 F+O | 14 F | 10 F | 14 F+O | 14 F |
| C43 | 10 F+O | 22 F+C | 18 F | 10 F | 14 F+O | 10 F+O | 10 F | 14 F+O |

### MODEL OUTPUT — formal-mix, estimated

| Candidate | FCFS | SPF | SPF+D1 | EDF | EDF+D1 | Least-slack | EDF w2 | EDF w2+D1 |
|---|---|---|---|---|---|---|---|---|
| C6 | 10 F+O | 18 F | 18 F | 10 F | 14 F+O | 10 F+O | 10 F | 14 F+O |
| C17 | 10 F+O | 22 F | 18 F | 14 F+O | 14 F | 10 F | 14 F+O | 14 F |
| C18 | 10 F+O | 18 F | 18 F | 10 F | 14 F+O | 10 F+O | 10 F | 14 F |
| C19 | 10 F+O | 18 F | 18 F | 10 F | 14 F+O | 10 F+O | 10 F | 14 F+O |
| C31 | 10 F+O | 18 F | 18 F | 10 F | 14 F+O | 10 F+O | 10 F | 14 F |
| C42 | 10 F+O | 26 F+C | 22 F | 14 F+O | 14 F | 10 F | 14 F+O | 14 F |
| C43 | 10 F+O | 22 F | 22 F | 10 F | 14 F | 10 F+O | 10 F | 14 F |

MODEL OUTPUT interpretation: statistical allowance changes some strict first failures without changing schedules. For example C42/SPF/formal-mix first fails strictly at 22 (F+C), but under the chosen estimate first fails at 26 (F+C). This **does not validate N=22 on hardware** and does not establish that the public implementation uses this interval. The more cold-heavy dev mix fails much earlier than formal-mix; none of these tables is a deployment capacity prediction.

## Draft, tests and reproduction

[003 design](../../patches/003-slo-aware-scheduling.md) documents the HTTP→GenerateReqInput→tokenizer→scheduler fields, state lifetime, queue/slack computation, aligned continuation reservation, single-partial guard, scope and rollback. [003 patch](../../patches/003-slo-aware-scheduling.patch) is against the ready `build/d2/b` base; it applies with 001/002 and also with D0. It has no output, timestamp-reporting, request-dropping or cache-flush shortcuts. D0's actual receive stamp is used only to calculate priority.

**VERIFIED / CPU:** SLO-01…07 pass. New 16 tests include independent count frontiers, 450 randomized actual-helper/model comparisons, real adder host-miss and D1 guards across all three policies, unchanged accounting ASTs, 56 FCFS paired scenarios, and clean patch-chain/byte parity. Regression evidence: 47 simulator, 29 scorer/ladder, 21 weighted/tool, 16 original D2, 15 original D1 tests pass (**144 total**). Full dependency imports and live tests SLO-08/09 remain todo; these CPU fakes cannot establish KDA logits, runtime memory lifecycle or real timing.

Reproduction:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_slo_calibration.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts -p test_slo_scheduling.py -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts -p 'test_sim*.py' -v
ARENA_FLUSH_ATTEMPTS=1 ARENA_FLUSH_SERVER_WAIT_S=0 PYTHONDONTWRITEBYTECODE=1 \
  python3 -m unittest discover -s scripts -p test_ladder_search.py -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts -p test_t16_tools.py -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts -p test_spf_scheduling.py -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts -p test_role_boundary_split.py -v
python3 scripts/check_records.py
```

The T14 flush-test environment is required: the initial run with default retries consumed the one-response-per-case fixture and failed; rerunning with its prescribed single attempt removes that harness mismatch. F40's refreshed public data also invalidated an old F35 whole-file count assertion; the test now selects the original saved R10 attempt IDs, retaining the same F35 counts/medians. The T20 runner is pinned to its original five schedulers to preserve its 560-run recipe after this extension. No calibrated candidate/old result was overwritten.

Receipts: `evidence/T25_slo/{unit_tests,simulator_regression,scoring_regression,weighted_regression,d2_regression,d1_regression}.log`, `calibration.log`, `validation.json` and `records.log`. Input hashes are verified against the simulation's read-only input manifest. No GPU, Trisol, engine launch, image build or submission by W12.

Evidence migration: `notes/t25_slo` is now `evidence/T25_slo` under the updated rule.md. `migration.json` records byte-identical movement; `validation.json` proves that the executed simulator/scorer/driver and original R10 input differ from current files only by reversible path substitutions listed in `evidence/MOVES.txt`. The 896-run result is unchanged; current-path simulator/SLO suites and a 16-arm one-level CLI smoke passed again.
