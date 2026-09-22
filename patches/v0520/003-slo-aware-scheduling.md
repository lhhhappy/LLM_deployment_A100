> 2026-09-22 归档：v0.5.20 线，打不上底包（决策 29）；仅作 L1 替身参考。

# 003 — SLO-aware prefill scheduling (T25 / W12)

2026-09-22. **Draft, opt-in, CPU validated. No live correctness, throughput or capacity claim.** The fixed R10 models do **not** justify replacing SPF with this draft: see [R12](../research/codex/R12_slo_aware_scheduling.md). This implements the requested I6 experiment, not a deployment recommendation.

## Base, application and rollback

`build/d3/a` snapshots the three modified files from W7's ready `build/d2/b`; other touched files come from read-only v0.5.20 `src/sglang`. `build/d3/b` contains 003. Neither D2's workspace nor either read-only tree is edited. Rebuild the frozen draft with `python3 build/d3/make_patch.py`; that helper is a development artifact, not an image build. Patch hashes and base provenance are in `evidence/T25_slo/validation.json`.

On a writable SGLang source copy, apply `001 → 002 → 003`, each using `patch --batch --fuzz=0 -p1`. CPU tests also apply `000 → 001 → 002 → 003` cleanly. At installed package root use `-p3`. The HTTP hunk intentionally has two context lines to compose with D0's receive-time hook without fuzz. Without D0, the scheduling-only timestamp starts at the typed handler; D0 is required for the intended pre-body arrival basis and interface-compliant measurements.

Select `--schedule-policy arena-edf`, `arena-least-slack` or `arena-edf-chain-weighted`. Default `fcfs` and existing `shortest-prefill-first` remain selectable with their previous decisions. D1 remains independently controlled by `SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS`. Roll back by selecting `shortest-prefill-first` (or `fcfs`) and restarting; reverse 003 on the writable copy for source rollback. No submission or Dockerfile flags were changed.

## Request information and classification

1. The `/generate` typed handler has the Starlette `Request` and headers. It calls `apply_slo_headers` from new `managers/arena_slo.py`, independently of the unrelated `SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES` option. It reads **only** `X-S1-Session-ID`, `X-S1-Cache-Namespace` and D0's server-owned `scope['arena_recv_perf']`; absent D0 it captures `time.perf_counter()` at that point. Unknown headers stay harmless. The body cannot override this scheduling timestamp on this route.
2. Three optional fields (`arena_slo_session`, `arena_slo_namespace`, `arena_slo_arrival`) pass through `GenerateReqInput`, its batch `__getitem__`, tokenizer construction of `TokenizedGenerateReqInput`, and Scheduler's normal generate path into `SchedulePolicy.register_slo_request`. These are separate from existing native `session_id` / `session_params`, which can change cache or prompt-session semantics. The patch does not enable session radix caching, rewrite prompts or alter routing.
3. Scheduler-local state uses `(namespace, session_id)` keys. First observed request gets a **chain_start proxy**; later requests get an **intra proxy**. Missing identity uses the conservative intra proxy. State is bounded to 65,536 LRU identities. Successful, idle `Scheduler.flush_cache` clears this bookkeeping alongside real cache clearing; failed busy flush leaves it intact. Namespace changes isolate replay levels. Retraction/requeue keeps the same request's original arrival and frozen target.
4. Existing `calc_priority` cache matching supplies observed `num_matched_prefix_tokens`. At the first scheduling decision, freeze target `L=30s` for a first session request; otherwise `L=3s` if actual uncached work `max(1, input+output−matched) ≤4096`, else `L=5s`. Later chunk progress cannot turn the same request into a new fast bucket. Subsequent work estimates can refresh with cache state; the target stays frozen.

**Limits:** neither frozen evaluation labels nor `uncached_expected` are available to the scheduler. Turn-start cannot be inferred reliably from session identity plus cache size, so the draft does **not** invent a 15-second bucket: later turn-starts get 3/5 seconds conservatively. Context reconstruction under an existing ID is likewise not reliably recognized. In the dev replay, 3 of 314 scored chain starts occur after the first observed request; the proxy sees only 311. Tokenizer completion can reorder concurrently issued same-session requests; tracker identity is scheduler observation order, not an authoritative client ordinal. Closed-loop serial sessions avoid that race. Evicted/restarted identities are first-seen again. These are classification limitations, not reasons to reject/drop requests.

The supported initial topology is single DP, TP8 on one host, ordinary chunked prefill, no HiCache/mixed-chunk (existing decisions #5/#16). Each TP rank sees the same requests and config. DP>1 would need session-affine routing or shared tracker state; multi-host clock domains, PD/PP, beam/DLLM/internal request producers and fully imported runtime integration require separate review. Non-HTTP callers lacking the stamp fall back to scheduler arrival for priority only. Do not claim exact first-of-session behavior across those modes.

## Queue keys and reservation

For request `i`, retain **nominal deadline** `d_i = arrival_i + L_i`. EDF uses `(d_i−now, arrival_i)`, with stable ties. Least-slack uses `(d_i−now−estimated_remaining_prefill_seconds_i, arrival_i)`. The prototype estimate is:

```
remaining_uncached * max(1, full_prompt_tokens / 32768)^alpha / prefill_tps + overhead_s
```

No future output length, future arrivals or evaluator gate label enters that estimate. It excludes future prefill/decode interference and needs real timing calibration. EDF does not depend on throughput estimates. `now` is common to the ordering operation and cancels between requests.

Weighted EDF retains nominal `d_i` for diagnostics and uses a separate scheduling deadline `arrival_i + w*30` for inferred chain starts. `w=2` by default; this is an explicit **priority delay of 30 seconds**, not a changed SLO, measured timestamp, permitted timeout or statistical guarantee. EDF and least-slack use `w=1`. The environment settings below are read only when a SLO policy is enabled:

| Environment variable (`SGLANG_ARENA_SLO_` prefix) | Default | Meaning |
|---|---:|---|
| `CHAIN_WEIGHT` | 2 | Weighted EDF scheduling deadline multiplier, ≥1 |
| `PREFILL_TPS` | 40000 | Least-slack estimated aggregate prefill rate, ≥1 |
| `LENGTH_ALPHA` | 1.2 | Least-slack prompt-length penalty, ≥0 |
| `OVERHEAD_S` | .002 | Least-slack remaining-forward overhead, ≥0 |

All must be finite. These defaults illustrate R10 C19; use that candidate's actual rate when comparing other candidates. The simulator reads rate/exponent/overhead directly from its `Engine`. Production estimates are separate configuration, not calibrated hardware constants.

Unlike SPF's absolute duplicate-prefix demotion, earlier SLO deadlines take precedence over that hint. Cache matching still runs, but same-prefix waiters cannot be delayed indefinitely solely by the duplication flag. Overdue requests stay eligible; there is no abandonment, request removal, timeout shortening or late-result suppression. Old cold requests can therefore regain priority. This property and long service estimates are reasons EDF/least-slack need not outperform SPF.

The existing `shortest_prefill_chunk_limit` scheduler hook dispatches to the new reservation helper in SLO modes. At each forward boundary:

- Compute the active continuation's priority using its **original** target/arrival and its current remaining prefill work.
- Scan the sorted waiting queue. Reserve page-ceiled **complete** work only for waiters with a strictly earlier priority key, stopping on the first nonqualifying or nonfitting waiter.
- Keep at least `lcm(page_size, truncation_align_size or 1)` tokens of continuation progress. Return an aligned cap on its existing budget. Never add capacity or preempt an active forward.
- Run the continuation first, then pass waiters through the existing slot/KV/Mamba/SWA/tile/delayer/host-load gates. Reservation is not a guarantee of admission; a rejected waiter can waste some reserved capacity, as in 002.

The minimum concerns reservation-induced shrinking of a long continuation. Natural final tails and stricter pre-existing memory constraints can be smaller. A huge early waiter that cannot complete is not skipped; admission retains D2's scan/break semantics rather than adding an unreviewed backfill strategy.

## Single partial, D1 and compliance

003 extends the existing ordinary-truncation opt-in guard to all three SLO modes. It reuses **the same** `_can_start_partial_prefill(has_chunked_req)` for normal truncation, host-miss re-selection and D1 boundary splitting. If the active continuation remains partial, admitted waiters must complete; if it finishes, only one new partial may occupy the slot. After a D1 split, full requests may still enter, but no second partial may start.

Every other `PrefillAdder` method is AST-identical to 002, including commit, token/memory accounting and D1 splitting. No changes to generation budgets, thinking, tools, tokens, cache counters or emitted time statistics. Flush still performs actual engine cache/pool reset; the added identity reset is auxiliary. Scheduling may change floating-point batching; CPU tests do not prove live output/numerical equivalence.

## Validation and remaining work

Registered in `tests/TEST_PLAN.md` as **SLO-01…SLO-09**. The 16 new CPU tests execute the real helper and actual policy/adder classes (AST-loaded with runtime dependencies stubbed), not reimplementations. They cover header and field propagation, identity lifecycle, deadlines, aging/weighting, freezing, aligned reservation, 450 randomized model/production order/cap comparisons, the existing D1/host-miss adversarial cases under each new policy, default-FCFS parity, clean patch chains and independent Decimal-binomial frontiers. Existing 47 simulator, 29 scoring/ladder, 21 weighted/tool, 16 D2 and 15 D1 tests also pass; see `evidence/T25_slo/`.

The old F35 regression now selects R10's original attempt IDs from the refreshed F40 data; the calibrated numeric recipes/results are unchanged (evidence paths were migrated). The T20 comparison runner explicitly retains its original five policies so adding SLO policies does not silently expand its recipe.

Live **SLO-08/09 remain todo**: complete-module imports on the intended dependencies, concurrent/abort/requeue/flush lifecycle, numerical/logits correctness, emitted timestamps/counters, live single-partial traces, TTFT/TPOT and same-engine ladders. R12's 896 simulations are MODEL OUTPUT only. No worker W12 GPU/Trisol service, image build or submission occurred.

During T25, Claude also drafted `004-role-boundary-final-chunk.patch` on 002. The validation above targets 003 on 002; it does not certify combined 003+004 behavior or the pending E2b numerical results.
