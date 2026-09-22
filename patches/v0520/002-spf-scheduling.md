> 2026-09-22 归档：v0.5.20 线，打不上底包（决策 29）；仅作 L1 替身参考。

# D2 — shortest-prefill-first on v0.5.20 + D1 v1.1

Author: Codex W7, T20, 2026-09-22. **Implementation and CPU validation; no serving-performance or KDA-numerics claim.** The priority decision is now **#18** in `notes/decisions.md` (the task prompt's #16 was renumbered). References: R4, R10, F38.

## 1. Base, scope, application and rollback

Port of upstream [SGLang #40024](https://github.com/sgl-project/sglang/pull/40024), merged 2026-09-18 as [`65ef55e2a8c51be0523723fcb98d7138b5839ebb`](https://github.com/sgl-project/sglang/commit/65ef55e2a8c51be0523723fcb98d7138b5839ebb). Frozen upstream patch and production hunks are under `build/d2/upstream/`; source hashes are in `notes/t20_d2/patch_validation.json`.

`build/d2/a` is v0.5.20 `94602c9c2b7cbdb8efd5c52802dac6a1c180089e` plus **D1 v1.1**: its schedule_policy.py was copied from `build/d1/b`; scheduler.py and arg_groups/fields/schedule.py were copied from the read-only base. `build/d2/b` contains the port. `002-spf-scheduling.patch` is the diff from a to b, covering those three files. Neither `src/sglang` nor `build/d1` was modified.

VERIFIED: clean base → 001 → 002 applies with `--fuzz=0`, and all three final files match `build/d2/b` byte for byte. **002 alone does not apply to clean base**: three schedule_policy.py hunks depend on replacing D1's guard/helper context. Do not partially apply it or force fuzz. D2 alone as an *enabled feature* is available by applying both patches and leaving the D1 environment variable unset.

```bash
# Run from a writable copy of the SGLang repository, never src/sglang.
patch --batch --fuzz=0 -p1 < /workspace/Agentic_science_challenge/patches/001-role-boundary-mamba-ckpt.patch
patch --batch --fuzz=0 -p1 < /workspace/Agentic_science_challenge/patches/002-spf-scheduling.patch
# Enable D2 with the upstream CLI value:
# --schedule-policy shortest-prefill-first
```

With cwd at the installed `sglang` package directory, the equivalent strip level is `-p3`, as for 001. Prefix caching must be enabled, matching upstream. No new environment knobs or reserve-fraction options. Default `schedule_policy="fcfs"` stays unchanged. For rollback, restart with `--schedule-policy fcfs`; unset `SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS` to disable D1 too. To recover exact D1 v1.1 source, reverse 002 on the copy. Candidate commands/Dockerfile payloads have not been updated or built by T20.

## 2. Scheduling semantics

VERIFIED / production code:

1. Refresh actual prefix matches using the existing cache-aware policy machinery. Order waiters by `(temporarily_deprioritized, max(1, input_len + output_len - matched_prefix_tokens), wait_queue_entry_time)`. This uses current cache work, never the workload's frozen gate label. Existing in-batch duplicate-prefix handling remains in place.
2. For an active continuation, scan the sorted queue. Reserve page-ceiled work for each waiter that is **strictly shorter** than the continuation and fits while leaving its minimum progress. Stop at the first equal/longer or nonfitting waiter, as #40024 does; there is no #39717 skip-ahead or 128-entry scan cap.
3. Set `adder.chunked_req_limit`, then run the continuation **first** with `min(existing memory/SWA limit, reserved-budget cap)`. The reservation uses the existing chunk budget; it never adds compute capacity. Normal upstream minimum is one page, with no percentage reservation. No qualifying waiter means no cap.
4. Walk the waiting queue with the original v0.5.20 admission gates. Reserved capacity is not a promise of admission: request slots, KV, Mamba, delayer, host load and tile gates still apply. A rejected waiter can leave unused capacity; the continuation is not replanned afterward.

The local `truncation_align_size` may enforce a larger deterministic/DSA grid. The optional helper argument and scheduler call use `lcm(page_size, truncation_align_size or 1)` for continuation minimum and endpoint. With default/one-page alignment this is exactly upstream's reservation formula. If there is insufficient budget for one joint unit plus one page, no reservation is made. The one-page guarantee concerns **reservation-induced** shrinking of a genuinely long continuation; a naturally shorter final tail or the pre-existing memory/SWA limit can still be smaller. This patch does not weaken those memory gates.

## 3. D1 interaction: one authority for partial eligibility

`PrefillAdder._can_start_partial_prefill(has_chunked_req)` is the sole definition of partial-slot availability: neither an existing unfinished continuation nor this round's `new_chunked_req` may hold the slot. Both ordinary truncation and D1's optional role split consult it.

- Ordinary truncation checks eligibility inside `_select_prefill_admission`, before delayer negotiation or load-back. It is enabled when SPF or D1 is enabled, so the fully disabled FCFS path retains legacy decisions.
- Both initial selection and the host-load-miss re-selection receive `has_chunked_req`. A miss that changes a predicted full waiter into a partial is rejected before commit; the earlier D1-only guard did not cover that second selection.
- D1 checks the same helper before proposing a role split. A full waiter sharing an unfinished continuation is kept full and increments `skipped_active_chunk`. After a D1 split, more full waiters may enter, but no second partial may enter.
- A load exposing *more* cached tokens can only shorten/complete the already approved work; it cannot turn a full admission into a partial. Thus all potentially partial-producing paths consult the shared authority before publication. `_commit_prefill_admission` remains unchanged, with no new rejection after successful materialization.

Selection/commit separation, both KV admission gates, page overhead, shared Mamba gap/slot reserve, host miss fallback, HIP tile gates, exact-chunk-fill mode, SWA constraints and new-candidate DSA alignment were retained from this local base. Tests compare the budget/commit helper ASTs to clean v0.5.20, and exercise actual selection and commit with fakes. No request output budget, prompt, tool, cache-count or timestamp contract was changed.

## 4. Deviations and limits

- The upstream partial guard only tests an existing continuation in the ordinary chunk branch. This port also protects a newly created D1/ordinary partial and reuses that decision for D1. It removes the earlier standalone D1 guard instead of stacking independent decisions.
- The continuation alignment argument is local, preserving DSA/deterministic alignment when it exceeds page size. It may reserve less usable waiter capacity or leave alignment slack. New-candidate alignment logic is unchanged.
- Production tests use local fakes and full production classes; no upstream allocator/runtime code was transplanted wholesale. #39717, HRRN interleaving, mixed-chunk, cache eviction policy and kernels are outside this patch.
- SPF has no aging guarantee: continuing work gets a page, but a long **waiting** request can still starve under sustained shorter arrivals. Memory/request-slot misses may waste reservations. CPU tests cannot establish actual cache lifecycle, cancellation/allocator cleanup, KDA state equality, long-tail latency or throughput. D2-11/12 remain live follow-ups.
- “Default FCFS unchanged” means default decisions, admitted ranges and accounting match clean base (56 paired scenarios = 112 executions, serialized snapshots identical), with the original fcfs CLI default. The source necessarily contains new opt-in branches. D1-enabled FCFS deliberately receives the host-miss safety correction above.

## 5. Tests and reproduction

See D2-01…D2-12 in `tests/TEST_PLAN.md`. Local tests compile the **actual complete policy/adder classes** from `build/d2/b` via AST, stubbing imports and cache/allocator/request objects, without translating scheduler code. An optional `D2_REAL_IMPORT=1` mode imports the complete module against the GPU box's installed dependencies before applying the same fakes. The tests include patch-chain parity so stale working copies cannot pass.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts -p test_spf_scheduling.py -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts -p 'test_sim*.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts -p test_role_boundary_split.py -v
PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_d2_sim_calibration.py
python3 scripts/check_records.py
```

VERIFIED: 16 D2 production tests; 47 simulator tests (original 41 + 6 new), including 500 random admission rounds checked against the real port; original 15 D1 tests also pass. Local evidence: `notes/t20_d2/unit_tests.log`, `all_sim_tests.log`, `sim_semantics_tests.log`, `d1_regression.log`. The small comparison artifact stores 70 arms × 8 N levels = **560 model runs**; no output-budget shortening. GPU-box CPU verification also passed all 16 production tests with the complete module imported under `/sjtu/linhang/arena/env/sgl` (Python 3.12, torch 2.13.0+cu130), keeping all work inside `/sjtu/linhang/arena/code/d2-worker-w7` and logs under `/sjtu/linhang/arena/runs/T20`. Receipt: `notes/t20_d2/full_import_tests.log`. Initial scp interruption left an incomplete fixture (recorded separately); after retransferring the fixture, the same test suite passed. These tests execute no GPU forwards.

Remote reproduction after syncing the fixture:

```bash
source /sjtu/linhang/arena/env.sh
cd /sjtu/linhang/arena/code/d2-worker-w7
PYTHONDONTWRITEBYTECODE=1 D2_REAL_IMPORT=1 /sjtu/linhang/arena/env/sgl/bin/python -m unittest discover -s scripts -p test_spf_scheduling.py -v
```

The direct `scripts/sim_closed_loop.py` CLI was also checked for all five policies at C19/N14; its output equals the wrapper results exactly (`cli_c19.log`, `cli_c19_parity.log`). Twelve complete FCFS/legacy-SPF simulations including traces are byte-identical to the pre-T20 implementation (`legacy_sim_parity.log`).

## 6. R10 simulator cross-check and stock HRRN/LPM

**MODEL OUTPUT / WHAT-IF**, not measured capacity. Reused the five main R10 candidates C6/C17/C18/C19/C31 and two stock-N18 neighbors C42/C43, fixed formal-mix seed 20260922, profiles/output budgets, page64, each candidate's timing parameters and N=2,6,10,14,18,22,26,30. No refit or changes to the gates. Full engine parameters, input/script hashes, per-N gates/metrics and first failures: `notes/t20_d2/calibration/comparison.json`; command receipt: `notes/t20_d2/calibration.log`.

The old simulator `spf` **already reserves complete short waiters; it is not pure ordering**. However, it admits those waiters first, uses available request slots during reservation, and schedules the continuation afterward. #40024 instead reserves without checking slots, runs continuation first, then applies admission gates. A deterministic counterexample is budget4096/page64, active16384, waiter512, max_running1: legacy spends4096 on continuation; upstream caps it at3584 and leaves512 unused when the waiter fails the slot gate.

New `--schedulers spf-upstream` models #40024's two-stage reservation/admission exactly under the simulator's stated ideal assumptions: fixed cache-work values, no duplicate-prefix deprioritization, no host misses, unlimited KV/Mamba capacity, normal page alignment. It is not a full SGLang simulator. The 500-round differential covers continuation/no continuation, arbitrary page/work/budget sizes, request-slot caps and second-partial prevention. Legacy `spf` remains available with unchanged behavior for R9/R10 reproducibility.

In the fixed R10 comparison, all **14 stock/D1 paired ceilings** agree between legacy SPF and upstream SPF. Almost all metrics also agree to rounding. C42/stock/N14 differs (same verdict): overall p95 3.57083→3.57325s, turn p95 4.36313→4.14778s, chain p95 20.77048→20.61486s. Request completion order within a common forward affects closed-loop chain assignment and subsequent arrivals; exact timestamps should not be assumed identical across the two models.

**MODEL OUTPUT — contiguous passing N, stock-cache / D1-cache proxy.**

| Candidate | FCFS | SPF upstream | HRRN stock policy | LPM stock policy |
|---|---:|---:|---:|---:|
| C6 | 6 / 6 | 14 / 10 | 6 / 6 | 2 / 6 |
| C17 | 6 / 6 | 14 / 14 | 6 / 6 | 6 / 6 |
| C18 | 6 / 6 | 14 / 14 | 6 / 6 | 6 / 6 |
| C19 | 6 / 6 | 14 / 10 | 6 / 6 | 6 / 6 |
| C31 | 6 / 6 | 14 / 14 | 6 / 6 | 6 / 6 |
| C42 | 6 / 6 | 18 / 18 | 6 / 6 | 6 / 6 |
| C43 | 6 / 6 | 18 / 14 | 6 / 6 | 6 / 6 |

New simulator values `hrrn` and `lpm` mirror the **unpatched v0.5.20 ordering**, without continuation reservation. HRRN uses processed-prefill-token aging `(counter - arrival_counter) / uncached`, zero-work priority, and rid ties; LPM sorts by absolute matched prefix count with stable ties. Both use the stock >128 waiters FCFS fallback. Their sort results were compared to the actual production methods. The simulator snapshots arrival counters at modeled scheduler admission boundaries and uses synthetic request IDs, static cache counts and no duplicate-prefix matching; overlap/runtime ranking can differ.

INFERRED from these fixed models: neither stock HRRN nor LPM approximates the SPF ceiling gain here. The continuing long chunk still consumes the budget, so merely sorting waiting requests is insufficient under this model. This supports measuring D2 next, and does not establish any real-engine ceiling or D1 improvement. A same-engine stock/HRRN/LPM/SPF experiment remains necessary before deployment choice.
