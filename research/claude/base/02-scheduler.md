# 02 — Scheduler source map (base image SGLang, fe236ea6c3)

> **2026-09-23 更正（8 卡实测 + Fable 审阅）**：① "冷预填充独占 GPU 伤 intra" 已被 8 卡证实为 N6 主瓶颈（intra 排队 p95 6.4s，F76），补丁 120 把它降到 0.36s（F77 起）。
> ② 120 v1 的续算封顶在无等待时也生效 → chain_start 尾部变差；S0 使用的当前补丁 120 在有等待时封顶，配合 chunk 16384 / cap 8192 / short 8192。当前文件与 026/035 的 120 逐字节等价（`evidence/T57/equivalence.log`）。
> ③ 底包默认 `max_running_requests=116=584/5`，但日志的 KV 百分比不等于物理余量；哪一池先限制更高并发尚未证实（R18 §1、§3）。④ NO_TOKEN 使 `batch_is_full` 粘住（`scheduler.py:3755,3581`），KV 吃紧时可能重现排队。

Source root: `build/base_exact/sglang/`. All paths are relative to `srt/`. "unverified" = not confirmed in code.
Deployment assumed: TP8, 8×A100, `--page-size 64 --mamba-radix-cache-strategy extra_buffer --schedule-policy lpm`. No mixed chunk, no priority, no DP-attention. MTP may or may not be on.

## 1. Event loop

- **Overlap is on by default.** `enable_overlap = not disable_overlap_schedule` (`managers/scheduler.py:465`). `disable_overlap_schedule=False` (`server_args.py:885`). `dispatch_event_loop` picks `event_loop_overlap` (`managers/scheduler.py:5478`). extra_buffer is compatible with overlap. Only `auto`→`no_buffer` forces overlap off (`arg_groups/overrides.py:587-596`).
- **Normal loop** (`scheduler.py:1832`): recv → `get_next_batch_to_run` → `run_batch` → `process_batch_result`.
- **Overlap loop** (`scheduler.py:1867`): the loop launches batch N, then processes the result of N−1 (`:1924-1926`). So a prefill's first token is emitted one CPU iteration later, after the next batch has been scheduled and launched. `is_disable_overlap_for_batch` (`:1941`) only syncs two back-to-back prefills when `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP` is set (default False, `environ.py:606`).
- **`get_next_batch_to_run`** (`scheduler.py:3364`), in order:
  1. Stash the previous chunk of `chunked_req` into the radix tree (`:3393-3401`, `stash_chunked_request` `:3269`).
  2. Merge the last extend batch, minus the chunked request, into `running_batch` (`:3423-3441`).
  3. Try a prefill (`:3456-3460`).
  4. **Prefill always wins.** `if new_batch is not None: ret = new_batch`. Decode runs only if no prefill batch was formed (`:3476-3484`).
- **Prefill/decode alternation.** None by default. While there is waiting work that fits, or a chunked request is in flight, consecutive rounds are all prefill and running decodes stall. With `chunked_req` set, `_get_new_batch_prefill_raw` always produces a batch (`:3581-3584`, `:3657-3659`). So a cold 100k prompt means about 13 back-to-back prefill rounds with **zero decode steps**. That hurts tpot_p95.
- **`prefill_decode_interval`** (default 0, `server_args.py:718`). After any extend batch, `_arm_prefill_decode_interval` (`scheduler.py:1263`) forces N decode-only rounds via `_should_defer_prefill` (`:1256`, used at `:3456`). The counter also defers a chunked-request continuation, and it is global, not per request.
- **Mixed chunk.** `enable_mixed_chunk` defaults to False (`server_args.py:903`). `is_mixed_chunk` also requires chunked prefill (`scheduler.py:1238`). When on, the whole running batch is folded into the prefill batch as 1-token extends (`:3839-3868`, `schedule_batch.py:2889`). The decode tokens are subtracted from `rem_chunk_tokens` and `rem_input_tokens` (`schedule_policy.py:504,512`). Spec decoding forces it off unless the algorithm is EAGLE/EAGLE3 (`arg_groups/speculative_hook.py:675`, `speculative/spec_info.py:135`).
- **Other knobs:**
  - `num_continuous_decode_steps` (`server_args.py:893`);
  - `scheduler_recv_interval` (`scheduler_components/recv_skipper.py:16`);
  - `enable_prefill_delayer` (off by default, `server_args.py:3408`);
  - `min_free_slots_delay` (off unless DFlash, `scheduler.py:1103`);
  - `prefill_max_requests` (None, checked at `schedule_policy.py:1191`).

## 2. Admission (`PrefillAdder`, `managers/schedule_policy.py:478`)

- **Construction** (`scheduler.py:3638-3656`):
  - `rem_input_tokens = max_prefill_tokens` (default **16384**, `server_args.py:728`; passed through unchanged, `tp_worker.py:552`);
  - `rem_chunk_tokens = chunked_prefill_size`;
  - `rem_total_tokens` is a property (`schedule_policy.py:640`): for hybrid SSM it equals `available_size + full_evictable_size − rem_total_token_offset`.
  - The offset starts with a reservation for every running request, `min(max_new − out, CLIP=4096) × new_token_ratio` (`:531`, `:630`, CLIP `:76`).
  - `cur_rem_tokens` (`:672`) is the same pool without the future-decode reservation.
- **`add_chunked_req`** (`:978`) continues the in-flight partial request with `min(rem_chunk_tokens, rem_total_tokens)` tokens (`:982`). It is never refused: if the result is ≤0 it falls back to `rem_chunk_tokens` (`:991-994`). It returns the request while it is still truncated (`:1027`).
- **`add_one_req`** (`:1182`):
  1. `total_tokens = extend + min(max_new,4096) + page (+mamba gap)`. If `total_tokens >= rem_total_tokens` it returns `NO_TOKEN` (`:1217`, rechecked under the node lock `:1256`).
  2. It fits whole if `input_tokens <= chunk_tokens_limit`, and then the full extend range is committed (`:1335-1359`).
  3. Otherwise it becomes the round's chunked request, truncated to a page-aligned `rem_chunk_tokens` (`:1360-1408`). This applies to cached requests too; only the ignore_eos path asserts no prefix (`:1166`).
  4. `budget_state` (`:815`) returns NO_TOKEN (KV/mamba), OTHER (input/chunk budget spent), or CONTINUE.
- **Budget charge** (`_update_prefill_budget`, `:838`): `rem_total_offset += ceil_page(extend) + max_new + page + mamba_gap`; `rem_input_tokens` and `rem_chunk_tokens` are reduced by `ceil_page(extend)`.
- **Mamba slots:**
  - The slot is allocated **at match time, before admission**. `init_next_round_input` (`schedule_batch.py:1390`) calls `match_prefix(cow_mamba=True)`, and the Mamba component allocates the destination slot, evicting a state if none is free (`mem_cache/unified_cache/components/mamba_component.py:191-212`; legacy `mamba_radix_cache.py:1181`). Slots are pre-grabbed with `alloc_group_begin(len(waiting_queue))` (`scheduler.py:3676`). If the request is rejected, its fresh slot is freed (`:3757-3770`).
  - `PrefillAdder` gates on mamba **only for the unified pool**. `_mamba_slot_cost` and `rem_mamba_slots` are non-zero only with `UnifiedMamba*Allocator` (`schedule_policy.py:555-579`), i.e. `--enable-unified-memory` (default False, `server_args.py:871`). On the default static pool, mamba capacity is enforced indirectly through `max_running_requests` (see below) and at `alloc_req_slots`. That function evicts up to 3 states per request, then fails loudly (`mem_cache/allocation.py:229-270`).
- **What blocks admission when memory is tight:**
  - **The loop `break`s on the first non-CONTINUE result** (`scheduler.py:3747-3771`). A head request that gets NO_TOKEN therefore blocks everything behind it (head-of-line).
  - NO_TOKEN sets `running_batch.batch_is_full=True` (`:3755`). The next rounds then skip prefill entirely (`:3581-3584`) until a request finishes, retracts or leaves (resets at `:3433,3908,3977`). The reset at `:3579` applies only to priority-preemption and hybrid-SWA, which does not include this model.
  - Request slots: `get_num_allocatable_reqs` (`:3514`) ≤0 ⇒ full (`:3604-3611`, `:3689-3694`).
- **`chunked_prefill_size` default** (`arg_groups/memory_hook.py`): A100-80GB (<90 GiB) ⇒ **8192** (`:102-110`); A100-40GB ⇒ 4096 (`:88-96`). There is no GLM/KDA model override. With extra_buffer a warning is printed if the value is below `mamba_cache_chunk_size` (`arg_groups/mamba_hook.py:128-139`). Effective per-round cap = min(8192, 16384) = **8192**.
- **`max_running_requests`** (`mem_cache/kv_cache_configurator.py:2160`):
  - The value is `min(clamp(token_capacity/context_len×512, 2048, 4096), token_capacity//2, max_mamba_cache_size // ratio)`.
  - `ratio = 3 + 2 = 5` for extra_buffer with overlap (`:2100-2129`, constants `:159-163`). The mamba cap is usually the binding one here (unverified numerically).
  - If a spec algorithm (EAGLE/MTP family) is on and the flag is unset, it is **forced to 48** (`arg_groups/speculative_hook.py:657`).
- **`schedule_conservativeness`** (default 1.0, `server_args.py:803`) only scales the initial `new_token_ratio`. The ratio is `init = min(0.7×c, 1)`, and it decays over 600 steps down to `0.14×init` (`scheduler_components/new_token_ratio_tracker.py:20-34`, `environ.py:559-561`). This ratio multiplies each running request's reserved future tokens in `rem_total_tokens`. In the public dev set, requested `max_output_i` has p50=198, p90=554, max=5644; the harness sends each request's `max_new_tokens` (`s1-dev/harness/s1_loadgen.py:103,258`). The reservation can therefore vary greatly across requests.

## 3. Policies

- **Where it runs.** `calc_priority` (`schedule_policy.py:240`) is called every prefill attempt (`scheduler.py:3614`). The schedule-policy default is fcfs (`server_args.py:745-759`).
- **LPM** (`_compute_prefix_matches` `:324`, `_sort_by_longest_prefix` `:384`):
  - For each waiting request it runs a full `match_prefix` against the radix tree (`:334-340`).
  - The sort key is `-num_matched_prefix_tokens`: absolute matched length, including host hits, not the fraction cached.
  - The sort is stable and in place.
  - A request's matched prefix is capped by mamba checkpoint availability (cache subsystem, see the cache map).
- **LPM → FCFS fallback** happens when `len(waiting_queue) > 128` (`:293-297`). FCFS is then a no-op `pass` (`:277-278`), so the queue keeps its previous LPM order plus appended arrivals. It is not true arrival order.
- **LPM cost.** It does O(queue) prefix matches, each O(prompt/page) on 36k–257k-token keys, every scheduling round in which the queue is non-empty and the batch is not full. Plus in-batch radix inserts. Even non-cache-aware policies do the full match when `supports_fast_match_prefix` (`:250-255`), so FCFS is not free either.
- **In-batch prefix caching.** Requests with ≤32 cached tokens (`IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD`, `:84`) are inserted into a simulated tree. A later waiting request that shares ≥32 tokens with one of them is sorted to the end (`inf`, `:349-368`, `:91`).
- **LPM starves cold requests.** Chain starts (0 hit) always sort after cached mid-chain requests, so they wait until the queue drains or reaches 128+.
- **Priority scheduling:**
  - It requires fcfs or lof; `lpm` is rejected (`arg_groups/validation_hook.py:142-146`).
  - The sort is `(priority×sign, wait_queue_entry_time)` (`:427`).
  - Preemption is on by default when enabled (`scheduler.py:1328`). It resets `batch_is_full` every round (`:3579`). `preempt_to_schedule` (`schedule_policy.py:1411`) evicts running requests whose priority gap exceeds `priority_scheduling_preemption_threshold` (default 10, `server_args.py:784`). Evicted requests are re-queued and their KV is freed without insert.
  - `retraction_policy=priority` is an option (`schedule_batch.py:3134`).
  - `"priority"` is accepted as a CLI choice but is not in `CacheAgnosticPolicy`, so passing it would raise (unverified at runtime).
- **Routing-key** (`:439`): the queue is sorted by how many running requests share the same `routing_key`, most first. Requests without a key go last.

## 4. Chunked prefill

- **Splitting a 100k+ cold prompt.**
  - Round 1: `add_one_req` truncates it to 8192 minus whatever earlier requests in the round consumed, page-aligned (`:1362-1381`), and sets `new_chunked_req`.
  - Next rounds: `add_chunked_req` gives it `min(8192, rem_total)` first (`scheduler.py:3657-3659`).
  - Each chunk is stashed into the tree between rounds.
  - The last chunk returns `None` (`schedule_policy.py:1027`) and reserves `max_new` only then.
- **Nothing else is admitted while a middle chunk runs.** A middle chunk consumes all of `rem_chunk_tokens`, so `chunk_budget_exhausted()` (`:810`) ends the waiting loop immediately (`scheduler.py:3679`). Other requests can share only the round of the *first* chunk (if they come before it) or of the *last* chunk (with its leftover budget).
- **Only one partial request per round.** Once a request is truncated, `rem_chunk_tokens` is ≤ one page and the loop stops. `assert self.chunked_req is None` (`scheduler.py:3789`) enforces a single global `chunked_req`.
- **Head-of-line blocking is real:**
  - A 100k cold prefill occupies about 13 consecutive rounds. Every mid-chain request that arrives in the meantime waits for all of them, even with a 1k extend.
  - All running decodes stall for the same period (§1).
  - Any mid-chain request whose extend exceeds the leftover chunk budget is itself split and gets its first token one round later.
  - LPM keeps cold requests at the tail, which limits how often a cold request starts. But once a cold chunked request starts, it is unconditionally continued.

## 5. Retraction and preemption during decode

- `update_running_batch` (`scheduler.py:3902`) calls `check_decode_mem` (`schedule_batch.py:3023`; one page per request crossing a page boundary). On failure, `retract_decode` (`:3034`) pops requests until the rest fit.
- **Retraction order** (`_get_decode_retraction_order`, `:3118`): the fewest output tokens go first, and among ties the longest input. That means freshly admitted long-context requests are retracted first.
- `release_req` frees KV with `is_insert=False` (`:2067`) and evicts `remaining×20` tokens from the tree. The request is re-queued (`scheduler.py:3972`) and loses its decode KV, although its prefix may still be in the tree.
- `new_token_ratio` jumps to `(decoded + 20·n)/Σmax_new` (`new_token_ratio_tracker.py:44-53`), which makes admission more conservative. Otherwise the ratio decays each decode step (`scheduler.py:3974`).
- Without priority scheduling there is no preemption at admission time.
- Other drops: `SGLANG_REQ_WAITING_TIMEOUT` (off, `environ.py:594`; `scheduler.py:3139`) and `max_queued_requests` (`:3091`).

## 6. Where a mid-chain cached request (~1–4k new tokens) waits

1. ZMQ recv → `_add_request_to_queue` → `waiting_queue` (`scheduler.py:3042`).
2. The next `get_next_batch_to_run` may skip prefill because:
   - `prefill_decode_interval` is active (off by default);
   - `batch_is_full` is sticky after a NO_TOKEN or a full request pool, until a request finishes (`:3581`);
   - request slots are exhausted, i.e. `max_running_requests` (48 if spec is on) (`:3604`).
3. In the adder loop, it can wait:
   - behind an in-flight middle chunk (the whole round is lost);
   - behind higher-LPM requests that use up the 8192 chunk budget (only about 2–7 mid-chain requests of 1–4k fit per round);
   - behind a head request that gets NO_TOKEN (`break`);
   - by being truncated itself into a 2-round prefill.
4. The GPU round time grows with the extend tokens plus the attention over 36k–257k prefixes on the 11 MLA layers (unverified magnitudes).
5. With overlap, the first token is sent after the next iteration's `pop_and_process` (`:1924`).
6. Retraction can send it back to step 1 after it has been admitted.

## Levers

1. **SLO/deadline ordering in `SchedulePolicy.calc_priority` / `_sort_by_longest_prefix`** (`schedule_policy.py:240,384`).
   - Change: sort by a class key first (mid-chain with uncached ≤4096 → deadline 3s; else 5s; cold → 30s), then by slack `deadline − (now − wait_queue_entry_time)`, then LPM as the tie-break. Also keep sorting past 128 instead of silently falling back.
   - Risk: lower radix locality; LPM matching cost is still paid; cold requests can starve unless slack becomes urgent.
2. **Cap cold chunks and let short cached requests join the same round.**
   - Change: in `PrefillAdder.add_chunked_req` (`:978`), limit the continuation to `cold_cap` (e.g. 2048–4096) when the waiting queue holds cached requests. Also change `_get_new_batch_prefill_raw` (`scheduler.py:3657-3679`) so it iterates the waiting queue **before** the chunked continuation, or reserves `rem_chunk_tokens − cold_cap` for it.
   - Risk: longer cold TTFT (still ≤30s); more rounds; extra_buffer requires chunks that are multiples of `mamba_cache_chunk_size` (keep them page/64-aligned); it must still always add the chunked request (memory-leak comment at `:989`).
3. **No mid-chain truncation.** In `add_one_req` (`:1360`), if the request is cached with extend ≤4096 and does not fit the leftover chunk budget, return `OTHER` for this round and try the next request (`continue` instead of `break` at `scheduler.py:3747`), or allow a one-off overshoot up to e.g. 12k.
   - Risk: activation memory is sized for 8192 (`memory_hook.py`); skip-ahead can starve large requests; the patch 101 role-boundary split must stay consistent.
4. **Replace `break` with bounded skip-ahead on NO_TOKEN/OTHER** (`scheduler.py:3747-3771`) for small requests, and do not set a sticky `batch_is_full` when a small cached request would still fit.
   - Risk: big requests starve; the mamba slot free-on-reject path (`:3757-3770`) must run for each skipped request.
5. **Decode interleaving during cold chunks.** Turn on `--enable-mixed-chunk` (only if the spec algorithm allows it), or add a "one decode every k chunks" rule next to `_should_defer_prefill` (`scheduler.py:1256`) that applies only while `chunked_req` is a cold request.
   - Risk: mixed batches change CUDA-graph and kernel paths (MLA/KDA mixed extend is untested); TTFT is traded for TPOT.
6. **Reservation estimate as an experiment.** The existing `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION` affects `rem_total_token_offset` (`schedule_policy.py:531,630`) and `add_one_req` (`:1201-1208`). Any lower cap needs an A/B that checks retractions and long outputs; the harness sends frozen per-request budgets and must not be changed for a performance gain.
7. **Retraction order** (`schedule_batch.py:3118`): prefer retracting cold/long-remaining requests over mid-chain ones that are near completion. `retraction_policy=priority` needs priority scheduling, which conflicts with lpm.
   - Risk: cold requests pay again at 30s.
8. **Raise `max_running_requests` explicitly** (it is 48 if spec is on) and check that the mamba cap `max_mamba_cache_size/5` is not the limit.
   - Risk: KV/mamba OOM, `alloc_req_slots` RuntimeError.
