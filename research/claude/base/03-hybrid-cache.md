# 03 — Hybrid prefix cache + KDA state tracking (base image SGLang, fe236ea6c3)

Source root: `build/base_exact/sglang/`. Paths are relative to `srt/` unless they start with `kernels/`. Deployment: TP8, 8×A100, `--page-size 64 --mamba-radix-cache-strategy extra_buffer`, overlap on, Python tree core. **VERIFIED** = read in code. **INFERRED** = derived from code, not run. **unverified** = not confirmed.

## 0. Which cache actually runs

- **`MambaRadixCache` (`mem_cache/mamba_radix_cache.py`) is dead code in this image.** Nothing constructs it: grep finds only type hints and comments. `default_radix_cache_factory` always ends in `_create_unified_radix_cache` (`mem_cache/registry.py:89,143`). That function builds a `UnifiedRadixCache` with components `[FULL, MAMBA]` (`registry.py:165`).
- The live code is:
  - `mem_cache/unified_radix_cache.py` (the controller);
  - `mem_cache/unified_cache/unified_tree_core.py` (the tree). The Python core is the default; `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND` defaults to "python" (`environ.py:643`);
  - `mem_cache/unified_cache/components/mamba_component.py`.
- Semantics are close to `MambaRadixCache`, with one difference that matters: **a leaf may hold FULL KV with no mamba state** (`unified_tree_core.py:1089-1093`, `_is_device_leaf` `:1815-1831`). The old cache required every leaf to carry a state.
- Constants for this model:
  - `mamba_cache_chunk_size = max(FLA 64, page 64) = 64` (`arg_groups/overrides.py:1840-1869`, `kernels/.../fla/chunk_delta_h.py:23`);
  - checkpoint grid = `lcm(64, 64) = 64` (`runtime_context.py:1920`);
  - decode track grid = `lcm(64, mamba_track_interval=256) = 256` (`runtime_context.py:1928`, `server_args.py:2641`).

## 1. match_prefix

**Walk** (`unified_tree_core.py:698-811`).
- The key is page-aligned to 64 (`:704`). The walk goes down children and adds each matched length to `full_kv_hit_length`.
- If the match ends inside a node, **the tree splits that node** (`:790`). The new parent is a mamba tombstone: `redistribute_on_node_split` sets its value to None (`mamba_component.py:310-320`).
- Validators: `best_match_device_node` is the deepest node where every component validator passes. For MAMBA that means `value is not None` (`mamba_component.py:142-153`). So the best match is the **deepest node on the path that holds a mamba state**.

**KV vs. mamba** (`unified_tree_core.py:813-868`).
- `device_indices = cat(value[:best_match_device_value_len])` (`:844`). **The KV hit is cut back to the deepest mamba node.**
- The KV between that node and `full_kv_hit_length` stays in the tree but is not used. The request **recomputes** it.
- When that request is inserted later, its freshly computed duplicate slots are freed and the tree's older copy is kept (`FreeDeviceKV`, `:1071-1076`).

**`mamba_branching_seqlen`** (`mamba_component.py:155-185`).
- It is set to `floor64(full_kv_hit_length)` when that is greater than the mamba boundary, and to None otherwise.
- The scheduler copies it onto the request: `schedule_policy.py:196-197` and `schedule_batch.py:1475-1484`, via `init_next_round_input` (`schedule_batch.py:1390`).

**CoW** (`mamba_component.py:187-216`).
- `cow_mamba` defaults to `tree_cache.supports_mamba()`, which is True (`schedule_batch.py:~1441`).
- If the request has no slot yet, it allocates one. On failure it locks the matched node, evicts one mamba slot, and retries.
- It sets `req.kv.mamba_cow_src_index = node's slot`.
- The copy itself is deferred to the forward stream: `_collect_deferred_mamba_cow_and_clear` (`schedule_batch.py:2860-2881`) copies conv+SSM from the node slot into the request's active slot.

**LRU touch on match.**
- MAMBA: only the best-match node is moved to MRU (`mamba_component.py:118-140`).
- FULL: `last_access_time` is refreshed along the whole path (`unified_tree_core.py:822-826`).

## 2. Tracking in extra_buffer mode

**Slots per running request.**
- `mamba_pool_idx`: the active state, taken in `HybridReqToTokenPool.alloc` (`mem_cache/memory_pool.py:1350-1400`).
- Plus a **ping-pong pair**, `_alloc_ping_pong_buffer` (`memory_pool.py:1465-1494`). The pair has 2 slots with overlap on and 1 with overlap off (`:1226`).
- **All of these are real mamba-pool slots**, taken from `mamba_allocator`.
- `keep_idx = mamba_last_track_idx` (`:1461-1463`).
- `donate_mamba_ping_pong_slot` hands the keep slot to the tree and puts a newly allocated slot in its place (`:1508-1529`).

**Prefill**: `_mamba_radix_cache_v2_req_prepare_for_extend` (`managers/schedule_batch.py:2766-2860`).
- Tracking is on only if the extend length is ≥ 64 (`:2794`).
- **Each extend tracks exactly one state per request.** The default is the chunk end rounded down to 64: `mamba_track_seqlen_aligned` (`:2813-2816`).
- If `mamba_branching_seqlen` falls inside this extend, is 64-aligned relative to the prefix, and is before the chunk end, the branch point **replaces** the chunk-end point (`:2836-2853`).
- `_force_track_h(i)` returns `i+1` (`:2779-2791`). The backend then sees an unaligned length and reads the intermediate `h[i/64]` instead of the final state.
- The tracked depth is kept in `req.kv.mamba_last_track_seqlen` (`:2854`).
- Backend index build: `_init_track_ssm_indices` (`layers/attention/hybrid_linear_attn_backend.py:356-402`). A length aligned to 64 reads the final state; an unaligned one reads `h[offset + len//64]`. The conv window is copied from raw `mixed_qkv` rows (`linear/kda_backend.py:~745-752`, `_init_track_conv_indices` `hybrid...:333-354`).
- Copy into the track slot: `_track_mamba_state_extend` (`hybrid_linear_attn_backend.py:857-876`).

**After prefill** (`managers/scheduler_components/batch_result_processor.py:346`): `maybe_cache_unfinished_req` → `UnifiedRadixCache.cache_unfinished_req` (`unified_radix_cache.py:931-1053`).
- The cache length is `mamba_last_track_seqlen` (`mamba_component.py:536-537`).
- The keep slot is donated after a replacement is allocated (`:589-595`).
- It inserts `key[:cache_len]` with the donated state. On a split tombstone this revives it (`mamba_component.py:235-247`).
- It re-matches, and `cache_protected_len` becomes the tracked depth (`unified_radix_cache.py:1001,1039`).
- KV past the tracked depth stays owned by the request.
- The same path runs for intermediate chunks (`scheduler.py:3270`, `chunked=True`). **Every chunk end, or the branch point in that chunk, becomes a tree state.**

**Decode tracking.**
- `prepare_for_decode` builds the mask from `seq_len % 256 == 0`: `mamba_track_grid` (`schedule_batch.py:3348-3378`). The interval default is `mamba_track_interval=256` (`server_args.py:2641`).
- The copy happens in the last KDA layer, `_track_mamba_state_decode` (`hybrid...:813-855`).
- After the step, `_mamba_prefix_cache_update` (`batch_result_processor.py:1213-1259`) swaps the ping-pong index and moves `mamba_last_track_seqlen` forward.
- **Decode states are not inserted into the tree along the way.** Only the latest one is, at finish.

**Finish** (`unified_radix_cache.py:838-929`).
- `effective_cache_len = mamba_last_track_seqlen`. It is None, treated as 0, if no 256-boundary was crossed since the post-prefill insert, because cleanup sets it to None at `mamba_component.py:650`.
- The insert covers only `[0, cache_len)`, using the keep slot.
- **All KV in `[max(cache_len, cache_protected_len), end)` is freed** (`:882-905`). That is up to 255 decode tokens plus the unaligned tail. If no decode boundary was crossed, it is everything past the prefill-tracked point: the prompt tail and the whole output.
- Leftover slots are freed in `cleanup_after_caching_req` (`mamba_component.py:632-641`).

## 3. Pools and sizing

**Solver** (`mem_cache/kv_cache_configurator.py:2265-2433`).
- `mamba_budget = rest × r/(1+r)`, with `r = mamba_full_memory_ratio = 0.9` (`server_args.py:2613`), so 47.4% of the budget.
- Slots = `(budget − per_slot)/per_slot` (`:2398-2405`). KV gets whatever is left (`:2433`).
- `--max-mamba-cache-size` overrides this (`:2323`).
- `rest` is the post-weights `mem_fraction_static` budget; see the scheduler note.
- **The running-request cap** is `max_mamba_cache_size // ratio` (`:2175-2199`). `_calculate_mamba_ratio` is 3 + 2 = **5** for extra_buffer with overlap, and 4 with lazy (`:2100-2129`).

**Per-state size** (`configs/mamba_utils.py:116-125,271-310`, `configs/glm5_next.py:264-280`).
- The SSM dtype defaults to **fp32** (`mamba_utils.py:71`); Glm5Next passes no dtype. The conv dtype is bf16.
- Per KDA layer per rank:
  - SSM: 8 heads × 128 × 128 × 4 B = 524,288 B;
  - conv: 3 × (1024+2×1024) × 2 B = 18,432 B.
- × 34 layers = **18,452,480 B/rank (17.6 MiB)**. With `--mamba-ssm-dtype bfloat16` it is 9.54 MB.

**KV per token.**
- On SM80 the DSA KV dtype defaults to **bf16** (`arg_groups/overrides.py:633`).
- MLA `kv_lora_rank` 512 × 2 B = 1024 B, plus the indexer K+scale (~132 B, **unverified exact layout**). About 1.16 KB × 11 layers ≈ **12.7 KB/token/rank**. MLA KV is replicated per TP rank.
- **One KDA state costs about as much as 1,450 KV tokens.**

**Measured pool sizes for the earlier default allocation.** The 8-card boot log gave 584 KDA slots, 10.05 GiB of KDA state and 943,360 KV tokens per rank; the extra-buffer running-request cap was 116 (F59, R18 §7). S0 explicitly sets `--max-mamba-cache-size 200` and `mem_fraction_static=0.75`, giving about 1.32M KV tokens (F80). These are different configurations. Pool sizes alone do not prove which resource limits a later concurrency level; include live slots, tree locks, evictions, decode reservation and prefill queue observations.

**Unaccounted.**
- The `--enable-int8-mamba-checkpoint` pool (`memory_pool.py:1281-1295`, default 2× the slots) does not appear in the configurator. Where its memory is charged is unverified.

## 4. Eviction

**Driver** (`unified_radix_cache.py:696-759`).
- Components are processed in order, FULL then MAMBA. `evict_for_alloc` stops once the allocator can serve the request (`:566-584`).
- **evict_full** (`components/full_component.py:194-228`):
  - it is a heap over `evictable_device_leaves`, keyed by `--radix-eviction-policy` (LRU by default);
  - evicting a leaf deletes the node (`_delete_unbacked_device_leaf`, `unified_tree_core.py:1429-1445`) and cascades to its mamba state;
  - childless tombstone ancestors are then deleted too (`:1759-1813`).
- **evict_mamba** (`mamba_component.py:360-431`):
  - it walks the MAMBA LRU from the tail and skips locked nodes;
  - a node in `evictable_device_leaves` is returned as a leaf and deleted with its KV;
  - any other node, internal or locked-leaf, is **tombstoned**: only the state is freed and the KV stays.
  - Priority rule: MAMBA=0 < FULL=2 on internal nodes, so evicting a mamba state never cascades to KV (`tree_component.py:482-506`, `_cascade_evict` `unified_tree_core.py:1615-1656`).
- **LRU order.** The mamba LRU is updated only on (a) insert of a new or revived state (MRU) and (b) a match that uses that node. Ancestors are deliberately not refreshed (`mamba_component.py:124-129`).

**Locks.**
- `inc_lock_ref(node)`: FULL locks the device-resident path up to the root (`full_component.py:263-300`). MAMBA locks only that node (`mamba_component.py:433-461`).
- A request holds the lock on `last_node` from admission, and again after the rematch in `cache_unfinished_req` (`unified_radix_cache.py:1016-1030`).
- `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK` skips the MAMBA lock during decode (`:1022-1028`).

**`mamba_max_states_per_path`** (default −1; `server_args.py:2595`; `mamba_component.py:252-308`).
- After each insert, it tombstones the shallowest single-child, unlocked, non-leaf, non-tail states beyond the cap.
- Forks and leaves are kept. It is a soft cap.

**Likely waste for this workload (INFERRED).**
- **(a) Decode states and the finish-insert.** They sit after the reminder. On a replace edge they are unreachable, yet they keep 18 MB plus the KV of reminder + output alive until LRU catches them. They are useful only on strict-append edges.
- **(b) Prompt-end states when no branch is present,** for example the first turn of a chain. These are also past the reminder.
- **(c) KV between the deepest state and this request's LCP.** It stays resident but this request cannot use it (§1); another branch may use it before eviction.
- **(d) One-turn staleness.** The branch tracked on turn N sits at `LCP(P_{N-1},P_N) = b_{N-1}`, not at `b_N`. So turn N+1 recomputes from `b_{N-1}`.
  - If turn N−1's KV past its state was freed, turn N sees no branch. It tracks its prompt end, and turn N+1 falls back even further.

## 5. flush

- `Scheduler.flush_cache` works only when fully idle (`managers/scheduler.py:4763-4792`). Otherwise it just logs a warning.
- When it runs it calls `tree_cache.reset()`. That rebuilds the root, LRUs and sizes and clears the host pool (`unified_radix_cache.py:355-384`, `unified_tree_core.py:440-494`).
- It also calls:
  - `req_to_token_pool.clear()`, which resets `mamba_allocator` to all slots and clears the int8 pool and the mappings (`memory_pool.py:1590-1604`, `allocator/mamba.py:93-96`);
  - `token_to_kv_pool_allocator.clear()`.
- **Everything is freed.** State tensors are not zeroed; a new request gets `mamba_needs_clear`.

## 6. extra_buffer_lazy vs extra_buffer

- **Lazy allocates only one ping-pong slot;** the other is −1 (`memory_pool.py:1471-1494`).
- Prefill skips the index swap (`schedule_batch.py:2828-2835`).
- At a decode boundary the second slot is allocated on demand, **without evict-retry**. If that fails, tracking falls back to writing in place (`schedule_batch.py:3197-3226`).
- After the step, `mamba_lazy_post_decode_at_boundary` frees the old slot (`batch_result_processor.py:1363-1376`).
- A finish can set `mamba_lazy_is_insert=False` and skip the insert (`:1139-1146,1192-1196`).
- The ratio is 4 instead of 5, which gives about 25% more running requests for the same slots. It needs no PD (`arg_groups/mamba_hook.py:110-115`) and overlap on (`kv_cache_configurator.py:2117-2122`).

## 7. Precision of tracked states

- The pool `temporal` is fp32. Triton KDA extend calls `chunk_kda_fwd` (`kernels/ops/attention/fla/kda.py:1083-1195`).
- **`h` is allocated as `k.new_empty(...)`** (`chunk_delta_h.py:349`), with `kg = empty_like(k)` (`kda.py:719`). So **h is bf16**.
- `h[i_t]` stores the fp32 accumulator, cast to bf16, *before* chunk `i_t` (`chunk_delta_h.py:152-172`). The final state is written to the pool at pool precision, fp32 (`:295-313`).

**bf16-rounded tracked states** (they then round-trip into the fp32 pool, `hybrid...:872-874`):
1. every **branch-point** state, because `_force_track_h` always sends it to h;
2. every **prompt-end** state whose extend length in that chunk is not a multiple of 64, which is about 63 in 64 prompts.

**Full fp32 states:**
- intermediate chunk ends, since the chunk size is 64-aligned and the state is the final one;
- prompt ends with an aligned tail;
- all decode-tracked states (pool copy);
- CoW copies.

With `--enable-int8-mamba-checkpoint`, cached states are int8.

Backends: FlashKDA/NVIDIA/CuteDSL have different `h` contracts (see the F-lines). On A100 the default Triton path applies.

## Current implementation and open questions

The preceding sections map the unpatched base. S0 uses 101 to track a role boundary and 140 to retain both the boundary and prompt-tail state, including an fp32 intermediate snapshot. Patch 140 also changes state retention priority. Read [their patch notes](../../../patches/README.md) before applying base-only suggestions from older reports.

The remaining measurable issue is how often a reachable state is evicted or skipped under pressure. R20 measured the maximum useful LCP gap on 026/N18 at 850,432 tokens (8.0% of actual prefill), so cache changes alone cannot be assumed to solve the N22 TPOT failure. Check each proposal against actual LCP, slot free/evict counts, skipped snapshots, KV use, and all 11 scoring gates. A lower-precision state or KV format also needs numerical and ability validation.
