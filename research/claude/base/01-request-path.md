# 01 — Request path: HTTP → TokenizerManager → Scheduler → back

Source: `build/base_exact/sglang` (fe236ea6c3). All paths below are relative to `sglang/`. Only the source was read; nothing was run against a live server. Each claim is marked VERIFIED (read in code) or unverified.

## Process topology (launch command as given, dp_size=1)
- The HTTP server is uvicorn with uvloop and `workers=tokenizer_worker_num` (default 1) (`srt/entrypoints/http_server.py:2726-2737`, default `srt/server_args.py:429-431`). The TokenizerManager runs in that **same single process and event loop** (`srt/entrypoints/engine.py:174`).
- With dp_size=1 there is **no DataParallelController**. It is used only if `dp_size>1` or EP scale-join (`srt/entrypoints/engine.py:852-854`). Tokenizer PUSH → TP rank 0 scheduler PULL on `scheduler_input_ipc_name` (`srt/managers/tokenizer_manager.py:562-564`, `srt/managers/scheduler_components/ipc_channels.py:36-38`).
- Scheduler rank 0 → DetokenizerManager process (`ipc_channels.py:62-65`) → TokenizerManager (`srt/managers/detokenizer_manager.py:131-133,177-185`).

## 1. /generate
- **Handler:** `generate_request` in `srt/entrypoints/http_server.py:900-951`. FastAPI parses the body directly into the `GenerateReqInput` dataclass. If stream=true, each chunk is sent as `b"data: " + orjson(out) + b"\n\n"`, followed by `data: [DONE]` (`:914-936`). Errors come as a `data: {"error":{...}}` chunk (`:925-934`). A background task runs 2 s after the stream ends and aborts the rid if it is still live (`srt/managers/tokenizer_manager.py:2154-2165`).
- **GenerateReqInput fields** (`srt/managers/io_struct.py:173-352`):
  - `rid` (kw_only, `:177`) and `session_id` (a stable identity that does not alter the prompt, `:180`)
  - `session_params` (`:262`) and `routed_dp_rank` / deprecated `data_parallel_rank` (`:286-291`)
  - `routing_key` (`:295`), `conversation_id` (`:297`), `priority` (`:311`), `extra_key` (`:313`)
  - `received_time` (`:329`) and `cache_salt` (cache namespace, `:352`)
  - A duplicate live rid raises `ValueError("Duplicate request ID")` (`tokenizer_manager.py:3455`).
  - Caveat: batch `__getitem__` does **not** copy `routing_key` (`io_struct.py:876-966`). Single requests are unaffected.
- **Header overrides:** `apply_header_overrides` runs only if env `SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES=1` (default False, `srt/environ.py:323`; gate at `http_server.py:909-910`). The map is `srt/entrypoints/request_headers.py:10-19`: `x-override-{rid, bootstrap-host/port/room, conversation-id, routed-dp-rank, disagg-prefill-dp-rank, priority}`. There is **no routing_key entry and no X-S1-\* header anywhere in srt/** (grep, VERIFIED).
- **meta_info** is built in `_handle_batch_output` (`tokenizer_manager.py:2234-2240`):
  - Always present: `id`, `finish_reason`, `prompt_tokens`, `weight_version`, `num_retractions`, and for text output `reasoning_tokens`, `completion_tokens`, `cached_tokens` (`:2280-2285`), plus `cached_tokens_details` when available.
  - With `--enable-metrics` (on in our launch), each chunk also gets scheduler stats `forward_entry_time`, `prefill_finished_time`, `queue_time` (`srt/observability/req_time_stats.py:1167-1181`, merged at `tokenizer_manager.py:2243-2245`).
  - `response_sent_to_client_ts` is written **only on the first yielded chunk** (`:1803-1808`; finish path `:1773-1778`).
  - The final chunk adds `e2e_latency` (`:2469`), spec metrics (`:2472`), `request_received_ts`, `api_server_dispatch_finish_ts`, `response_sent_to_client_ts`, `request_finished_ts` and `decode_throughput` (`req_time_stats.py:488-515`).
- **Where the timestamps are set:**

  | Timestamp | Where it is set | Note |
  |---|---|---|
  | `created_time` | `_init_req_state` → `set_created_time(obj.received_time)` (`tokenizer_manager.py:3423,3461`) | `/generate` never sets `received_time` (only OpenAI/Anthropic do, `srt/entrypoints/openai/serving_base.py:79,99`). So `request_received_ts` is taken **after body parse and `normalize_batch_and_arguments`**, not at socket accept. |
  | `tokenize_finish_time` | `:1449` | |
  | dispatch / dispatch_finish | `:1568,1576` | |
  | `wait_queue_entry_time` | scheduler (`srt/managers/scheduler.py:3050`) | |
  | `forward_entry_time` | first scheduled chunk (`scheduler.py:3795`, `req_time_stats.py:759-762`) | |
  | `prefill_finished_time` | `srt/managers/scheduler_components/batch_result_processor.py:316,423` | |
  | `first_token_time` | when the tokenizer receives the first output (`tokenizer_manager.py:2460-2461`) | Not emitted in meta_info. |

  All values are perf_counter converted with a per-process offset (`req_time_stats.py:70-73`).

## 2. TokenizerManager
- **Where tokenization happens:**
  - Call chain: `generate_request` (`:773`) → `_tokenize_one_request` (`:962`) → `_tokenize_texts` (`:890-960`).
  - Default path: `self.tokenizer(tokenizer_input)` runs **synchronously on the asyncio event loop** (`:926-938`). There is no executor.
  - `--enable-dynamic-batch-tokenizer` (default False, `srt/server_args.py:3507-3511`) routes single strings to `AsyncDynamicbatchTokenizer`, a **1-thread** ThreadPoolExecutor with batching of up to 32 and a 2 ms wait (`srt/managers/async_dynamic_batch_tokenizer.py:23,44`; `tokenizer_manager.py:910-924`). This frees the loop but stays serial.
  - `--tokenizer-backend fastokens` is also available (`server_args.py:419-428`).
- **Cost (unverified, not measured):**
  - HF fast (Rust) BPE typically runs at a few hundred k tokens/s single-threaded. Guess: roughly 0.05–0.3 s for 36k tokens and 0.4–2 s for 250k tokens.
  - While it runs, the loop is blocked for **every** request: no SSE is sent, no detokenizer output is drained and no other request is tokenized. That inflates the TTFT of concurrent arrivals and the TPOT of streams already running.
  - Also sent: `input_text` (the whole string) plus `array('q')` ids, about 8 B/token, i.e. ~2 MB ids + ~1 MB text at 250k (`:1347-1349,1381`).
  - Measure on the GPU box with `s1-dev/glm_tok/tokenizer.json`. I could not measure locally: there is no usable `tokenizers` build.
- **Tokenization cache:** none. Each request fully re-encodes its text. There is no prefix-aware or cross-turn tokenization cache; the only cache is `mm_processor`'s preprocess cache, which applies only to multimodal inputs (VERIFIED by reading `_tokenize_texts`).
- **Send to scheduler:**
  - `_send_one_request` (`:1558-1579`) → `_dispatch_to_scheduler` → `sock_send` over a ZMQ PUSH socket (`:578-581`).
  - IPC is **pickle by default**: `SGLANG_USE_PICKLE_IPC=True` (`srt/environ.py:349`) makes `send_pyobj(protocol=HIGHEST)` the wire format; setting it to False switches to msgpack (`srt/managers/io_struct.py:2468-2473`). The send is synchronous on the loop.
  - Rank 0 drains the socket non-blocking with no per-poll cap (`SGLANG_SCHEDULER_MAX_RECV_PER_POLL=-1`; `srt/managers/scheduler_components/request_receiver.py:111-137`).
  - Rank 0 then **re-pickles and gloo-broadcasts the whole request list to the other 7 TP ranks** (`request_receiver.py:215-221` → `srt/utils/common.py:2467-2512`), and each rank unpickles it.
- **`--incremental-streaming-output`** (`server_args.py:1436`, read at `tokenizer_manager.py:426`):
  - Each chunk carries delta `text` and `output_ids`, and the per-token meta keys are sliced to the delta (`:2356-2375`, `:309-319`).
  - If several chunks queue up before the HTTP coroutine wakes, they are coalesced (`:1744-1751`, `:1621-1656`).
  - Without the flag, intermediate chunks defer the full text (`:2384-2391`).
  - `stream_interval` defaults to 1 (`server_args.py:1421-1425`), so the first token is streamed immediately.
- **Detokenizer path:**
  - The scheduler's `output_streamer` sends `BatchTokenIDOutput` to the DetokenizerManager process, which runs one blocking loop (`detokenizer_manager.py:177-185`) with incremental decode (`:301-420`).
  - It returns `BatchStrOutput` to the TokenizerManager, where `handle_loop` (`tokenizer_manager.py:2192-2205`) calls `_handle_batch_output`.
  - The TokenizerManager sets request events in batches of `batch_notify_size=16` and yields with `sleep(0)` (`:2500-2504`, `server_args.py:1426-1430`).

## 3. /flush_cache
- **Handler:** `http_server.py:977-991`, GET or POST, with an optional `?timeout=` (default 0.0).
- **Return format (VERIFIED): it does NOT return JSON.** On success it is a plain-text `Response` with body "Cache flushed.\nPlease check backend logs..." and status 200. On failure the body is `ret.message` or "Flush cache failed." with **status 400**. The harness's `{"success": true}` check will fail against stock.
- **Fan-out:** `TokenizerManager.flush_cache` (`srt/managers/tokenizer_control_mixin.py:304-312`) awaits a `FanOutCommunicator` with `fan_out = dp_size` (`:161-173`). It returns **only `[0]`**, so with DP>1 another rank's failure is masked.
- **Scheduler side:**
  - The request is a control message broadcast to all TP ranks.
  - `SchedulerFlushWrapper.handle` (`srt/managers/scheduler_components/flush_wrapper.py:24-38`): with timeout 0 it flushes **only if `is_fully_idle()`**, otherwise it returns `success=False` immediately (`scheduler.py:4763-4791`).
  - With timeout>0 it defers the flush and re-checks every loop (`flush_wrapper.py:40-64`, called at `scheduler.py:2025`).
  - Only rank 0 has a real `send_to_tokenizer`; other ranks use `SenderWrapper(None)` (`ipc_channels.py:67-73`).
- **What gets cleared (`scheduler.py:4765-4779`):**
  - `tree_cache.reset()`, which for MambaRadixCache rebuilds the root and resets the full and mamba LRU lists and counters (`srt/mem_cache/mamba_radix_cache.py:486-500`).
  - `req_to_token_pool.clear()`, i.e. `HybridReqToTokenPool.clear`: free slots, the mamba allocator and the int8 checkpoint pool (`srt/mem_cache/memory_pool.py:1591-1600`).
  - `token_to_kv_pool_allocator.clear()`, `reset_aux_cache_allocator()`, the grammar cache, metrics, the draft (MTP) cache pool (`clear_cache_pool`), and `empty_cache()`.
  - State tensors are freed via the allocators, not zeroed (unverified that no zeroing happens elsewhere).

## 4. routing_key flow
The field flows end to end:

`GenerateReqInput.routing_key` (`io_struct.py:295`) → `TokenizedGenerateReqInput.routing_key` (`tokenizer_manager.py:1413`; `io_struct.py:1036`) → `Req(routing_key=…)` (`scheduler.py:2673`; `srt/managers/schedule_batch.py:1026`).

Where it is consumed:
- Only with `--schedule-policy routing-key`, which sorts the waiting queue by that key's count in the running batch (`srt/managers/schedule_policy.py:287-289,439-469`).
- In metrics (`srt/managers/scheduler_components/metrics_reporter.py:980-991`).
- In the session controller (`srt/session/session_controller.py:313`).

The DP controller never reads it.

What can set it:
- **/generate can set it from a JSON body field** `"routing_key"`. It is a plain dataclass field, so FastAPI populates it (VERIFIED).
- **No header maps to it for /generate.** `x-smg-routing-key` maps only in the OpenAI/Anthropic endpoints (`srt/entrypoints/openai/serving_base.py:259-262`, used at `srt/entrypoints/openai/serving_chat.py:1162`) and in a Prometheus middleware gauge (`srt/utils/common.py:2695-2718`).
- `X-S1-Routing-Key` is ignored.
- Under our `--schedule-policy lpm` the key has **no scheduling effect**. Also, LPM falls back to FCFS when the queue holds more than 128 requests (`schedule_policy.py:294`).

## 5. DP (dp_size>1, DP attention)
- **Inbound path:** TokenizerManager → the DataParallelController process PULL (`srt/managers/data_parallel_controller.py:156-159`) → one PUSH per DP rank (`:579-584`). This is **another full unpickle and re-pickle hop**.
- **Event loop:** a busy-spin with non-blocking recv and no sleep (`:806-815`).
- **Methods** (`:85-91`):
  - `auto` resolves to `round_robin` unless PD prefill (`srt/arg_groups/serving_hook.py:200-212`).
  - `round_robin` (`:759-778`), `follow_bootstrap_room` (`:780-789`), `total_requests` (`:791-795`), `total_tokens` (`:797-804`, cost = `len(input_ids)`).
- **`routed_dp_rank`:** every method first calls `maybe_external_dp_rank_routing` (`:744-757`), which sends directly to `workers[rank]`. The rank is validated in the tokenizer (`tokenizer_manager.py:792-802`). It can be set from the body or from the `x-override-routed-dp-rank` header (with the env set).
- **Load snapshots:**
  - Schedulers write through `ShmLoadSnapshotWriter` to /dev/shm (`srt/managers/load_snapshot.py:314-370`) from `publish_load_snapshot` (`scheduler.py:839-857`).
  - A write is forced on every extend batch and every `load_snapshot_publish_interval=15` decode iterations (`server_args.py:1520-1524`, `scheduler.py:4398`), and on stalls with a wall-clock floor (`scheduler.py:4528-4536`).
  - The controller reads at most once every 20 ms (`data_parallel_controller.py:306-321`). In between it adds speculative +1 request / +tokens (`:120-135`) and skips stale timestamps (`:109-118`).
  - These snapshots are used only by `total_*`.
- **Minimal "route by key" change:**
  1. Add a `ROUTING_KEY` member to `LoadBalanceMethod` (`:85`).
  2. Add `routing_key_scheduler(req)` in `data_parallel_controller.py`: call `maybe_external_dp_rank_routing` first, then `active[zlib.crc32(req.routing_key) % len(active)]`, and fall back to `total_tokens` when the key is None. Register it in `dispatch_lookup` (`:164-169`).
  3. Add the choice string at `server_args.py:981-987`.
  4. Populate `routing_key` from the header (Lever L3).

  `routing_key` already reaches the controller inside `TokenizedGenerateReqInput`. A cheaper alternative with no controller code: the proxy or HTTP layer computes `routed_dp_rank = hash(key) % dp` and sets `obj.routed_dp_rank` in `generate_request` (`http_server.py:907`).

## 6. TTFT latency sources in this path
1. **Single event loop doing tokenization synchronously** (`tokenizer_manager.py:926-938`). Arrivals of 36k–250k-token prompts serialize here, and output streaming stalls during each encode. This is likely the largest front-end cost (unverified, estimated at 0.1–2 s per long prompt).
2. **Pickle and IPC of about 3 MB per 250k-token request:** tokenizer → rank 0, then a gloo broadcast to 7 ranks (`common.py:2467-2512`) inside the scheduler's `recv_requests`, which **blocks the scheduler loop**. Estimated tens of ms per long request (unverified). With DP there is one more hop.
3. **Request pickup waits for the scheduler iteration:** a new request is only received between forward steps (`request_receiver.py:82-108`), so queue time is bounded below by the current batch time (e.g. a long chunked-prefill step).
4. `request_received_ts` omits the body JSON parse (~ms per MB, unverified) and the time spent waiting for the event loop, because it is stamped late (§1). The TTFT the server reports therefore understates what the client sees when the loop is congested.
5. The detokenizer is a single process. A second parse also happens in `_handle_batch_output` (cheap per token).

## Levers
- **L1 — JSON `/flush_cache` (required).** In `http_server.py:977-991` `flush_cache`, return `ORJSONResponse({"success": ret.success, "message": ret.message})`, and default `timeout` to e.g. 30 s so that a level boundary with draining requests waits instead of returning 400. In `tokenizer_control_mixin.py:304-312`, reduce with `all(r.success for r in results)` instead of `[0]`.
- **L2 — take tokenization off the loop.**
  - Minimal: in `_tokenize_texts` (`tokenizer_manager.py:926-938`), wrap the encode in `await loop.run_in_executor(pool, …)` using a multi-thread pool. HF Rust releases the GIL (unverified for this build).
  - Or launch with `--enable-dynamic-batch-tokenizer`: no code change, but only one thread.
  - Or `--tokenizer-worker-num N` (multi-process; check that `/flush_cache` and metrics still behave).
  - Or `--tokenizer-backend fastokens` if the package is present.
- **L3 — honor X-S1 headers.** In `request_headers.py:10-19`, add `"x-s1-routing-key": ("routing_key", str)` (and optionally `"x-s1-session-id": ("session_id", str)`), and launch with `SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES=1`. Or set these unconditionally in `generate_request` (`http_server.py:907-910`).
- **L4 — route by key across DP:** see §5 (`data_parallel_controller.py` `LoadBalanceMethod`, a new `routing_key_scheduler`, `dispatch_lookup`; plus `server_args.py:981-987`). This keeps prefix families on one radix/mamba cache.
- **L5 — prefix-cache-friendly scheduling.** `--schedule-policy routing-key` is available once L3 is in (`schedule_policy.py:439`), but it drops LPM. A hybrid would need edits to `SchedulePolicy.calc_priority` (`:240-291`).
- **L6 — shrink IPC.**
  - In `_create_tokenized_object` (`tokenizer_manager.py:1381`), drop `input_text` when the text is not needed downstream. Unverified: `Req.origin_input_text` may be used by some features.
  - Try `SGLANG_USE_PICKLE_IPC=0` (msgpack) and measure.
- **L7 — truthful timestamps.** In `generate_request` (`http_server.py:907`), set `obj.received_time = time.perf_counter()` at handler entry, mirroring `serving_base.py:79,99`, so that `request_received_ts` includes front-end queueing.
- **L8 (unverified, large).** `SGLANG_RUST_SERVER=1` (`srt/environ.py:1622`, `srt/rust_server/server.py`) replaces the Python api-server, tokenizer and detokenizer with Rust threads inside the scheduler process. Its `/generate`, `/flush_cache`, meta_info and header behavior were not checked.
