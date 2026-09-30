# 179: device gather for ordinary Mamba prefill track slots

`SGLANG_AX_MAMBA_PREFILL_TRACK_GPU=1` enables a default-off path for ordinary
prefill with batch size >= 8 using the static `HybridReqToTokenPool`, non-lazy `extra_buffer`, CUDA
int64 slot mapping, and no speculative/mixed/dual-snapshot mode. Other modes
retain the original scalar slot read and upload.

`alloc_for_extend` first allocates the request slots and refreshes the pool
mapping. The new path freezes each request's `mamba_next_track_idx` before the
existing per-request helper updates it. The helper still calculates mask and
track length and updates Req fields at the same point. The existing device
gather reads the frozen positions from the authoritative pool mapping into a
batch-owned tensor. It leaves the decode-only `mamba_track_buffer_indices`
field unchanged. Slot donation and replacement continue through the existing
pool methods; scheduler code does not infer or translate physical slot IDs.

CPU mask and length values, sampling and allocation behavior are unchanged.
The enabled branch uploads mask/length through fresh per-batch pinned sources
with nonblocking H2D; ScheduleBatch.copy retains those immutable sources.
Pageable synchronous uploads would otherwise undo the avoided slot-read wait. Mapping or
position validation failure selects the original path. The gate checks host
metadata only; it does not inspect CUDA contents or capture new graphs.
Persistent host staging is not introduced: the reused gather helper creates
fresh pinned staging for each asynchronous position upload.

Validation receipts live in `evidence/execution-0930/tokenspeed-host-review/`.
Boundary timings are developer GPU0 measurements, with synthetic token bodies
and real CUDA slot buffers; they are not model or service performance proof.
Full preparation timing includes remaining mask/length upload behavior and is
required before deciding whether to deploy this flag. No Pod test or commit
has been made for this candidate.

Full preparation measurements retained the earlier synchronization in
HybridReqToTokenPool.alloc: CUDA mapping writes indexed by the host select_index
list (memory_pool.py:1394,1397). This candidate does not change allocation or
claim that the full preparation boundary becomes asynchronous. The measured
fixed overhead did not amortize for B1/B2; B4 was noise and B6 weak, so one
conservative >=8 threshold preserves their original path.
