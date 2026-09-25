# 120: preserve continuation checkpoint alignment under KV pressure

Follow-up to 120, based on frozen 759a6eb for the 071/072 comparison.
When the continuation budget is smaller than the protection grid and cannot
finish the request, keep the existing partial parked. The scheduler can run
decode and reclaim capacity; a later continuation retains its aligned prefix.
A complete short final chunk is still allowed. No second partial, extra memory
reservation, cache lookup, or checkpoint-coordinate rounding is introduced.
Protection off retains the original policy. This prevents new misalignment;
it does not repair an already misaligned prefix or change cache eviction.

072 enables `SGLANG_AX_CHUNK_ALIGNMENT_TRACE=1` (default 0). CPU metadata
observations on TP0 record the first sub-grid deferral, resume with deferred
decision count and elapsed time, the first subsequent planned checkpoint, and
the final prefill checkpoint of affected requests. An unexpected unaligned
prefix is recorded once per request, plus its final checkpoint plan.
The 2 MiB lifetime output budget includes per-line overhead; exhaustion is
explicit. No per-token logs, GPU reads, synchronization or new cache buffers.
`checkpoint_plan` means selected in batch preparation, not publication or H2D
completion. Elapsed deferral time is wall time, not GPU service time.

Per the user's current workflow, validation is code review followed directly
by TP8: same host64/122on, N30, frozen 5601 requests, rep16 warmup and real flush
as 071. Check progress after deferral, checkpoint positions, actual cache reuse,
errors and every SLO. No CPU probe is an admission prerequisite. No performance
improvement is claimed before the complete run.
