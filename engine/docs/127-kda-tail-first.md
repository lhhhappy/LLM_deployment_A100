# 127 — keep the prompt-end KDA checkpoint over the branch point in a request's last prefill chunk (candidate)

## Why
Prefill saves one KDA state per extend (`ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend`): at the
extend end rounded down to the checkpoint grid (256 with HiCache and the compressed DSA index), unless the
request's radix match found a full-KV hit deeper than its deepest KDA state (`req.mamba_branching_seqlen`), in which
case the branch point replaces the extend-end checkpoint (base behaviour; 140's dual snapshot, which would keep
both, is off under MTP and HiCache). In chains, the next request resumes at the previous prompt's end (4439 of 5601
transitions in the long-chain set end exactly there), so a branch point saved in a request's last chunk leaves the
next request with KV up to the prompt end but a KDA state only at the branch point: it recomputes that span and in
turn saves a branch point instead of its own prompt end, and the lag propagates. Run 071 (N30): about 1.66M
prefill tokens (~9% of prefill) recomputed this way (R30, `research/claude/R30_071_online_bottlenecks.md`).

## What it does
Off unless `SGLANG_AX_KDA_TAIL_FIRST=1`. In the extend that reaches a fresh request's fill end (its last prefill
chunk; a retracted request, whose fill includes output no successor resumes from, keeps the base choice), a branch
point at or beyond `SGLANG_AX_KDA_TAIL_FIRST_MIN_SHARE` (default 0.5) of the prompt no longer replaces the
prompt-end checkpoint. The two kinds of branch point differ:
- deep (the tree's KV covers most of the prompt): in chains, where the previous request of the same chain stopped
  matching, typically where its generated output diverges from the replayed turn; no later request resumes there,
  while the next request of this chain resumes at this prompt's end — the prompt end wins;
- shallow (a shared system prompt or tools, as among chain starts): siblings fork there — the branch point is kept.
Earlier chunks keep the branch point as before. It chooses between two positions the code already tracks, through the
same `_force_track_h` path: no new state memory, no extra round, no kernel change; decode tracking (and MTP verify)
is untouched; the choice depends only on request fields, identical on every TP rank. Because
`cache_unfinished_req` inserts KV only up to the tracked depth, the prompt's KV is published (and written through to
the host tier) at prefill instead of at finish.
Counter `schedule_batch.AX_TAIL_FIRST_STATS["kept"]` (only when the saved position changes), logged at 1, 2, 4, …;
the mechanism line shows `127=on`.

## Evidence
CPU regression on the real radix cache, pools and match (`scripts/tests/hicache180/test_127_tail_first.py`, run on the dev box where torch is installed): see the test's cases.
Not validated: TP8, the recomputed-token reduction (R30's cache ledger on raw `cached_tokens`), chain-start reuse
lost to siblings, and the effect on the gates.
