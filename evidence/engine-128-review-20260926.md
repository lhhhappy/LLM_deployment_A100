# Engine 128 review (3aad9c4f)

Verdict: **PASS for the v3/N26 opening probe** against 104. This is a scheduling hypothesis, not a measured improvement yet.

The LPM holdback path records each follower's shared prefix length. `link_families` now requires matching token prefix, `extra_key`, and `cache_salt`, consistent with `RadixKey`; the first revision (`dd64e9a8`) could credit a different cache namespace and was corrected in `3aad9c4f`. Links are cleared on a successful `/flush_cache`. The unit-work value changes only the within-tier order of a family leader; 124 still judges whether that leader can meet its deadline using its own remaining work. Rank 0 computes the plan and broadcasts the RID order, so rank-local clocks cannot split TP ranks. Enabling 128 without 124 or LPM is rejected at startup. With 128 off, the new shared-length record is unused.

Verified in the frozen worktree: `test_ax_deadline.py` 32/32, `test_ax_admission_scheduler.py` 24/24, `test_sched_protect_chain.py` 32/32, and `git diff --check 7c6cb634 3aad9c4f`. The probe script `v3_open_124_128_n26.sh` differs from 104 only by engine commit, `SGLANG_AX_DEADLINE_FAMILY=1`, and its mechanism expectation.

Limits to measure on the Pod: LPM observes only in-batch holdbacks; real cache hits do not enter a family. Family unit work assumes followers run after the leader and does not check each follower's deadline. Its first prefix comparison costs CPU time; the waiting-queue policy falls back to FCFS above 128 requests. Assess chain repairs and new misses on identical request IDs, plus turn/TPOT and scheduler wall time, before retaining it.
