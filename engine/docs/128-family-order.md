# 128 — rank a prefix-sharing family by its work per request (candidate, inside 124)

## Why
Opening probe 104 (v3, N26, 124 on) still missed 13 of the 27 first-minute chain starts. Its opening table
(`evidence/L104-v3_open_124_n26/window/raw.jsonl`) shows 124 running small starts first, and then families of
chain starts arriving within 1–4 s yet starting at 41–56 s: family 62bf0fac had four starts of 35–38k tokens
sharing about 31k (the same system prompt and tool definitions). LPM holds the followers back until one of them,
the leader, has computed the shared prefix; 124 ranked that leader by its own 35k of work, behind single starts of
14–19k, so the whole family missed. After the leader, each follower has only 4–7k left.

The opening queue model (arrival-order or shortest-first execution at the measured prefill rate) reproduced the
measured misses of 081, 074, 103 and 104 (21.4/22, 17.4/18, 16/17, 12/13). On 104's own work and shared
prefixes (25 starts in 11 families of sizes 4/4/4/3/3/2/1×5, 10.3k tok/s) it puts per-request shortest-first at
12 misses and ranking families by work per request at 7. These are estimates; the mechanism is judged on the
pod against 104 on the same request IDs.

## What it does
Off unless `SGLANG_AX_DEADLINE_FAMILY=1`; requires 124 (`SGLANG_AX_DEADLINE_TIERS=1`) and `--schedule-policy lpm`,
and refuses to start otherwise.

- `SchedulePolicy._compute_prefix_matches` (base LPM in-batch prefix sharing) additionally records, for each
  request it holds back, how many leading tokens it shares with a request queued before it (`ax_shared`). Nothing
  else changes there; with 128 off the record is unused.
- `ax_deadline.link_families` maps each held request to the first waiting request that is not held back and whose
  prompt starts with the same tokens (token-by-token comparison of the shared length; links are cached per held
  request while the leader still waits, since prompts do not change).
- `ax_deadline.family_unit_work`: for each leader, (its remaining work + each follower's remaining work minus the
  shared tokens) / family size.
- `ax_deadline.tier_order` ranks a leader by that value instead of its own remaining work. Tiers are unchanged:
  whether a leader can still make its budget is judged on its own work; followers stay last, as LPM holds them;
  once the leader is admitted they match the cache and rank by their small remaining work.
- Computed on request-plane rank 0 inside 124's plan and broadcast with it. The 30 s log line adds
  `family_leader_rounds` (plan rounds × leaders ranked by family work); the mechanism line shows `128=on`.

## Known limits
- A family is what LPM's in-batch check sees: prompts sharing at least `IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD`
  (32) tokens with an earlier waiting request whose own cache match was at most the check threshold. Requests that
  already hit the real cache are not grouped.
- The family value assumes the followers will be served right after the leader; it does not model their own budgets.
- Comparing prefixes costs up to the shared length per held request once (then cached).

## Evidence
CPU tests: `tests/test_ax_deadline.py` (`FamilyOrder`: linking by token content, family work, ordering against a
smaller single start, a hopeless leader staying hopeless, link reuse) and `tests/test_ax_admission_scheduler.py`
(`FamilyOrder`: the real admission plan admits the leader first with 128 on and the smaller single start with it
off; refusals without 124 or LPM). Pod: pending (v3 N26 opening probe, single change versus 104).
