# T29 — E2 v1.1 residual checks

Random qfull Kimi-Linear, TP1, page64/chunk8192/extra_buffer, D0+001 v1.1.
Not a full GLM, numerical-correctness, or performance/SLO result.

- `cpu_tests.log`: initial 10 CPU checks.
- `cpu_tests_v2.log`: 11 checks after adding the missing-workload counterexample.
- `first/if08.json`: W6 live IF-08 pass, including busy failure, waiting success,
  and post-flush zero cached tokens.
- `first/completion_report.json`: D1-10 pass with actual previous role split at
  90624 and post-flush cached=0. **Its D1-08 `pass` is INVALID**: the mixed replay
  never started because a 256733-token prompt exceeds context131072. Do not
  promote that field to a test verdict. D1-05 likewise did not exercise N4.
- `first/mixed_replay.log`: retained pre-send rejection; no prompt truncation.
- `first/hashes.txt`: original model/config/source/diagnostic hashes.

Remote first run: `/sjtu/linhang/arena/runs/E2_T29_20260922/`.
Mixed-only follow-up: `E2_T29_20260922_large_v2`, context262144/KV524288,
same weights/patches/other flags. `large/` contains the completion report,
manifest, summary, mixed case IDs and complete 144KB scheduler trace:
105 requests/0 errors; 306 workload rounds, max partial1; pools restored.
Stats105 equal new admissions105, not attempts215 or all commits334. The tool's
strict all-commits gate reports fail; TEST_PLAN keeps D1-05 overall incomplete
pending an explicit denominator. No second partial or crash was observed.
`cpu_regression.log`: 50 CPU/mock checks pass.
The intervening `E2_T29_20260922_large` launch refused because the original
service had not yet stopped; it is not a run. No other sessions were terminated.

Use `notes/experiments.md` T29 and TEST_PLAN for the qualified final verdicts.
