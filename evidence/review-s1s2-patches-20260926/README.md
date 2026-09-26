# S1/S2 patch review evidence — 2026-09-26

Status: **DIAGNOSTIC**. This directory contains source inventory, reproducible CPU counterexamples, paired CPU/GPU measurements and complete single-GPU probe logs. It is not an original-harness full-cohort score.

The [review report](../../notes/reports/review-s1s2-patches-0926.md) is the interpretation and per-commit coverage ledger. [manifest.json](manifest.json) records the six engine commits, producer/source hashes, commands, test data, warmup and coverage limits. [SHA256SUMS](SHA256SUMS) covers every evidence file except itself.

Use `optimized-tests.log`, `optimized-cpu-benchmark-final.json`, and `gpu/summary.json` for final validation. `existing-tests.log`, `new-regressions.log`, `family-cpu.json`, `optimized-cpu-benchmark.json`, and earlier GPU log versions are preserved iteration receipts with narrower coverage or older source hashes. No full-model or official score can be inferred from them. `gpu/log-sha256.json` and `gpu/transfer-receipt.json` verify copies against the original developer-machine logs.

The original historical run files were read only; `comparisons.json` records their paths and SHA256 and the neighboring CSVs preserve per-request comparisons. Window comparisons do not replace complete original raw/service/harness records. The finalized evidence and report are also copied to the developer-machine archive listed in the manifest.
