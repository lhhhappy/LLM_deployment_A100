# Pod execution audit, 2026-09-28

Read-only diagnostic of six archived, drained 40-minute admission runs. Not a full-cohort or official score.

- Analysis and conclusions: [report](../../notes/reports/pod-execution-0928.md).
- Reproducible tool: [pod_execution_audit.py](../../scripts/analysis/pod_execution_audit.py).
- `audit.json`: source paths and hashes, runtime configuration, window statistics, common-ID comparisons, per-request failures, cache pressure and decode/prefill overlap cases.
- `*-chain.csv`: all chain heads with lifecycle intervals. These intervals are not isolated GPU kernel time.
- `*-timeline.csv`: 10-second bins of logged prefill work, GPU samples, KV usage and queue length.
- `validation.txt`: agreement with archived harness summaries and original percentile method.
- `cap-validity-audit.json`: capability counts separated into completed, transport errors, truncated and wrong nontruncated responses. The latest capability job was interrupted; its aggregate percentages are not a complete quality verdict.
- `live-*.txt/json`: timestamped Pod snapshots. Text captures may end with the transport's `exit_code: 0` footer.

Source archives remain in `/workspace/Agentic_science_challenge/evidence/L130ezn*/`; source identity is recorded in `audit.json`. No GPU activity was initiated and no Pod processes were stopped. Live sampler snapshot only examines the identified metrics processes and their output files.
