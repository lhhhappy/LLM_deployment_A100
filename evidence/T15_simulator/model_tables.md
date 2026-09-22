**MODEL OUTPUT table: maximum passing tested N (dev ten gates plus TPOT).** Each cell is **stock / D1**. `≥30` means all tested levels through 30 pass and the ceiling is unobserved; `none` means no tested level passes. This is neither a measured capacity nor a confidence bound.

| Prefill tok/s | Base decode ms | FCFS dev stock / D1 | SPF dev stock / D1 | FCFS formal-mix WHAT-IF stock / D1 | SPF formal-mix WHAT-IF stock / D1 |
|---:|---:|---:|---:|---:|---:|
| 20,000 | 30 | 6 / 6 | 6 / 6 | 22 / 26 | 22 / 26 |
| 20,000 | 50 | 2 / 2 | 2 / 2 | 14 / 14 | 10 / 14 |
| 20,000 | 80 | 2 / 2 | 2 / 2 | 2 / 2 | 2 / 2 |
| 40,000 | 30 | 14 / 14 | 14 / 14 | ≥30 / ≥30 | ≥30 / ≥30 |
| 40,000 | 50 | 10 / 10 | 10 / 10 | 22 / 26 | 22 / 22 |
| 40,000 | 80 | 2 / 2 | 2 / 2 | 6 / 6 | 6 / 6 |
| 80,000 | 30 | 26 / 26 | 26 / 26 | ≥30 / ≥30 | ≥30 / ≥30 |
| 80,000 | 50 | 18 / 18 | 18 / 18 | ≥30 / ≥30 | ≥30 / ≥30 |
| 80,000 | 80 | 2 / 2 | 2 / 2 | 6 / 6 | 6 / 6 |

**MODEL OUTPUT table: which gate binds first.** All tests here use the same full-budget workload. “No TTFT fail ≤30” means none on the tested ladder, not a continuous-N guarantee.

| Mix / scheduler / cache | Prefill / decode | First failing N | Gates failing at that N | First TTFT-only failing N and gate |
|---|---|---:|---|---|
| dev / fcfs / stock | 20,000 / 30 ms | 10 | fast_intra + TPOT | 10: fast_intra |
| dev / spf / stock | 20,000 / 30 ms | 10 | TPOT | No TTFT fail ≤30 |
| dev / fcfs / D1 | 20,000 / 30 ms | 10 | fast_intra + TPOT | 10: fast_intra |
| WHAT-IF formal-mix / fcfs / stock | 20,000 / 30 ms | 26 | TPOT | No TTFT fail ≤30 |
| WHAT-IF formal-mix / fcfs / D1 | 20,000 / 30 ms | 30 | TPOT | No TTFT fail ≤30 |
| dev / fcfs / stock | 40,000 / 50 ms | 14 | TPOT | No TTFT fail ≤30 |
| WHAT-IF formal-mix / fcfs / stock | 40,000 / 50 ms | 26 | TPOT | No TTFT fail ≤30 |
| WHAT-IF formal-mix / fcfs / D1 | 40,000 / 50 ms | 30 | TPOT | No TTFT fail ≤30 |

At 20k/30 ms with FCFS, the dev stock and D1 proxies first fail fast_intra and TPOT together at N=10. Replacing the load composition with the synthetic formal mix moves the first stock failure to TPOT at N=26 (D1: N=30). This isolates a possible cold-prefill-load effect; it does not establish formal behavior. At 40k/50 ms, dev first fails TPOT at N=14, while the formal-mix stock/D1 first failures are N=26/N=30. SPF is not uniformly better: the formal-mix 20k/50 ms stock ceiling falls from 14 to 10, and the 40k/50 ms D1 ceiling falls from 26 to 22. The ordering changes feedback and decode stalls.

**MODEL OUTPUT table: per-request aggregate diagnostics for the representative 40k/50 ms FCFS runs.** Times are simulated seconds except wall minutes. All four runs pass dev plus TPOT.

| Cache | N | fast TTFT p95 | overall TTFT p95 | turn TTFT p95 | chain TTFT p95 | TPOT p95 | Wall min |
|---|---:|---:|---:|---:|---:|---:|---:|
| stock proxy | 6 | 0.2819 | 1.0586 | 2.8839 | 3.1086 | 0.0811 | 47.96 |
| stock proxy | 10 | 0.8394 | 1.1085 | 4.4080 | 3.2232 | 0.0973 | 33.64 |
| D1 proxy | 6 | 0.1321 | 0.9244 | 1.1004 | 3.0735 | 0.0787 | 47.70 |
| D1 proxy | 10 | 0.7458 | 1.0800 | 1.1029 | 3.2163 | 0.0978 | 33.36 |

**MODEL OUTPUT table: output-length sensitivity on dev, FCFS; maximum passing tested N.** The source-length variant contains 167,606 modeled output tokens after budget clipping and 12 missing-count budget fallbacks. All other settings are unchanged.

| Prefill tok/s | Decode ms | Full-budget stock / D1 | Source-length proxy stock / D1 |
|---:|---:|---:|---:|
| 20,000 | 30 | 6 / 6 | 2 / 6 |
| 20,000 | 50 | 2 / 2 | 2 / 2 |
| 20,000 | 80 | 2 / 2 | none / none |
| 40,000 | 30 | 14 / 14 | 10 / 10 |
| 40,000 | 50 | 10 / 10 | 6 / 6 |
| 40,000 | 80 | 2 / 2 | 2 / 2 |
| 80,000 | 30 | 26 / 26 | 22 / 22 |
| 80,000 | 50 | 18 / 18 | 14 / 14 |
| 80,000 | 80 | 2 / 2 | 2 / 2 |

Shorter modeled outputs can reduce the passing N: workers return to prefill sooner and each prefill stall is diluted over fewer output tokens in request-average TPOT. This is why measured output lengths are required. All 90 completed grid groups were monotone in their pass/fail sequence across this tested ladder; the simulator still checks and reports nonmonotonic cases rather than assuming they cannot occur.
