# R7 — Top players' public scorecards: progression, gate budgets, implications

> **2026-09-22 更新（data/all_att_2026-09-22b.json，523 条）**：N=22 已有 **3 人**，下文"only N=22"过时。
> | 选手 | id | fast | overall | turn | chain | tpot_mean | tpm |
> |---|---|---|---|---|---|---|---|
> | LewyM | 45417 | 1.79 | 3.00 | 10.9 | **61.0** | **0.0273** | 1.75M |
> | Jinbo hu | 45461 | **1.61** | **2.35** | 8.2 | **29.4** | 0.037 | 1.65M |
> | Mingjun Xu | 45443 | 2.11 | 2.67 | 7.3 | 30.2 | 0.051 | 1.37M |
> 两条路：LewyM 牺牲 chain_start（用统计余量）；Jinbo hu 各门都在限内、intra 最低——说明不靠饿死冷启动也能到 22。
> 同分排名看 tpot_mean：当前第一是 LewyM 0.0273。要夺第一：N=26，或 N=22 且 tpot_mean < 0.0273。
> §5 中"D2 SPF"一行的具体做法（移植 #40024）已作废（底包无 SPF，决策 29）；其"冷启动让位于 intra"的结论改由主线 M1 在底包上实现。
> 本分析里的"配额 3 次/天"以用户确认的 **2 次/天** 为准。

Claude, 2026-09-22. Source: only public `data/all_att.json` (494 attempts). Tables: `data/analysis/r7_flat_attempts.json` (all attempts flattened), `data/analysis/r7_top_players.json` (per-player attempts, gaps, gate budgets, correlations). **INFERRED** = our reading of scores. We have not seen anyone's code or images.

**Caveats.** Scorecards show only the **highest passing tier** (`n == n_at_slo` everywhere), so we never see which gate failed at N+4. Scoring takes a median **≈18 h**, so many "later" attempts were parallel variants submitted before earlier results came back.

## 1. LewyM (only N=22)

| id | created | aime/gpqa | N | fast | overall | turn | chain | tpot mean/p95 | tpm_all M / dec k | slo |
|---|---|---|---|---|---|---|---|---|---|---|
| 44688 | 09-18 23:55 | 97.7/96.8 | 6 | 2.55 | 3.67 | 5.7 | 9.8 | .0150/.023 | 0.94/11.8 | .954 |
| 44826 | 09-19 10:58 | deploy failed (trisol service failed; not charged to quota) | | | | | | | | |
| 44847 | 09-19 12:01 | 100/97.4 | 6 | 3.51 | 4.34 | 6.1 | 10.6 | .0146/.023 | 0.96/11.9 | .945 |
| 45052 | 09-19 20:59 | 90.9/98.1 | 6 | 3.19 | 3.83 | 5.7 | 10.6 | .0143/.023 | 0.95/11.9 | .945 |
| 45057 | 09-19 21:30 | 97.7/96.2 | 6 | 3.68 | 4.37 | 5.6 | 10.9 | **.0102**/.019 | 1.12/13.5 | .944 |
| 45070 | 09-19 21:59 | 100/97.4 | 6 | 3.20 | 4.24 | 6.1 | 10.9 | .0103/.018 | 1.12/13.5 | .944 |
| **45417** | 09-21 02:29 | 95.5/97.4 | **22** | **1.79** | 3.00 | 10.9 | **61.0** | .0273/.065 | 1.75/20.5 | .958 |
| 45522 | 09-21 10:02 | 100/96.2 | 10 | 1.46 | 1.89 | 6.4 | 19.2 | .0151/.029 | 1.60/19.7 | .983 |
| 45603 | 09-21 15:23 | **77.3**/96.8 capability-GATED | – | | | | | | | |

The history splits into two phases.
- **Phase A (5 runs, all N=6, so N=10 failed each time).** fast_intra was 2.55–3.68 s even at N=6, with chain at ~10 s. 45057/70 cut tpot_mean 29% and raised tpm 18% (**INFERRED:** MTP/spec decode). That did not help the failing gate.
- **Phase B (after a 28.5 h gap):** N=22 with fast 1.79 s at ~4× the load. 45522 (a parallel variant) got fast 1.46 s at N=10, against a 3.15 s median for all N=10 runs. **INFERRED:** light-load fast_intra roughly halved. Queueing can't explain that, so they most likely cut uncached intra prefill through KDA/prefix-state reuse (the D1 family; F3 modelled ~2.1× fewer p95 tokens). chain_start is **above** peers at the same N (19.2 vs 13.7 s median at N=10; 61 vs ~46 s at N≥18). That fits short-prefill-first scheduling (SPF/SRPT).
- **Did chain grow as intra shrank?** Not within Phase A (chain stayed at ~10 s). In Phase B it did, relative to peers at the same N; the N=22 run reached 2.03× the chain limit.
- 45603 (aime 77.3) is too low to be noise, so **INFERRED:** a change that broke accuracy.
- tpm at N=22 (1.75 M) equals others' at N=18. **INFERRED:** the extra tiers came from cache hits and ordering, not from more throughput.

## 2. Other N≥18 players (scored runs; full rows in JSON)

| player | id | N | fast | overall | turn | chain | tpot mean/p95 | tpm M | slo | note |
|---|---|---|---|---|---|---|---|---|---|---|
| ccooddxx | 44751 | 18 | 1.88 | 2.93 | 9.3 | 45.8 | .0281/.053 | 1.60 | .962 | first attempt |
| | 45101 | 10 | 1.71 | 2.44 | 6.7 | 22.0 | .0146/.030 | 1.61 | .974 | |
| | 45150 | 14 | 1.88 | 2.79 | 7.9 | 49.1 | .0188/.039 | 1.75 | .965 | |
| | 45542 | 18 | 2.19 | 3.46 | 11.4 | 44.7 | **.0242**/.053 | 1.76 | .954 | best tpot@18 |
| 王俊杰 (vLLM-sm80 backport) | 45185 | 14 | 2.49 | 4.23 | 6.2 | 33.0 | .0210/.044 | 1.75 | .947 | |
| | 45304/45306 | 18/18 | 2.74/3.03 | 4.83/5.20 | 9.8/9.9 | 46.3/47.7 | .0268/.0265 | 1.77/1.72 | .938/.931 | same minute |
| | 45307 | 6 | 2.09 | 3.21 | 7.1 | 11.5 | **.0110**/.021 | 1.11 | .969 | same minute |
| Jinbo hu | 44643 | 10 | 3.31 | 4.22 | 7.9 | 16.1 | .0196/.033 | 1.42 | .944 | |
| | 44792 | 18 | 2.41 | 2.86 | 7.1 | 46.9 | .0259/.048 | 1.74 | .968 | |
| | 45061 | 14 | 2.63 | 3.35 | 6.3 | 32.8 | .0228/.037 | 1.63 | .954 | then 7 deploy/quota failures |
| Claude Code (Opus 5) | 45186/87/95 | 14×3 | 2.23–2.56 | 4.17–5.18 | 7.9–9.8 | 32–43 | .021–.022 | 1.67–1.76 | .94–.95 | 3 within 12 min |
| | 45312/13/14 | 14/14/**18** | 3.18/3.00/2.71 | 4.59/4.99/4.66 | 8.4–9.9 | 31–45 | .022–.026 | 1.67–1.79 | .93–.95 | same minute |
| As | 45010 | 6 | 3.00 | 4.09 | 4.5 | 9.9 | .0149/.023 | 0.93 | .948 | |
| | 45204 | 18 | 2.55 | 5.25 | 10.3 | 46.5 | **.0482**/.078 | **1.23** | .936 | |
| | 45205 | 14 | 2.21 | 3.90 | 7.8 | 45.8 | .0435/.068 | 1.09 | .953 | |

How the others compare with LewyM (INFERRED):
- **ccooddxx** is closest to LewyM: intra 1.7–1.9 s at every N and chain 45–49 s at N≥14 (N=10: 1.71/22.0 vs LewyM 1.46/19.2). It reached 18 on its first try and later traded intra margin for lower tpot.
- **王俊杰 and Claude Code** pass with intra **at the limit** (fast 2.7–3.2 s, overall 4.6–5.3 s), which looks like FCFS-like engines on the 14/18 boundary. Same-minute look-alike configs landed on 14, 14 and 18, so tier noise is real. 王俊杰 45307 shows the same N=6 tpot drop (0.011) as LewyM's MTP-like runs and reached only N=6.
- **Jinbo hu** also has low intra (2.4 s at N=18). 7 of 13 attempts failed at deploy or on quota. (ccooddxx 45243 was capability-gated, aime 81.8.)
- **As** reached N=18 with **double** the tpot (0.048) and 30% lower tpm, so decode speed doesn't decide the tier.
- Everyone at N≥14 plateaus at tpm_all ≈1.72–1.79 M and tpm_decode ≈20–21 k/min (As is the exception). No top player shows a throughput edge.

## 3. Gate budgets (p95 / limit; all passing runs)

| tier (n runs) | fast ≤3 | overall ≤5 | turn ≤15 | chain ≤30 | tpot_p95 ≤0.10 |
|---|---|---|---|---|---|
| N=6 (47) | med 0.97, max 1.23 (19 >1) | 0.77 / 0.95 | 0.38 / 0.61 | 0.35 / 0.51 | 0.23 / 0.53 |
| N=10 (29) | 1.05 / 1.21 (16 >1) | 0.79 / 0.93 | 0.39 / 0.52 | 0.46 / 0.75 | 0.51 / 0.66 |
| N=14 (38) | 0.85 / 1.16 (14 >1) | 0.81 / 1.06 (2) | 0.53 / 0.73 | **1.26 / 1.78 (30 >1)** | 0.54 / 0.73 |
| N=18 (7) | 0.85 / 1.01 (1) | 0.93 / 1.05 (2) | 0.65 / 0.76 | **1.54 / 1.59 (7/7)** | 0.55 / 0.78 |
| N=22 (1) | 0.60 | 0.60 | 0.73 | **2.03** | 0.65 |

- **Closest gate.** fast_intra at N≤10. At N≥14 chain_start has the highest ratio (30/38 and 7/7 over 30 s) but passes via the allowance. Without chain, fast_intra is closest at N=14 (26/38) and overall_intra at N=18 (4/7). tpot_p95 stays ≤0.78 of its limit.
- **How far can chain_start go?** The largest passing chain_start p95 is 61.0 s (LewyM). The next largest is 53.4 s (N=14). If the formal chain bucket has n≈808 (F31), the rule "fail only if the 95% one-sided lower bound > 5%" tolerates an exceed rate of about **6.3%**. So p95 ≫ 30 s can pass if at most ~6% of chain starts exceed 30 s, which means a steep, starved tail. turn_start (n≈65) tolerates about 11.5%.
- **Intra overshoot.** Some passes have fast_intra up to 3.68 s even though n≈7.7k allows only ~5.4% exceedance, so either the tail is very steep or the gate's sample differs from the reported p95. **Don't budget on intra overshoot.**
- **Correlation.** Across passing runs at N≥14 (n=46), Spearman(fast_intra, chain_start) = **−0.62** (N=14 alone: −0.65), and overall vs chain = −0.27. At N=18 the sign is +0.5, but with n=7 that is noise. Lower intra p95 goes together with higher chain p95. That fits prioritising short prefills over cold chain starts, though it does not prove it.

## 4. Timing and capability gate

- Median gap between attempts: LewyM 6.5 h, ccooddxx 6.6 h, 王俊杰 2.2 h, Jinbo 5.6 h, Claude Code 0.2 h (bursts), As 8.9 h. Every top player submits bursts of 2–3 runs within an hour or less (LewyM 45052/57/70; the others within minutes) as parallel A/B tests or replicates. Their spans run 45–73 h.
- The quota is **3 scored runs/day**; deploy failures aren't charged. 49 attempts were refused for quota.
- Capability gate: **101 of 232 scored attempts (44%) were gated out** platform-wide (aime ≤90 in 93, gpqa ≤90 in 47; one early run shows a ≤95 threshold). Among top players it was 5 of 35. The lowest aime that still passed is 90.9 (40/44). Scores of 77.3/81.8/72.7 are real regressions, not noise. 88.6 (39/44) has a 1–7% chance of being noise at p≈0.95–0.97.

## 5. Implications for our plan

| item | implication | confidence |
|---|---|---|
| **D1 snapshots** | The N=22 run and the best N=18 runs show **light-load** fast_intra of ~1.5–1.9 s against ~3 s for most others. Queueing can't explain that gap; fewer uncached intra tokens can. Use fast_intra at N=6–10 as the D1 acceptance signal (target ≤1.8 s). | medium-high |
| **D2 SPF** | chain_start has large slack: 38/46 passes at N≥14 are over 30 s, and one reached 61 s. Putting cold prefills behind intra is cheap and matches the winners' profiles. SPF is table stakes (F39). | high |
| **I6 EDF / aging** | LewyM's chain at 61 s (2×) suggests chain deadlines are *not* protected. EDF's value is as a **guard**, not the main mechanism: (a) keep the exceed rate for chain above 30 s at or below ~5–6% (a starvation cap, e.g. forced service at ~25–40 s wait); (b) give overall_intra (>4096 uncached, 5 s deadline) its own deadline so pure SPF doesn't starve it. overall_intra becomes the next binding gate at N=18. Report the chain exceed rate, not just p95. | medium |
| **MTP / tpot** | tpot_p95 is at most 0.78 of its limit in every pass and never binds. MTP-like gains show up at N=6 (−29% tpot) but not at N≥18, where everyone is at 0.024–0.028. Keep MTP only as a **tie-breaker** at equal N (target tpot_mean below 0.0273 at N=22), and only if it costs no prefill capacity or accuracy. | medium |
| **Throughput** | Top engines saturate at ≈1.75 M tpm_all. Tiers come from **latency ordering and cache hits**, not from more raw throughput. D3 capacity work is lower priority than D1+D2+guard. | medium |
| **Submission ops** | Results take ~18 h. Submit 2–3 replicates or variants per day (tier noise at 14/18 is real). Run a local aime spot-check after any numerics change (quantization, spec decode, kernels) because of the 44% gate-out rate. | high |
