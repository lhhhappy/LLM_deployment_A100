# 官方 47043 N30 与新包本地 N30 数据包

2026-09-28。**chain p95：官方旧版 25.608 s；新包本地 18.422 s。两份测试集与范围不同，不能据此计算新包的正式提升。** 新包本地为 ezno、引擎 bf6b66fa、0928c 配置。新提交的正式结果不在此包中。

## 完整主指标

| 指标 | 47043 官方 N30 | 新包 ezno 本地 N30 |
| --- | ---: | ---: |
| chain TTFT p95，s | 25.607907 | 18.421743 |
| fast TTFT p95，s | 3.445378 | 1.699653 |
| overall TTFT p95，s | 4.043678 | 2.216685 |
| turn TTFT p95，s | 4.934348 | 12.167546 |
| TPOT mean，ms/token | 37.682613 | 51.048157 |
| TPOT p95，ms/token | 59.581739 | 94.626981 |
| TPOT>100ms 条数 | 平台未返回 | 83/1869 |
| 四桶样本数/超标数/统计允许数 | 平台未返回 | 见下表 |
| 最高通过 N | 官方返回 30 | 定时诊断不能给出正式 N@SLO |
| TPM all，token/min | 1,727,881.883 | 原摘要 null，不构造 |
| TPM decode，token/min | 19,375.167 | 原摘要 null，不构造 |
| SLO attainment | 0.9440715884（平台原数） | 不跨口径构造 |

| 本地 TTFT 门 | 目标 | 超标/允许/桶样本数 |
| --- | --- | --- |
| chain | 30 s | 6 / 19 / 253 |
| fast | 3 s | 28 / 83 / 1394 |
| overall | 5 s | 13 / 92 / 1545 |
| turn | 15 s | 3 / 7 / 71 |

统计允许数按仓库原 `score_formal.allowed_over`，不是拿 p95 是否低于目标另造判分规则。四桶关系来自原 harness；fast 是 overall 的子集，不能加总当总请求数。

## 完整性、数据身份与时长

官方原 JSON 返回 N30 PASS、n_at_slo=30 和最终状态；**没有返回逐请求 raw、四桶大小/超标数、TPOT慢条数、本档完成请求数、本档测量时长、数据哈希或更高失败档的门**。这些缺项在 metrics.json 是 null，在 CSV 是空白；不从 SLO attainment 或协议目标反推。

本地冻结数据是 `s1-dev-longchain-v5g-tail-rot150`，cohort哈希 `b78593bdea138f58`、workload哈希 `97175a1e2ea92d15`，完整集合311链/5601请求。本次仅40分钟准入，1869派发全部排空、1869唯一行、0错误，flush成功，warmup=rep16-v1。TIMED_WINDOW的首派发到排空2546.632991秒，其中准入后排空146.632991秒；另一个 run.wall_s计时器为2545.781498秒，summary.wall_s=2567秒，保留各自定义，不混用分母。

原 `score.log` 对已准入子集打印 PASS；归档 `level_verdict.json` 因另一取证步骤用了不匹配的数据根而保留 INVALID（missing/extra ID），二者原文都保留。正确范围仍是 **DRAINED / fixed-duration diagnostic，full_cohort_complete=false**，不是完整5601请求的正式PASS。

## 能力验证不等价

官方47043能力门通过：AIME26 **43/44，97.7273%**；GPQA Diamond **151/156，96.7949%**。新包本地只有真实权重能力冒烟 **12/12**，不是AIME/GPQA全测，也不代表新提交已经通过官方能力门。精确原数在官方JSON中。

## 真正同条件的旧/新本地对照

官方没有请求级raw，**不存在官方raw与新本地raw的逐ID配对**。这里附的是旧版47043配置的本地eznb与新包本地ezno的1689个共同ID：

| 共同1689请求 | eznb旧配置 | ezno新包 |
| --- | ---: | ---: |
| chain 超标 | 12 | 6 |
| fast 超标 | 248 | 26 |
| overall 超标 | 209 | 13 |
| turn 超标 | 11 | 2 |
| TPOT mean，ms | 54.904810 | 52.191383 |
| TPOT p95，ms | 98.386834 | 96.730300 |
| TPOT>100ms | 83 | 78 |

chain修好6、新坏0、持续6；全部 gate 的修/新/持续和逐ID时间在 `local-paired-1689/`。这些数字不能混用上面 ezno 全1869请求的28/13/3和94.627ms。两次本地运行仍是单次定时诊断，不表示每项差值都有重跑支持的稳定性。

## 文件说明

- `metrics.json`：完整可用聚合值、null缺项、数据范围、时长、能力、原件哈希、1689-ID摘要。
- `metrics.csv`：方便表格软件读取的官方/本地主指标；空白表示该口径未返回。
- `official/`：官方最终JSON原件。
- `local-ezno-N30/`：完整请求raw、run、summary、loadgen/TIMED_WINDOW、flush、score/verdict/fetch_status原件、运行指标metrics.jsonl与GPU采样。`timed-window-extracted.json`从loadgen抽取，不是伪装原生成的verdict；`selected-receipt-lines.txt`仅摘录commit/12题smoke/排空证据。
- `local-paired-1689/`：旧/新本地配对CSV、chain坏例CSV和完整摘要。
- `SHA256SUMS`：包内数据逐文件哈希。没有复制启动环境、配置凭据、提交密钥或token。

全包仅包含本地已存在的证据与其派生摘要，没有触发新的GPU试验。
