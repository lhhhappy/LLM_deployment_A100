# eznq N34：TPOT 超标形状与今晚单参数候选

2026-09-28，Codex；离线 raw/server 核对。**chain 仍是原来的 6 条；建议只试现成 `--prefill-decode-interval 2→4`，保留开场 125 的 interval=0 和 131 风险保护 interval=1。** 这是未实测的参数假设，须在同 N34、相同 ID 上先看 chain 修好/新坏/持续；不能预报 TPOT 收益。未动 GPU、Pod、队列或共享文件。

已读 [最新综合复盘](../../notes/reports/0928-official-local-synthesis.md)。[原始 raw](../../evidence/L130eznq-tail_rot150_n34_chainmax16k_fix_mamba400_118_scatter_moetune_kda_40m/N34/raw_s1-dev-longchain-v5g-tail-rot150_N34_1790605927.jsonl)、[server.log](../../evidence/L130eznq-tail_rot150_n34_chainmax16k_fix_mamba400_118_scatter_moetune_kda_40m/N34/server.log)、[启动配置](../../evidence/L130eznq-tail_rot150_n34_chainmax16k_fix_mamba400_118_scatter_moetune_kda_40m/N34/job.log) 与 [排空收据](../../evidence/L130eznq-tail_rot150_n34_chainmax16k_fix_mamba400_118_scatter_moetune_kda_40m/N34/loadgen.log) 为本次依据。1960 条唯一请求、0 error、40 分钟准入后 DRAINED；102 条 TPOT>100 ms，整体 p95=103.322 ms。它不是完整 cohort 的正式 N34 结果。下表由原 `score_formal.load_harness()`/`window_gates` 读取，分位数不另造算法；按每条请求的 client_dispatch 相对第一条派发时刻分窗。

## 102 条超标是什么请求

| 派发窗口 | 完成请求 | >100 ms | TPOT p95 ms |
| --- | ---: | ---: | ---: |
| 前5分钟 | 248 | 51 | 398.805 |
| 5–10分钟 | 247 | 14 | 105.737 |
| 10分钟后 | 1465 | 37 | 79.132 |
| 全部5分钟后 | 1712 | 51 | 82.700 |

短/中/长输出在此明确指 ≤128、129–512、>512 token，都是原输出预算与实际 output_tokens，不改预算：

| 输出范围 | 前5min 请求/坏 | 5min后 请求/坏 | 全部 请求/坏 |
| --- | ---: | ---: | ---: |
| ≤128 | 62/19 | 454/41 | 516/60 |
| 129–512 | 133/29 | 795/10 | 928/39 |
| >512 | 53/3 | 463/0 | 516/3 |

102 条中 phase=intra 74、session_start 14、turn_start 10、context_reset 4。全部请求的输出中位数 230 token，超标请求中位数 105；5 分钟后 51 条坏例有 41 条输出≤128，输出>512 的 463 条没有坏例。说明后续风险集中于短生成期间遇到长 prefill 的请求，不能概括为持续稳态 decode 算力不足。前5分钟有长输出坏例3条，不能说问题仅限短输出。

最慢一条 `scimaster:canon:zlBy3aBiTqMNBYqPsCGRf:llm:0` 开场派发，输出29 token，TPOT=2.594 s；服务首字约第2.7秒、客户端完成约第75.3秒。这是首字后长等待，与 chain TTFT 是两种不同的损失，不能拿 TTFT 代替 TPOT。

## 现有日志直接证明的范围

测量起点为 **2026-09-28 14:32:07.804 UTC**，排除了冒烟与预热日志。日志显示：

- **14:32:08→14:33:17**：125 relief 开启，随后因 backlog 回落关闭；两端 slow 都是0/0。首60秒有100条 prefill batch 日志，合计1,085,952 new tokens，其中65批≥8192；首条 decode 统计直到14:33:25出现（13.45 tok/s，running=33）。源码解释了 slow0/0：只在 produced>1 才计入 `note_tpot`，首次纯 prefill 阶段的慢 decoder 还没有进入这项护栏。
- **14:33:36→14:33:40**：125再次开启/关闭，slow38/43→41/43；**14:34:40→14:34:41**：slow78/116→80/116，到80后 guard latch，之后不再开启 relief。故调低 `BACKLOG_MAX_SLOW` 不能消除最初69秒窗口，主要只会改变后面很短的 relief 段，不是本轮首选。
- **14:49:33→14:49:49**：相邻 decode 统计间隔16秒，其中有23次 prefill，合计281,792 new tokens；后一条 decode 统计56.94 tok/s、running=25。125早已停止，14:49:45又有131 risk日志。
- **14:43:10→14:43:23**：相邻 decode 统计间隔13秒，其中15次 prefill、216,064 new tokens，末端75.91 tok/s。类似区间不只发生在最初的 relief 模式。

一个可对齐的后续短输出例子：`biomaster:canon:lc_20260924_194:llm:0002` 第688.699秒派发，输出15 token，TPOT435.594 ms；服务首字第689.586秒到客户端完成第695.688秒之间有7批 prefill、114,688 new tokens。另一个同链 `...:0008` 输出15 token，TPOT370.942 ms，其第861.545→866.792秒生成窗口内有6批 prefill、93,760 new tokens。它们直接支持“短生成暴露于重 prefill 区间”，但日志没有每个 token 的逐步完成事件，不足以精确分摊其每毫秒等待。

**日志采样限制：** bf6b66fa 的 `scheduler_components/metrics_reporter.py:788–802` 每 `decode_log_interval` 个 decode forward 才打印，吞吐以该统计窗口内的 generated tokens / wall gap 计算。16秒是相邻统计点的间隔，绝不是“完全16秒没有任何decode”；秒级日志时间戳也不能给出单个GPU步骤耗时。prefill条数/新token量、running、低窗口吞吐才是这里直接可核的事实。

## bf6b66fa 有效参数与唯一推荐

job.log 的 ENGINE_COMMIT 精确为 `bf6b66faf3ffbe10e593e858d3bbedd1c48cefb2`，MECHANISMS OK。有效配置为：`--prefill-decode-interval 2`、chunk16384、running48、mamba400、scatter；118 prefill、MoE up/down、KDA三个开关开启；131 interval1/chunk-auto；125 backlog interval0、coldcap16384、high/low15/5s、max_slow80/gate0.10；132 chain-first、124 load1.05；133与124m未开启。源 job 是 [tail_rot150_n34_chainmax16k_fix_mamba400_118_scatter_moetune_kda_40m.sh](../../scripts/pod/jobs/tail_rot150_n34_chainmax16k_fix_mamba400_118_scatter_moetune_kda_40m.sh)。

推荐只改 **`--prefill-decode-interval 2`→`4`**，其他参数不动。源码依据是 bf6b66fa `scheduler.py:1969–1995`：先取 configured interval；125 relief 时覆盖成0；131命中再取 `min(interval,1)`。因此增加的是普通 prefill 后的 decode 轮数；现有 opening relief 和 at-risk 保护规则仍然存在，避免133曾经在开场向纯prefill通道强塞decode的直接做法。它也能作用于125已永久关闭后的51条短输出坏例所在区间。

这里的“保留规则”不等于承诺 chain 不变：执行节奏会改变后续到达、缓存、风险模型命中以及队列，尚未到131风险区的长头可能延后。2→4 是有意留下可测差别的单臂，不声称最优。继续用同N34、同冻结数据/预热/flush、40分钟准入后排空来对齐ID，chain出现净恶化即否决；TPOT按前5分钟/之后及输出≤128分开看，整窗p95仍必须报告。由于开场51条未被这个参数直接针对，不能保证短窗整体p95会跨回100ms。

相比直接缩小16k冷块，这个候选保留当前8k/16k核路径与纯prefill效率，首先检验现有间隔控制是否足以减少普通长prefill对短decode的阻塞；不加入新机制、不重试133默认或slack调参。N30/N34的差值不用于本次参数因果判断。

原始文件 SHA256：

- raw：`6125d6629e69f4766eb514222bbac5032e3f80d17f4793b2b5b9b8e01f94db86`
- server.log：`dda1477d8fef719fbe6803eefa4a565aa3691851d297e78f6ef76934333995a6`
- job.log：`4f4fc2654b625b3253729ef19244b8bfb6ce29a7e409b4c7d8873a7c321a8ab4`
