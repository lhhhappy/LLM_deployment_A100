# 最近 Pod 运行：chain、执行速度与缓存内存审查

2026-09-28，Codex。状态：只读诊断；原始数据与原 harness 校准完成，尚未经另一位参与者独立复核。

**chain 第一：N30 钉池对照的同 ID chain 超标为 12→7，修好 5 条、新坏 0 条；同时 TPOT p95 为 98.39→120.94 ms，超过 100 ms 硬门。剩余 7 条全在开场，开始执行前已等待 37.7–67.0 秒。** 这是短测观察，不能据此宣布 N30 通过或直接采用钉池配置。

本次把任务收窄到用户要求的运行审查：读取 Pod 当前快照，重算最近六份已排空诊断的 raw、服务日志、metrics 和 GPU 采样，核对执行源码。没有启动 GPU 探针、改动或部署引擎。独立工作区基于 `20a58da9`；新增的是分析脚本和证据。

## 数据和口径

主对照是 `130eznb` / `130eznc`，均为 `20a58da9`、TP8、no-MTP、DCP=1、running=48、16k chunk、117 ON、118 OFF、119 OFF。任务实际配置的唯一旋钮差异为 `--max-mamba-cache-size 400`；自动分配时实际为 779。两者均关闭 attention-TP input scatter，prefill CUDA graph disabled，decode graph enabled。不要沿用任务文件复制下来的旧注释判断配置。

- 冻结负载：`s1-dev-longchain-v5g-tail-rot150`，workload hash `97175a1e2ea92d15`，cohort `b78593bdea138f58`。40 分钟派发，随后排空；总 cohort 5601 请求没有全部回放。
- `eznb`：UTC 01:36:08–02:18:37，1689 请求；`eznc`：02:35:31–03:17:54，1726 请求。共同 1687 请求，其中共同 chain 238 条；各自独有 2 / 39 请求。
- 辅助对照：`ezne/eznf`（124m ON/OFF，同 rot150），`ezn3/ezn5`（N26，118 OFF/ON，原始 tail 排序）。不同数据集不横向比较成绩。
- 全部六份 raw 都满足唯一请求 ID、`rows = dispatched = n_attempted`、无请求 error、生命周期时间戳有序、输入和输出 token 数与冻结字段一致。配对还核对 prompt/output budget、chain、phase、expected uncached、edge type。
- chain selector 直接调用原 `s1_common.in_ttft_gate`；p95 使用原 harness 的 `sorted[min(n-1, floor(0.95*n))]`，不插值。六份 chain p95 均与保存的 harness 摘要相符。TPOT 是客户端首 token 至结束的平均每 token 时间，不是 GPU 单步耗时。
- 开场按请求到达前 120 秒定义，稳态为 120–2400 秒。metrics 和 GPU 采样按实际 raw 起止时间裁切；累计 counter 取边界相邻采样之差，约 10 秒粒度。服务批次日志只有整秒时间戳；GPU 采样约 5 秒粒度。

原始证据：[eznb](/workspace/Agentic_science_challenge/evidence/L130eznb-tail_rot150_n30_chainmax16k_fix_40m/N30/summary.json)、[eznc](/workspace/Agentic_science_challenge/evidence/L130eznc-tail_rot150_n30_chainmax16k_fix_mamba400_40m/N30/summary.json)。可复算输出为 [audit.json](../../evidence/pod-execution-0928/audit.json)，含 raw/log/metrics SHA256、逐请求坏例和批次重叠记录；[校准收据](../../evidence/pod-execution-0928/validation.txt)。这些数据的状态都是 **DRAINED_DIAGNOSTIC**，没有完整成绩的含义。

## 1. chain 改善伴随 TPOT 退化，不能只看均值

以下请求指标均为共同 1687 ID；显存、缓存量为各自完整诊断窗口，后者工作量不完全相同。

| 指标 | eznb：自动 779 槽 | eznc：固定 400 槽 |
| --- | ---: | ---: |
| chain 超标 / 共同 chain | 12 / 238 | 7 / 238 |
| chain 开场 / 稳态超标 | 8 / 4 | 7 / 0 |
| chain p95 | 30.305 s | 24.969 s |
| TPOT mean | 54.93 ms | 60.00 ms |
| TPOT p95 | 98.39 ms | **120.94 ms** |
| TPOT >100 ms 请求 | 83 | 118 |
| 共同 ID 实际未缓存输入 | 11,719,982 | 11,636,270 |
| 实际 KV 池容量 | 1,260,160 tokens | 1,810,112 tokens |
| KV locked 占池峰值 | 99.89% | **98.25%** |
| H2D cache restore 累计量 | 3342.11 GiB | 419.38 GiB |
| 每卡显存采样最大值 | 76,526 MiB | **78,042 MiB** |

修好类型：开场 `context_reset/append-only` 1 条；稳态 `context_reset/append-only` 2 条、`intra/system-tools-changed` 1 条、`intra/append-only` 1 条。请求 ID 全部在 `audit.json → pairs → 130eznb_130eznc → chain_start`。

本次观察支持“缓存驻留改变能改善稳态 chain”。共同 ID 实际新算输入只减少 0.71%，而恢复传输减少 87.45%，更接近减少搬运和内存压力的解释。它没有证明孤立算子变快；同 ID 后续请求到达时刻也会随前序完成时间改变，因此不是相同时间轴上的逐 kernel A/B。

TPOT 的退化不只来自开场：按各自完整请求集合，600–1200 秒到达的请求 TPOT p95 为 94.96→143.21 ms，>100 ms 数量为 18→41。这个分窗集合不相同，作为定位用，不替代同 ID 比较。

## 2. 一个能追到批次的 decode 坏例

共同请求 `scimaster:canon:lc_20260924_241:llm:0047`，输出均为 35 tokens：

| 观测 | eznb | eznc |
| --- | ---: | ---: |
| 请求到达相对时间 | 1229.8 s | 1219.1 s |
| TTFT | 15.92 s | 1.35 s |
| 客户端首 token 后耗时 | 1.071 s | **18.307 s** |
| TPOT | 31.4 ms | **538.3 ms** |
| 首 token 至完成期间记录的 prefill 批次 | 3 | 17 |
| 同期间记录的新算 prefill tokens | 1,408 | **259,136** |

eznc 的这段记录中，15 个批次为 16,384 tokens，另有 12,800 和 576；运行请求约 20–21，KV 使用约 60–74%，队列通常为 0–1。期间仍有 decode 日志，故不能说“18 秒没有 decode”。服务日志与客户端时间共同支持 **长 prefill 与 decode 干扰** 的解释，不能把这 18 秒归因为某个慢 decode kernel，也不能用低 queue 值判断服务没有阻塞。

另一个共同请求 `biomaster:canon:lc_20260924_194:llm:0003`，输出 5 tokens：首 token 后耗时 0.169→2.306 秒；后者期间有两个 16k prefill 批次。短输出对少数长批次非常敏感。

这些例子为纯计算优化提供目标：缩短真正插在 decode 之间的 prefill 执行，也能改善 TPOT。现有日志没有逐 token 时间戳或近期完整 GPU trace，尚不能区分其中的 launch、通信、内存等待和算术各占多少。事件和原请求字段保存在 `audit.json → selected_decode_overlap`。

## 3. 缓存压力减轻了，但没有消失

**OBSERVED：** eznc 的 KV 池增大 43.64%；稳态 `full_token_usage > 95%` 的采样点从 93/227 降到 3/227。其峰值仍为 1,778,496 / 1,810,112 = 98.25%，发生在约 1077 秒，不是“最高只有 80%”。这些近满采样点没有观测到排队，也没有记录 retraction，不能直接称为内存事故。

更容易看错的是 mamba 状态：`mamba_used` 峰值两边都是 129，但它只覆盖不能回收的使用量。400 槽配置的 `used + evictable` 曾达到 **400/400**，有 7 个采样点 available=0；例如 273.6 秒为 used=72、evictable=328。多数可回收，故不是 400 个槽全被锁死，也不是 OOM；同样不能据“活动只有 129”继续把状态池缩到接近 129。需要计入缓存检查点的保留成本。

H2D 恢复 CUDA event 累计时长为 330.51→39.75 秒；这是累计传输事件时间，异步传输可以重叠，**不能声称整轮墙钟省了 290.76 秒**。数值按本日志可见 metrics collector 口径报告，不擅自乘 8。有关计数定义已核对 `metrics_collector.py` 与 `unified_radix_cache.py`。

GPU 显存峰值在钉池后反而高 1516 MiB，达到每卡 76.21 GiB，距 80 GiB 约 3.79 GiB。这是采样最大值，不保证捕获最短瞬时峰值。对新算子，临时张量、graph pool 和常驻 workspace 都应与 KV 池损失一起验收，不能只报 kernel 毫秒数。

## 4. 开场仍有明确的执行产能缺口

eznc 剩余 7 条超标都在 0.68–5.13 秒到达：

| prompt tokens | TTFT s | 开始执行前 s | 开始执行至首 token s |
| ---: | ---: | ---: | ---: |
| 52,218 | 42.13 | 37.67 | 4.46 |
| 81,051 | 50.51 | 44.39 | 6.12 |
| 92,306 | 46.98 | 40.97 | 6.01 |
| 122,206 | 64.87 | 55.93 | 8.93 |
| 87,331 | 59.06 | 52.18 | 6.88 |
| 83,312 | 54.74 | 50.34 | 4.40 |
| 194,991 | 80.26 | 66.97 | 13.28 |

前 120 秒日志统计的 prefill 吞吐为 13,623→13,661 tokens/s，几乎相同，且这段 H2D 恢复增量两边都是 0。这里是窗口内总工作量/墙钟，不是固定形状的 GPU 单块计时。它说明这次钉池主要改善稳态，没有解决开场冷头执行产能。

新算子加速前方计算有价值；但最近的一条剩余坏例 TTFT 已是 42.13 秒，不能承诺“prefill 快 5–10% 就净少一条”。5–10% 应仍作为研发目标，通过配对实验检查最终落在哪些请求上。

## 5. 当前真实算子形状：大块和大量小块都要测

最近这些运行 `enable_attn_tp_input_scattered=False`。源码 `AttnTpContext.init_context` 需要显式开关才允许 scatter。因此当前 mHC 的 8k/16k prefill 主形状仍是每 rank 完整 token 行数，不能套用“每卡只有 1k/2k 行”的测试方案。indexer 自己的 row shard 是另一机制，不能混淆。

eznc 有 2736 个记录的 prefill 批次，其中 ≤4k 为 1979 个（72.3%），≤1k 为 1321 个。所有记录的 prefill 都没有使用 CUDA graph。≤4k 批次承载的新算 tokens 仅占 16.34%，所以 **72.3% 是次数占比，不是 GPU 耗时占比**。大块控制冷头产能，小块的固定开销也值得测；仅追一个 16k 跑分不足以代表服务。

结合已存在的实现，纯推理优化应优先回答下列具体问题：

| 对象 | 已核实的实现事实 | 应验证的候选，不是已获得的收益 |
| --- | --- | --- |
| mHC | 已有 norm/pre 融合；`MHCState.attn_to_mlp` 仍调用 post 再 pre。历史单独重写 post 的 8k 探针没有变快 | 在 256/1k/4k/8k/16k 实际形状测 post→pre 边界的数据读写与跨算子融合；保留 BF16 中间舍入、residual 和 fp32 mixing 语义 |
| MoE | 当前117走 Humming；indexed 路径先写 down_output，再 weighted sum；已有172主要针对旧 Marlin，不应重复计收益 | 先量路由、down-output 加权归并、workspace、host enqueue 的独立成本，再决定调 tile 或融合；不要默认重写 GEMM 有大收益 |
| mHC 分片/通信 | 119 已有 large-extend threshold 候选，当前关闭；会更换 collective 组合 | 属于执行布局优化，先复用119做当前栈 TP8 验证；不能当作新发明或假设把 mHC 直接加速8倍 |
| 小块执行 | 70–72% 的 prefill 批次数≤4k，当前没有 prefill graph | 小形状算子和 launch/graph 路径分开测，确认 graph 显存成本；不能把批次数当热点时间 |

MoE 源码形状推算：16,384×9×4096×2 字节的 BF16 down_output 为 **1.125 GiB/层调用**；完整写出再读入的名义数据量为 2.25 GiB。这是实现形状的算术，不是 Nsight 测得的 DRAM 流量；workspace 可复用，不应乘层数当常驻显存。能否省去、实际省多少，取决于归并数值顺序、路由分布和 kernel 设计。

已有硬加速并非空白：N26 `ezn3/ezn5` 的共同 1572 ID 中，118 的 chain 超标仍为 6→6，chain p95 为 24.756→23.303 秒；TPOT p95 为 87.83→82.48 ms。它支持执行余量改善，没有证明净救回 chain。原 TP8 单请求 16k 目标块 1084.9→989.2 ms 的证据在 [118 报告](/workspace/Agentic_science_challenge/build/worktrees/prefill-sm80-0927/notes/reports/prefill-sm80-0927.md)。

## 6. 新发现：10 个旧任务的采样进程仍在污染旧目录

UTC 07:28 左右只读 `/proc`，找到 10 个 `metrics_sampler.py`，PPid 全为 1，目标分别属于已经失败/停止的 `072r、086、090、097、108、130ec、130ef、130ezd、130ezl、130ezn6`。12.008 秒后二次采样，**每个文件增加 23,472 字节，最后记录时间均为当前时间**。不是仅仅遗留了空闲进程。

源码中采样器无限循环；`dev_ladder_template.sh` 仅在 runner 正常返回后清理 monitor，没有退出 trap。`stopjob_inpod.sh` 计算了 children，却未使用该变量清理采样器。这与当前孤儿进程吻合；具体每次退出的历史原因仍需原 job 收据确认。

影响首先是证据：旧 run 目录的 metrics 尾部可能已经属于别的引擎、别的负载，累计计数还可能跨服务重启。直接整文件均值或末值相减会误判。此次分析固定在 raw 起止窗口；被分析六轮也不是这十个遗留输出目录。

十个 sampler RSS 合计约 232.4 MiB；12 秒中 CPU tick 增量合计仅 0.03 秒。没有证明这会造成明显 GPU 性能下降，服务端处理 metrics 的 CPU 成本也未单独量化。应修监控生命周期和归档截窗，不能把它宣传成算子优化收益。未终止任何进程。原始快照：[live-orphan-samplers.json](../../evidence/pod-execution-0928/live-orphan-samplers.json)。

## 7. 能力复核中断：不能宣称已经保证效果

本轮审查期间，`130ezng` 写出 `CAP_FULL aime=25/30 gpqa=99/197 errors=88 truncated=6`。逐行核实后：

| 数据集 | 总请求 | 成功取得响应 | 正确 | 连接错误 | 达到长度上限 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AIME | 30 | 30 | 25 | 0 | **5** |
| GPQA | 197 | 109 | 99 | **88** | **1** |

5 条 AIME 未通过记录全部为 `finish=length`、60,000 tokens，约1100秒；GPQA也有1条60,000-token截断，另外9条取得非截断响应但答错。88个错误为12个 `Remote end closed connection without response` 和76个 `Connection refused`。

时序证据：

- 06:41:20：worker 启动能力任务。
- **07:29:14：worker 记录该任务退出 rc=143，归入 failed。**
- **07:31:10：worker 启动下一轮 N34；旧能力引擎同时记录 SIGTERM。**
- 07:32:08：旧引擎仍报告12个请求未完成；约07:32:10引擎停止，能力客户端随后写出错误汇总。

这说明能力客户端在 job 结束后仍继续工作；下一任务切换引擎时能力请求尚未闭合。`lib.sh` 的引擎替换逻辑与这条时序一致。**已查明的是生命周期失配；没有定位07:29终止任务的具体发起者，不能臆称某个超时或某位参与者导致。** 这些退出均为审查前后其他运行流程的记录，本次只读工具没有发送退出信号。

GPQA的99/197混入了88个连接错误，不是一次完整的模型能力测量；仅取99/109也有完成者选择偏差。AIME的25/30是当前60k预算下的结果，5条长输出是必须继续检查的真实现象，不能因连接问题而一并忽略。它们可能涉及采样、停止行为、预算或数值差异；此处没有相同条件的118 OFF参考，无法判断118是否有责任。

还发现脚本注释称“no output cap that truncates answers”，但实际 `max_tokens=60000`，并且确实发生截断；脚本捕获连接异常后记录 `ok=False`，最后只打印汇总，未按 errors/truncated 返回失败。调用模板仅 grep `CAP_FULL`，也没有完整性与能力阈值判定。这次队列本身记录了143失败，但一般情况下，打印出 `CAP_FULL` 不能当作能力门通过。

需要补齐同条件118 OFF/ON、相同输出预算和完整请求生命周期的复核，先保证错误与截断被单独拒收，再比较真实能力。本记录不支持“118能力已通过”，也不支持“118使GPQA准确率降到50%”。任何mHC/MoE新实现都还需要自己的数值、真实权重和能力验证，不能继承118的结果。

证据：[能力逐行有效性审计](../../evidence/pod-execution-0928/cap-validity-audit.json)、[原始结果](../../evidence/pod-execution-0928/live-cap-full-results.txt)、[worker时序](../../evidence/pod-execution-0928/live-worker-transition.txt)、[SIGTERM](../../evidence/pod-execution-0928/live-cap-exit-events.txt)、[退出时服务日志](../../evidence/pod-execution-0928/live-cap-server-end.txt)。

## 8. Pod 快照：区分能力负载与压力回放

UTC 07:27 的快照正在跑 `130ezng` 的118能力复核。每卡显存约 72,480–72,980 MiB、GPU util 100%；cgroup 使用约 978.6 / 1509 GiB，memory failcnt=0。此前六个实际测量窗口没有发现选定的 OOM/Traceback 标记；服务日志尾部的 Traceback 位于排空后的 SIGTERM 退出阶段，不能算作测量期失败。cgroup 含 page cache/shmem，不能与 tmpfs 再加一遍。

07:31 的能力负载为 decode batch=12、队列0、KV约23%、CUDA graph=True，总生成速度约672–675 tokens/s。等分到12条约17.8 ms/token，只是该批次的有效墙钟速度。它不同于压力回放的混合负载，也不证明核心达到峰值算力；100% busy 不等于计算效率100%。快照：[容量](../../evidence/pod-execution-0928/live-capacity.txt)、[服务尾部](../../evidence/pod-execution-0928/live-engine-tail.txt)。

截至本轮稍后的只读状态快照，Pod 已在运行 `130eznh` N34 压测；本报告的六轮性能比较只使用已排空归档，未把这轮仍开放的数据混入。

## 复算与下一步验收

```bash
python3 scripts/analysis/pod_execution_audit.py \
  --repo /workspace/Agentic_science_challenge \
  --out evidence/pod-execution-0928 \
  --cap-results evidence/pod-execution-0928/live-cap-full-results.txt
```

脚本不调用引擎，只读取归档证据。每轮输出全体 chain 的 CSV、10秒时间线 CSV，集中 JSON 包含三个配对、错误标记的时间窗、内存近满采样点和首 token 后的批次重叠。

下一轮纯算子验收以当前实际形状为准：算子数值和显存 → 完整 prefill 块时间 → 长头执行墙钟 → 相同 ID 的开场/稳态 chain 净修好数；同时守住 TPOT p95≤100 ms 和能力门。这里提出 mHC/MoE 测量目标，尚未交付或宣称新 kernel 的加速结果。
