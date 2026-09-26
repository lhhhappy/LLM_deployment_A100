# DCP local extend：修正 JIT 与机制观测，定位选键差异

2026-09-26，Codex。分支 `codex/dcp-prefill-local-kv`，修正代码提交 `6976639e`。回应 Fable 的[独立审查](/workspace/Agentic_science_challenge/notes/reports/review-dcp-local-extend-0926.md)；完整审查反馈和收信确认已收到，会话联络登记已刷新。原提交 `f34c7ac4` 的 B1、B2 属实：原预热计时遗漏了新形状 JIT，启动机制行也不足以识别新路线。本报告记录修正及定案证据，不改写原始失败记录。

## B1：尺寸不再触发 JIT

ROWS/COLS/S0/S1/OUT_COLS 全部改为运行时参数，并显式 `do_not_specialize`。只删掉 `constexpr` 仍不足以排除隐式整数特化，因此直接数实际 Triton kernel cache 中的版本，覆盖连续行数、奇数列宽、不同步长和大块。

| 开发机实测 | rank 0 | rank 1 |
|---|---:|---:|
| 原实现，前 8 个 T 的编译版本 | 8 | 8 |
| 原实现，第 2–8 个 T 首次调用中位 | 56.34 ms | 58.23 ms |
| 修订，543 种形状/步长组合的版本 | 1 | 1 |
| 修订，通用版本首次调用 | 61.30 ms | 61.44 ms |
| 修订，后续首次形状调用最大 | 0.166 ms | 0.164 ms |

这列是包含 host launch、输出分配、同步及可能编译的调用墙钟时间，不是纯 compiler 时间。原实现第一调用还包含初始化，约 865 ms，不拿它与已初始化的修订作加速比。

测试逐个覆盖 T=1…512、513/1024/2048/8192，以及列宽 1/3/5/63/64/65/127/2048/4099 的非连续输入。oracle 从**未压缩**表过滤全部 owner token，核对每行保留数、顺序、地址和负值/padding；两个 rank 均通过。静态 owner 几何的第一次加载仍有成本，没有宣称服务完全不再 JIT。

原始逐形状记录：[rank 0](../../evidence/dcp-local-extend-20260926/review-fixes/review_jit_v1.rank0.json)、[rank 1](../../evidence/dcp-local-extend-20260926/review-fixes/review_jit_v1.rank1.json)。工具：[dcp_local_indices_jit.py](../../scripts/analysis/dcp_local_indices_jit.py)。

## B2：区分配置生效与实际触发

`_ax_mechanism_report` 直接读取 target ModelRunner 已初始化的 eager policy，输出 `dcp_local`、`dcp_local_max`、`dcp_local_large`。例如 ON 大块臂的 `G_EXPECT` 可声明：

```text
dcp_local=on dcp_local_max=512 dcp_local_large=8192
```

只设置环境变量、但 runner 没有对应 policy，不会打印 on。关闭主开关时输出 off/0/0，且不创建路由计数器。

每个 target/draft runner、每个 rank 单独计数 `gather_kv`、`local_short`、`local_large` 的 batch、padded Q 和 prefix token 总量，保存最近 T/P；普通 eager forward 返回后才计数。首次遇到每条路线立即打印 JSON，之后最多每 30 秒一条累计快照。带 Unix 时间，无 GPU 同步、新 GPU buffer 或随形状增加的字典。不能把不同 rank 的副本加成请求数，也不能把累计 prefix 当唯一 KV token。

启动 token 只证明配置；[dcp_route_audit.py](../../scripts/analysis/dcp_route_audit.py) 另要求指定路线在**每个预期 rank**有非零计数。传 `--since START --until END` 时，只用测量窗口内两个累计快照之差，warmup 的触发不会误作测量触发。窗口两端未采样的部分不计入，因此这是保守的活动证明，不是完整整窗账本。CPU 回归包含“on 但一个 rank 未触发”和“只在 warmup 触发”两种拒绝情况。

最终完整 Engine 启动对照 `review_mtp_info_base/local` 已实测：原 scheduler 机制行分别输出 **off/0/0** 与 **on/512/8192**。target/draft 的 rank 0、1 均记录三个路线，累计已打印快照为 gather 4 / short 1 / large 1；只取启动后请求窗口内两个快照之差，四个 runner 均得到 **gather 3 / short 1 / large 1**。这是已观测区间的计数，不是全部 churn/flush 请求的完整账本。关闭臂没有路由计数日志。

证据：[原机制行](../../evidence/dcp-local-extend-20260926/review-fixes/review_mtp_info_mechanisms.json)、[target 窗口](../../evidence/dcp-local-extend-20260926/review-fixes/review_mtp_info_window_target.json)、[draft 窗口](../../evidence/dcp-local-extend-20260926/review-fixes/review_mtp_info_window_draft.json)、[完整 ON 日志](../../evidence/dcp-local-extend-20260926/review-fixes/review_mtp_info_local.log)。复算示例：

```bash
python3 scripts/analysis/dcp_route_audit.py SERVER_LOG --tp-size 2 \
  --role target --require local_short --require local_large \
  --since START --until END
```

START/END 取该臂 `measurement_window.json`，draft 同样独立执行。首次 `review_mtp_base/local` 数值诊断也通过，但离线 `sglang.Engine` 默认 `log_level=error`，没有打印机制/计数，路由审计正确拒绝。没有将那组退出码当作观测通过；诊断脚本显式设置 INFO 后补跑上述两臂。生产计算代码未为补跑改变。

调用图已增量同步，见[实际 caller 查询](../../evidence/dcp-local-extend-20260926/review-fixes/codegraph-callers.json)；机制配置与记录规则见 [115 文档](../../engine/docs/115-dcp-local-extend.md)。

## N1 与 N2

`from_runner` 独立检查 KPool=4 和 4 对齐的 page size；`COMPACT_TOPK` 关闭也不能绕过本地索引压缩的几何前提。

主开关只覆盖短尾。**P=0 的冷首块和超过短尾上限的大块不会触发。** 冷请求后续很短的尾块可能命中，但不能据此期待解决开场 chain 的主要 prefill 工作。要评估开场收益，必须单独启用大块选项，实测其覆盖比例、执行时间和显存峰值；不能把短尾百分比套到 cold chain。

## 第 3950 行：基线 A/A 已复现

新增诊断在原生 top-k 返回后保存指定行的 FP32 logits、query 字节、gate weights、pooled index-K 字节及 scale、选键和映射。只读捕获，不固定或修改生产选键。114 的行分片下，第 3950 行由 rank 0 计算 logits；两个 rank 都保存相同全局行的选键与 index-K。不能把 rank 1 没有计算该 logit 行解释成缺证。

**OBSERVED（两臂各 10 个独立进程，全部正常退出）**：layer 7 第 512/513 名分数均为 `0.0012001374270766973`，FP32 差为 0；对应物理位置 21028–31 与 23340–43。两组 key 彼此不同，但同一组跨全部对照的有效 pooled index-K、scale、query、gate weights 和 FP32 logits **逐字节相同**。80 条 rank/layer 记录完整，无缺项；logit 行由 owner rank 计算，选键与有效 pooled K 在两个 rank 均核对。

关闭新开关的 `review_base_02/03` 同样翻到 21028–31，attention 行相对参考差约 5.419%；`review_local_02` 约 5.338%。完整矩阵中，layer 7 两种选择在 base 和 local **各出现 5/10 次**，两个 rank 一致；layer 3 各臂 10/10 选键相同，候选该行最大相对差约 0.6173%。20 次 greedy 输出一致，但它不单独构成数值通过判据。证据：[完整逐字段诊断](../../evidence/dcp-local-extend-20260926/review-fixes/review_indexer_final.json)、[重复运行收据](../../evidence/dcp-local-extend-20260926/review-fixes/review_repeats.jsonl)。这是底包 A/A 的直接反例，原生输出超过 1% 不能单独证明本地索引或 LSE 合并错误。

源码也符合这个现象：`kpool_topk_transform.cuh` 最后一个 radix round 对完全相等的分数用 `atomicAdd(&s_last_remain, -1)` 竞争剩余位置，没有稳定索引次序。该行只有 3035 个有效 pooled key，低于 4096 候选缓存上限；此用例不需要“候选桶溢出”解释。数学上并列分数允许两种 top-k 选择，而 attention 的 V 不同，会把离散换组放大成输出差。

所有 owner 行的 `selection_regret = max(unselected)-min(selected) <= 0`；layer 7 为 0。这排除了该行选了更低分组的解释。两臂观察到相同的 5/5 次数，**不证明新路径绝不改变平局选择概率**；样本有限，采集本身也影响执行时序。

脚本交错两臂启动顺序，第一对保留全量 extend attention，后续只保存调查的第 3950 行，以限制诊断文件量。所有前向仍执行完整模型；缩小捕获范围不改变输入、输出或计算。原始二进制保存在开发机 `/sjtu/linhang/arena/runs/dcp-prefill-local-kv-20260926/`，不归档或清理。复现入口：[采集](../../scripts/analysis/dcp_indexer_capture.py)、[重复实验](../../scripts/analysis/dcp_indexer_repeats.py)、[逐字段复算](../../scripts/analysis/dcp_indexer_audit.py)。

## 修订后大块时间账

`review_local_10` 在数值采集完成后，交错切换两条路线，各执行 40 次完整 `ModelRunner.forward`。TP2/W2、每 rank H8、8 层缩小模型，T=8192、P=14336、同一 GPU 进程，保留 Humming 和图配置。rank 0 中位 93.165→86.454 ms（少 7.20%），rank 1 为 93.115→86.433 ms（少 7.18%）。两 rank 完整前向临时峰值均为 1312.96→1290.85 MiB。

这是修复 B1 后的开发机实测；重复前向会修改 KDA state，仅用于计时，数值判断来自此前独立前向。该轮计时的计数器尚未加入快照时间戳，40 对期间没有触发周期快照；最终带时间戳版本另由完整 MTP 启动验证。不能把这约 7.2% 外推为 TP8、chain 或 N@SLO 收益。证据：[rank 0 全部样本](../../evidence/dcp-local-extend-20260926/review-fixes/review_local_10.pt.json)、[rank 1](../../evidence/dcp-local-extend-20260926/review-fixes/review_local_10.pt.rank1.json)。

## 判据与交付边界

保留原生 8K FAIL 记录，并撤回把它当作新 kernel 错误证据的解释。**没有调大容差**：固定同一组选键的 attention 数学比较继续使用 1%；自由选键的整段比较必须先分离精确平局和真实选错。真实模型还需用 A/A 噪声对照、选键有效性、greedy/logprob 和任务能力共同判断，dummy 模型的一行不能替代 TP8 质量验证。

CPU 当前 **10 项** local policy/观测合同、**7 项**既有 DCP/MTP 回归通过。最终两卡 Engine 对照使用 H8 缩小模型、DCP W2、NEXTN 3/1/4、decode graph、HiCache 4 GB、COMPACT_TOPK=1，主开关分别 0/1，大块上限均设置为 8192（关闭臂有效值仍为 0）。**882 条敏感 trace 全部通过**原 1% 判据，最大逐行相对 L∞ 为 **0.68494%**；target EXTEND/VERIFY、draft EXTEND/DECODE/EXTEND_V2 覆盖完整，7 个请求的输出 token 一致。

两臂均证实 device repeat 为 device=4096/host=0，真实回载为 device=0/host=4096，回载与设备命中的敏感输出对照通过；严格 flush 成功后 cached_tokens=0。四个 target/draft、rank 0/1 容量收据在两臂逐项相同，physical latent rows=32768、logical allocator size=65536。该次自然接受只覆盖接受长度 1，不冒充本次重新覆盖 1–4；此前 controlled-proposal 与接受跨页证据仍见首轮报告。详见[882 条对照](../../evidence/dcp-local-extend-20260926/review-fixes/review_mtp_info_comparison.json)、[完整运行收据](../../evidence/dcp-local-extend-20260926/review-fixes/review_mtp_info_receipt.jsonl)、[回载](../../evidence/dcp-local-extend-20260926/review-fixes/review_mtp_info_local/host_restore_verdict.json)、[flush](../../evidence/dcp-local-extend-20260926/review-fixes/review_mtp_info_local/flush.json)。

TP8 实测及修订后的独立复核仍待完成；主开关和大块选项保持默认关闭。下一项探针必须保持同一引擎、相同 KV/KDA 池、相同调度，记录各 rank 显存峰值与测量窗口内路由触发；8K 每层约 0.75 GiB 临时内存的边界仍成立。

S1 修正 + DCP 的整合由既有 `e464d8ab` 提供，Fable 已安排对应任务；本轮不重复建任务或操作共用 Pod 队列。
