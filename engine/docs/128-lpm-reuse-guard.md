# 128g：原生 LPM 暂缓须有新增可复用前缀的可能

2026-09-27，Codex。默认关闭；CPU 复现与回归、Fable 独立代码审查通过。TP8 S6 筛选已闭合，同 ID chain 8→8，暂不进入候选，详见[配对归因](../../notes/reports/lpm-reuse-guard-0927.md)。属于 128 准入／held 协调修正，独立于 128p 的家族成本排序。

## 问题与证据

原生 `SchedulePolicy._compute_prefix_matches` 用 `RadixCache.create_simulated()` 建立等待队列树，默认 `page_size=1`。共享至少 32 token 就进入 `temporary_deprioritized`。实际 UnifiedRadixCache 按真实 KV 页匹配，extra-buffer KDA 检查点还须满足 `mamba_checkpoint_grid(tree_cache.page_size)`；DCP 会扩大树页。因此共享 50–58 token 可以触发暂缓，却无法形成一页新增可复用缓存。

130ezm7/130ezm8 的同一 19,335-token 冷头分别有 43/61 个 `effective_held=true` 决策，深度全部为 50/58；首次放行为第 45/64 轮，首次入批为第 50/71 轮，TTFT 为 25.420/32.255 秒。详见[原始轨迹复核](../../notes/reports/lpm-reuse-guard-0927.md)。这些是修复前的 118 OFF/ON 对照，不是 128g 的效果。

## 开关与实现

```bash
SGLANG_AX_LPM_REUSE_GUARD=1
```

任务 `G_EXPECT` 加 `128g=on`；关闭臂显式设环境变量为 `0`，要求 `128g=off`。实际机制行带 `128g=on:<grid>`；130ezn2a / S6 的实测为 `on:256`，其他配置仍须读取实际 grid，期望字符串不得写冒号。默认关闭时保持原生匹配、代表插入和 held 决策。

开启后仍先执行真实缓存匹配与原生模拟树匹配。仅对原本将 held 的请求，计算保守的**新增可复用量上界**：

1. `raw_shared` 为原生模拟树匹配的长度。
2. 按消费者的最终 logits、logprob 和 SWA 重算范围裁剪共享跨度。
3. 向下对齐真实缓存页；extra-buffer Mamba 再按实际 checkpoint grid 对齐。`no_buffer` 路径保持它的一 token 页语义。
4. 对齐后的上界减去当前 device prefix。仅当新增上界为零时解除原生暂缓。

上界大于零仍使用原来的 held，不声称缓存已经 READY；128p 的 READY 仍必须来自真实 KV/KDA validator。被释放的请求照常插入模拟树，使其真正的深前缀兄弟仍能找到代表。缓存域 `extra_key/cache_salt` 继续由原生 RadixKey 隔离。

开关要求启用缓存的 LPM，且原生 in-batch check 未关闭；bigram radix tree 在开启时报错。部署的 `kv_cache_builder` 使用 `spec_algorithm.is_eagle()` 设置 `is_eagle`，因此当前补丁不能与 EAGLE MTP 同开；本轮 S6 无 MTP。队列超过 128 时保留原生临时 FCFS 回退。此机制不新增 deadline、超时计数或请求身份规则，不改变 124/132 或 128p 家族评分，也不改 KDA/页分配、停车预算和一个 partial 的约束。

## 日志、资源与验证

128p 原有每轮 `[ax-prefix-decision]` 的候选行增加 `lpm_hold`：`shared/grid/reuse_upper/gain_upper/decision`；`decision` 是 `keep` 或 `release_zero_gain`。仅开启 128g 且本轮触发原生暂缓判断时出现。下一轮清空，避免 FCFS 回退、命中变化或代表消失后的陈旧观测。`ax-prefix-stats` 增加相应计数，口径是最多 64 个候选的逐轮观测次数，不是请求数或等待时长。无 128p 时机制可独立工作，但没有该逐请求 trace；本次探针保留 128p。

沿用 128p 的 120 秒／2048 轮默认诊断预算，每轮最多 64 行；没有新增无界日志。无新增 GPU buffer 或 KDA 槽。被释放的请求可能增加模拟树中代表数量及 CPU 临时 token/value 存储，仍受原生 LPM 的 128 请求回退限制；须在 TP8 检查 host 调度耗时，不能把 CPU 功能测试当性能结论。

验证命令：

```bash
python3 -B -m unittest discover -s tests -p test_lpm_reuse_guard.py
python3 -B -m unittest discover -s tests -p test_prefix_producer.py
python3 -B -m unittest discover -s tests -p test_sched_protect_chain.py
```

新增 14 项测试执行真实 policy、RadixKey、模拟 RadixCache 和调度器/PrefillAdder；只替换 tensor 存储、物理缓存与 GPU 执行。覆盖 19k 被浅前缀压后、代表轮换、深前缀保留、64/128/256 页和非整除 checkpoint grid、logits/logprob/SWA 上限、已有 device 命中、域隔离、FORCE_MISS、关闭路径和不支持组合。原 128p 35 项与调度器 32 项也通过。调度器机制报告测试修正为读取 working tree，并加载真实 DCP token 函数，避免旧提交测试及缺失 import 掩盖机制行漂移。

严格单变量实测使用同引擎 OFF/ON、同数据/cohort、同 S6 配置、同派发窗口和排空口径；除开关外不加 118、132 或块预算变化。本轮由 Fable 完成 ON 对旧 S6 的跨提交筛选，OFF 备用，比较边界见[交接](../../notes/handoffs/lpm-reuse-guard-0927.md)。实测救回一个无实际缓存命中的 79k 头，但新增一个稳态 reset 超时；局部前置没有净减少 chain，不能从短测外推正式 N@SLO。稳态逐请求 trace 未覆盖该等待窗口，具体准入阻塞仍待证据。
