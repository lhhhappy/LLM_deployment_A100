# Prefill 执行层：118 的 DCP 普通续算入口

2026-09-27，Codex。状态：候选实现、CPU 合同和单卡 GPU 矩阵完成，TP8 待测，交 Fable 独立审查。
分支 `codex/prefill-sm80-0927`，基于组合引擎 `791453ca` 的独立 worktree。当前未改准入、排序、缓存或队列。

## 为什么先做这里

目标仍是提高正式 N@SLO，重点降低开场和稳态冷 chain 的 prefill 执行时间。没有把“快 10% 少一条
chain”当作结论：同样的省时落在不同请求和不同等待位置，过门收益可以不同。

本轮用现行 `scripts/pod/verify/prof_ledger.py` 在 Pod CPU 上重算两份 rank0 原 trace：单请求目标模型
完整 8k 块 591.9 ms（5 块），负载下 584.1 ms（28 块）；草稿独立列为 27.1 / 25.1 ms。负载窗口
98.7% 在 extend、decode 为 0%，开场几乎没有 decode 时间可再交换。原先目标/草稿混合的均值不能使用。
[重算账本](../../evidence/prefill-sm80-0927/profile-ledger.json) 保留各形状的 span、busy 与计数。
其粗粒度 kernel 分类会漏掉匿名 DSA/Humming 核，不能把 `other` 或 `kda` 百分比直接当作真实算子归因。
Fable 人工映射的目标块摘要仍是：DSA `main_kernel` 21.7%、mHC 三核约 13.9%、all-reduce 11.7%；
本轮独立重算确认的是完整块时间与执行阶段，未将上述人工分类冒充本脚本输出。

**本轮 GPU0 实测，未推进成生产改动：** 113 indexer 在 114 已分摊 query rows 后，8192-token/TP8
对应每卡 1024 行。固定 tile 的 LOOP 4→8，在 128k 上下文约 2.130→2.034 ms，在 256k 上下文约
4.303→4.100 ms；256k 改 GROUP=64 的一组为 4.073 ms。这只是单卡合成算子、同进程交替计时，尚不是
真实模型或服务收益。128k 即使按目标 11 层相加也只约 1 ms，因此本轮不为这点收益增加生产旋钮。
原 JSONL 留在开发机 `/sjtu/linhang/arena/codex/prefill-sm80-0927/results/tune-1024-{32768,65536}-causal.jsonl`；
可复现工具为 [indexer probe](../../scripts/analysis/bench_prefill_indexer.py)。

**历史可核实的候选证据：** [118 原始 bench](../../evidence/dsa118/bench.jsonl) 的单卡 H8、D512、8192 query rows：
49k 上下文 10.587→6.585 ms，首个 8k 块 9.943→5.145 ms。测量是 CUDA graph 内随机页/索引的算子计时，
不是当前 DCP 完整 KV 布局、更不是本轮 TP8 结果。旧 [118 数值与资源说明](../../engine/docs/118-dsa-sparse-triton.md)
已覆盖掩码、尾列、fp32 参考、图重放及预热，但整个开关拒绝 DCP，尚未在当前服务栈兑现。

本轮选择重用这份数值核，补齐受限的 prefill 入口。若当前 TP8 也能每层省 4 ms，11 个目标 DSA 层是
约 44 ms/块、相对 592 ms 约 7.4%——这只是选择方向的算术，不是速度预测或 chain 条数承诺。

## 调用契约与具体改动

CodeGraph 用于定位 `_forward_tilelang` 的 `forward_extend` / `forward_decode` 调用关系；最初根目录索引
较旧，因此所有关键边和条件均回到本 worktree 的完整函数核实，并在本 worktree 引擎建立独立索引。

普通 DCP prefill 的实际路径：`forward_extend` → 已有 `_ax116_dcp_extend_rows` → 完整 KV 和重映射后的索引
→ `_forward_tilelang`。TP8/DP1/CP1 下本卡 8 头；DCP 大小不会改变 `attn_tp_size`，普通 prefill 没有 Q 头聚合。
局部续算和 target verify/draft-extend-v2 则提前进入 partial/LSE 分支；decode 有独立调用。

新增默认关闭的 `SGLANG_AX_DSA_SPARSE_TRITON_PREFILL=1`：

- 只有普通 `ForwardMode.EXTEND` 的完整 KV 调用显式传 `is_prefill=True`；wrapper 还要求 `return_lse=False`。
- 本地续算、partial/LSE、verify、draft-extend-v2、MIXED 和 decode 不选新入口。普通草稿模型 EXTEND 若走完整
  KV 路径，同样适用；不能笼统声称所有草稿前向都保持原核。
- 旧的 all-phase 开关仍拒绝 DCP；两个开关同时设置时报错。dtype、RoPE、HiSparse、确定性模式的守卫保留。
- 继续在后端初始化时预热本卡头数的全部 split 变体，不将预热移入 forward。
- 机制行报 `118=on:prefill`；首次真实进入时每个 backend 记一行 `118 route=full_kv_prefill`，带 tokens、heads、
  index width 和 DCP 状态。它证明入口被走到，不能作为 CUDA graph replay 次数。

关闭两个开关后张量运算沿用原路径。没有改 kernel 数学、top-k、owner 公式、DCP gather/LSE 通信或调度策略。

## 显存账

没有新增持久张量。在 H8/8192 行时，省去原 TileLang 的 2051→2112 列 padding 副本，算术为 66 MiB；
输出仍为 64 MiB，不能把输出也算成节省。小行数普通 EXTEND 会使用既有 split 临时缓冲，按历史 A100
324 个驻留 program 的测量上限约 5.1 MiB。真实 KV 池、启动 kernel 代码开销和 graph pool 的变化均待实测。

## GPU 新测量（2026-09-27，开发机，不是 TP8）

已验证：新增 3 项 CPU 测试运行真实源码提取的方法，覆盖 27 个路由/启动组合；原 10 项 DCP local-extend
合同通过；Python 语法、shell 语法和 diff 空白检查通过。CPU 测试使用 kernel recorder，不能证明 GPU 数值。

用户恢复 SSH 后，在独立目录使用 GPU0 完成测试与成本矩阵。A100-SXM4-80GB、torch 2.13.0+cu130、
Triton 3.7.1、TileLang 0.1.12，候选执行代码为 `3caadef4`。没有停止、重启或部署 Pod 服务。

本轮结果：

- [GPU 118 测试](../../tests/gpu/test_dsa_sparse_118.py)：15 项通过（152.4 秒），包括 14 个数值形状、4 个
  CUDA graph 形状、路由/启动守卫、负例以及两次新进程预热；两次都是预热加载 14 个 kernel，此后
  21 种行数 × 两种 width × 连续/切片索引表额外加载为 0。
- [普通 prefill 成本曲线](../../scripts/analysis/bench_prefill_sparse.py)：复用既有 fp32 oracle，完整连续 KV，
  8k–262k 上下文、333–16384 query rows、四请求混合批。交替测 OFF/ON 的 eager GPU 时间、墙钟和临时峰值，
  保留全部样本；包含 kpool 边界与请求边界的参考行。合成索引不能替代真实 top-k 局部性。
- [开发机运行入口](../../scripts/analysis/run_prefill_devbox.sh)：使用现成 m0 环境，GPU0，源码和全部新增缓存限于
  `/sjtu/linhang/arena/codex/prefill-sm80-0927`。不会安装第二套环境或修改共享开发源码。

成本矩阵 9 个形状全部通过 fp32 参考界限；ON 的平均误差不超过 OFF 的 1.25 倍加 1e-7。
实际为同进程真实 wrapper OFF/ON 随机交替 5 轮、每轮 5 次；L2 在每次计时前清理，保留所有样本。
以下是 eager CUDA event 的中位数，包含原 wrapper 的 padding；没有 DCP collective 或真实 top-k。

| 上下文 / 新算行数 | OFF ms | ON ms | 单次算子加速比 |
| --- | ---: | ---: | ---: |
| 首块 8192 / 8192 | 10.525 | 5.277 | 1.99× |
| 16384 / 8192 | 10.603 | 6.023 | 1.76× |
| 49152 / 8192 | 11.252 | 6.485 | 1.74× |
| 131072 / 8192 | 11.637 | 8.316 | 1.40× |
| 262144 / 8192 | 12.242 | 9.298 | 1.32× |
| 49152 / 1024 | 2.625 | 1.120 | 2.34× |
| 30011 / 333 | 1.551 | 0.598 | 2.59× |
| 49152 / 16384 | 21.711 | 12.747 | 1.70× |
| 四请求混合 / 合计 8192 | 11.382 | 7.556 | 1.51× |

8k 行时 ON 的临时峰值增量为 64 MiB（输出），OFF 通常为 130.25 MiB（16384 上下文一组为 131.00 MiB）；
16k 行为 128 对 260.5 MiB。未新增持久张量。成本矩阵预热单个 width 加载 9 个核，此后额外加载 0。
部分 eager 样本有 host/launch 长尾，全部保留；表中位数不能充当服务尾延迟。更长上下文时节省变小，也不能
用 49k 的加速比代表 262k。完整日志为 [GPU suite](../../evidence/prefill-sm80-0927/118-prefill-tests.log) 与
[成本矩阵](../../evidence/prefill-sm80-0927/118-prefill-cost.jsonl)。

复现（候选源码已上传到独立目录）：

```bash
PREFILL_ROOT=/sjtu/linhang/arena/codex/prefill-sm80-0927
bash "$PREFILL_ROOT/scripts/analysis/run_prefill_devbox.sh" "$PREFILL_ROOT" \
  tests/gpu/test_dsa_sparse_118.py -v
bash "$PREFILL_ROOT/scripts/analysis/run_prefill_devbox.sh" "$PREFILL_ROOT" \
  scripts/analysis/bench_prefill_sparse.py
```

## Fable 复核与 TP8 接口

先独立核普通 EXTEND 与 partial 分支的边界、预热头数、OFF 路径和机制行。候选环境仅增加
`SGLANG_AX_DSA_SPARSE_TRITON_PREFILL=1`，旧 `SGLANG_AX_DSA_SPARSE_TRITON=0`；ON 的 `G_EXPECT`
包含精确 token `118=on:prefill`。仍保持已选对照的 DCP、MTP、running、块大小、数据与派发窗口，避免混合归因。

TP8 先查真实能力冒烟、8 个 rank 的入口与无晚编译、KV 池/显存；用同一 49k 提示/8k 块测目标层与整个
forward，不把 draft 平均进去。再用同数据、同派发窗口的开场和含稳态冷段首探针，逐 ID 核 TTFT 与所有 SLO 门。
不能把历史 118 的 decode 加速写成这个 prefill-only 开关的收益。

负载验证还须区分源会话起点与回放链首；用户新提供的是源业务历史缓存统计，不是正式运行缓存。
参见 [链首来源核实](chain-origin-audit-0927.md)。本候选按执行阶段生效，稳态普通 prefill 同样覆盖，
不依赖开场计时、固定请求 ID 或源会话标签。

## mHC post 探索：本轮不推进生产改动

在 GPU1 用 [独立算子探针](../../scripts/analysis/bench_mhc_post.py) 对现行 TileLang post 做了直接 Triton
读写原型。它没有接入引擎，不改变运行开关。简单逐项 FMA 与原核有少量末位差异；把首个乘加改为
`fma(post, x, mix0 * residual0)` 再依次加入其余 residual，在本次 8192/128 行、H4096 的全输出比较中
逐位相等，且 64 个参考行满足独立 float64 误差界限。这个局部结果不代表所有输入逐位等价。

关键负结果：8k 行 TileLang 为 0.3648 ms，最好的两个直接读写原型为 0.3656 / 0.3660 ms，没有大块收益。
128 行有较短计时，但 eager 波动显著，不能拿来声称目标大块提速。因此本轮不增加 mHC 生产旋钮；
后续应关注 post/pre 融合减少中间张量搬运，而不是仅重写同一 post 的存取。原始两次测量：
[初始表达式](../../evidence/prefill-sm80-0927/mhc-post-probe.jsonl)、
[调整乘加顺序](../../evidence/prefill-sm80-0927/mhc-post-fma2-probe.jsonl)。

119 的大块 scatter 候选已经存在，应先区分其收益与新增 kernel 的收益；不重复开发已有机制，
不把孤立算子百分比直接相加。当前最直接可交 TP8 的仍是 118 prefill 入口。

Fable 提供的原 trace：

- `/tmp/ax/runs/130ezc-tp8_prefill_profile_dcp2/tp8_8k/traces/*TP-{0..7}*.trace.json.gz`
- `/tmp/ax/runs/130ezd5-v3_open_S1dcp_profile_n34/N34/traces/`

两目录中的旧 ledger 混合了 target/draft；本轮重算文件在 Pod `/tmp/ax/codex/prefill118-profile-ledger.json`，
只对上述两个 rank0 trace 作 CPU 分析。队列安排仍由共享 queue 与任务入队者管理，本提交没有新增 Pod 任务。
