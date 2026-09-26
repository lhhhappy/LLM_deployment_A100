# DCP / NextN 开发机诊断证据

2026-09-26，Codex；源码自审，尚待另一参与者独立复核。这里没有正式成绩，也没有 TP8 或 N34/N38 通过结论。目标和技术方案见 [DCP计划](../../notes/plan-dcp-8card.md)。

## 代码和环境

- 开发起点 `aebaff56`；独立分支 `codex/dcp-mtp-n34`。115 提交 `a90ac349`，180 提交 `f254c64f`。
- 当前8卡候选分支为 **`codex/dcp-mtp-stack2-n34`**，将相同引擎补丁整合到现行服务基线；`fix5`为其冻结源码，全部4,697文件逐项一致，见[源码收据](phase3/fix5-source-verification.json)和[整合差分](integration-patch-receipt.json)。开发机完整矩阵`tp2_13`使用该候选源码。
- GPU 原始目录：`/sjtu/linhang/arena/runs/dcp-mtp-20260926`。`base` 是开发起点；`fix` 是第一版补丁；`fix2` 把 indexer 容量校验移到实际构造器；`fix3` 加可选 KPool 列选择及 DCP-off 原路径；`fix4` 再修 SM80 的 DCP4 共享内存布局。
- `fix4` 的全部 4,693 个受控源码文件与 `f254c64f` 逐项 SHA256 一致。预期清单：[engine-f254c64f-sha256.json](engine-f254c64f-sha256.json)；GPU 收据在上述目录的 `fix4-source-verification.json`。
- A100-SXM4-80GB；开发环境复用 `arena/env/m0`，PyTorch 2.13.0+cu130、TileLang 0.1.12。完整环境变量由 [驱动](../../scripts/analysis/run_dcp_devbox.sh) 设置，CUDA compatibility 库、JIT/cache/临时文件均在 `arena/`。
- 缩小 GLM 为8层（2 DSA + 6 KDA）、NextN 1层、attention头数8、hidden4096。`model-mtp-r2` 修正了缩层后18条量化排除规则的层号。用参数名播种的有量纲权重，DSA敏感观测为 `o_proj` 输入；不能单凭该模型 logits 一致判正确。配置与生成规则见 [fixture脚本](../../scripts/analysis/dcp_mtp_fixture.py)、[配置](model-mtp-r2/config.json)。

## 已闭合诊断及边界

| 记录 | 观察 | 能支持的结论 |
|---|---|---|
| [CPU契约](contract_fix2.json) | 33/40、33/34补齐，三阶段路由，页内indexer搬运及HiCache组合检查 | 原函数契约；没有真实通信或模型前向 |
| [move1](move1.json) | 2 rank NCCL，24项，BF16/uint8逐字节一致 | 跨owner、重叠循环、重复源、空目标rank、零padding；不证明压缩KPool树式MTP |
| [core4](compare_core4.log) | TP2 W1/W2；cold ≤1.408e-4，prefix extend ≤1.961e-3，decode ≤5.376e-3 | 真实普通前向，低地址，eager |
| [mtp4严格复核](phase2/compare_mtp4_strict.json) / [图复核](phase2/compare_mtp_graph4_strict.json) | 每臂504条DSA观测，最大相对L∞约7.576e-3；输出token相同；三种图均有重放证据 | TP2+W2完整NextN低地址路径，**自然接受长度仅1**；当时hybrid目标indexer尚未按虚拟容量扩容 |
| [host_gpu7b](phase2/host_gpu7b_fix.log) | fix2真实构造器；W2/4/8，rank0及末rank共6项，CUDA往返逐字节一致 | 目标/草稿latent和indexer、KDA/conv异址恢复，真实CUDA mover；不是完整模型续算 |
| [host_cpu10](phase2/host_cpu10_fix.log) | 同6项通过，CPU仅替代裸字节mover | 与GPU用例相同的实际池构造和控制器路径；旧构造器因虚拟indexer容量不足失败 |
| [sparse10](phase2/sparse10.json) | 真实KPool输出；6组rank/width边界检查、18组attention+LSE+FP32抽样+graph通过；最大逐行相对误差0.007752 | 可选列选择的局部正确性；H32生产wrapper已验证eager/graph；不含TP8通信 |
| [move13](phase3/move13.json) | 当前候选源码，2 rank NCCL，24项逐字节一致 | 跨rank owner变化的MLA/indexer搬运；压缩KPool任意token压实仍明确拒绝 |
| [mtp13](phase3/compare_mtp13.json) | 同TP2，W1 eager对W2 graph，504条观测；最大有效行相对误差0.007519；三类图各96次rank重放 | 当前真实构造器和完整NextN自然提议路径，输出token相同 |
| [accept13](phase3/compare_accept13.json) | W1/W2都开graph；264条观测，最大有效行相对误差0.007353 | 四种输入、每个rank的真实verifier均实际接受1/2/3/4；[接受收据](phase3/mtp_tp2_accept_dcp13/acceptance_verdict.json)；三类图各48次rank重放 |
| [compact13](phase3/compare_compact13.json) | W2开启/关闭列选择，264条有效观测逐位一致、输出相同 | 列选择接入完整缩小MTP；仍不代表TP8通信已验证 |
| [host13](phase3/compare_host13.json) | W1/W2各756条观测；最大有效行相对误差0.007519；三类图各144次rank重放 | 真正挤出后host命中4096、device命中0，恢复续算输出相同；[恢复](phase3/mtp_tp2_host_dcp13/host_restore_verdict.json)与[真flush](phase3/mtp_tp2_host_dcp13/flush.json)成功，清空后cached_tokens=0 |
| [high_graph13](phase3/compare_high_graph13.log) | 物理32768行，逻辑65536，最大loc55484；普通decode图8次；最大attention相对误差0.005376 | 当前容量修复后的高地址普通前向；本用例不含MTP |

以上是已闭合的开发诊断，[调度日志](phase3/jobs/dcp_0926_tp2_13.log)为`DONE rc=0`。可用[汇总脚本](../../scripts/analysis/dcp_receipt.py)从JSON收据重新生成[开发摘要](development-summary.json)，不调用或冒充原harness判分。

实际池几何也已留证：[target](phase3/mtp_tp2_host_dcp13/capacity_target0.json)两层latent共67,239,936 B、indexer共17,335,296 B；[draft](phase3/mtp_tp2_host_dcp13/capacity_draft0.json)单层分别33,619,968 B与8,667,648 B。两者latent均32768物理行加64 padding，复制式indexer覆盖65664逻辑行；不能把探针固定容量下的这些字节当成真实权重服务的新增KV容量。

`mtp3` 不能用作通过证据：量化排除规则与前缀检查点不对齐，504条观测里26条超门槛。`host_gpu1` 的旧测试手动重建了 indexer，会掩盖生产构造错误；现行夹具已去掉该重建。`high_graph5` 在 NCCL 初始化失败，未运行到高地址前向。`gpu9` 的 MTP 因另一个服务重启占用显存而 OOM，没有数值结果。失败日志保存在 GPU 原目录及分期归档中。

`mtp_tp1_host12`恢复数值已通过，但结束时探针未正确序列化flush返回类型而退出1；不能将它记作完整通过。`kit13`修正收据序列化后，`host13`的两个rank、W1/W2与flush才全部闭合。`accept10`对输出末尾提前发起的一轮错误套用了已耗尽的oracle，`accept12/13`明确保留该轮而标为unchecked，不增加覆盖计数。

## 性能观察的范围

`sparse10` 每个形状交替次序测16对，每次10次调用，保留全部 eager/graph 样本。T=136/152模拟N34/N38每路验证4行，H=16/32/64模拟TP8、W2/4/8。此次 graph p50：

| W | 原列列表 → 列选择（T136/152、rank0/末rank范围） | 比值范围 |
|---|---|---|
| 2 | 0.267–0.325 ms → 0.156–0.190 ms | 1.71× |
| 4 | 0.408–0.411 ms → 0.131–0.148 ms | 2.77–3.11× |
| 8 | 0.690–0.694 ms → 0.214–0.236 ms | 2.94–3.23× |

这是单卡局部核函数加索引映射的观察，包含不了通信、完整模型、prefill等待或服务吞吐。H32两条计时路径均采用能在A100运行的单stage布局；旧默认布局要求196608 B共享内存，无法作为可运行速度参考。前两轮 `sparse7b/9` 不作独占GPU的速度依据。

## MTP 数值观测合同

[探针](../../scripts/analysis/dcp_mtp_probe.py)保存实际forward mode、角色、rank、层、位置、seq_lens、原始DSA采样行及graph重放编号。比较器要求输入/元信息匹配、有限值、有非零参考信号，缺文件或不完整运行不能通过。

`DRAFT_EXTEND_V2` 总会计算完整4行窗口；worker只取 `accept_lens-1` 的预测，KPool更新也读取 `num_accept_tokens`。主机恢复后，拒绝行可能保留不同的暂存内容。`host10` 中6条不一致均只发生于拒绝行，真实接受的第0行逐位一致。现行探针从运行时接受长度捕获 `valid_rows`，同时保留全部原始行及全行误差；有效行门槛仍为1%，并检查逐行误差。不能根据哪里误差大来推断掩码。`host11` 的失败来自探针在图捕获中使用CPU索引列表，已改为设备切片；不算引擎图失败。

[接受长度诊断](../../scripts/analysis/dcp_mtp_acceptance.py)仅替换送入真实tree builder的proposal，来源为同TP自然贪心参考。目标logits、真实verifier、KV写入、KDA提交都保持原路径。每个输入必须实际覆盖接受1/2/3/4并产生相同用户输出。异步调度在最后一个输出仍在发送时可能预先运行末尾一轮；超过保存的参考输出范围时保持自然proposal，明确记为unchecked，不能用它增加已验证覆盖或报告接受率。该诊断不用于性能结论。

## 复现与归档

所有开发机运行需显式选择可用卡。单卡用GPU1 UUID；只有跨rank测试需要两卡。命令示例（模型与冻结source必须已存在）：

```bash
R=/sjtu/linhang/arena/runs/dcp-mtp-20260926
DCP_DEVICES=GPU-0fd597b1-c02f-3a31-8fa7-8ecb15791ca8 \
DCP_TP=1 DCP_PROBE=mtp DCP_HICACHE=1 DCP_RESTORE=1 \
  bash "$R/run_dcp_devbox.sh" "$R" "$R/fix4/engine" new_host_arm 1 1 0
```

- 每臂输出目录拒绝覆盖；源码与控制脚本分别冻结在 `fix*` 与 `kit*/audit*`，SHA256收据保存在GPU原目录。控制脚本改进不改写原始运行。
- 第一批压缩原始档案在共享仓库忽略目录 `build/scratch/dcp-mtp-20260926/phase1-evidence.tgz`，39,154,894 B，SHA256 `92dbb122f1009329bdf3a9364d81ecdfd00bb6253f0b0ec552b30839abe0355a`。[逐文件收据](phase1-local-receipt.json)与[GPU交叉验证](phase2/phase1-cross-verification.json)：2573同路径匹配，3个控制文件匹配其冻结快照，零不符。
- 第二批已闭合证据的[本地收据](phase2/local-receipt.json)列出20个文件及哈希；保留未纳入git的压缩包和完整 `.pt`。这里的原始文件名不因归档而改变。
- 第三批包括完整成功矩阵及上述失败探针，GPU永久归档`/sjtu/linhang/arena/archives/dcp-mtp-20260926/phase3-evidence.tgz`，50,073,526 B，SHA256 `45238e76b6e8eee5501155af7a33a9ebf6098df15f99a1abaca0daddcc595cdc`。4,651文件、116,572,396 B原始内容全部验证；[逐文件清单](phase3/remote-manifest.json)、[传输收据](phase3/local-receipt.json)。本地同名压缩包在`build/scratch/dcp-mtp-20260926/`，完整`.pt`解压于本目录，git仅记录文本证据和哈希。
- 顶层空的 `compare_mtp4_strict.log`、`compare_mtp_graph4_strict.log` 是本地缺环境的重定向产物。有效严格复核在 `phase2/`，不能混用。
- 运行dispatch/drain、完整请求计数和原harness评分不适用于这些定向开发探针；不生成或冒用SLO成绩文件。TP8能力、全量N34/N38、实际显存收益和另一参与者复核仍需独立收据。
