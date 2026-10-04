# 从底包到最新 N42：探索历程与逐提交阅读索引

2026-10-03，Codex 整理。源码阅读对象是 `engine-base..ca5d646c`；正式源码与配置身份见 [提交记录](submissions.md)。下面保留原始 commit ID，将问题、实现、验证和取舍接起来。

## 建议阅读顺序

先读 [源码地图](architecture.md)，再按此表进入一个阶段。想理解单次修改，在下方逐提交索引点开 commit，再看它的机制文档、测试和证据。正式成绩是整个源码与配置组合的结果，单个补丁没有独立的 N@SLO 分数。

| 阶段 | 当时要解决什么 | 代表提交 / 原始标签 | 看结论的地方 |
| --- | --- | --- | --- |
| 1. 固定底包和接口 | 怎样正确接入评测并如实计时、计数与 flush？ | `engine-base`、000、101、106 | [接口说明](../engine/docs/000-interface-compliance.md)、[赛规](../llm-challenge-arena-v1/task.md) |
| 2. A100 基础执行 | DSA / FP8 MoE 在 sm80 上怎样工作，怎样减少 TP 重复计算？ | 110、111、114；`official-A-0923a` | [引擎机制](../engine/README.md)、[底包研究](../research/README.md) |
| 3. 缓存与资源 | 长链 KV、KDA 状态、DSA index 如何保持一致，host tier 能省多少重算？ | 180；`image-lh-img-0925a` | [HiCache](../engine/docs/180-hicache-glm-dsa.md)、[缓存研究](../research/codex/R18_cache_loss_and_capacity.md) |
| 4. 等待与冷启动 | 大冷头把谁堵住了，怎样分配冷/热预算和 decode 份额？ | 117、120、124、125；`image-lh-img-0926b` | [124](../engine/docs/124-deadline-admission.md)、[125](../engine/docs/125-opening-mode.md)、[实验记录](experiments.md) |
| 5. 探索容量与共享 | DCP 在本地很有效，为什么没有成为最新配置？共享前缀怎样形成真实收益？ | 115、128；`image-lh-img-0926c` / `0926d` | [DCP](../engine/docs/115-dcp-sm80.md)、[prefix producer](../engine/docs/128-prefix-producer.md)、[数据边界](knowledge.md) |
| 6. chain-first | 如何救冷链首，并保证 ranks 节奏一致？ | 131、132、`20a58da9`；`image-lh-img-0927a` | [chain-risk](../engine/docs/131-chain-risk-interval.md)、[chain-first](../engine/docs/132-chain-first.md)、[审查](../evidence/chainmax-review-0927/contracts.json) |
| 7. prefill 执行成本 | 调度之后，能否把同样的 8k/16k 工作算得更便宜？ | 118、117 follow-up、KDA prefill；`image-lh-img-0928c` | [KDA prefill](../engine/docs/170-kda-sm80-prefill.md)、[09-28 复盘](reports/0928-official-local-synthesis.md) |
| 8. 09-29 单变量筛选 | 间隔、冷块、短请求阈值、状态池哪些值得继续？ | `bf6b66fa` 上的多份配置；分数不能归因新源码 | [夜间方案](reports/night-n34-tuning-0928.md)、[N42 chain 筛选](reports/chain-night-0929.md)、[正式记录](submissions.md) |
| 9. 09-30 执行改动 | 减少 decode 和 host 元数据的重复工作，再选择两份正式配置 | `d18933a9..ca5d646c`；`image-lh-img-0930a` | 本页下一节、[最终选择](../evidence/submission-0930-execution/decision.json)、[正式终态](submissions.md) |

## 09-30 执行改动

| 机制 / commit | 为什么改 | 47798 配置选择 | 原始设计与验证边界 |
| --- | --- | --- | --- |
| 174 / `d18933a9` | packed KDA decode 候选 | 关闭 | [174](../engine/docs/174-kda-safe-packed-decode.md) |
| 175 / `6b629210` | 小 M Humming 下投影选用不同 K tile / CTA 配置 | 请求启用；不支持的形状回退 | [175](../engine/docs/175-moe-small-m-down-tuning.md) |
| 176 / `400adf7d` | DSA paged logits 在连续四页间复用 query | 请求启用；小网格保留原核 | [176](../engine/docs/176-dsa-sm80-decode-page-loop.md) |
| 177 / `f3a99079` | 启动期捕获 decode metadata graph | 关闭 | [177](../engine/docs/177-metadata-glue-startup.json) |
| 178 / `8f19417f` | 同一 forward 跨层共享 prefix-state mask | 请求启用 | [178](../engine/docs/178-kda-shared-prefix-mask.json) |
| 179 / `86b82497` | GPU gather prefill track，异步准备元数据 | 请求启用；受类型/模式守卫 | [179](../engine/docs/179-mamba-prefill-track-gpu.json) |
| 181 / `7f87346f` | 对齐条件满足时复用 BF16 query view | 请求启用 | [181](../engine/docs/181-dsa-query-view.json) |
| 182 / `ce395da5` | 两次 Mamba 索引赋值共享一次上传 | 请求启用；保留所有权与 stream 契约 | [182](../engine/docs/182-mamba-alloc-gpu-index.md) |
| 184 / `ca5d646c` | HiCache 操作复用同一个只读索引 Tensor | 请求启用；只复用完整 Tensor identity | [184](../engine/docs/184-hicache-index-alias.json) |

这些机制文档记录当时的单卡/AST/算子验证范围。后续服务组合对照看 [09-30 实验记录](experiments.md#09-30-执行组合与最后两包)，最终正式结果看提交记录。正式 CAP 和 TPOT 共用全部源码，只改变两项环境变量；TPOT 的能力失败原因不能从这个配置差异直接推断。

## 探索中放弃或未采用的路线

| 路线 | 已有记录支持什么 | 最新配置的选择 / 边界 |
| --- | --- | --- |
| DCP 扩容量 | 本地短 gap 造成的 KV 压力下有效；正式 S3/S4 没有建立相对 S1 的同条件 chain 净收益 | 未启用。先复核本地负载与正式的差异，见 [115](../engine/docs/115-dcp-sm80.md) 与 [知识库](knowledge.md) |
| 122 TPOT pacing | 有正式和本地对照；后续替代激进 125 的窗口增加 chain 坏例 | 关闭，历史结果留在 [实验记录](experiments.md) |
| 126 demand cold cap | 两次早期对照方向不一致，尚无可复现的 chain 净收益 | 关闭；[126](../engine/docs/126-demand-cold-cap.md) |
| 133 decode budget | 09-28 单变量加入后新增 chain 坏例，原记录否决该臂 | 未合入正式候选；[夜间审计](reports/night-n34-tuning-0928.md)，原分支仍可查 |
| 09-29 冷块、短阈值与停车筛选 | 五组 N42 指定派发窗口排空；汇总未建立超过两次基线噪声的稳定收益 | 见 [chain-night](reports/chain-night-0929.md)，窗口不能当完整正式晋档 |
| 174 / 177 / metadata fusion | 单独的能力、路由、生命周期或成本探针有各自边界 | 最新冻结配置关闭；不把“进入源码”当作“进入运行” |
| 47800 / TPOT | 平台返回能力门未通过，未做压测 | 保留失败配置、原始 ZIP 和官方回报；原因仍未知，见 [提交记录](submissions.md) |

## 逐个读 65 个引擎 commit

此表由 `git log --reverse --format='%H %s' engine-base..ca5d646c -- engine/sglang` 对照生成，按 Git 输出顺序列出全部源码提交。每行的中文说明解释改动要解决的问题；验证范围应回到该 commit 的测试、机制文档与关联实验。

| 序号 | 原始 commit | 这一步要理解什么 |
| --- | --- | --- |
| 01 | [da281c67](https://github.com/lhhhappy/LLM_deployment_A100/commit/da281c6797c886951422ddd4bc7cd4cf3f67cf62) | 接口返回真实计时、计数与可清空缓存 |
| 02 | [0ae3be67](https://github.com/lhhhappy/LLM_deployment_A100/commit/0ae3be67cd1136d8ac284e3c28f2bf1916066929) | 在角色边界保存可复用的 KDA 状态 |
| 03 | [7da95c6e](https://github.com/lhhhappy/LLM_deployment_A100/commit/7da95c6efa620b802b43351cb291b3ba63062ec3) | KV 不足时推迟续算，避免 prefill OOM |
| 04 | [96fc86d3](https://github.com/lhhhappy/LLM_deployment_A100/commit/96fc86d39d77ba156ed780ddfabf2a45b4003fac) | 让 DSA 索引打分能在 A100 上执行 |
| 05 | [8f4e4c5e](https://github.com/lhhhappy/LLM_deployment_A100/commit/8f4e4c5e7790877fe8f1d60276c5b2d2a58efd5b) | 用 Marlin 执行 A100 上的 FP8 MoE 权重 |
| 06 | [cf7e00a5](https://github.com/lhhhappy/LLM_deployment_A100/commit/cf7e00a5c29a61fc75259110978d00d362697b6f) | 把索引查询行分给 TP ranks，减少重复计算 |
| 07 | [bdeca5c3](https://github.com/lhhhappy/LLM_deployment_A100/commit/bdeca5c3a9fb234d56b4acf487a7807c2a7e4833) | 限制长冷 prefill 对短请求与 decode 的阻塞 |
| 08 | [179eb466](https://github.com/lhhhappy/LLM_deployment_A100/commit/179eb466e187daaccbe86478f1d084a44bca2195) | 已有 decode 时继续限制冷块大小 |
| 09 | [c1b6e1ae](https://github.com/lhhhappy/LLM_deployment_A100/commit/c1b6e1aeb9286cb73105126b2d920c56b2f9d530) | 把完整分词移出 HTTP 事件循环 |
| 10 | [ad931d6b](https://github.com/lhhhappy/LLM_deployment_A100/commit/ad931d6b25e55f12962e4cec33b926dc35dee057) | 保留角色边界与 prompt-end 的 FP32 KDA 快照 |
| 11 | [6a9bea37](https://github.com/lhhhappy/LLM_deployment_A100/commit/6a9bea37aa949ec798fffe9bf8eb2f8e16eed091) | 启动期覆盖代表性请求形状 |
| 12 | [ed1d37f9](https://github.com/lhhhappy/LLM_deployment_A100/commit/ed1d37f9631bfc2039f68925d243ee7ff4d58b0b) | 兼容 A100 的 NEXTN 与统计 |
| 13 | [152617cb](https://github.com/lhhhappy/LLM_deployment_A100/commit/152617cb3c9dce796d03406636e68c3b7cea44ff) | 探索可以打断的 prefill CUDA graph |
| 14 | [0cb0eab6](https://github.com/lhhhappy/LLM_deployment_A100/commit/0cb0eab6909b8dd354d22a11c41ceedebc818cfc) | 探索 DCP 扩大每卡 KV 容量 |
| 15 | [fe121b23](https://github.com/lhhhappy/LLM_deployment_A100/commit/fe121b23c22d90c6694e8a24a4fae498b97ed784) | 根据 TPOT 调整 prefill 预算 |
| 16 | [015d5bd2](https://github.com/lhhhappy/LLM_deployment_A100/commit/015d5bd2c8e060d0f5173659f4ef27d29063fd87) | 按剩余 prefill 工作量排序，并限制饥饿 |
| 17 | [c2132b63](https://github.com/lhhhappy/LLM_deployment_A100/commit/c2132b6349b1a837b59d1936ee5c813442c60b82) | 探索 KDA BF16 投影融合 |
| 18 | [9f53ac59](https://github.com/lhhhappy/LLM_deployment_A100/commit/9f53ac5997c6fd4482e36a88e71ce8f57e9a2db9) | 探索 MoE clamped-SwiGLU 融合 |
| 19 | [f7bcf752](https://github.com/lhhhappy/LLM_deployment_A100/commit/f7bcf7523e517ca12448cc324439998e8724c4aa) | 把 DSA 索引和状态所有权纳入主机缓存 |
| 20 | [e43e2dcf](https://github.com/lhhhappy/LLM_deployment_A100/commit/e43e2dcf2934a8b878ba405b451027b306964069) | 报告启动时实际机制及关闭原因 |
| 21 | [8a44a24d](https://github.com/lhhhappy/LLM_deployment_A100/commit/8a44a24d8424736200842d443b2b9d90c6513ec5) | DCP 关闭时保留原生稀疏注意力路径 |
| 22 | [759a6ebb](https://github.com/lhhhappy/LLM_deployment_A100/commit/759a6ebb8e31723519ad5daf438e26e24b32501a) | 只为当前确实能入批的命中保留空间 |
| 23 | [37e90023](https://github.com/lhhhappy/LLM_deployment_A100/commit/37e9002368390fefc92621ea8b18f73e86429596) | 在 TP ranks 之间同步准入顺序 |
| 24 | [26002a14](https://github.com/lhhhappy/LLM_deployment_A100/commit/26002a149d0271e18971640234a5fbf9aa183ece) | 引入 Humming 的 A100 FP8 MoE 执行路径 |
| 25 | [928b9e18](https://github.com/lhhhappy/LLM_deployment_A100/commit/928b9e186e19c3fdecb07971e31a26578b6d2ec1) | 让 attention-TP scatter 可以按块长门控 |
| 26 | [9f483efc](https://github.com/lhhhappy/LLM_deployment_A100/commit/9f483efce4b83c469d40911df7af3a9cb9fe5a22) | 修正小于 checkpoint 网格的续算处理 |
| 27 | [14097601](https://github.com/lhhhappy/LLM_deployment_A100/commit/14097601b29475c78c9d19008dcee4bdc25f79b4) | 只记本批实际选中的续算资源 |
| 28 | [d9e61323](https://github.com/lhhhappy/LLM_deployment_A100/commit/d9e613239bd5ff0d9741a9c82a1f9cc651fb07d4) | 按冷/热预算分层准入并允许块间停车 |
| 29 | [18a41e6e](https://github.com/lhhhappy/LLM_deployment_A100/commit/18a41e6e88ffbdfc0ecc0d567f243c931c839f99) | 冷积压高时扩大 prefill 份额 |
| 30 | [b0776ae5](https://github.com/lhhhappy/LLM_deployment_A100/commit/b0776ae5d6cc7656c642a2a80c6d0b3badc3cf10) | 按等待短命中需求调整冷块 |
| 31 | [7c6cb634](https://github.com/lhhhappy/LLM_deployment_A100/commit/7c6cb6349f088de3af0e4d32440bdfbac6ee7941) | 在 A100 上增加 Triton 稀疏注意力候选 |
| 32 | [83c52de5](https://github.com/lhhhappy/LLM_deployment_A100/commit/83c52de59dd5e742107451899a39c16a111536be) | 积压缓解时保留需求侧的短请求座位 |
| 33 | [b38e5b24](https://github.com/lhhhappy/LLM_deployment_A100/commit/b38e5b247d9fdcc0bba50a51c79aa3e8bfc82808) | 探索共享前缀 family 排序 |
| 34 | [5c30b9ed](https://github.com/lhhhappy/LLM_deployment_A100/commit/5c30b9ed8b13097b3cb1e0692d5ac7c8af97eb38) | flush 同时复位保留座位与 family 状态 |
| 35 | [51237af7](https://github.com/lhhhappy/LLM_deployment_A100/commit/51237af7c00b34d412caee00636b452437f6bc3e) | 饥饿边界放行被 family 暂缓的请求 |
| 36 | [4f9d1f0b](https://github.com/lhhhappy/LLM_deployment_A100/commit/4f9d1f0b9c71522ad8e97413924baf513e6f4357) | 区分 family 暂缓与原生 LPM 暂缓 |
| 37 | [84dcca0e](https://github.com/lhhhappy/LLM_deployment_A100/commit/84dcca0ed84f949cf44acd0a5d2427b171d317a2) | 分别限制热请求与冷请求最长等待 |
| 38 | [efe837c8](https://github.com/lhhhappy/LLM_deployment_A100/commit/efe837c8f0b395ef881fe90eae441b608a436a31) | 只为可救等待者停车，被拒 turn 立即恢复 |
| 39 | [f67f40eb](https://github.com/lhhhappy/LLM_deployment_A100/commit/f67f40ebce226ca321830a69e40436306af7ddb1) | 索引 shim 关闭时保留原生 DeepGEMM |
| 40 | [5c897b42](https://github.com/lhhhappy/LLM_deployment_A100/commit/5c897b4212a9f43b9c4a6b0a513ec03f362fde37) | 有界缓存 Humming 形状配置元数据 |
| 41 | [f546934e](https://github.com/lhhhappy/LLM_deployment_A100/commit/f546934eff67f7b1d9f9cc5516c5a89377aaeb71) | 释放 HiCache host sidecar 和 staging 引用 |
| 42 | [ebfcaaa9](https://github.com/lhhhappy/LLM_deployment_A100/commit/ebfcaaa980c67e5a696930bd264ec8c561060571) | 修正 DCP、NEXTN 与状态迁移对齐 |
| 43 | [cd4a0a4c](https://github.com/lhhhappy/LLM_deployment_A100/commit/cd4a0a4c02ce9994fff406d4089c68fe66de6a59) | 正确核算 DCP 复制索引的 host 池 |
| 44 | [e464d8ab](https://github.com/lhhhappy/LLM_deployment_A100/commit/e464d8abe63f60f93fcaf3d0fc5a86c81d70747e) | 处理 DCP/NEXTN 未解析 top-k 的守卫 |
| 45 | [f2b6425e](https://github.com/lhhhappy/LLM_deployment_A100/commit/f2b6425eb74b4a417b19156bb3b39ddc2f283423) | 按热请求大小分配等待预算 |
| 46 | [741f3eda](https://github.com/lhhhappy/LLM_deployment_A100/commit/741f3eda75bc6abd51f0bf8e933ba16dd70d5546) | 首次进入时冻结冷/热类别 |
| 47 | [c0fcd486](https://github.com/lhhhappy/LLM_deployment_A100/commit/c0fcd48670281885a0af8898b9304eee8e60c97a) | 跟踪前缀生产者，给就绪兄弟请求保留准入 |
| 48 | [7d6099ac](https://github.com/lhhhappy/LLM_deployment_A100/commit/7d6099acc7d6cac61339228d147ae9c2647ab703) | 探索 DCP 的本地 KV prefill 续算 |
| 49 | [791453ca](https://github.com/lhhhappy/LLM_deployment_A100/commit/791453ca86545a04f9a2d6c943d6f1f7feb4b6a7) | 限制本地 extend JIT 组合并报告路由 |
| 50 | [450e8580](https://github.com/lhhhappy/LLM_deployment_A100/commit/450e8580ca63b601fa45daa3dc0bf8aaa0abaef8) | 冷链首接近预算时减少穿插 decode |
| 51 | [c1fa4877](https://github.com/lhhhappy/LLM_deployment_A100/commit/c1fa4877d9337d78c122a930d53bd5946c1d7a81) | 在分层准入中优先仍能救下的冷链首 |
| 52 | [b3f8c3c0](https://github.com/lhhhappy/LLM_deployment_A100/commit/b3f8c3c0aeb7c294a53aa6903bebe1044fd4a0c0) | 同步 ranks 的节奏，并计入本批实际计划工作 |
| 53 | [20a58da9](https://github.com/lhhhappy/LLM_deployment_A100/commit/20a58da99737c70f21c801ff587e94d8f54022be) | 报告 chain-first，拒绝缺少依赖的配置 |
| 54 | [0ed5fc95](https://github.com/lhhhappy/LLM_deployment_A100/commit/0ed5fc95b181bab11b35a0df06e3f2d25537b8a4) | 将 118 接到普通 full-KV prefill，保持默认关闭 |
| 55 | [78e2bed3](https://github.com/lhhhappy/LLM_deployment_A100/commit/78e2bed34e50f310cb08b82e386fdc859e8a2e44) | 为 8k/16k MoE 投影选择有守卫的配置 |
| 56 | [bf6b66fa](https://github.com/lhhhappy/LLM_deployment_A100/commit/bf6b66faf3ffbe10e593e858d3bbedd1c48cefb2) | 减少 KDA prefill 准备与逐层标量读取 |
| 57 | [d18933a9](https://github.com/lhhhappy/LLM_deployment_A100/commit/d18933a91999eba66f4106b9e2efad34342fb3aa) | 探索带守卫的 packed KDA decode |
| 58 | [6b629210](https://github.com/lhhhappy/LLM_deployment_A100/commit/6b629210f8c62f9cb5d512d2eb1f4ecce9a9cd9a) | 调整小 M MoE 下投影的 K tile 与 CTA 数 |
| 59 | [400adf7d](https://github.com/lhhhappy/LLM_deployment_A100/commit/400adf7d99d5b26c5a889d1668c78ee8afc4b28f) | DSA decode 在四页间复用 query 与权重 |
| 60 | [f3a99079](https://github.com/lhhhappy/LLM_deployment_A100/commit/f3a990790d64f201f115e687c9aba12fd6e1d1c5) | 探索启动期 metadata graph 捕获 |
| 61 | [8f19417f](https://github.com/lhhhappy/LLM_deployment_A100/commit/8f19417f9b2f067c149a51f66a446dbb13749fa7) | 跨层共享同一个 KDA prefix-state mask |
| 62 | [86b82497](https://github.com/lhhhappy/LLM_deployment_A100/commit/86b82497b11d2dbfe73c8ed57c95c01d3f0c3871) | 在 GPU gather prefill track，异步上传批元数据 |
| 63 | [7f87346f](https://github.com/lhhhappy/LLM_deployment_A100/commit/7f87346f53b20e2ef23dfcd0552acafd6725f178) | 对齐条件满足时复用 BF16 query view |
| 64 | [ce395da5](https://github.com/lhhhappy/LLM_deployment_A100/commit/ce395da510f59922014d30bfa3517541b7f7b5c5) | 两次 Mamba 分配共享同一次异步 index 上传 |
| 65 | [ca5d646c](https://github.com/lhhhappy/LLM_deployment_A100/commit/ca5d646c252688177480c7e67cec0a901e4b7069) | 同一 HiCache 传输复用完全相同的索引 Tensor |


## 造数据的探索也在 Git 中

下列 commit 保留方法和修正过程。当前可执行顺序、参数与外部素材要求见 [生成配方](../scripts/longchain/recipes/README.md)。Git 中的聚合验证收据用于说明检查范围；原始输入和生成正文不随这些提交发布。

| 原始 commit | 这一步要理解什么 |
| --- | --- |
| [a2d3f1ef](https://github.com/lhhhappy/LLM_deployment_A100/commit/a2d3f1ef3f942f7112d87a58cc6998d3408d6521) | 对齐主办方链的 token 结构，生成 v3 |
| [45708cec](https://github.com/lhhhappy/LLM_deployment_A100/commit/45708cec8b2fbc2c270f5f0143faa68d9d4136f8) | 审查修复 reset 追加、分歧边界与改写账本 |
| [e786cb7f](https://github.com/lhhhappy/LLM_deployment_A100/commit/e786cb7f60135d0cd0287375f650261fa30c66a4) | 按公开模板和每步真值构造，冻结提醒与抽样 |
| [b93a9a12](https://github.com/lhhhappy/LLM_deployment_A100/commit/b93a9a12d0cc3fadf67a275a7b91870edf8b783d) | v4：将改写点提前，补充计划内未命中计算 |
| [d9778b61](https://github.com/lhhhappy/LLM_deployment_A100/commit/d9778b6180a1c3928c30f88fd42f2ae7ae84680d) | 将 prompt 总量缩放纳入 deficit；保持追加块大小 |
| [30fa1338](https://github.com/lhhhappy/LLM_deployment_A100/commit/30fa13385de6f7c8c36d47b1d501775ec7c3c3a6) | 派生发布补齐正文链接、manifest 与来源账本；只延长等待长尾 |
| [ded6a189](https://github.com/lhhhappy/LLM_deployment_A100/commit/ded6a1899db1180a72b7d04f242a75f38e118a1b) | 记录原 Pod 上的发布与验证范围 |
| [b5cda215](https://github.com/lhhhappy/LLM_deployment_A100/commit/b5cda2156c5db46183737fb0d981c729197a43fa) | 补充独立审查和不能证明正文全量通过的边界 |

长尾规则和旋转顺序都是本地诊断假设；结构检查通过、token 总量接近正式，都不能证明等同隐藏负载。最新参数文件是 `scripts/longchain/recipes/0930.json`，旋转实现是 `scripts/longchain/rotate_cohort.py`。

## 在本地怎样只看一件事

```bash
# 读一个改动的说明、源码与测试
git show --stat 6b629210
git show 6b629210 -- engine/sglang engine/docs tests

# 看 09-28 源码到 09-30 源码的完整增量
git diff image-lh-img-0928c image-lh-img-0930a -- engine/sglang

# 看实验与决策如何推进；这类 commit 不必改变引擎源码
git log --oneline -- notes/experiments.md notes/submissions.md
```

`engine NNN:` / `engineNNN:` 说明实现机制，`tests:` 与 `review:` 说明验证和审查，`evidence:` 说明运行记录，`docs:` / `reports:` 说明解释与结论，`queue:` 是当时的派发安排。编号和 commit 类型帮助定位，但它们本身不是采用或通过的证明。

原始分支、镜像标签和 `archive/wip-20260930/*` 保留探索现场；恢复方法见 [仓库归档记录](reports/repository-consolidation-0930.md)。未提交工作区快照可能包含不完整方案，阅读时先找最终结论及冻结配置。当前默认分支提供最新成果和阅读导航，机制原始提交仍可逐一访问。
