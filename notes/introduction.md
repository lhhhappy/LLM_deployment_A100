# 当前项目介绍

我们在8×A100-SXM4-80GB上部署GLM-5.3-Flash推理服务。赛规只看 [task.md](../llm-challenge-arena-v1/task.md)：能力先过AIME26/GPQA Diamond，压测再按最高通过并发、该档TPOT均值、TPM、提交先后排名。正式回放341条链、5150个请求，正文隐藏。不能关闭thinking、压短输出、截历史、删除tools或伪造时间戳和token数；`/flush_cache`必须真清。

模型有45层，包含KDA线性注意力、DSA稀疏注意力、MoE、mHC与MTP。复用长前缀需要KV、索引键/KPool、KDA与draft状态在同一合法位置完整可用；显卡上的MoE计算、缓存丢失导致的重算、prefill/decode与不同请求的服务顺序都会影响并发。[源码与机制入口](../research/README.md)

现行SGLang基线源码为`759a6ebb8e31723519ad5daf438e26e24b32501a`，TP8、MTP、host64。正式46251以cold cap4096通过N22；46364只把冷块上限改到6144，已提交待结果。本地069完整N30仅chain门失败31/29，其他10门通过。当前081/082把46364配置完整回放于N30/N34，分析定在测量后12/60/90分钟，见 [队列](queue.md)。

本地长链集是311链、5601请求，其中部分正文合成；只用于相同配置的机制比较，不能直接推断正式通过档。过去被撤回的缓存归因和历史实验不在本页延续；当前已成立的证据与边界见 [knowledge.md](knowledge.md)，完整结果见 [experiments.md](experiments.md)，判分合同见 [evaluation.md](evaluation.md)，正式提交见 [submissions.md](submissions.md)。

Claude在开发机研究EP8执行成本，Codex管理8卡回放与结果分析。已授权的工程和实验直接推进，按改动选择必要验证；8卡服务保持运行。工作分工和联系见 [collaboration.md](collaboration.md)。
