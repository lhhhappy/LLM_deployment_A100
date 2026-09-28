# 轻量 MTP：固定短验证窗先行（2026-09-28）

状态：已完成冻结引擎 bf6b66fa 的 draft 循环及参数解析静态核对；两个可运行 job 已准备且 shell 语法、参数差异检查通过，尚未入队、无本轮 TP8 性能或正确性结果。已有3/1/4臂保持原计划。

建议首测 NEXTN steps=1/topk=1/draft_tokens=2，次测2/1/3；与关闭MTP及3/1/4比较。同一引擎、池400、max-running48、graph48、DCP1、cadence2、118/KDA/调参开关、数据与一小时晋档规则不变。轻量臂相对3/1/4只改steps和必须联动的draft_tokens。角色机制101随MTP关闭，三种MTP之间相同，与非MTP的比较属于组合收益。

| 模式 | draft-extend以外的草稿前向次数/轮 | target verify窗口 | KDA验证临时状态推算/卡 |
| --- | ---: | ---: | ---: |
| 3/1/4 | 2 | 4 | 3.368 GiB |
| 2/1/3 | 1 | 3 | 2.526 GiB |
| 1/1/2 | 0 | 2 | 1.684 GiB |

资源公式：(48+1) × D × 34 × (8×128×128×4 + 3×3072×2) 字节。仅针对R实际为48、dense KDA验证SSM/conv临时状态；不含草稿权重、KV、图池或激活，不能当总显存节省。源码位置：srt/mem_cache/memory_pool.py speculative intermediate 分配；参照 R17。第一份proposal由draft-extend产生，eagle_worker_v2.py draft循环在最后一次迭代forward前break，所以steps1仍有draft-extend，并非免费草稿。speculative_hook.py topk1将draft_tokens规范化为steps+1。

收益条件：单候选接受概率p、普通decode一轮成本C、轻MTP整轮成本L，则隔离稳态下平均输出约1+p，只有 L/C < 1+p 才有计算收益。端到端TPOT还受prefill阻塞、排队影响。窗口减半不表示整机快两倍。报告验收顺序：同ID chain新增坏例为零，再看TPOT p95≤100ms、其他TTFT门与错误；同时看加权接受长度、真实KV容量/峰值、重算和每轮耗时。

当前所有臂SGLANG_OPT_FUSED_KDA_VERIFY=0，故不能以T2不命中T≥3融合路径为由否决1步臂。未来若打开融合必须重新比较，不假设短窗必然更快。

底包已经包含adaptive_spec_params.py自适应步数，不需要先另写控制器。但默认候选最大steps7，缓存按候选上界分配，不能直接视作省显存模式；如果静态1步有效，再审查限制候选{0,1,2}的图捕获、KDA状态切换、TP一致性。当前不启用该路径，不以存在参数证明GLM组合可用。

暂不做：删草稿层（已有单层）、降低接受标准、取消target验证、缩输出预算、压缩KDA状态精度。暂不叠加DCP或降低max-running；先把短窗的收益和代价测清楚。
