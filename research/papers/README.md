# 一手资料存档

2026-09-24 为 [R24](../claude/R24_glm53flash_official.md) 抓取。网页会变，这里存的是当天快照；结论以 R24 的引用和标注为准。抓取日期均为 2026-09-24（UTC）。

| 文件 | 来源 | 为什么和 serving 有关 |
|---|---|---|
| glm5-tech-report-2602.15763.pdf | https://arxiv.org/pdf/2602.15763 （v2） | GLM-5 技术报告（744B 大模型，不是 Flash）；DSA、MTP 参数共享与接受长度、多轮 agent 推理的 DP 亲和路由、PD 分离防 prefill 干扰 decode |
| glm53flash-blog.md，glm53flash-blog-img/ | https://z.ai/blog/glm-5.3-flash （JS 页面，正文从其 JS 包抽取；架构图原图另存） | 官方说明为何用"线性注意力 + 稀疏注意力"混合、IndexPool 4 合 1、KV 比 GLM-5.3 小 4.4 倍；国产芯片上的 ReplaySSM、缓存量化、EPD 分离 |
| glm53flash-model-card.md | https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/README.md | 官方模型卡：320B/18B、推荐框架、reasoning_effort 与 clear_thinking 默认值 |
| kimi-linear-kda-2510.26692.pdf | https://arxiv.org/pdf/2510.26692 | KDA（Kimi Delta Attention）原论文：固定大小状态、3:1 混合比例、MLA 层不用位置编码、prefill/decode 速度曲线 |
| deepseek-v3.2-dsa-2512.02556.pdf | https://arxiv.org/pdf/2512.02556 | DSA（lightning indexer + top-k 选择）原论文：indexer 仍是 O(L²) 但便宜、可用 FP8；短序列 prefill 用 masked MHA |
| deepseek-mhc-2512.24880.pdf | https://arxiv.org/pdf/2512.24880 | mHC 原论文：残差流扩成 n=4 路，访存按约 n 倍增长，训练额外 6.7% 时间 |
| sglang-prs-glm53flash.md | GitHub REST API，sgl-project/sglang 的 19 个 PR/issue 正文（未含评论与 diff） | GLM-5.3-Flash 支持（#36507）、HiCache 缺 indexer 导致主机命中出错（#40915/#40134/#38212）、FP8 KV 不可用（#36830）、DCP（#40433/#40434）、tail-replay RFC（#40865）等 |
| sglang-docs/ | 复制自只读的 refs/sglang-fe236ea6c3（docs/cookbook、docs/src/snippets、docs/docs/advanced_features、.claude/skills） | 本地参考版本的 cookbook、部署面板配置与基准说明、HiCache 设计文档、双池比例计算 skill |
| sglang-cookbook-glm53flash-online.txt | https://cookbook.sglang.io/autoregressive/GLM/GLM-5.3-Flash （HTML 去标签） | 线上最新版 cookbook，与 refs 版有差异（MTP 改为固定 5/1/6、提到 breakable prefill graph） |
| vllm-recipe-glm53flash.txt | https://recipes.vllm.ai/zai-org/GLM-5.3-Flash | vLLM 官方 recipe：仅支持 Hopper 及更新；Hopper 不支持本模型 FP8 KV；KV offload 验证范围 |
| vllm-hybrid-kv-cache-manager.md | https://raw.githubusercontent.com/vllm-project/vllm/main/docs/design/hybrid_kv_cache_manager.md | vLLM 混合模型前缀缓存的设计：各层类型分别求命中再取交集 |
| vllm-automatic-prefix-caching.md | https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/automatic_prefix_caching.md | vLLM 对 Mamba 类状态的前缀缓存语义：只能在块边界恢复，共享前缀分叉点可加检查点；前缀匹配粒度须是稀疏 MLA 压缩比的倍数 |

没能取到的：z.ai 博客里的基座对比表与各基准图只存为图片链接，未转文字；docs.z.ai 的 API 文档页已下载但未作为依据（与 serving 无关）；PR 评论区（例如 #38212 作者和 H100 复核结果）未抓取。
