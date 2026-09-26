# 长程测试集工具

当前成品：[`data/s1-dev-longchain/`](../../data/s1-dev-longchain/)，311链、5601请求；已完成全量CPU验收。交付与分布见[`data/README.md`](../../data/README.md)。本目录集中维护生成、校验、回放入口与Phoenix观测，不在`scripts/analysis/`保留旧副本。

64链快速试跑集已生成：`data/s1-dev-longchain-lite`，1123请求，按用户要求未跑自检。原runner只换root/set/cohort，具体参数见[`data/README.md`](../../data/README.md)。复现命令：`python3 -B scripts/longchain/longchain_subset.py`。

## 直接回放当前成品

在仓库根目录运行，换成实际引擎根地址和本次N；每次使用全新的输出目录。下面N=14只是命令示例，不是CPU推断出的推荐通过档。

```bash
uv run --with-requirements scripts/longchain/requirements-longchain.txt \
  python -B scripts/longchain/longchain_replay.py \
  --root data/s1-dev-longchain \
  --base-url http://ENGINE_HOST:8000 \
  --n 14 --out runs/longchain-n14-01
```

鉴权服务追加`--api-key`；引擎地址带`/v1`会由原runner剥离。该入口先离线验收，再调用原checked runner：preflight → warmup → 真正flush KV → 完整回放 → 原评分和题面补充门。它不启动或修改引擎。API key不写日志。复用同一引擎JIT预热时可加`--skip-warmup`，每档flush仍执行。

只替换负载时，现有队列/原runner使用这三个参数即可，发压和评分算法不变：

```text
--root data/s1-dev-longchain
--set s1-dev-longchain
--cohort data/s1-dev-longchain/cohort.json
```

评分必须指向`data/s1-dev-longchain/requests.jsonl`；使用新的运行目录避免原harness复用别的数据的body cache。不要加`--max-chains`、`--no-gap`或`--include-all`后把结果当同一份负载。

## 文件职责

| 文件 | 用途 |
|---|---|
| `longchain.py` / `longchain_events.py` | 源素材、冻结事件计划与不可变历史编译 |
| `longchain_subset.py` | 从成品按链长、pack分层抽取64条完整链；正文和等待不变，不运行自检 |
| `longchain_check.py` | 独立正文、工具组、token/LCP、预算和来源验收 |
| `longchain_replay.py` | 离线防护后调用原runner与评分，不是另一套回放算法 |
| `longchain_material_inventory.py` | 公共素材盘点 |
| `longchain_distribution.py` | 分布、素材重复与链首共享前缀诊断 |
| `longchain_timing.py` | 仅读元数据，对比dev/full/lite等待分位数、事件条件分布与长等待连续性 |
| `dev_distribution_observe.py` | 原开发集与048固定记录的来源分布审查 |
| `requirements-longchain.txt` | 原GLM Renderer的冻结CPU依赖 |
| [`phoenix/`](phoenix/README.md) | Phoenix只读采集、连续事件分析与目录展示 |
| [`longchain.md`](longchain.md) | 唯一现行生成设计与复现命令 |

采集与生成分开：已有成品回放不需要连接Phoenix，也不需要恢复已删除的原文cache。生成只读取s1-dev和冻结结构profile。旧验收证据保留当时路径与源码hash，目录整理不重写历史收据，也不改变成品数据。
