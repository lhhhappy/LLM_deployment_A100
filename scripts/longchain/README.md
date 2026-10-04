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
| `workload_compare.py` | 只读元数据：按评分器的门归属比较各数据集的请求占比与新增token，并逐链对比主办方三总数 |
| `repair_metadata.py` / `longchain_metadata.py` | 复用正文，增量修订输出预算、明确来源的等待，或只恢复指定 cohort 顺序；更新来源和完整 manifest |
| `rebudget.py` / `regap.py` | 输出长尾 / 按链放大等待的敏感性臂；不代表恢复了真实逐请求负载 |
| `requirements-longchain.txt` | 原GLM Renderer的冻结CPU依赖 |
| [`phoenix/`](phoenix/README.md) | Phoenix只读采集、连续事件分析与目录展示 |
| [`longchain.md`](longchain.md) | 唯一现行生成设计与复现命令 |

采集与生成分开：已有成品回放不需要连接Phoenix，也不需要恢复已删除的原文cache。生成只读取s1-dev和冻结结构profile。旧验收证据保留当时路径与源码hash，目录整理不重写历史收据，也不改变成品数据。

## 增量修订 v4 / v5，不重新生成正文

在拥有完整 v4 正文的机器上，用 v4 的完整 manifest 和现有 v5 输出预算修复发布契约：

```bash
python3 -B scripts/longchain/repair_metadata.py \
  --parent cache/s1-dev-longchain-v4 \
  --outputs-from cache/s1-dev-longchain-v5/requests.jsonl \
  --out cache/s1-dev-longchain-v5-repaired --set s1-dev-longchain-v5-repaired
```

正文使用文件链接；请求身份、链内顺序、phase、token/LCP 标签均保持不变。
cohort 默认继承，不重新分层抽样。需要比较 v3/v4 同一开场时，另建单变量臂，
只加 `--cohort-from cache/s1-dev-longchain-v3/cohort.json`，不要同时改变预算或等待。
新目录必须不存在，不能写进父数据集。原始正文路径在回放期间须保留。

只有元数据快照时可加 `--metadata-only`，产物明确标为 `METADATA_ONLY_INCOMPLETE`，不能据此声称验收或开跑成功。
完整数据仍走原校验/回放入口；验收报告与运行输出放在数据集以外。

等待补丁接受 JSONL：`{"chain_id":"原 chain_id","idx_in_chain":1,"gap_ms":1234}`，
也可用 `req_id`；索引从 0 起，补丁不允许修改链首。
必须明确 `--gap-basis capped-replay|raw-end-to-start-sensitivity|synthetic-sensitivity`。
前者是已按工具/思考规则处理的间隔，后两者只能作为诊断；仍由原 harness 处理每链累计上限。
只有 P50/P95 等摘要时，本工具不会推断逐请求时间、工具/思考拆分或分配位置。

实现审查、已修 bug 与数据偏差见[审查报告](../../notes/reports/longchain-incremental-repair-0927.md)。
