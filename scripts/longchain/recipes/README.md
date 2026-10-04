# 09-30 本地诊断数据：只发布生成方法

2026-10-03，Codex 整理。Git 中保存生成器、[参数配方](0930.json)、校验和阅读说明；主办方输入、tokenizer 文件、正文分片、生成的 requests/cohort/provenance 和大型缓存留在本地。`data/` 只跟踪说明，`cache/`、`s1-dev/` 和 `build/base_exact/` 已忽略。

最新探索使用 `s1-dev-longchain-v5g-tail-rot150`。它是有限假设的本地诊断负载，正式 N42 来自平台隐藏集；不能把此配方当作正式数据生成器。

## 输入从哪里来

按 [task.md 的公开开发集说明](../../../llm-challenge-arena-v1/task.md) 取得官方包：

```bash
wenyon-cli dataset download s1-dev-combined-v1 -o ./s1-dev
```

实际输入为 `s1-dev/data/dev-combined-v1/` 的公开请求与链统计，`s1-dev/harness/` 的原始 Renderer / runner，和 `s1-dev/glm_tok/` 的冻结 tokenizer。需要数据平台访问权限。生成器的 Read 填充语料使用冻结底包 `build/base_exact/`；它按固定顺序读取文本，文件身份进入构建 manifest。该语料及原始数据不在本次 Git 发布中。

若已有完整 v4 成品，可以从第 2 步开始；若已有 v5 / v5g 和完整父 manifest，可以从第 4 步开始。派生发布始终写新目录并复用正文，避免覆盖冻结父数据。

## 每一步改变什么

| 步骤 | 改动 | 保持不变 |
| --- | --- | --- |
| v3 基础构建 | 从公开素材冻结事件计划、编译各链独立的历史；真实渲染并计算 token/LCP | 官方公开部分的身份与来源；按链追踪目标和实际量 |
| v4 rewrite top-up | 将计划中的历史分歧点提前，补充未命中新算工作 | 追加块大小；不通过改计数标签补量 |
| v5 rebudget | 对合成请求按重尾分布重分配输出预算，按链保持原总量 | 正文、prompt、cohort、真实公开请求的输出预算 |
| v5g 提案 | 按链时长估计拉长合成等待并逐步封顶 | 正文与输出预算；这是敏感性提案，未复原真实时序 |
| v5g-tail | 只采纳原本已在 P90 之后的合成非链首等待延长 | v5 的普通 P50/P90、其余字段、请求与链内顺序 |
| rot150 | `chains[150:] + chains[:150]` | 完整链成员、链内顺序、正文、等待、输出预算 |

v3/v4 的计划位置和部分历史属于合成；改写、输出再分配和等待变换会改变联合分布。源链总量接近、结构校验通过都不证明它等于隐藏负载。详细机制见 [longchain.md](../longchain.md)，发布修复与偏差见 [审查报告](../../../notes/reports/longchain-incremental-repair-0927.md)。

## 可执行配方

在仓库根目录运行。以下所有输出都位于忽略目录，目标数据目录须不存在。命令描述当前可复用生成链路；每次生成用自己的 manifest 和哈希标识输入与产物。

### 1. v4：基础生成加 rewrite top-up

```bash
uv run --with-requirements scripts/longchain/requirements-longchain.txt \
  python -B scripts/longchain/longchain.py build \
  --source-root s1-dev/data/dev-combined-v1 \
  --harness-dir s1-dev/harness --tok-dir s1-dev/glm_tok \
  --read-corpus build/base_exact --chains 311 --seed 20260924 \
  --max-context-tokens 524288 --rewrite-topup \
  --set s1-dev-longchain-v4 --out cache/s1-dev-longchain-v4
```

### 2. v5：输出预算

```bash
python3 -B scripts/longchain/rebudget.py \
  --parent cache/s1-dev-longchain-v4 --public s1-dev/data/dev-combined-v1 \
  --gamma 1.6 --cap 16384 --seed 20270101 \
  --out cache/s1-dev-longchain-v5
```

### 3. v5g：等待提案

```bash
python3 -B scripts/longchain/regap.py \
  --src cache/s1-dev-longchain-v5 \
  --organizer s1-dev/data/dev-combined-v1/chains.jsonl \
  --ttft-s 1.5 --tpot-s 0.03 --gap-cap-s 310 --chain-cap-s 3600 \
  --set s1-dev-longchain-v5g --out cache/s1-dev-longchain-v5g
```

保持默认 `allow-scale-down=false`、`redistribute=false`。旧 v5g 的中位等待被过度拉长，所以作为长尾候选的提案来源，不作为最新主负载。

### 4. 只采纳等待长尾，修复发布契约

```bash
python3 -B scripts/longchain/finalize_v5g.py \
  --source-root cache --out-root cache/review-0930 --variant v5g-tail
```

该脚本先核对 v4/v5/v5g 请求 ID 和顺序，按条件选择长尾，再经 `longchain_metadata.publish` 生成新 manifest、来源账本、哈希和正文链接。没有正文的 metadata-only 快照不能当作可回放数据。

### 5. 完整链旋转 150 位

```bash
python3 -B scripts/longchain/rotate_cohort.py \
  --parent cache/review-0930/s1-dev-longchain-v5g-tail-review-0927 \
  --offset 150 --set s1-dev-longchain-v5g-tail-rot150 \
  --out cache/s1-dev-longchain-v5g-tail-rot150
```

10-03 对保留的父 cohort 按这个规则复算，得到历史记录中的 `cohort_sha256=b78593bdea138f58`。完整原始请求与正文没有因此进入 Git。只有输入身份完全相同才应要求相同产物哈希；否则记录新的版本，不把结构或规则相同写成字节相同。

### 6. 独立验收、分布比较与回放

```bash
mkdir -p build/scratch/data-repro
uv run --with-requirements scripts/longchain/requirements-longchain.txt \
  python -B scripts/longchain/longchain_check.py \
  --root cache/s1-dev-longchain-v5g-tail-rot150 \
  --harness-dir s1-dev/harness --tok-dir s1-dev/glm_tok \
  --out-json build/scratch/data-repro/check.json

python3 -B scripts/longchain/workload_compare.py \
  --root dev=s1-dev/data/dev-combined-v1 \
  --root generated=cache/s1-dev-longchain-v5g-tail-rot150

uv run --with-requirements scripts/longchain/requirements-longchain.txt \
  python -B scripts/longchain/longchain_replay.py \
  --root cache/s1-dev-longchain-v5g-tail-rot150 \
  --out build/scratch/data-repro/self-check --self-check
```

实际发压沿用 [longchain_replay.py](../longchain_replay.py)，另建运行目录，并提供 `--base-url`、`--n`。每档核对原 harness、有效配置和真正 flush。历史 v5g-tail 的发布/结构审查保留在 [验证收据](../../../evidence/longchain-incremental-repair-0927/)，其中 `STRUCTURAL_OK` 不等于独立全量正文渲染通过。

## 没有原始数据也能检查哪些内容

```bash
PYTHONPATH=tests python3 -B -m unittest \
  test_longchain_metadata test_longchain_check test_longchain_replay
```

这些测试在临时目录中构造小样本，检查字段边界、输入不可覆盖、完整性、cohort、tail 规则和回放保护。生产数据需要外部素材和冻结 tokenizer，Git 本身只交付方法与验证代码。
