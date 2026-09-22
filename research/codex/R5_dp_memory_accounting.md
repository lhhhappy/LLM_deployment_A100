# R5 — TP8 下 DP1 / DP2 / DP4 显存账

作者：Codex；2026-09-22 UTC。**配置/源码尺寸计算，不是 GPU 实测，不是最终启动参数**。接受 §G 的容量任务，单列本文，不混入镜像访问或调度报告。

## 1. 固定假设与结论

本地 SGLang commit `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`；冻结 [config.json](../../s1-dev/glm_tok/config.json)，SHA256 `bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f`。TP=8、DP=D、attention TP=8/D；不改变 MoE/EP 分片。34 层 KDA + 11 层 MLA/DSA，BF16 latent KV、FP32 KDA recurrent + BF16 conv state；无 MTP、HiCache、DCP，静态分池，未打开 int8 checkpoint。以下源码路径相对 `src/sglang/python/sglang/srt/`。

**VERIFIED（尺寸/源码）**：DP 减少 MLA KV 的组内复制，但 KDA 单 slot 每 rank 成本和 attention 权重常驻量会增加。静态池缺省 ratio=0.9 时，候选缓存预算约 **47.37% 给 KDA，52.63% 给 KV**；不能把权重之后的剩余显存全部除以 KV 的每 token 大小。

**INFERRED**：DP2/4 仍值得保留；现阶段不能宣布 ×2/×4 的实得容量，更不能据此推导 N@SLO。真实底包、可用池预算、路由均衡及 decode 代价都未知。

## 2. 可精确确认的几何

**VERIFIED（F8；`memory_pool.py:3876,4798–4855`、`index_key_cache.py:32`、`configs/mamba_utils.py:324–365`）：**

```text
c_KV = 11 × (512 × 2 + 128 + 4) = 12,716 B / token / attention rank
S(D) = 34 × [(64/(8/D)) × 128 × 128 × 4
             + 3 × (64×128/(8/D)) × (4−1) × 2]
     = 18,452,480 × D B / KDA slot / rank
```

indexer 当前按完整物理 token slot 分配 132 B/层，`index_kpool=4` **不让这项实际分配自动除四**。每份 128 Ki-token KV 约 1.552 GiB/rank；padding、临时 buffer 另计。

| TP8 配置 | attention TP | KV 每 token 每 rank | 一个 34 层 KDA slot / rank | 相同净 KV 字节预算下的逻辑容量倍率 |
|---|---:|---:|---:|---:|
| DP1 | 8 | 12,716 B | 17.598 MiB | 1 |
| DP2 | 4 | 12,716 B | 35.195 MiB | 2 |
| DP4 | 2 | 12,716 B | 70.391 MiB | 4 |

逻辑容量按 DP 组相加，组内看最小可用 token 数；不是把 8 张卡的重复份数相加。只有分组预算相等且请求放置均衡时，才近似 `D × B_KV/c_KV`。同一会话迁移、共享前缀跨组重复保存、热组先满都会降低收益。

## 3. 池划分：默认设置带来的重要修正

**VERIFIED（`mem_cache/kv_cache_configurator.py:2148–2196,2433–2545`）：** 引擎先取已加载权重之后的可用显存，扣 `pre_model_load_memory × (1−mem_fraction_static)` 及多模态等保留，再为 Mamba 状态分池；不是简单 `80 GB − 文件大小/8`。

令 `B(D)` 为上述扣减后、尚未划分 KDA/KV 的每 rank 候选预算，`r` 为最终解析的 `mamba_full_memory_ratio`。无 MTP / replay scratch 时：

```text
M_budget = B(D) × r/(1+r)
K(D) = floor(M_budget / S(D)) − 1        # 可用状态槽；另有一个 padding slot
B_KV(D) = B(D) − [K(D)+1] × S(D)
T_total ≈ D × B_KV(D) / 12,716           # 再按 page、用户上限等裁整
```

`arg_groups/fields/schedule.py:203` 的 fallback 为 r=0.9，须检查最终解析值及镜像 override。忽略 slot 取整时，若 **B=24 GiB**，KV 仅约 12.63 GiB，KDA 约 11.37 GiB；不能使用“24 GiB 全给 KV≈2.03M token”的数字。

若显式设置全局 `max_mamba_cache_size=K_global`，源码先按 `floor(K_global/D)` 分到各 DP worker，单 slot 再乘 D；固定总逻辑 slot 数的主状态内存因此大致不变。不要将同一个 CLI 数字误读为“每 DP group 各有这么多”。加 D1 快照可能先消耗预分配池中的空 slot，未必立即增加进程 VRAM；真正影响是可同时保留的其他状态和驱逐。

**VERIFIED**：`enable_unified_memory` 缺省 false；动态共享池另要求 Triton attention/linear/Mamba backend，且限制 HiCache、prefill graph 及非 DSPARK speculation（`arg_groups/fields/memory.py:80`）。**INFERRED**：不能把它当成本赛 CUDA DSA 可直接启用的“免分池”开关。

## 4. Attention 权重复制的可算部分

**VERIFIED（结构/量化排除表）**：`models/glm5_next.py:306–485` 的 KDA qkv/b/f_b/g_b/o 等按 attention TP 分片；f_a/g_a 已是 replicated，不是新增 DP 倍增项。冻结 FP8 config 的排除表使这些 KDA 投影保持 BF16；conv 权重、dt_bias、A_log 在实现中为 FP32。

按源码张量形状，不含后端重排/临时驻留，34 层可随 attention TP 改变的 KDA 权重总字节：

```text
W_K = 34 × [4×4096×8192×2 + 2×128×8192×2 + 4096×64×2
            + 3×8192×4×4 + 8192×4 + 64×4]
    = 9,301,729,792 B = 8.662911 GiB
ΔW_K(D) = (D−1)/8 × W_K
```

**VERIFIED（部分张量尺寸）**：MLA 的 q_b、kv_b、o 同样按 attention TP 分片（`models/deepseek_v2.py:1752–1866`）。按 config 的 q_b/o FP8、kv_b BF16 原始张量计：

```text
W_M_major = 11 × [1536×64×256×1 + 512×64×512×2 + 4096×64×256×1]
          = 1.2890625 GiB
ΔW_M_major(D) = (D−1)/8 × W_M_major
```

| 相对 DP1 的每 rank 增量 | DP2 | DP4 |
|---|---:|---:|
| KDA 上述分片权重 | 1.082864 GiB | 3.248592 GiB |
| MLA 上述主要投影 | 0.161133 GiB | 0.483398 GiB |
| **已列张量合计** | **1.243997 GiB** | **3.731990 GiB** |

**不是实际加载显存差值，也不是严格下界**：MLA absorb 后可能释放/保留不同表示，FP8 scales、Marlin packing、w_kc/w_vc、副本、vision encoder、workspace、graph、MTP 均需按真实后端重新盘点。indexer 的主要投影及 MLA qkv_a 已 replicated，不应不加区分地再次乘 D。328,326,771,576 B 是 checkpoint 序列化总量，不是可直接除八的进程显存。

## 5. 条件算例：收益仍可能明显，但不是标称 DP 倍率

**INFERRED / 纯算术例子**：假设 DP1 的 `B(1)=24 GiB`，切 DP 时只扣 §4 列出的额外权重，其他成本完全不变；r=0.9。采用连续近似，忽略 slot/page 取整、padding 尾差及其他未列开销：

| 配置 | 候选 KDA+KV 预算 / rank | KDA 预算 / rank | 净 KV 预算 / rank | 逻辑 KV token 合计 | 相对 DP1 |
|---|---:|---:|---:|---:|---:|
| DP1 | 24.000 GiB | 11.368 GiB | 12.632 GiB | ≈1.067M | 1.00 |
| DP2 | 22.756 GiB | 10.779 GiB | 11.977 GiB | ≈2.023M | 1.90 |
| DP4 | 20.268 GiB | 9.601 GiB | 10.667 GiB | ≈3.603M | 3.38 |

这不是宣称机器恰有 24 GiB 可用；改变 r、state dtype、max_mamba_cache_size、MTP 或 backend 都要重新算。将“剩余显存”明确区分为**划分前混合预算**或**划分后净 KV 预算**，旧的 token 容量估计才可比较。

## 6. 请求槽、D1 与 MTP 必须独立记账

**VERIFIED（`kv_cache_configurator.py:2210–2239,2263–2295`）：** 显式 `max_running_requests=R` 先做 `R//D`；还受 KV token 和 KDA slot 上限限制。若 R=22、DP4，则每 worker 上限先变为 5、合计 20，不能据此配置宣称有 22 个同时运行槽。闭环 N 是逻辑并发会话，未必要求所有会话都同时在 engine running；若目标确实是至少 22 个 running 槽，应选合适的 D 倍数并考虑热组负载。

Mamba 槽位估算还含运行状态/锁/extra buffers：默认 base ratio=3，overlap extra_buffer 再加 2，lazy 再加 1；skip-decode-lock 等另改 ratio。这些是容量限制系数，不等于一条会话仅有一份快照，也不能覆盖无限历史缓存。

**VERIFIED（F8/F12 算术）**：22 个会话各多保留一份 D1 checkpoint，均衡分组时，DP1/2 最忙 rank 增量约 0.378 GiB；DP4 因 6/6/5/5 分组，最忙 rank 约 0.412 GiB。全机合计均约 3.025 GiB（3.25 GB），不是 DP 后白送；历史多快照、padding、锁定状态另计。

MTP 未纳入本账。源码按 draft token 数额外保留中间状态；近似形如 `(K+1)S + (K/ratio+1)×draft_tokens×S`，实际按 request cap 截断；还要加 draft 权重/KV、验证 workspace/graph。不能用“加一层”或“全部显存翻倍”代替源码账。

## 7. 下一步需要的证据（不执行）

**INFERRED（优先顺序）**：先保持 DP1/2/4 候选，不提前定 DP8。

1. R3 的底包 digest/source/backend 收据；确认 attention/KDA 的 TP 分片与 FP8 实际表示。
2. 最终解析的 pool/并发参数、权重加载后 free bytes、slack、每 rank 的 KDA slot 与 KV token capacity；不把磁盘权重当 VRAM。
3. D1 预计活跃/缓存/锁定 slot 峰值；cap=2 不保证保护角色点，见 R4。
4. Claude session 路由设计需报告热组负载、前缀重复与迁移；静态 session hash 不能保证平衡。
5. 获准后再测逐组缓存命中/驱逐、排队、decode 通信与所有 SLO。两卡裁层能检查 DP 接口/池行为，不能外推八卡满模型容量与成绩。

本轮仅公式和源码核对，无模型加载、模拟回放、GPU 测量或配置变更。
