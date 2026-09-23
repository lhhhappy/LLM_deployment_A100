# 116 — DCP 地址协议修复（GLM DSA：latent 真分片 + 读路径换算 + indexer K 虚拟空间复制）

叠加在 `000→101→105→106→110→111→112→113→114→115` 之后（RELEASE 顺序：`…→115→116→140→120→130→150→160`，全栈 fuzz=0 已在本地与开发机验证）。
不开 `--dcp-size`（W=1）时所有改动都是恒等变换：kernel 的 `loc%1==0`、`loc//1`，读路径的 DCP 分支不进入，indexer 比例 ×1。
代码注释统一标 `[ax] 116`。任务 T50/T50b；根因证据 `evidence/T50/address_protocol.txt`。

## 问题（8 卡 025 崩溃的真正原因）

DCP 下调度器与分配器使用**虚拟 loc**（`[0,(rows+64)·W)`，页 64·W，容量按 ×W 计，`scheduler.py:2249,2423`、`kv_cache_configurator.py:1995`），而 DSA 池（latent 与 indexer K）是**每卡大小**。

- **latent 写入没分片**：GLM `qk_rope_head_dim=0`，写入走 `set_mla_kv_buffer_kernel_norope`（`kernels/ops/kvcache/mla_buffer.py:87-114`）。这个 kernel 没有 DCP 规则，每卡都把每个 token 写到虚拟 loc。只有带 rope 的 kernel（:42-44）按 `loc%W==rank → loc//W` 分片。
- 读路径（tilelang 稀疏 prefill/decode 的 `page_table_1`、indexer 的 `real_page_table`、112 的 `_paged` 不设上界）和 indexer K 写入也全用虚拟 loc。
- 结果：原栈 DCP 实际上是“每卡复制全部 KV，按虚拟 loc 访问一个每卡大小的缓冲区”。
  - 所有活跃 loc 都小于每卡行数时，结果正确（开发机实测 orig_dcp 与 orig_ref 一致），但容量没有增加（F78 的“逻辑 ×7.8”并不存在），decode 还在每卡对全部 token 算 64 头（W 倍冗余）。
  - 分配器高水位一旦超过每卡行数（025 为 818,112），读写就越界 → illegal memory access（推断；025 当时活跃约 0.46–0.65M token，加上碎片）。
- 与前缀命中无直接关系：长前缀树只是把高水位推高。冷探针从没到达这个区间。
- 110–115 都没有引入这个问题。

## 改动（4 个文件）

1. `kernels/ops/kvcache/mla_buffer.py`：`set_mla_kv_buffer_kernel_norope` 加 `DCP_RANK/DCP_WORLD_SIZE`（默认 0/1），规则与 rope 版相同：只存 `loc%W==rank`，写到本卡行 `loc//W`。发射处传入 `get_parallel()` 的值。**latent 从此真分片**。
2. `srt/layers/attention/dsa_backend.py`（仅 tilelang 分支）：
   - **extend（含前缀命中、chunked prefill）**：底包 forward_mla 在 DCP extend 时已把本批每个请求的前缀从各卡分片 all-gather 进 `attn_dcp_metadata.dcp_kv_buffer`，并拼接新 token（`dcp/comm.py:267`，布局见 `dcp/planner.py:48`），但原来只有 FlashInfer-MLA 读它。116 让 DSA 也读这块缓冲区：
     - 每批建一次“虚拟 loc → 缓冲区行”的 int32 表（每卡 `(rows+64)·W+1` 项；无效项进垃圾槽，置 -1）；
     - 把 top-k 转出的 `page_table_1` 查表换成行号；
     - 每卡用本卡的头对完整 KV 做注意力，不需要 LSE 合并。
     - DCP extend 若缺少 `attn_dcp_metadata`（例如将来被图捕获的 prefill 跳过了 planner），直接报错，不静默读错行。
   - **decode / target-verify**：`page_table_1` 只保留本卡拥有的 loc，换成 `loc//W`，其余置 -1（kernel 按 `>=0` 掩码）；输出 partial + LSE，由底包的 LSE 合并完成。
     - 某行本卡没有 key 时 kernel 得 0/0=NaN，而 NaN×0 会穿过合并，所以用 `nan_to_num` 清零（这一行的 LSE 为 -inf/NaN，合并时权重为 0）。
     - 全部是张量运算，可以录进 CUDA graph。
3. `srt/mem_cache/kv_cache_configurator.py`：DCP 目标池的 `index_buf_size = (rows+64)·W`。indexer K **保持复制**（每卡本来就为全部 token 计算 index K，top-k 也需要全部 token），按虚拟 loc 寻址，不再越界。与 hisparse 放大 index 缓冲的做法同构。
4. `srt/model_executor/pool_configurator.py`：`_compute_dsa_indexer_cell_size` 对目标池 ×`attn_dcp_size`，让显存核算与第 3 条一致（draft 分支原本就自己 ×dcp，不重复乘）。

## 容量（算术，未在 8 卡实测）
每层每个逻辑 token：latent 1024 B（bf16 512）+ index 132 B。非 DCP 每卡存 1156 B。DCP8+116 每卡存 1024/8+132=260 B，所以逻辑容量约 **×4.4**（不是原来宣称的 ×7.8）。025 同配置（mem 0.75）下每卡行数估计从 818k 降到约 450k，逻辑约 3.6M token。开发机 TP2 实测 pool_rows 从 9,224,896 降到 8,279,488（W=2 时与算术一致）。
要再往 ×8 靠：indexer K 也分片（本卡算 logits + 本卡 top-k → all-gather 候选再取全局 top-k），并处理 kpool 压缩跨卡的问题。另立任务。

## 验证（开发机 2×A100，TP2，`scripts/analysis/devbox_dcp_check.sh`）
- harness：`scripts/analysis/dcp_check.py`（真实模型代码、缩小版 GLM：8 层中 2 层 DSA，topk 2048）。
  - 两个请求先冷预填充 8192/6144，再在缓存前缀上续算 1000/700（前缀命中路径），然后 decode 4 步。
  - 可选 filler 占住 60% 的分配器，把请求推到虚拟 loc > 每卡行数（高 slot）。
  - 比较器 `dcp_compare.py`，门槛为相对 L∞ ≤ 1e-2。
- **判据要点（实测教训）**：底包 dummy 权重（±1e-3，所有参数同一个种子）下 logits 与注意力无关，任何 DCP 变体都与参考逐位相同，DSA 输出约 1e-13，是纯噪声。因此：
  - harness 用按参数名播种的 well-scaled 初始化（DSA 注意力 2D ~N(0,1/fan_in)、norm=1、embedding ~N(0,1)，两臂一致）；
  - 以每个 DSA 层 `o_proj` 的输入（DCP 合并后、本卡头）作为判据；
  - 并用 `AX_ATTN_BOOST=100` 放大 DSA 输出，让 logits（含 CUDA graph decode）也敏感。
- r3（well-scaled，116 v1 未改写入 kernel）：
  - orig_dcp vs orig_ref：DSA 输出 cold/ext/dec 相对 L∞ ≤1.9e-3，**通过** → 证实原栈是复制式、低 slot 下正确；
  - fix_ref vs orig_ref（非 DCP 下 116 恒等）：≤2.0e-3（同为运行间噪声水平），logits 逐位相同；
  - fix_dcp（v1：读路径按分片换算但写入未分片）：ext/dec 相对误差 1.2–1.7 → **不通过**，这正是发现 norope 写入 kernel 的依据。
- r5（116 最终版，含写入分片；全矩阵 orig/fix/full × ref/dcp/dcp_hi/graph）：**待跑**（开发机让给 T52b，gjob `t50_dcp6` 在 `T52b/DONE` 出现后自动运行）。结果写入 `evidence/T50/devbox_r5_summary.txt`。

## 风险
- 只修了 tilelang DSA 路径（A100 唯一可用的路径）。flashmla/fa3/trtllm/aiter 在 DCP 下仍按原样（Hopper 路径，这里不用）。
- MTP/NEXTN + DCP：verify 走了 decode 同款换算，但 draft 池是复制式（底包 loc_space_scale），未测试。
- `SGLANG_DSA_FUSE_TOPK` 的 v2 plan（`SGLANG_OPT_USE_TOPK_V2=1`）未测；A100 强制 v2=0。
- extend 每层做一次前缀 all-gather（底包原本就做，原来白做），外加一张 `(rows+64)·W` 的 int32 表（8 卡约 15 MB）。extend 批次会临时分配 `seq_lens_sum×1 KB` 的 `dcp_kv_buffer`（22 万前缀约 230 MB），占激活余量。
- 底包 planner 假定每个请求的前缀长度是 W 的倍数（前缀按页对齐，101 切点按 64 向下取整，W≤8 时成立）；非对齐的前缀会让 gather 错位（静默错，不崩）。
- `offload_kv_cache`（PD decode / 优先级抢占的 CPU 备份）按虚拟 loc 拷贝 latent，DCP 下不对；我们的配置不走这条路。
- 容量只有约 ×4.4，而且 decode 仍要对 64 头 all-gather Q。性能与 TPOT 要看 8 卡梯子。

## 8 卡复验方案（交 Claude 执行，本任务不提交）
前提：r5 全部 PASS。patch 列表 = RELEASE TIER1 + `114 115 116`（+ 原 025 的 120）。
1. **冷探针 + 能力冒烟 + 容量**（仿 `scripts/pod/jobs/coldprobe_b115_dcp8.sh`）：
   ```
   CP_NAME=b116dcp
   CP_PATCHES="000-interface-compliance.patch 101-d1v12-on-base.patch 105-role-split-single-partial.patch 106-defer-chunk-on-no-kv.patch 110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 112-sm80-indexer-kernels.patch 113-sm80-prefill-indexer.patch 114-indexer-row-shard.patch 115-sm80-sparse-attn-many-heads.patch 116-dcp-dsa-address.patch"
   CP_ARGS="--dcp-size 8 --mem-fraction-static 0.75"
   ```
   看点：
   - `KV Cache is allocated #tokens` 约 450k/卡（算术预期），`max_total_num_tokens×8` 约 3.6M；
   - 20k/60k/190k 冷预填充时间与 F78（b115dcp 2.04/5.89/15.58 s）对比（写入分片后多了 gather，预期接近）；
   - 能力冒烟 ≥10/12。
2. **025 原崩溃序列 + 梯子**（`scripts/pod/jobs/ladder_dcp.sh` 换 patch 列表）：
   ```
   G_NAME=b116dcp120
   G_PATCHES="000-interface-compliance.patch 101-d1v12-on-base.patch 105-role-split-single-partial.patch 106-defer-chunk-on-no-kv.patch 110-sm80-dsa-indexer.patch 111-sm80-fp8-moe-marlin.patch 112-sm80-indexer-kernels.patch 113-sm80-prefill-indexer.patch 114-indexer-row-shard.patch 115-sm80-sparse-attn-many-heads.patch 116-dcp-dsa-address.patch 120-sched-protect-chain.patch"
   G_ARGS="--chunked-prefill-size 16384 --mem-fraction-static 0.75 --dcp-size 8 --cuda-graph-max-bs-decode 64"
   G_ENV="SGLANG_AX_SCHED_COLD_CAP=8192 SGLANG_AX_SCHED_SHORT_TOKENS=8192"
   LADDER="10 14 18 22"
   ```
   - N10 的活跃 token（约 0.5–0.7M）已超过每卡行数（约 450k），必然用到 loc ≥ 行数，正好覆盖原崩溃区间。
   - 看点：无 engine crash；与 025b（非 DCP 最佳）比 formal_est / tpot。
   - 另外 grep `server.log` 里的 `full token usage`，确认实际超过 1/8。
3. （可选）**逐 token 对照**：同一 60k 真实提示 + 追加 600 token 的前缀命中请求，DCP8+116 与非 DCP（同 patch 去掉 `--dcp-size`）各跑一次 temperature=0、`logprobs`。判据：前 64 个 token 的 top-1 一致率 ≥95%，所选 token 的 |Δlogprob| 中位数 < 0.05。需要新写一个 verify 脚本，本任务未提供。
