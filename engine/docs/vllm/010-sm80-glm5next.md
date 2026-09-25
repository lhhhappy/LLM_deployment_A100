# vllm 010 — GLM-5.3-Flash 稀疏注意力层在 A100（sm80）上运行

官方 vLLM 底包在 A100 上启动 GLM-5.3-Flash 会失败，原因全在 11 个 DSA（稀疏注意力）层和同为 DSA 的 MTP 层：

1. 稀疏 MLA 注意力后端全部要求 SM90+，后端选择报 "No valid attention backend found"（`vllm/platforms/cuda.py` sm8x 分支）。
2. 索引器的打分（MQA logits）只有 DeepGEMM 实现（SM90/SM100/SM120）；官方包自带 DeepGEMM，所以 `has_deep_gemm()` 为真，会一直走到 DeepGEMM 调用才失败。
3. kpool 索引器的三个 Triton 内核直接向 fp8 指针写数；Triton 在 SM89 以下拒绝 fp8e4nv 转换。

其余部件在底包里已有 sm80 路径（按源码阅读）：路由专家与稠密层 FP8 权重走 Marlin（W8A16）、KDA 走 Triton（FlashKDA 限 SM90+）、mHC 走 TileLang、通信走 NCCL。

## 改动（来源：github.com/wtdcode/vllm-backport master 559be900，Apache-2.0，逐文件注明）

| 文件 | 改动 | 来源 |
|---|---|---|
| `vllm/v1/attention/ops/fp8_sm80.py`（新） | Triton 用的 e4m3fn 软件编码（就近舍入到偶数、饱和）与 256 项查表解码；SM89+ 编译期仍用硬件转换 | 分支原样 |
| `vllm/models/glm5next/nvidia/ops/kpool_compress.py` | 三个内核加编译期常量 `NATIVE_FP8`：SM89+ 路径代码不变；A100 上把编码后的字节写入同一缓冲的 uint8 视图 | 分支 13 个 hunk，按上下文精确匹配应用（只有行号偏移） |
| `vllm/v1/attention/ops/mqa_logits_triton.py`（新） | DeepGEMM `fp8_mqa_logits` / `fp8_paged_mqa_logits` 的 Triton 实现（fp8 解码成 bf16 后做 `tl.dot`，fp32 累加） | 分支内核；4 个调优环境变量改为分支默认值常量，未改 `envs.py` |
| `vllm/models/glm5next/nvidia/sparse_indexer.py` | `_use_triton_mqa_logits()`：CUDA 且 `support_deep_gemm()` 为假（即 A100）时 prefill/decode 打分改用 Triton；A100 上 MXFP4 索引缓存启动即报错 | 按分支思路重写到当前文件（上游已把该文件搬入模型目录） |
| `vllm/v1/attention/backends/mla/indexer.py` | DeepGEMM 调度元数据只在 `support_deep_gemm()` 为真时计算 | 同上 |
| `vllm/v1/attention/ops/common.py` | `pack_seq_triton` 遇到 fp8 且硬件无原生转换时按字节打包 | 分支思路，限定在 SM89 以下生效 |
| `vllm/v1/attention/ops/triton_mla_sparse_kernel.py`（新） | split-KV 稀疏 MLA Triton 内核，NoPE（512）与 RoPE（576）两种头宽 | 分支原样 |
| `vllm/v1/attention/backends/mla/triton_mla_sparse.py`（新）＋registry/cuda.py | `TRITON_MLA_SPARSE` 后端，只接受 `capability.major == 8`，排在 sm8x 候选表最后 | 分支后端按当前接口改写，见下 |

相对分支的两处必要改写（否则在当前底包上出错）：

- **top-k 宽度按共享缓冲的实际列数**（2176），不是 `index_topk`（2048）。GLM 的 kpool 索引器在 2048 个选中 token 后追加最近一个未满 pool 的 `kpool−1` 个 token；
  XPU 基类只换算前 2048 列，会让注意力看不到最新的几个 token。
- **`record_logical_topk_ready()` 为空操作**：当前底包的 MLA 层对每个稀疏层都会调用它（为共享 index group 服务，GLM 不用），分支后端没有这个方法，第一次前向就会报错。

## H100 及以上不变的依据

- 所有新分支都由硬件能力决定：`support_deep_gemm()`（SM90/SM100/SM120 为真）、`supports_fp8()`（SM89+）、后端 `supports_compute_capability`（只收 major 8）。
- 选择 Triton 打分的条件是硬件不支持 DeepGEMM，而不是分支用的 `is_deep_gemm_supported()`；后者还受 `VLLM_USE_DEEP_GEMM` 影响，会在 H100 的非默认配置下改变路径。
- 新后端排在 sm8x 候选表末尾；H100 上前面的 FlashMLA/FlashInfer 稀疏后端先被选中，且新后端自身拒绝 major≠8。
- `kpool_compress.py` 在 SM89+ 上 `NATIVE_FP8=True`，编译出的内核体与底包相同（常量分支）。

## 验证（层级与结果）

| 层级 | 内容 | 结果 |
|---|---|---|
| 单卡 A100，单元测试 | fp8 编码对 torch 转换逐字节一致（含全部 256 个码、就近偶数的中点、饱和、次正规数）；索引器 Q 量化 `fwht128_quant_fp8` 对独立 torch 参考逐字节一致；kpool decode 写入对独立 torch 参考、与 prefill 写入一致 | 48 通过（1 项 AMD 专用跳过） |
| 单卡 A100，单元测试 | Triton prefill/paged 打分对 torch 参考（含部分屏蔽、`clean_logits=False`、大块号 int32 溢出、q 预解码与内核内查表逐位相同）；稀疏 MLA 内核 NoPE/RoPE 对 torch 参考、split-KV 与单遍一致 | 123 通过；另 1 项测的是 ROCm 的 torch 参考实现（分支改过、未移植、不在 CUDA 路径）已从移植的测试中去掉 |
| 单卡 A100，单元测试 | mHC pre/post 的 TileLang 内核在 GLM 形状（hidden 4096、4 路残差、20 次 Sinkhorn）对 torch 参考 | 8 通过 |
| 两卡 A100，TP2、真实配置截成 8 层（6 KDA + 2 DSA）+ MTP、dummy 权重 | 启动日志确认：索引器走 Triton 打分、注意力 `TRITON_MLA_SPARSE`、MoE `MARLIN`；PIECEWISE/FULL（含解码）CUDA graph 全部捕获成功；接口探针全部通过 | 见 [探针记录](../../../research/claude/vllm/README.md) |
| 八卡 TP8 真实权重 | 数值、能力、性能 | 未做 |

测试文件：`tests/kernels/attention/test_fp8_sm80.py`、`tests/kernels/test_kpool_fwht_quant.py`（新写）、`tests/kernels/test_kpool_decode_update_batched.py`（底包原有）、
`tests/kernels/attention/test_mqa_logits_triton.py`、`test_triton_mla_sparse_nope.py`、`test_triton_mla_sparse_kernel.py`（分支）。
数值上，Triton 打分对 fp8 值精确解码后做 bf16×bf16、fp32 累加，与 DeepGEMM 的 fp8 MMA 舍入不同，接近打平的 pool 在 A100 与 H100 上可能排序不同；这是预期差异，不单独说明有错。
