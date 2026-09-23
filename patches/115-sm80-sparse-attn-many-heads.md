# 115 — sm80 tilelang 稀疏 MLA：支持每卡 64 头（DCP 模式），并去掉无用的 O_shared
- 问题：`--dcp-size 8` 时每卡持有全部 64 个注意力头，`sparse_attention_fwd_kernel_v1` 需要 Q 64×512 + 双缓冲 KV + O_shared > 256KB 共享内存，A100 上限约 164KB → 启动失败（pod 探针 012e：`Failed to set the allowed dynamic shared memory size to 262144`）。
- 改动（`kernels/ops/attention/dsa/tilelang_kernel.py`，仅 v1 kernel + 分发）：
  - 删除 `O_shared`（写入后从未读取）；新增 `max_heads_per_block` 参数（原硬编码 64）。
  - sm80 且头数 ≥64 时用 `block_I=64, num_stages=1`（开发机扫描 7 种组合中最快且能编译：T=512 2.33–2.87ms；其余布局冲突或超限）。
- 验证（开发机 A100，`build/p115_test.py`、`build/p115_sweep.py`）：H=8 与原版**逐位相同**、速度不变；H=64 相对误差 ≤3.4e-3、CUDA graph 正常；原版 H=64 按预期失败。
- 观察：64 头一起算（2.3–2.9ms/512 token）远快于 8 次 8 头（≈7.4ms）→ 同一 KV 读一次供 64 头用，印证 TP 按头切分的重复读 KV 问题。
