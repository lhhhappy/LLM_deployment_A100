# T50 DCP 前缀命中寻址修复

- 状态：active　负责人：Codex W24 → Claude subagent T50b（09-23 接手）　创建：2026-09-23
- 关联：dispatch T50，F78/F83，R18 §8.3

## 目标
补丁116修复A100补丁栈DCP前缀命中的寻址协议，并用开发机TP2+DCP2验证。

## 范围
- 包含：底包DCP/DSA/indexer协议、两次extend复现、补丁与生成器、数值及全栈验证、Claude的8卡方案。
- 不包含：镜像、提交、8卡GPU任务或服务生命周期。

## 前置条件与约束
- 用户已授权开发机2卡测试；长任务用gjob，文件限/sjtu/linhang/arena。
- pod仅pread/pexec_codex，底包与原harness只读；不得停止服务。

## 风险与缓解
| 风险 | 缓解 | 回滚 |
|---|---|---|
| DCP虚拟地址与物理池混用 | 写清协议，高虚拟地址、前缀、decode分别验证 | 撤销116并关闭DCP |
| 数值/集体通信错误 | 同随机权重TP2非DCP oracle，logits误差<1e-2 | 保留原栈证据 |
| 开发机环境与真实模型不同 | 明确L1边界，提供TP8完整验证方案 | 不提前宣称L2通过 |

## 里程碑
1. 地址协议和原栈复现。
2. 修复与TP2+DCP2数值验证。
3. 全栈fuzz0、文档、记录、交付。

## 验证方式
- CPU：生成器确定性、完整000→101→105→106→110→111→112→113→114→115→116→140→120→130→150→160 fuzz=0。
- GPU：先长前缀再续算；DCP2对非DCP logits相对L∞及绝对误差，门1e-2；边界/高地址/解码。
- L2：由Claude执行8卡数值、能力、N10原崩溃序列与缓存压力回归。

## 进度记录
- [x] 规则/交接/已有证据阅读，任务in-progress。
- [x] （T50b）CPU 侧：底包 DSA 读路径与 indexer K 全按虚拟 loc 访问每卡大小的池。证据 `evidence/T50/address_protocol.txt`（rev2）。
- [x] （T50b）开发机 r1/r2：dummy 权重下 logits 与注意力无关（所有变体逐位相同）→ 改为 well-scaled 初始化 + DSA o_proj 输入判据。
- [x] （T50b）r3 实测纠正根因：GLM rope 维=0，latent 写入走 `set_mla_kv_buffer_kernel_norope`（`mla_buffer.py:87-114`），**没有 DCP 分片**，每卡按虚拟 loc 写全部 token。原栈 DCP = 复制 KV（低 slot 正确、无容量增益、高水位超过每卡行数时越界 → 025 崩溃）。116 v1（只换算读路径）在 ext/dec 上不通过（相对误差 1.2–1.7）。
- [x] （T50b）116 最终版：norope 写入按 owner 规则分片 + extend 读 dcp_kv_buffer + decode/verify 转本卡行 + indexer K 虚拟空间复制 + 显存核算。全栈 fuzz=0（本地、开发机）。`patches/116-dcp-dsa-address.{patch,md}`（含 8 卡方案）。
- [ ] r5 全矩阵（orig/fix/full × ref/dcp/dcp_hi/graph）：开发机让给 T52b；gjob `t50_dcp6` 在 `T52b/DONE` 出现后自动跑。
- [ ] 根据 r5 回填 116.md、evidence，交付。
## 决策记录
- 2026-09-23：先确认各池地址域，禁止用冷请求通过替代前缀命中正确性。
