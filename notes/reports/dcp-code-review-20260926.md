# DCP / MTP 调用关系与实现审查

2026-09-26，Codex。范围：`codex/dcp-mtp-n34`，起点 `aebaff56`。
这是源码自审记录；运行结果与判定见实验记录，工程目标与验收见 [DCP方案](../plan-dcp-8card.md)。尚待另一参与者对照原始记录复核。

## CodeGraph

按用户要求安装 [CodeGraph](https://github.com/colbymchenry/codegraph)，固定 `v1.6.0`。独立运行时在共享仓库 `build/tools/codegraph/runtime`；只索引本 worktree 的 `engine/sglang`，数据库约306MB，已通过本地 Git exclude 排除。

入口：`scripts/analysis/codegraph_dcp.sh`。遥测关闭，CLI 使用时关闭后台 daemon；改源码后先 `sync` 再查询。首次索引4,038文件、105,930节点、316,850关系，[原始状态](../../evidence/dcp-mtp-20260926/codegraph_status.txt)。未改 agent 的全局指令或 MCP 配置；本会话通过CLI直接使用。

可复跑查询：

```sh
scripts/analysis/codegraph_dcp.sh sync
scripts/analysis/codegraph_dcp.sh callers move_kv_cache --json
scripts/analysis/codegraph_dcp.sh callers is_dcp_mla_decode_phase --json
scripts/analysis/codegraph_dcp.sh node is_deepseek_dsa --file srt/configs/model_config.py
```

图是定位工具。Python动态分派、属性访问、Triton间接kernel调用需直接核源码；`loc_space_scale`的callers为空并不代表未使用。宽泛自然语言查询会匹配到无关的同名`gather`，本轮用精确符号和文件限定复查。

## 跨文件影响

| 改动 | 源码确认的调用方 / 约束 | 验证要求 |
|---|---|---|
| MLA `move_kv_cache` | MTP前缀分支复制、接受压实；图还列出multi-ended allocator压实 | 后者实际用UnifiedMLA覆写，参数是物理页位置，不能套用虚拟loc协议；当前补丁只改普通MLA实现 |
| `IndexKeyCache.move` | DSA移动latent后移动indexer | DCP开启时页内key区与scale区分别寻址，一次性快照源；关闭DCP保留固定基线原路径，现行topk=1不调用接受压实 |
| `is_dcp_mla_decode_phase` | CUDA和ROCm的prepare/core均调用 | 本轮实测范围是A100 TileLang；ROCm不能据本轮结果宣称通过 |
| `_should_return_dsa_dcp_lse` | DSA extend、decode、TRTLLM分支 | Q gather和LSE返回/合并必须一起覆盖DRAFT_EXTEND_V2；其他backend须独立验证 |
| 草稿 `loc_space_scale` | pool_page_size和多种pool创建器；是property | DSA判定包含`Glm5NextForConditionalGenerationNextN`；草稿物理latent及逻辑indexer必须与预算一致 |
| HiCache indexer镜像 | hybrid Mamba host stack的声明式sidecar | indexer覆盖anchor逻辑容量；latent传输按W分片，indexer不分片；总host预算计入W倍indexer |
| DSA indexer设备分配 | 普通factory、hybrid目标池、NextN构造三条路径 | 必须在`DSATokenToKVPool`统一执行容量约束；只改`_create_dsa_pool`会漏掉当前GLM目标 |

[move调用图](../../evidence/dcp-mtp-20260926/codegraph_move_callers.json)，[阶段判断调用图](../../evidence/dcp-mtp-20260926/codegraph_phase_callers.json)。

### 搬运协议与边界

源/目标描述符是所有DCP rank一致的虚拟loc。每层先读源位置快照，仅源owner保留字节，再用NCCL uint8 SUM传播；每个字节只有一个非零贡献，保存BF16/FP8位模式。目标owner写本地行，空目标rank仍参与。单层临时量约`搬运token数 × latent行字节数`，逐层复用；不依赖整个KV池大小。本轮对整池大规模压实不作性能承诺。

当前topk=1链式MTP不触发接受后压实。压缩kpool的一个indexer条目代表多个token，还带请求尾状态；DCP模式下任意token压实明确报错，发生在latent变更之前。原语通过不能升级为“树式MTP已支持”。

### MTP与首字路径

源码顺序：target prefill → `future_map.publish` → draft prefill → worker返回 →结果D2H（copy stream等待forward stream）→服务输出。publish用于下轮调度准备，不等于首token已经发给客户端。因此草稿prefill可能直接进入TTFT关键路径，具体毫秒数需要trace；另外还有KV占用、块间decode占时和闭环到达密度的间接影响。130f的开关消融只能评价组合净效果，不能直接分离这几项。
