# R22 — KDA BF16 投影融合：171 的首轮算子筛选

2026-09-24，开发机 A100-SXM4-80GB GPU0。对应[用户交接](../../notes/codex-handoff-执行层并行任务.md)，以改善首 token 前的执行成本为目标。已完成单机制候选和开发机筛选；实际 TP8、真实权重、整模型 TTFT/TPOT 尚未验证。8 卡 046r 与现有服务没有改动。

## 改动与依据

[171 patch](../../patches/171-kda-bf16-proj-fusion.patch) 只改 `glm5_next.py`，默认关闭。模型级 FP8 配置存在时，逐项检查原 KDA 投影是否免量化，再启用底包已有融合模块。没有改变 MoE、卷积、KDA recurrence 或 MTP 接受算法。加载布局、开关、兼容条件见[同名说明](../../patches/171-kda-bf16-proj-fusion.md)。

checkpoint 配置的 34 个 KDA 层满足条件。实际基线输入侧为 qkv、b、f_a、g_a 四次 linear，融合后一次；f_b、g_b 再由两次 linear 合成一次 bmm。CPU profiler 验证原路径 `aten::linear=6, aten::mm=6`，候选 `linear=1, mm=1, bmm=1`，没有额外 clone/contiguous 出现在这段投影中。此 profiler 仅用于计调用，速度另测。

## 正确性证据及覆盖边界

使用[通用算子测试](../../tests/gpu/test_kda_fusion_171.py)，导入实际候选模型类、FP8 配置和完整 `load_weights` 方法；单层容器仅省掉模型其他层。权重为真实形状的随机 BF16，全部参数先填 NaN，再以两种打乱顺序加载。各 rank 与 checkpoint 切片逐位比较，覆盖 f/g 一级复制、二级分片。一个 GPU 依次模拟 rank 0–7，**没有执行 TP8 collective**。

| 检查 | 本轮结果 |
|---|---|
| 加载 | 8 个模拟 rank 全部无未加载参数；融合分片与 checkpoint 切片逐位一致 |
| 投影形状 | 1/4/33/37/63/65/256/1024/4096/8192/16384 行；误接 f/g 的负对照被拒绝 |
| 图重放 | 上述形状，基线/候选各捕获投影图；每图 3 次更换输入，图输出与同输入 eager 逐位一致 |
| 状态连续性 | 两请求冷输入 273/65；各 decode 1；再追加 37/63。用真实 conv、chunk KDA、recurrent、norm 及本 rank o_proj GEMM |
| MTP | 保留上述真实历史状态，两个请求各 verify 4；原 SSM 不变；真实 scatter 对接受 1/2/3/4 分别提交，结果等于对应 scratch 步 |
| dtype | checkpoint BF16、实际分配 FP16 时，两级 dtype 一致且能前向 |

状态算子使用实际生产布局的 conv 池 `[slot,3,3072]` 转置视图。局部层输出没有 o_proj 的跨 rank 归约；没有通过完整 attention backend/scheduler 构建元数据。MTP 检查对应正式 A 的非融合 verify 算子，未跑 draft 模型、完整验证 graph、retract 或缓存淘汰。

最终运行中，conv/window 对照逐位一致；SSM 最大绝对差约 `3.82e-4`（含 verify/accept），连续性检查中局部层输出相对 L2 最大约 `0.00392`。这些是随机权重算子误差，**不是可直接移用于 TP8 真实模型的容差**。测试的 `rel_l2≤.01, max_abs≤.05` 是本轮筛选阈值；必须另测真实模型基线重复误差、逐层状态与输出。

## 速度：投影有收益，整模型尚无结论

同设备、相同输入、相同形状；每臂预热 10 次，每组 11 次样本，每样本小块 10 次调用、大块 3 次。依次 base→fused、fused→base 两组，原始 wall/CUDA-event 样本均保存。表中墙钟包含同步等待；eager 的 CUDA event 区间也可能包含 host 发射间隙，不能称为纯 kernel 时长。权重反复使用，未模拟整模型跨层的 cache 压力。

| 行数 | eager 基线→候选（µs，合并样本中位数） | 投影 graph 基线→候选（µs） |
|---:|---:|---:|
| 256 | 141.3 → 87.1 | 76.1 → 49.3 |
| 1024 | 波动明显，见下 | 160.5 → 118.7 |
| 4096 | 587.6 → 477.4 | 562.3 → 452.7 |
| 8192 | 1104.5 → 945.4 | 1098.1 → 929.2 |
| 16384 | 2081.9 → 1871.5 | 2069.5 → 1861.9 |

1024 行 eager 两组中位数分别是 329.4→226.8µs、174.8→126.2µs，早期探针为 174.5→124.3µs。两臂存在共同 host 波动；不使用合并后的约 2× 比值作稳定收益。其他行也保留两组值和 p95，未只选最好一次。

不能把此表乘 34 当整模型 TTFT 实测：其他 kernel、跨层依赖、通信、CPU/GPU 重叠及排队都没计入。局部收益是继续做模型对照的依据，尚不够决定部署，更不证明任何并发档晋级。

## 内存与成本接口

每个 KDA 层参数为 **36,294,944 bytes/rank**，融合前后相等。没有额外持久 BF16 权重副本、KV 或 SSM 池。融合输出让 qkv/beta 共享包含 f/g 中间值的底层存储；临时张量生命周期会改变。

独立 eager 探针相对基线的峰值 allocated 增量：c=256 为 +64KiB；c=4096/8192/16384 为 +1/+2/+4MiB。两套层与图在同一进程中，记录的是每臂调用前后 peak 增量；**不是服务端峰值、图池总量或 KV 容量损失**。graph 的 allocated 增量受 workspace/捕获顺序影响，不用它宣称节省整模型显存。

JSONL 明确 `scope=kda_projection_only`、`emulated_ranks=true`；P、请求 batch、scatter、MTP 服务状态、KV/KDA 容量为 null。状态测试的历史是真实计算得到的 273/65、274/66、311/129 token；投影计时本身无 P。调度方不能直接拿这张表替换 `T(c,P,B)`。

## 下一轮的判据

1. 由队列负责人安排正式 A 原路径 flag=0/1 的实际 TP8 数值和状态对照，打印各层融合选择与实际 dtype。保留 MTP、原调度和池设置。独立核对 scatter/非整齐行、长前缀、多请求、graph 与接受路径。
2. 扩展现有 `extend_check.py` 或同一通用探针，测模型实际块成本、最慢 rank、workspace/图内存、稳定 KV/KDA 容量。局部收益若未兑现或被内存代价抵消，记无净收益。
3. 完整负载同时看 fast/overall/turn/chain 四道 TTFT、TPOT、完整性与其余硬门。用户指出的首 token 约束是主验收项之一；算子加速不替代完整过门。170 v2 的 TP8 正确性和小块固定成本继续独立验证。

源码/算子结论待对方对照原始数据复核后用于部署决策。

## 复现与证据

- 最终[原始 JSONL](../../evidence/kda171/final.jsonl)、[stderr](../../evidence/kda171/final.err)、[job 完成记录](../../evidence/kda171/job.log)、[汇总](../../evidence/kda171/summary.json)。补丁与源码 SHA256 均写在首行，并在本地与当前候选核对相等。
- [早期完整探针](../../evidence/kda171/initial-result.jsonl)在 dtype 跟随修正前，BF16 数学路径相同；没有 MTP/FP16 检查，仅用于保留测量波动证据。当前结论以 final 为准。更早两个试跑因测试上下文配置失败，未得到完整结果，不计通过。
- CPU 的 3 个资格测试通过；底包/S0/正式 A 栈以 fuzz=0 应用，AST 语法检查通过。没有新增正式提交或 8 卡 SLO 结果。

开发机复现（仅在已分配 GPU 上，`src` 为从只读底包应用当前 171 的工作副本）：

```bash
cd /sjtu/linhang/arena
source code/e1_env.sh
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/sjtu/linhang/arena/runs/kda171/src:/sjtu/linhang/arena/cache/T48/deps
env/m0/bin/python runs/kda171/kit/test_kda_fusion_171.py \
  --config runs/kda171/kit/config.json \
  --patch runs/kda171/kit/171-kda-bf16-proj-fusion.patch
```
