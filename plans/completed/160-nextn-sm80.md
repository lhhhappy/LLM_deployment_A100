# T48 M4 NEXTN sm80

- 状态：completed　负责人：Codex W22　创建：2026-09-22
- 关联：M4，dispatch T48

## 目标
交付可叠加指定底包栈的160、路径审计、A100算子证据与未执行8卡任务。

## 范围
- 包含：NEXTN draft/target verify、DSA/KDA/Marlin、状态与graph、采集和估算。
- 不包含：8卡运行、镜像、提交、完整权重加载。

## 前置条件与约束
- 用户已授权2卡开发机算子测试；仅arena目录、使用前确认GPU空闲。
- base_exact等四目录只读；不调用bohr/Trisol/pod。

## 风险与缓解
| 风险 | 缓解 | 回滚 |
|---|---|---|
| 140与spec状态生命周期未经验证 | NEXTN关闭角色功能，保留原extra_buffer提交 | 撤160/重启非spec |
| 开发机依赖不同 | 记录版本及缺失，专属目录补依赖 | 删除T48自有目录 |
| 算子通过不等于TP8通过 | 明列未测集成，交Claude执行脚本 | 不晋级发布 |

## 里程碑
1. 源码路径与sm80分支审计。
2. 160和CPU配置/补丁栈验证。
3. 开发机随机算子、数值、graph；8卡脚本与交接。

## 验证方式（机械可检查的优先）
- 命令：make_160、verify_160、test_mtp_sm80_160、解析器测试。
- 通过标准：全栈fuzz0/py_compile/反向还原；数值与graph对照；明确每项未测。

## 进度记录
- [x] 已读规则和基线；两卡空闲。
- [x] TileLang缺libz3.so.4.15已在T48独立目录补依赖。
- [x] 160兼容策略/精确统计，12补丁fuzz0、3624源码+8工具编译与reverse；CPU10项及真实resolve/draft mapper通过。
- [x] KDA/DSA/kpool/indexer/seed/accept/EH/mHC/MoE clip10数值与graph通过。
- [x] dense FP8 Marlin 9形状/27动态graph通过；7个本轮JIT build误落根盘的问题已迁回arena并正确配置重跑，原始证据保留。
- [x] 8卡脚本和接受统计采集器准备完毕，未执行。

## 决策记录
- 2026-09-22：首轮使用topk1线性链，保留fp32 KDA状态；不引入ReplaySSM。

- [x] F69/D37、P160-01…06与Claude交接已落盘；L2仍未执行。
