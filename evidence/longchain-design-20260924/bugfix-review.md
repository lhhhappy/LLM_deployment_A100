# 三路bug审查与修复汇总

2026-09-24。用户要求三个subagent重点核查bug，随后明确授权修复；8卡侧继续原数据布局和原runner，只换root/set/cohort。没有修改只读s1-dev/harness、task.md、旧冻结数据或GPU队列。

| 问题 | 本轮处理 | 验证 |
|---|---|---|
| 工具ID替换改坏role/工具名、级联替换、漏tool_call_id | 单次带边界替换，结构路径保护，ID字段准确映射；重复定义拒绝 | 短ID、级联、alternate ID、业务字段同名引用与结构content.type反例 |
| 损坏父数据经polish重新冻结为VALID | 读取/写出前验证父artifact全集、路径和hash | budget从10篡成777，修前被洗白，修后在生成前拒绝 |
| 连续两次polish误判来源 | adapted父代沿最初source hash/frozen labels验证 | 三代fixture均VALID，祖先账保持不变 |
| 缺provenance/错kind绕过合成标签检查 | generated强制完整来源文件与kind枚举 | 删除来源和拼错kind坏例均INVALID |
| 缺session回放KeyError、链内换session | 必填身份字段/phase校验；同链session稳定及链摘要一致 | 缺失和切换session坏例均INVALID |
| 新工具组复用历史/同suffix调用ID | 只对新增suffix检查重复及历史碰撞，保留源历史兼容 | 两种重复位置反例均拒绝 |
| 合成追加偷偷改system/tools | 当前追加机制下显式拒绝 | system和tools坏例；未来合法版本变更需专门事件与独立验证 |
| 生成实例继承同一个source session | 新build每chain独立接收session，provenance保留source session | 两条共源session链生成8请求，原load_index/freeze接受8/8 |
| 首合成排序早于上一已知end | 排序起点尊重已知end；后续仅保序，不虚构执行时长，provenance声明ordering-only | offset 10/60/105/110，end未知仍未知；gap独立 |
| 原self-check缺body也PASS、body cache可读旧材料 | 原harness不改；可选guard先完整checker验收，自检禁body cache，回放fresh out并复用原checked runner | 缺body/渲染不完整/旧out拒绝；runner失败不评分；显式新requests评分 |

独立报告：[生成器](review-generation-bugs.md)、[checker](review-checker-bugs.md)、[harness](review-harness-bugs.md)。修前与修后JSON及最小复现脚本均在本目录。反例没有证明旧96链/1718请求成品已损坏；旧集来源/哈希/完整渲染收据继续按当时范围解释。

最终相关回归测试 **53项通过**；CLI帮助及本地链接检查通过。

验证命令：

```sh
python3 -m pytest -q tests/test_longchain.py tests/test_longchain_check.py tests/test_longchain_replay.py
python3 evidence/longchain-design-20260924/repro-generation-bugs.py
python3 evidence/longchain-design-20260924/reproduce-checker-bugs.py --fake-renderer
python3 evidence/longchain-design-20260924/repro-harness-bugs.py
```

本轮极小端到端fixture使用明确的CharacterRenderer测试替身，只证明控制流、来源和结构；不是GLM tokenizer/全量数据验收，更不是模拟性能分数。本机搬迁venv不能导入冻结依赖，未用不兼容版本冒充验证。新成品必须在正确tokenizer环境执行全量渲染和原harness验收。

边界：跨session借素材本身兼容固定轨迹回放。新事件计划、真实重建与重建后增长、素材来源真实性核验和跨session克隆审计仍需实现；超大gap即使结构合法，也不代表等待分布有观测依据。正文变化对MTP/路由及闭环交错的实际影响仍需基线测量，不承诺与官方分数等价。

可选guard不是部署要求或新schema。已有队列可继续原入口，只需同等执行完整验收、全新输出目录和当前root的判分；无需服务端识别“生成/原始”来源。
