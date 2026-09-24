# v2事件生成核心：独立bug审查

2026-09-24。只读审查`longchain_events.py`及build集成；新增独立回归`tests/test_longchain_events.py`，不修改核心。只用CPU fixture，不触碰源数据、旧冻结集或GPU。

**最终独立复核：3项bug及随后发现的“摘要内引用旧system-reminder”分支均已修复，13/13回归通过（含新增短历史重建重试）；本次审查范围未发现剩余阻断bug。** 不等于完整真实token成品或GPU验收通过。下面的行号/行为描述对应发现时版本。

## P1：重建摘要在下一次正常增长时被删除（普通内容及引用reminder分支均已修）

位置：`longchain_events.py:108–110`、`:248–250`、`:269–278`。

重建把持久摘要存成user/system-reminder；当没有retained tail时，它就是末尾消息。下一次intra或turn_start无条件把末尾reminder替换掉，于是刚建立的压缩历史消失，后续只从opening task增长。该操作没有独立重建事件，破坏“重建后保留新历史再增长”的契约，也减少后续实际prompt长度和工作量。

实测反例：一条task加一段1920字符旧assistant历史→重建无尾部→分别接intra和新user事件。两个测试都发现summary已被替换。修复将持久摘要改为普通user中的固定格式；不再命中临时reminder替换规则。intra与turn_start两条普通内容回归均保住整个重建后前缀。

追加反例：摘录中包含旧user的`<system-reminder>Original controller note</system-reminder>`。虽然外层summary已是普通user，`lc.reminder`仍因内容任意位置有该子串而返回True；下一intra或turn_start再次把summary删除。两条测试起初均FAIL。后续修复在临时reminder替换条件中明确排除固定摘要头部；独立复跑两条均PASS，摘录里的原reminder文本仍保留。

## P1：context pressure的重建fallback在最需要时不可达（已修）

位置：`longchain_events.py:260–262`与`longchain.py:568–579`。

`event(intra)`在没有donor满足估计room时直接抛ValueError；build只在event返回并渲染之后发现真实超限才进入forced rebuild。room<=0或所有donor估计太大时，整轮提前终止，即使当前历史完全可以压缩到可用范围。

端到端fixture：2公开请求，最后prompt501，下一输出100，上限650，room=-207；可用摘要短于550，因此存在合法显式重建。修前build却直接在event抛“no complete compatible material fits context budget”。修复使用专门NoMaterialFits异常进入显式重建，不吞掉其他结构错误。端到端fixture现成功输出3请求，最后phase=context_reset、预算仍100，保留前两条预算10，来源明确displaced_planned_kind=intra。

## P2：被丢弃候选仍消耗素材usage（已修）

位置：`longchain_events.py:240–241`、`:270–271`与`longchain.py:575–580`。

event在真实渲染和预算验收前已修改global_usage/chain usage。若候选最终超限，build丢弃它并改成context_reset，但计数没有回滚，后续选择惩罚及`unique_donor_blocks/max_donor_reuse`会把未出现在成品的素材算成已使用。应在接受候选后commit，或完整回滚失败尝试的usage（必要时明确随机数消费语义）。修复把计数移到真实渲染/预算通过后的compiler.commit。新增intra和turn_start两条独立回归：丢弃候选不加计数，接受一次后对应素材各计1；reset不消耗素材。build压力fixture只commit一次context_reset，产物unique_donor_blocks为0。

## 已验证的不变量

完整近期assistant调用+tool结果组按原子边界保留；输出深拷贝不修改传入旧body的测试通过。跨session query的选择明确排除receiver源session；当前产物用receiver历史，不直接导入donor整份prompt。素材来源和query适配方式写入receipt。

源phase约束按原请求中已出现的turn/reset扣除后分配，reset和turn位置不重叠；cap受剩余请求数限制。尚不能据此声称所有重建位置都在真实token下可行，短历史、连续重建和输出大尾还须让真实CPU生成暴露不可行组合并明确处理。

## 执行记录

`.venv-longchain/bin/python -m pytest -q tests/test_longchain_events.py`：修前3 failed / 3 passed；修后独立复跑6/6通过，再补计数回归后**8/8通过**。覆盖摘要持久性(intra、turn_start)、pressure fallback及commit次数、原子工具组/不可变输入、跨模板拼接100请求链的缺失事件计数、跨session借query的receiver历史与来源、丢弃/接受候选的usage。pressure fixture使用CharacterRenderer，只核控制流，不是实际GLM token验收。

追加全量数据边界：只读扫pilot-v2b已落盘455行/5reset以及当时生成中v2的408行/8reset，均未发现空tail且summary含system-reminder的组合。后一结果仅生成前缀，不能证明完整产物未触发。

最终再次运行同一测试文件：10 passed in 0.01s。两类普通摘要/引用reminder摘要都能跨intra和turn_start保留；计数commit、pressure fallback与其余不变量全部通过。源码修复不自动证明修前已启动的成品使用了新代码，完整成品的实现hash与摘要持久性仍须按实际生成版本核对。

## 短历史重建缩短重试：追加独立复核

build新增“初次摘要不比旧prompt短时，保持同一次context_reset与行为reference，按keep_fraction=0、摘录120×4→32×1→1×1逐级重编译/渲染”的分支已只读review。接受前仍检查真实长度严格下降、prompt+output不超限；没有把失败reset偷偷改成intra，commit仍只在最终通过后调用。若最小摘要仍无法缩短，则明确失败，不声明成功。

复用CharacterRenderer端到端build fixture新增3档：旧内容180、70、35字符，分别在120、32、1字符摘录限制处成功。180档额外带近期尾部，初次只替换中间历史并保留尾部，实测摘要使总prompt变长；重试清除tail并重新选摘录后缩短。三档均断言：

- 初次重建渲染长度确实大于原prompt，最终严格变短；每次尝试使用同一个reset及相同reference。
- opening task逐字保留；两条公开body不变，原预算各10不变，合成输出预算仍100。
- 只commit一次context_reset；`rebuild_render_adjustment`明确记录最终keep_fraction/摘录长度/数量，且尝试顺序与实际分支一致。
- 没有新增或丢失请求：每个fixture仍是2公开请求+1合成请求。

最终独立运行：`.venv-longchain/bin/python -m pytest -q tests/test_longchain_events.py`，**13 passed in 0.03s**。该新分支范围未发现阻断bug；fixture仅证明控制流与不变量，完整GLM成品验收由实际产物报告负责。
