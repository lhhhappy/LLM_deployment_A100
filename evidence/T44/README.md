# T44 / W18 — 113交付证据

最终 **PASS**：独立113叠加112，六档8192query预填充达131.4–185.7等效TFLOPS，比112快6.19–6.69×。完整说明 `patches/113-sm80-prefill-indexer.md`，事实F65、决策D34。

| 文件 | 内容与复现 |
|---|---|
| `final_all.log` | 最终源码/oracle/112/测试SHA、环境、全部112用例与新增形状；222数值对照、30动态graph、六档性能及decode对照；末行PASS。`scripts/test_sm80_indexer_113.py --mode all` |
| `summary.json` / `summary_check.log` | 机械核验最终源码/oracle/测试/patch/compiler/profile SHA与decode PTX相同，数值门/性能目标；`python3 scripts/summarize_113.py` |
| `performance_table.md` | 从同一最终日志生成的旧新ms、等效TFLOPS与加速；JSON保留全样本、有效区间pair比例 |
| `generate_receipt.json` / `generate_apply.log` / `generate.log` | 最终生成基线、kernel与补丁SHA；`python3 scripts/make_113.py` |
| `stack_receipt.json` / `full_stack_apply.log` / `verify.log` | 000→101→105→110→111→112→113→120→130全fuzz0、3623编译、确定生成、整栈反向字节还原、只读底包不变、decode/fallback函数字节一致；`python3 scripts/verify_113.py` |
| `compiler/` / `inspect.log` | 从最终入口实际dispatch捕获的PTX/TTGIR和编译收据；主kernel162寄存器/0spill/48KB shared、sm80 bf16 MMA；decode PTX SHA与T43原值全同 |
| `profile/` / `profile.log` | torch CUDA profiler：112/113表格、JSON与Chrome trace；190k causal 112平均624.397ms、113主kernel96.102ms、两个预解码合计0.104790ms/调用。不是ncu硬件counter |
| `gpu_initial.log` / `gpu_final.log` | 开发机使用前/收尾占用；仅GPU0算子，本任务进程已退出，收尾两卡4MiB/0% |
| `prototypes/prefill.py` | 完整扫描原型：raw/predecoded、query-major/key-major、BQ/BK/warp/group/loop/stages。`scripts/tune_sm80_indexer_113_refine.py --mode layout/focused` |
| `prototypes/query_major.py` | 首轮query-major原型，`scripts/tune_sm80_indexer_113.py --path query_major.py`；最终脚本已适配Triton计时返回值 |
| `tune1.log` | 首轮计时脚本把单quantile返回的float按list索引，计时记录失败，数值抽检通过；主动结束后修正返回值处理 |
| `tune2.log` | 修复计时后的raw/predecoded扫描；部分大tile极慢，W18主动提前中止并转入有spill门控的布局扫描，不宣称首轮全部完成 |
| `layout190k.log` | 190k布局/BQ/BK/warp扫描完整完成，跳过有spill组合，最佳query-major BQ2/BK128/4warps |
| `focused190k.log` | 190k分组/loop/stages完整扫描；选GROUP32/LOOP4/stages1，stages3无稳定收益 |
| `verify_pre_cleanup.log` / `stack_receipt_pre_cleanup.json` | 清理不使用的原型分支前版本也通过CPU全栈；最终SHA以无后缀receipt为准 |
| `finding_id.txt` / `decision_id.txt` | 本轮取号收据F65/D34；账本正文在notes |
| `check_records.log` | 收尾记录一致性检查 |

测试ID P113-01…04已pass，P113-05（实际服务/L2）由Claude审阅安排。沿用112所有测试，分别对未修改的110和112，加新tile边界与六个8192×32k/95k/190k的causal/ragged矩阵。最大逐行相对L∞/L2为3.956824e-5，topk最低99.951171875%；输出fp32，clean=False仍全宽，clean=True仍完整-inf写回。

数值输入是随机激活、真实模型形状，不是实采q/K/weights。等效TFLOPS按任务指定全宽 `2*nq*nk*32*128/秒/1e12`，剪枝也贡献收益，不宣称实际MMA峰值利用率。计时含解码及分配，排除JIT与输入生成；七轮交替旧新顺序取CUDA event中位数。decode eager有Python发射/时钟波动；graph约0.1024/0.5925ms，源码与PTX都保持112，历史T43数据不改。

GPU代码在 `/sjtu/linhang/arena/code/T44`，原始记录在 `runs/T44`；runner只使用已有m0环境和CUDA13兼容库。没有安装软件，没有操作bohr/Trisol/pod、服务/队列、镜像或提交；没有修改RELEASE。初次汇总曾在scp未完成时读到空receipt而报JSON解析错误，待传输结束后同一完整证据汇总通过；不是kernel测试失败。
