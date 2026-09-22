# patches/

- 现行（打进提交镜像）：`000-interface-compliance`、`101-d1v12-on-base`；清单见 `RELEASE`。
  写法：照 `build/base_exact/`（= L3 实际代码，F54）生成 `a/python/sglang/...` 形式、`patch -p3 --fuzz=0` 可打。
  新补丁编号 1xx，叠在 000 之上。
- `drafts/`：未定稿（102a 单点版，存在 strict-append 退步与 bf16 精度问题，见 plans/active/102-role-track.md）。
- `v0520/`：v0.5.20 线（001/002/003/004），打不上底包，只作 L1 替身参考。

- `130-async-tokenize`（T42，CPU 已验证、待交叉审阅/L2）：叠在 000→101→110→111 上，完整分词单线程移出 HTTP loop、S1 routing key 接入；开关 `SGLANG_AX_ASYNC_TOKENIZE=0` 回退分词。未加入 RELEASE，说明与证据见同名 `.md` / `evidence/T42/`。

- `120-sched-protect-chain`（T41，CPU 已验证、待交叉审阅/GPU）：叠在 000→101→110→111 上，默认开启 decode 交替、对齐的长请求分块上限、短命中共享批预算；`SGLANG_AX_SCHED_PROTECT=0` 回退。未加入 RELEASE，复现 `scripts/make_120.py`；说明见同名 `.md` / `evidence/T41/`。

- `113-sm80-prefill-indexer`（T44，A100算子/全栈验证通过，待Claude交叉审阅/L2）：**叠加112**，预填充fp8预解码到bf16再用分组MMA；六档8192×32k/95k/190k causal/ragged达131–186等效TFLOPS，decode源码/PTX保持112。未加入RELEASE；生成器 `scripts/make_113.py`，说明与证据见同名 `.md` / `evidence/T44/`。

- `150-startup-warmup`（T46，启动期代表形状预热）：完整000→101→105→110→111→112→113→140→120→130后叠加，`--warmups ax_shapes`开启；真实请求/真flush与池断言，异常失败启动，MTP跳过。CPU与key审计见同名说明，GPU算子证据在`evidence/T46/`；有限采样不保证新长度零JIT。未加入RELEASE。
