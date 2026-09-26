# 正式提交核对清单（2026-09-26，Sonnet 子代理只读整理，fable 采纳）

候选 S1 = 46364 配置 + 引擎 4f9d1f0b + 10 个 env（117 Humming；124；激进 125 七项；`SGLANG_AX_DEADLINE_WARM_S=15`）。草案在 `evidence/submission-0926-s1/candidate.json`，差异表 `config-audit.json`。command 与其余 16 个 env 与 46364 逐字节相同。

## 关键事实（实测）
- 镜像不需要中间层：`git diff 759a6eb 4f9d1f0b -- engine/sglang` 11 个文件 +1493/−31，原始 92,678 B，gzip+base64 36,140 B，按 `scripts/build_image.sh` 模板重建的 Dockerfile 36,969 B < 65,536 B。直接 `FROM lh-img:0925a`（源码 759a6eb = 标签 image-lh-img-0925a）。
- `git merge-base --is-ancestor 759a6eb 4f9d1f0b` 成立（脚本第 47 行自检）。

## 步骤
1. `scripts/build_image.sh 4f9d1f0b --from-image registry.dp.tech/dptech/dp/native/prod-4727808/4650601/lh-img:0925a --from-ref 759a6eb --name lh-img:0926a --project-id <ID>`：脚本自带两项本地核对（嵌入 payload 逐字节等于 diff；打到 759a6eb 树上逐文件等于 4f9d1f0b 树）。产物 `build/scratch/image/Dockerfile`。
2. `bohr image build --dockerfile <DF> --name lh-img:0926a --project-id <ID> --wait --yes -o json` → `status: success`、`imageUrl`；收据格式参照 `evidence/submission-0925a/build-receipt.json`。
3. `git tag image-lh-img-0926a 4f9d1f0b`。
4. 镜像内核对：起 hang 容器，`cat /opt/ax/engine_commit` = `4f9d1f0b…`；全树摘要对 `scripts/engine/tree.py 4f9d1f0b`（参照 `evidence/submission-0925a/image-source-verification.json` 口径）。当前 `build_image.sh` 的 RUN 步骤没有 `ENGINE_SOURCE_VERIFIED` 断言，要手动做。
5. `candidate.json` 的 `image` 换成真实 `imageUrl`（钉 digest）；`python3 scripts/check_submission.py --final --trace <stub> <path>`。
6. 提交前自查（task.md 426–436）：`/v1/models` 200；真实 `/chat/completions`；真实 `/generate` SSE 且 `meta_info` 三个计数齐全；`/flush_cache` 两次验证命中上涨、flush 后归零。
7. `playground submit … --dry-run --bundle-out <zip>`：三份 submission.json 副本字节一致。
8. `bash scripts/submit_official.sh <ARM>`：attempt 建立、上传、`notes/submission_attempts.log` 追加；首查 `queued`。
9. `notes/submissions.md` 追加一行；`scripts/official_status.sh <attempt>` 查终态。

## 提交前必须闭合的
- 12 题能力冒烟：117+124+125 组合此前从未做过（112 关了 SMOKE_GATE）；已改 130a/130b 打开（`SMOKE_GATE=1`）。117 单独 084 为 12/12；124/125 不碰数值路径（推断）。AIME/GPQA 只能由平台跑，能力门不过则压测不计分——这是最大风险。
- 130b（A′ 60 分钟对 112）确认 turn 回落（112 的 turn 12→24 由 124 的 5 s warm 预算造成）。
- 机制行核对：本地任务 G_EXPECT 应含 `128=off`；镜像内 `[ax] mechanisms:` 应为 `117=on:43_layers … 124=on 125=on 126=off 128=off`，`101=off:role_ids_unset` 是 MTP 下的已知耦合。
- 125 护栏 MAX_SLOW=80 只占 5150 条的 1.6%，比官方 5% 保守；task.md 未说 tpot_p95 是否享受统计余量（未核实）。
- Humming 冷缓存首启：112 实测 `ENGINE_READY after 195s`，但复用了同 Pod 的调优表缓存；全新镜像冷启动按文档最坏 43 层 × 10 s 也远小于 3600 s（推算）。
- 写盘要求（notes/submissions.md 第 35 行）自 46364 起未落实，S1 与 46364 相同，需另定。
