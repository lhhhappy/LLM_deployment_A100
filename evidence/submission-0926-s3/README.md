# 0926 S3：attempt 46757（arm C）——S1 + DCP2

用户 2026-09-27 00:20 UTC 决定直接提交（"线上 GPQA 题目没这么多，应该更简单，直接提交，别跑参照了"）。
- 镜像 `registry.dp.tech/dptech/dp/native/prod-4727808/21221/lh-img:0926c`（id 166139，FROM 0925a，Dockerfile 49,989 B，本地自检 payload=diff、0925a 树+diff=e464d8ab 树）。
- 引擎 e464d8ab（tag `image-lh-img-0926c`）= f546934e（46676 引擎：修正 124/110/117/180）+ DCP 三补丁（115 NextN 池/阶段/搬运、180 主机池、180 guard），逐字节同 Codex 分支上 b261cbd9 的 DCP。
- 相对 46676：命令加 `--dcp-size 2`、`--max-mamba-cache-size 418`，running/graph 32→48（让梯子不在 N34 被并发上限卡住）；env 加 `SGLANG_AX_DCP_COMPACT_TOPK=1`、`SGLANG_AX_DSA_SPARSE_TRITON=0`。其余逐字节同 46676。
- 本地依据：130ed 对 130ee（同引擎 dcp 2/1，N30）chain 21→15、turn 31→2、overall 455→72、fast 514→72；130ee5 对 130eez（N34）chain 33→18、turn 33→2、overall 646→86、fast 696→91；TPOT 均值 +6–11 ms。能力：公开 AIME 2026 28/30、GPQA-Diamond 178/197（130eezzy），冒烟 12/12 五次。
- 未核实：平台部署（46676 同血统镜像部署失败原因未知，46677 仍排队）；S1 设置下的 DCP 对照对（130ez1/2/3）尚未跑完；GPQA 参照被用户叫停。
- 文件：submission.json、candidate.json、bundle-audit.json、build-receipt.json、Dockerfile。
