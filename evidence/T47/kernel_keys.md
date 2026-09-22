# T47 v2 编译键审计

显式constexpr只保留固定模型/tile常量。Triton3.7默认把值1特化为constexpr，其他整数按16整除类别特化；这两个有限类别行为未关闭。精确NQ/NK/P/S/R等值>1不再进入常量表。

| 版本 | kernel | 显式constexpr | runtime scalar |
|---|---|---|---|
| 112 | _paged | H, D, PAGE, HH, DD | N, P, S, QB, QN, QH, QD, WB, WH, CB, CN, TB, TP, KB |
| 112 | _ragged | H, D, CLEAN, BQ, BK, HH, DD | NQ, NK, QQ, QH, QD, KK, KD, SS, WQ, WH, KSS, KES |
| 113 | _paged | H, D, PAGE, HH, DD | N, P, S, QB, QN, QH, QD, WB, WH, CB, CN, TB, TP, KB |
| 113 | _ragged | H, D, CLEAN, BQ, BK, HH, DD | NQ, NK, QQ, QH, QD, KK, KD, SS, WQ, WH, KSS, KES |
| 113 | _unpack_prefill | H, D, BLOCK | R, XR, XH, XD |
| 113 | _prefill | H, CLEAN, BQ, BK, HH, GROUP, LOOP | NQ, NK, QQ, QH, QD, KK, KD, SS, WQ, WH, KSS, KES |

实际每个JIT miss的signature/constants/divisibility/options记录见`cache.log`的`jit_miss`行；生产代码没有autotune。随机阶段必须无miss、无实际编译、无磁盘命中。这里只覆盖H32/D128/PAGE64与测试中的dtype/布局/共享context枚举，H8/16/64等原数值用例不等于已包含在这次预热里。

113基于shape的fallback门限只选择两个固定kernel族（nq<32或nk<1024回退112）；大形状的GROUP32/LOOP4固定。改变grid大小不会单独触发编译。
