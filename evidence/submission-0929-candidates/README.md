# 2026-09-29 两份正式提交

用户明确授权两份上传，后确认接受高并发包short4096+cold12k的已知chain代价。原拟DEMAND候选未上传，保留配置仅供审计。

|包|attempt|worker job|相对47266的唯一变化|回执|
|---|---|---|---|---|
|FAST高并发探索|47606|25662|SCHED_COLD_CAP=12288、SCHED_SHORT_TOKENS=4096|submitted/queued|
|PDI4低TPOT探索|47607|25663|prefill-decode-interval=4|submitted/queued|

都沿用0928c镜像、bf6b66fa引擎，无新代码或模型改动。FAST本地N38 3006请求零错误，chain10/25、fast114/131、overall102/145、turn4/9，TPOT mean58.405/p9596.338ms；chain相对356参考ID新增1、修好0、缺1，不能宣称不退化。PDI4本地N34通过原268ID保护；N38 fast164/132失败，TPOT mean57.014/p9590.436ms，不保证保住最高并发或稳定降低TPOT。

两包各自schema、28 flags、48 env、trace检查0错误0警告；CLI dry-run成功，zip三份服务配置逐项一致，平台回执bundle SHA与本地SHA一致。FAST/PDI4各创建一次attempt，上传退出均0。配置、zip、dry-run、upload日志、receipt及独立status保存在本目录。未停止或修改本地八卡任务。
