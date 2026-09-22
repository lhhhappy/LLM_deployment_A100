FROM registry.dp.tech/dptech/dp/native/prod-3732438/20675/arena-sglang-glm53:260918@sha256:f24781f02a81d96f51e27741fc67d7056302c23a37d6e87b9d7c1ebe7e8580ab
RUN set -u; cd /sgl-workspace/sglang/python; \
    printf '%s\n' sglang/srt/models/deepseek_nextn.py sglang/srt/speculative/eagle_worker_v2.py sglang/_version.py > /tmp/list; \
    echo "DIAG nfiles=$(wc -l </tmp/list)"; tar cf - -T /tmp/list | xz -9e > /tmp/src.txz; \
    echo "DIAG txz_sha256=$(sha256sum /tmp/src.txz | cut -d' ' -f1) bytes=$(stat -c %s /tmp/src.txz)"; \
    echo DIAGTGZ_BEGIN; base64 -w 200 /tmp/src.txz | sed 's/^/DIAGB64 /'; echo DIAGTGZ_END; exit 1
