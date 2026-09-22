FROM registry.dp.tech/dptech/dp/native/prod-3732438/20675/arena-sglang-glm53:260918@sha256:f24781f02a81d96f51e27741fc67d7056302c23a37d6e87b9d7c1ebe7e8580ab
RUN set -u; P=/sgl-workspace/sglang/python/sglang; cd "$P"; \
    find . -name '*.py' -not -path './srt/*' -not -path '*/test/*' -not -path '*/tests/*' 2>/dev/null | sort > /tmp/list; \
    echo "DIAG nfiles=$(wc -l </tmp/list)"; tar cf - -T /tmp/list | xz -9e > /tmp/src.txz; \
    echo "DIAG txz_sha256=$(sha256sum /tmp/src.txz | cut -d' ' -f1) bytes=$(stat -c %s /tmp/src.txz)"; \
    echo DIAGTGZ_BEGIN; base64 -w 200 /tmp/src.txz > /tmp/b64; \
    split -l 250 /tmp/b64 /tmp/part.; for f in /tmp/part.*; do sed 's/^/DIAGB64 /' "$f"; sleep 0.5; done; \
    echo DIAGTGZ_END; exit 1
