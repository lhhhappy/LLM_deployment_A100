FROM registry.dp.tech/dptech/dp/native/prod-3732438/20675/arena-sglang-glm53:260918@sha256:f24781f02a81d96f51e27741fc67d7056302c23a37d6e87b9d7c1ebe7e8580ab
RUN set -u; cd /sgl-workspace/sglang/python; \
    find . -type f -not -path '*/__pycache__/*' -not -name '*.pyc' | sort | xargs -d '\n' sha256sum > /tmp/manifest; \
    echo "DIAG nfiles=$(wc -l </tmp/manifest) manifest_sha256=$(sha256sum /tmp/manifest | cut -d' ' -f1)"; \
    echo DIAGTGZ_BEGIN; xz -9e < /tmp/manifest | base64 -w 200 > /tmp/b64; \
    echo "DIAG txz_sha256=$(base64 -d /tmp/b64 | sha256sum | cut -d' ' -f1) bytes=$(base64 -d /tmp/b64 | wc -c)"; \
    split -l 250 /tmp/b64 /tmp/part.; for f in /tmp/part.*; do sed 's/^/DIAGB64 /' "$f"; sleep 1; done; \
    echo DIAGTGZ_END; exit 1
