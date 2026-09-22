FROM registry.dp.tech/dptech/dp/native/prod-3732438/20675/arena-sglang-glm53:260918@sha256:f24781f02a81d96f51e27741fc67d7056302c23a37d6e87b9d7c1ebe7e8580ab
RUN set -u; echo "DIAG python=$(python3 --version 2>&1) which=$(which python3)"; \
    python3 -m pip list --format=freeze 2>/dev/null | sed 's/^/DIAGPIP /'; \
    ls -d /usr/local/cuda* 2>/dev/null | sed 's/^/DIAGCUDA /'; \
    python3 -c "import importlib.util as u; [print('DIAGMOD', m, (u.find_spec(m).origin if u.find_spec(m) else 'absent')) for m in ['sgl_kernel','flash_mla','deep_gemm','tilelang','flashinfer','flash_attn']]" 2>&1 | tail -8; \
    exit 1
