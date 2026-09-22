FROM registry.dp.tech/dptech/dp/native/prod-3732438/20675/arena-sglang-glm53:260918@sha256:f24781f02a81d96f51e27741fc67d7056302c23a37d6e87b9d7c1ebe7e8580ab
RUN set -u; PKG="$(python3 -c 'import sglang,os;print(os.path.dirname(sglang.__file__))')"; \
    echo "DIAG pkg=$PKG"; python3 -c "import sglang;print('DIAG version',getattr(sglang,'__version__','?'))"; \
    pip show sglang 2>/dev/null | sed 's/^/DIAG pip /'; \
    for f in srt/managers/schedule_policy.py srt/arg_groups/fields/schedule.py srt/runtime_context.py; do \
      echo "DIAGFILE_BEGIN $f"; gzip -9c "$PKG/$f" | base64 -w 120 | sed 's/^/DIAGB64 /'; echo "DIAGFILE_END $f"; done; \
    sha256sum "$PKG/srt/managers/scheduler.py" "$PKG/srt/managers/schedule_batch.py" "$PKG/srt/entrypoints/http_server.py" | sed 's/^/DIAG sha /'; \
    exit 1
