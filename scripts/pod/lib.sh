# Sourced by pod jobs. Versioned code: never edit /sgl-workspace; copy it and apply repo patches.
BASE_PKG=/sgl-workspace/sglang/python/sglang
PORT=${PORT:-30000}
prepare_src() {   # prepare_src <name> [patch files under $AX/patches ...] -> $AX/src/<name>; rebuilt if any patch changed
  local name=$1; shift; local dst=$AX/src/$name
  local want; want=$( (echo "$*"; cd $AX/patches && sha256sum "$@") | sha256sum | cut -c1-16)
  if [ "$(cat $dst/PATCHES_SIG 2>/dev/null)" != "$want" ]; then
    rm -rf $dst; mkdir -p $dst && cp -a $BASE_PKG $dst/sglang
    for p in "$@"; do patch -p3 -d $dst/sglang --fuzz=0 --no-backup-if-mismatch -s < $AX/patches/$p || { echo "PATCH_FAIL $p"; rm -rf $dst; return 1; }; done
    echo "$*" > $dst/PATCHES; echo "$want" > $dst/PATCHES_SIG
  fi
  echo "src $name: $(cat $dst/PATCHES) sig=$want"
  export PYTHONPATH=$dst
}
stop_engine() { pkill -f "sglang.launch_server.*--port $PORT" 2>/dev/null; for i in $(seq 1 30); do pgrep -f "sglang.launch_server.*--port $PORT" >/dev/null || return 0; sleep 2; done; pkill -9 -f "sglang.launch_server.*--port $PORT"; }
start_engine() {  # start_engine <extra launch args...>; waits until ready (model load ~21 min) or failure
  stop_engine
  SGLANG_OPT_USE_TOPK_V2=0 setsid nohup python3 -m sglang.launch_server --model-path /mnt/models --host 0.0.0.0 --port $PORT \
    --tp-size 8 --served-model-name default --enable-metrics --incremental-streaming-output --page-size 64 \
    --mamba-radix-cache-strategy extra_buffer --reasoning-parser glm45 --tool-call-parser glm47 "$@" \
    > $RUN_DIR/server.log 2>&1 < /dev/null &
  local pid=$!; echo $pid > $RUN_DIR/server.pid
  for i in $(seq 1 540); do   # up to 45 min
    curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { echo "ENGINE_READY after $((i*5))s"; return 0; }
    kill -0 $pid 2>/dev/null || { echo "ENGINE_DIED"; grep -n "Error\|Exception\|Unsupported" $RUN_DIR/server.log | tail -5; return 1; }
    sleep 5
  done; echo "ENGINE_TIMEOUT"; return 1
}

ensure_engine() {  # ensure_engine <src name> <launch args...>: reuse the running engine if code+args unchanged
  local name=$1; shift
  local sig="$name $(cat $AX/src/$name/PATCHES_SIG 2>/dev/null) | $* | ${SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS:-}"
  if [ -f $AX/engine.sig ] && [ "$(cat $AX/engine.sig)" = "$sig" ] && curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; then
    echo "ENGINE_REUSED: $sig"; return 0
  fi
  rm -f $AX/engine.sig
  start_engine "$@" || return 1
  echo "$sig" > $AX/engine.sig; cp $RUN_DIR/server.log $AX/engine_current.log 2>/dev/null; echo "ENGINE_STARTED: $sig"
}
