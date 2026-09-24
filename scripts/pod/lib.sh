# Sourced by pod jobs. Versioned code: never edit /sgl-workspace; copy it and apply the engine commit's diff.
BASE_PKG=/sgl-workspace/sglang/python/sglang
PORT=${PORT:-30000}
prepare_src() {   # prepare_src <commit> -> $AX/src/<commit>: pod base + $AX/engine/<commit>.diff (scripts/engine/export.sh)
  local commit=$1; local dst=$AX/src/$commit
  [ -n "$commit" ] && [ -f "$AX/engine/$commit.diff" ] || { echo "ENGINE_DIFF_MISSING $commit"; return 1; }
  if [ "$(cat $dst/COMMIT 2>/dev/null)" != "$commit" ]; then
    rm -rf $dst; mkdir -p $dst && cp -a $BASE_PKG $dst/sglang
    patch -p3 -d $dst/sglang --fuzz=0 --no-backup-if-mismatch -s < $AX/engine/$commit.diff || { echo "ENGINE_APPLY_FAIL $commit"; rm -rf $dst; return 1; }
    echo "$commit" > $dst/COMMIT
  fi
  echo "ENGINE_COMMIT $commit"
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

ensure_engine() {  # ensure_engine <commit> <launch args...>: reuse the running engine if code+args unchanged
  local name=$1; shift
  # env is part of the reuse key (engines differing only by SGLANG_AX_*/NCCL_* must not be reused)
  local envkey; envkey=$(env | grep -E '^(SGLANG_AX_|SGLANG_ARENA_|NCCL_|SGLANG_MAMBA|SGLANG_OPT_)' | sort | tr '\n' ' ')
  local sig="$name | $* | $envkey"
  if [ -f $AX/engine.sig ] && [ "$(cat $AX/engine.sig)" = "$sig" ] && curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; then
    local log_path; log_path=$(cat "$AX/engine_log_path" 2>/dev/null)
    [ -n "$log_path" ] && [ -f "$log_path" ] || { echo "ENGINE_REUSE_LOG_MISSING: no live log reference"; return 1; }
    if [ ! -e "$RUN_DIR/server.log" ]; then
      ln -s "$log_path" "$RUN_DIR/server.log" || return 1
    elif [ ! "$RUN_DIR/server.log" -ef "$log_path" ]; then
      echo "ENGINE_REUSE_LOG_CONFLICT"; return 1
    fi
    echo "ENGINE_REUSED: $sig"; return 0
  fi
  rm -f $AX/engine.sig
  start_engine "$@" || return 1
  readlink -f "$RUN_DIR/server.log" > "$AX/engine_log_path" || return 1
  echo "$sig" > $AX/engine.sig; cp $RUN_DIR/server.log $AX/engine_current.log 2>/dev/null; echo "ENGINE_STARTED: $sig"
}
