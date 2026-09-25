#!/usr/bin/env bash
set -eu
AX=/tmp/ax
RUN_DIR=$AX/runs/071-official_b_host64_full_n30_shortwarm
export AX RUN_DIR
#!/usr/bin/env bash
# L070: paired 122 test against L069, keeping host64 and all other settings.
# Re-enable fixed paced prefill (tau .085); this also replaces fixed interval/cold cap.
G_COMMIT=759a6ebb8e31723519ad5daf438e26e24b32501a
# NEXTN is the CLI name; the base normalizes it to EAGLE before logging effective config.
G_EXPECT="120=on 122=on 123=off 140=off 180=on spec=EAGLE dcp=1 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1"
G_ARGS="--kv-cache-dtype bfloat16 --linear-attn-backend triton --linear-attn-verify-backend triton --speculative-algorithm NEXTN --speculative-draft-model-path /mnt/models --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --max-running-requests 32 --cuda-graph-max-bs 32 --prefill-decode-interval 2 --enable-hierarchical-cache --hicache-size 64 --hicache-write-policy write_through --mem-fraction-static 0.87"
G_ENV="SGLANG_AX_KDA_DUAL_SNAPSHOT=0 SGLANG_AX_SCHED_PROTECT=1 SGLANG_AX_SCHED_SHORT_TOKENS=8192 SGLANG_AX_ASYNC_TOKENIZE=0 SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_OPT_FUSED_KDA_VERIFY=0 SGLANG_AX_INDEXER_ROW_SHARD=1 SGLANG_AX_SCHED_COLD_CAP=4096 SGLANG_AX_PACE_TPOT=0.085 SGLANG_AX_KDA_FUSE_PROJ=0 SGLANG_AX_MOE_FUSE_SWIGLU=0 SGLANG_AX_SM80_INDEXER=1 SGLANG_AX_SM80_FP8_MOE_MARLIN=1"
unset G_MEASURE_SECONDS
G_WARMUP_PROFILE=rep16-v1
LADDER_UP="30"
G_DATA_ROOT="$AX/data/s1-dev-longchain"
G_DATA_SET=s1-dev-longchain
G_COHORT="$G_DATA_ROOT/cohort.json"
unset LADDER_DOWN SGLANG_AX_SRPT_AGING SGLANG_AX_SHORT_RESERVE
unset SGLANG_AX_NUMTRACE_DIR SGLANG_AX_NUMTRACE_DUMP_LAYER
SMOKE_GATE=0

# Dev self-test on one engine: run the levels in LADDER_UP (e.g. "22" or "22 26"); each level is scored by
# verify_kit/level_verdict.py (complete data + harness scorer + task.md rules); stop at the first failure. Wrapper sets:
#   G_COMMIT (engine commit id, exported by scripts/engine/export.sh), G_ARGS, G_ENV, G_EXPECT, optional LADDER.
#   G_EXPECT lists the effective mechanism states the job relies on, e.g. "120=on 122=off 180=on"; the engine's
#   "[ax] mechanisms:" line must match every entry ("off" also matches "off:<reason>") or the job measures nothing.
source $AX/bin/scripts/pod/lib.sh
prepare_src "$G_COMMIT" || exit 1
export SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS=154827,154829 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
[ -n "${G_ENV:-}" ] && export $G_ENV
# Reattach to this run's already-launched process. Never call the original
# start_engine/stop_engine or launch another model.
start_engine() {
  python3 -B - "$RUN_DIR" "$G_COMMIT" "$PORT" "$@" <<'CHECK'
import hashlib, json, os, re, sys, time, urllib.request
from pathlib import Path
run, commit, port = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
extra = sys.argv[4:]
ax = Path('/tmp/ax')
assert ax.resolve() == Path('/dev/shm/arena-runtime/ax')
assert (ax/'src'/commit/'COMMIT').read_text().strip() == commit
assert (run/'data-verification.json').is_file()
assert json.loads((run/'data-verification.json').read_text())['status'] == 'VERIFIED'
pid = int((run/'server.pid').read_text())
assert pid == 20041, 'unexpected original engine PID'
proc = Path('/proc')/str(pid)
def identity():
    stat = (proc/'stat').read_text().rsplit(')',1)[1].split()
    assert stat[0] != 'Z', 'engine is a zombie'
    return stat[19]
start = identity()
cmd = (proc/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
base = ['-m','sglang.launch_server','--model-path','/mnt/models','--host','0.0.0.0',
        '--port',port,'--tp-size','8','--served-model-name','default','--enable-metrics',
        '--incremental-streaming-output','--page-size','64','--mamba-radix-cache-strategy',
        'extra_buffer','--reasoning-parser','glm45','--tool-call-parser','glm47']
assert cmd[1:] == base + extra, 'active engine command differs from frozen job'
actual = dict(s.decode().split('=',1) for s in (proc/'environ').read_bytes().split(b'\0') if b'=' in s)
assert actual.get('PYTHONPATH') == os.environ['PYTHONPATH']
key = re.compile(r'^(SGLANG_AX_|SGLANG_ARENA_|NCCL_|SGLANG_MAMBA|SGLANG_OPT_)')
wanted = {k:v for k,v in os.environ.items() if key.match(k)}
got = {k:v for k,v in actual.items() if key.match(k)}
# launch_server receives this one command-local env assignment in lib.sh.
expected_child = {**wanted, 'SGLANG_OPT_USE_TOPK_V2':'0'}
assert got == expected_child, 'active engine mechanism environment differs'
receipt = dict(pid=pid,proc_start_ticks=start,commit=commit,argv=cmd,
               environment_sha256=hashlib.sha256(json.dumps(got,sort_keys=True).encode()).hexdigest(),
               verified_at=time.time(),previous_waiter_timed_out=True)
(run/'adopt-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('ADOPT_VERIFIED '+json.dumps(receipt),flush=True)
begin = time.monotonic()
for n in range(181):
    assert identity() == start, 'engine PID changed'
    try:
        with urllib.request.urlopen('http://127.0.0.1:'+port+'/v1/models',timeout=3) as response:
            data = json.load(response)
        assert any(m.get('id') == 'default' for m in data.get('data',[])), 'unexpected model endpoint'
        receipt.update(ready_at=time.time(),extra_wait_s=time.monotonic()-begin)
        (run/'adopt-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print('ENGINE_READY adopted existing PID '+str(pid),flush=True)
        break
    except (OSError, ValueError):
        if n == 180: raise TimeoutError('adoption wait ended; existing engine retained')
        if n % 30 == 0: print('ADOPT_WAIT seconds='+str(round(time.monotonic()-begin)),flush=True)
        time.sleep(10)
CHECK
}
ensure_engine "$G_COMMIT" --schedule-policy lpm --dsa-prefill-backend tilelang --dsa-decode-backend tilelang $G_ARGS || exit 1
printf 'PRELOAD_READY: adopted original process after verified cold-compile wait\n' >> "$RUN_DIR/preload.log"
python3 -B - "$RUN_DIR" <<'RELEASE'
import hashlib, json
from pathlib import Path
ax=Path('/tmp/ax'); run=Path(__import__('sys').argv[1]); job=run.name+'.sh'
assert (ax/'engine.sig').is_file()
worker=int((ax/'worker.pid').read_text())
proc=Path('/proc')/str(worker)
assert (proc/'stat').read_text().rsplit(')',1)[1].split()[0] != 'Z'
assert b'podq_worker.sh' in (proc/'cmdline').read_bytes()
pending=ax/'queue/pending'/job
if pending.exists():
    assert not list((ax/'queue/running').iterdir()), 'another job is running'
    assert hashlib.sha256(pending.read_bytes()).hexdigest() == 'd038f3f12189e945e9381e1a54e64c7fd120ad434ab7935233f1bae8efd30df8'
    (ax/'queue/PAUSE').unlink(missing_ok=True)
    print('REPLAY_RELEASED_071',flush=True)
else:
    assert (ax/'queue/running'/job).exists(), 'unexpected queue state'
    print('REPLAY_ALREADY_RELEASED_071',flush=True)
RELEASE
