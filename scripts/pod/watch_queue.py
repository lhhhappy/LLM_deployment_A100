#!/usr/bin/env python3
"""Read-only pod queue watcher. See --help for one-shot and notification checks.

The relay's SUBMITTED response means terminal submission, not that the agent read it.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = ROOT / "build/scratch/coordination/queue-watch"
GPU_STATE_DIR = ROOT / "build/scratch/coordination/queue-watch-gpu"
GPU_ROOT = Path("/sjtu/linhang/arena")
STATES = ("running", "pending", "done", "failed", "cancelled")
JOB = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.sh$")
REMOTE_STATUS = ('cd /tmp/ax/queue && for d in running pending done failed cancelled; '
                 'do echo "$d: $(ls $d 2>/dev/null | tr "\\n" " ")"; done')


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_bounded(argv, timeout, cwd=None):
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True, cwd=cwd)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise RuntimeError(f"timeout after {timeout}s: {Path(argv[0]).name}")
    if proc.returncode:
        raise RuntimeError(f"{Path(argv[0]).name} exit {proc.returncode}: {err.strip()[-300:]}")
    return out


def validate_gpu_location(root, cwd, state_dir):
    for label, path in (("ROOT", root), ("cwd", cwd), ("state-dir", state_dir)):
        if not path.resolve().is_relative_to(GPU_ROOT):
            raise ValueError(f"--gpu-record {label} must be under {GPU_ROOT}")


def gpu_status(timeout):
    # GPU-box local -> pod. This never calls pread/gssh back into the GPU box.
    command = 'source scripts/pod/common.sh >/dev/null 2>&1 && PEXEC_TIMEOUT=35 bexec "$1"'
    return run_bounded(["bash", "-c", command, "queue-watch", REMOTE_STATUS], timeout, cwd=ROOT)


def event_ids(path):
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text().splitlines():
        if line.strip():
            item = json.loads(line)
            ids.add(item["id"])
    return ids


def append_event(path, item, seen_ids):
    # The state outbox is written before delivery. A crash after append but before
    # clearing the outbox retries this id; the on-disk id prevents a duplicate row.
    if item["id"] in seen_ids:
        return
    with open(path, "a") as f:
        f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    seen_ids.add(item["id"])


def parse_status(output):
    """Fail closed on partial/ambiguous remote output."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    # pread appends this transport receipt after the five queue sections.
    if lines and lines[-1] == "exit_code: 0":
        lines.pop()
    found = {}
    for line in lines:
        m = re.fullmatch(r"(running|pending|done|failed|cancelled):\s*(.*)", line)
        if not m:
            raise ValueError(f"unexpected status line: {line[:120]}")
        kind, names = m.groups()
        if kind in found:
            raise ValueError(f"duplicate status section: {kind}")
        found[kind] = names.split()
    if set(found) != set(STATES):
        raise ValueError(f"incomplete status sections: {sorted(found)}")
    jobs = {}
    for kind, names in found.items():
        for name in names:
            if not JOB.fullmatch(name) or name in jobs:
                raise ValueError(f"invalid or duplicate job name: {name[:120]}")
            jobs[name] = kind
    return jobs


def empty_state():
    return dict(version=1, initialized=False, health="unknown", jobs={}, outbox=[], next_id=1)


def load_state(path):
    if not path.exists():
        return empty_state()
    state = json.loads(path.read_text())
    if state.get("version") != 1 or not isinstance(state.get("jobs"), dict) or not isinstance(state.get("outbox"), list):
        raise ValueError("unsupported or corrupt queue watcher state")
    return state


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def enqueue(state, message):
    state["outbox"].append(dict(id=state["next_id"], text=message, at=utc_now()))
    state["next_id"] += 1


def observe_success(state, jobs):
    if state["health"] == "down":
        enqueue(state, "[queue watcher] pod status 已恢复；继续监控队列。")
    if state["initialized"]:
        old = state["jobs"]
        for name, kind in sorted(jobs.items()):
            if old.get(name) != kind:
                before = old.get(name, "new")
                enqueue(state, f"[queue watcher] {name}: {before} → {kind}。请用 pread status 和 job.log 核查。")
    else:
        state["initialized"] = True  # baseline: historical failed/done are not replayed
    state["jobs"] = jobs
    state["health"] = "up"
    state["last_success"] = utc_now()


def observe_failure(state, detail):
    if state["health"] != "down":
        enqueue(state, f"[queue watcher] pod status 不可用：{detail[:240]}；恢复后会补报状态变化。")
    state["health"] = "down"
    state["last_failure"] = utc_now()


def deliver(state, state_path, sender):
    while state["outbox"]:
        item = state["outbox"][0]
        sender(item["text"])
        state["outbox"].pop(0)  # only after SUBMITTED; not a read receipt
        atomic_json(state_path, state)


def poll_once(state, state_path, fetch, sender):
    try:
        jobs = parse_status(fetch())
    except Exception as exc:
        observe_failure(state, str(exc))
        fetched = False
    else:
        observe_success(state, jobs)
        fetched = True
    atomic_json(state_path, state)  # persist transitions before attempting delivery
    try:
        deliver(state, state_path, sender)
        delivered = True
    except Exception as exc:
        print(f"notification pending (will retry): {exc}", file=sys.stderr, flush=True)
        delivered = False
    return fetched and delivered


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-dir", type=Path, help="persistent state directory (mode-specific default)")
    ap.add_argument("--gpu-record", action="store_true", help="on GPU box: record pod changes locally; never send relay messages")
    ap.add_argument("--interval", type=float, default=60, help="poll period in seconds (default 60)")
    ap.add_argument("--status-timeout", type=float, default=45, help="pread status timeout in seconds")
    ap.add_argument("--notify-timeout", type=float, default=20, help="relay submission timeout in seconds")
    ap.add_argument("--once", action="store_true", help="one status poll; persist snapshot; exit")
    ap.add_argument("--notify-test", action="store_true", help="send one explicit test notification; no status poll")
    args = ap.parse_args()
    if min(args.interval, args.status_timeout, args.notify_timeout) <= 0:
        ap.error("interval and timeouts must be positive")
    if args.gpu_record and args.notify_test:
        ap.error("--notify-test is for the local relay mode; use --once with --gpu-record")
    args.state_dir = args.state_dir or (GPU_STATE_DIR if args.gpu_record else DEFAULT_STATE_DIR)
    if args.gpu_record:
        validate_gpu_location(ROOT, Path.cwd(), args.state_dir)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock = open(args.state_dir / "watch.lock", "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("queue watcher already running", file=sys.stderr)
        return 2
    started_at = utc_now()
    mode = "gpu-record" if args.gpu_record else "notify-test" if args.notify_test else "notify"
    atomic_json(args.state_dir / "process.json", dict(pid=os.getpid(), started_at=started_at,
                                                        heartbeat=started_at, mode=mode))
    state_path = args.state_dir / "state.json"

    def send(message):
        out = run_bounded([sys.executable, str(ROOT / "scripts/agent_message.py"), "--to", "lead",
                           "--from-agent", "lead", "--text", message], args.notify_timeout)
        if "SUBMITTED destination=lead" not in out:
            raise RuntimeError("relay did not confirm terminal submission")

    if args.notify_test:
        send("[queue watcher] 通知通路测试：终端已提交；请主会话确认实际收到。")
        print("notification SUBMITTED; awaiting human/agent receipt confirmation")
        return 0

    state = load_state(state_path)
    if args.gpu_record:
        events_path = args.state_dir / "events.jsonl"
        seen_ids = event_ids(events_path)
        def send(message):
            item = state["outbox"][0]
            assert item["text"] == message
            append_event(events_path, item, seen_ids)
        fetch = lambda: gpu_status(args.status_timeout)
    else:
        fetch = lambda: run_bounded([str(ROOT / "scripts/pod/pread"), "status"], args.status_timeout)
    while True:
        started = time.monotonic()
        ok = poll_once(state, state_path, fetch, send)
        atomic_json(args.state_dir / "process.json", dict(pid=os.getpid(), started_at=started_at,
                                                            heartbeat=utc_now(), mode=mode, health=state["health"],
                                                            pending_notifications=len(state["outbox"])))
        print(f"{utc_now()} health={state['health']} jobs={len(state['jobs'])} pending={len(state['outbox'])}", flush=True)
        if args.once:
            return 0 if ok else 1
        time.sleep(max(0, args.interval - (time.monotonic() - started)))


if __name__ == "__main__":
    sys.exit(main())
