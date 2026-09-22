#!/usr/bin/env python3
"""T22: approval-gated official submission queue; stdlib only.

Production: --once or a 600-second loop. --dry-run runs an isolated, in-memory
mock scenario (no credentials, network, APPROVED files, or production writes).
See notes/experiments.md, Tools / T22 for approval format and crash recovery.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
CHALLENGE = "llm-challenge-arena-v1"
API = "https://play.bohrium.com/api"
SHANGHAI = ZoneInfo("Asia/Shanghai")
ITEM_RE = re.compile(r"[0-9]{2,}-[a-z0-9][a-z0-9-]{0,79}\Z")
FINAL = {"succeeded", "failed", "cancelled", "failed-precheck"}
UNCERTAIN = {"submitting", "submission-unknown"}
STRESS_FIELDS = ("n_at_slo", "tpot_mean", "tpot_p95", "chain_start_p95",
                 "turn_start_p95", "overall_intra_p95", "fast_intra_p95",
                 "evaluation_status", "reason")


class SafeError(Exception):
    """Only static, non-secret diagnostic codes may be used as messages."""


class Paused(SafeError):
    pass


def now_utc():
    return dt.datetime.now(dt.timezone.utc)


def timestamp(value):
    return value.astimezone(dt.timezone.utc).isoformat(timespec="seconds")


def parse_time(value):
    try:
        result = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError()
        return result
    except (ValueError, TypeError):
        raise SafeError("invalid-timestamp") from None


def sha(data):
    return hashlib.sha256(data).hexdigest()


def clean(value):
    """Whitelist persisted data elsewhere; additionally redact secret-like text."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if not isinstance(value, str):
        return value
    token = os.environ.get("PLAYGROUND_TOKEN")
    if token:
        value = value.replace(token, "[REDACTED]")
    value = re.sub(r"(?i)Bearer\s+\S+|(?:asp_|sk-)[A-Za-z0-9._-]+", "[REDACTED]", value)
    return " ".join(value.split())[:2000]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".ledger-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(clean(value), f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        dfd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextlib.contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SafeError("daemon-already-running") from None
        yield


def append_once(path, key, line):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and any(key in line for line in path.read_text().splitlines()):
        return
    # Key is a complete marker line for summaries. Events use their exact line.
    with path.open("a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def parse_approval(text, slug, submission, trace):
    fields = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if not sep or key in fields:
            raise SafeError("approval-format")
        fields[key] = value.strip()
    required = {"approved_by", "approved_at", "what", "submission_sha256", "trace_sha256"}
    if set(fields) != required or not all(fields.values()):
        raise SafeError("approval-format")
    parse_time(fields["approved_at"])
    if fields["what"] != slug:
        raise SafeError("approval-wrong-item")
    if fields["submission_sha256"] != sha(submission) or fields["trace_sha256"] != sha(trace):
        raise SafeError("approval-content-changed")
    return fields


def read_approval(item, submission, trace):
    approval = item / "APPROVED"
    if not approval.is_file() or approval.is_symlink():
        raise SafeError("approval-required")
    return parse_approval(approval.read_text(), item.name, submission, trace)


def item_snapshot(item):
    if item.is_symlink() or (item / "submission.json").is_symlink():
        raise SafeError("queue-symlink")
    try:
        submission = (item / "submission.json").read_bytes()
        trace = (item / "stub-trace.jsonl").read_bytes()  # a symlink is a supported pointer
        approval = read_approval(item, submission, trace)
    except (OSError, UnicodeError):
        raise SafeError("queue-files-missing") from None
    return submission, trace, approval


def attempt_id(payload):
    if isinstance(payload, dict):
        result = payload.get("attempt_id", payload.get("id"))
        if result is None and isinstance(payload.get("attempt"), dict):
            result = payload["attempt"].get("id")
        if not isinstance(result, bool) and isinstance(result, (str, int)) and re.fullmatch(r"[A-Za-z0-9_-]+", str(result)):
            return str(result)
    raise SafeError("attempt-id-missing")


def api_state(attempt):
    state = str(attempt.get("status", "")).lower()
    scoring = attempt.get("scoringState") or {}
    scoring_state = str(scoring.get("state", "")).lower()
    # A failed worker can still be evaluating/retrying; do not free the slot yet.
    if scoring_state in {"evaluating", "queued", "running", "scoring"}:
        return "running"
    if state in {"cancelled", "canceled"} or scoring_state in {"cancelled", "canceled"}:
        return "cancelled"
    if state in {"failed", "error", "rejected"} or scoring_state in {"failed", "error"}:
        return "failed"
    if state in {"scored", "completed", "succeeded"} or scoring.get("scoreIsFinal") is True:
        return "succeeded"
    return "running"  # unknown statuses block, never imply a free slot


def scorecard(attempt):
    card = attempt.get("scorecard") or {}
    raw = card.get("scorewheel_raw_result") or {}
    stress = raw.get("stress") or card.get("scorewheel_stress") or {}
    datasets = card.get("scorewheel_datasets") or {}
    def points(name):
        result = raw.get(name) or {}
        value = result.get("points")
        if value is None and isinstance(datasets.get(name), (float, int)):
            value = datasets[name] * 100
        return value if isinstance(value, (int, float)) and math.isfinite(value) else None
    return clean({"aime26": {"points": points("aime26")},
                  "gpqa": {"points": points("gpqa-diamond")},
                  "gate_passed": raw.get("gate_passed", card.get("scorewheel_gate_passed")),
                  "gate_reason": raw.get("gate_reason"),
                  "stress": {key: stress.get(key) for key in STRESS_FIELDS}})


def rows_from_payload(payload):
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = None
        for key in ("attempts", "items", "data"):
            if key in payload:
                value = payload[key]
                rows = value if isinstance(value, list) else rows_from_payload(value)
                break
    else:
        rows = None
    if not isinstance(rows, list) or not all(isinstance(a, dict) and "id" in a for a in rows):
        raise SafeError("api-list-schema")
    return rows


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise SafeError("api-redirect-refused")


def credential_env():
    """Read only the named token; never source executable credentials or log them."""
    try:
        text = (Path.home() / ".config/playground/credentials.env").read_text()
        match = re.search(r"^PLAYGROUND_TOKEN=(.*)$", text, re.M)
        if not match:
            raise ValueError()
        value = match.group(1).strip()
        # CLI writes JSON-quoted values; support common single-quoted env files too.
        token = json.loads(value) if value.startswith('"') else shlex.split(value)[0]
        if not isinstance(token, str) or not token or "\n" in token:
            raise ValueError()
        os.environ["PLAYGROUND_TOKEN"] = token
        os.environ["NODE_USE_ENV_PROXY"] = "1"
    except (OSError, ValueError, IndexError):
        raise SafeError("credentials-unavailable") from None


class Backend:
    def __init__(self, root, timeout=180):
        self.root = root
        self.timeout = timeout
        self.pause_check = lambda: None

    def command(self, argv, cwd=None):
        self.pause_check()
        try:
            proc = subprocess.Popen(argv, cwd=cwd or self.root, env=os.environ.copy(),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    start_new_session=True)
            try:
                out, _ = proc.communicate(timeout=self.timeout)
            except BaseException:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.communicate()
                raise
        except (OSError, subprocess.TimeoutExpired):
            raise SafeError("command-failed-or-timeout") from None
        if proc.returncode:
            # stdout/stderr may contain secrets. Deliberately never persist them.
            raise SafeError("command-nonzero")
        return out

    def cli(self, args, cwd=None):
        return self.command(["bash", "--noprofile", "--norc", "-c",
                             'source "$1" >/dev/null 2>&1 && export NODE_USE_ENV_PROXY=1 && '
                             'shift && exec playground "$@"',
                             "submit-daemon", str(self.root / "env.sh"), *args], cwd)

    def get_json(self, route):
        self.pause_check()
        token = os.environ.get("PLAYGROUND_TOKEN")
        if not token:
            raise SafeError("credentials-unavailable")
        request = urllib.request.Request(API + route, headers={
            "Authorization": "Bearer " + token, "Accept": "application/json"})
        try:
            with urllib.request.build_opener(NoRedirect()).open(request, timeout=self.timeout) as response:
                return json.load(response)
        except Exception:
            raise SafeError("api-unavailable") from None

    def owner_ids(self):
        credential_env()
        payload = self.get_json("/auth/me")
        if not isinstance(payload, dict):
            raise SafeError("api-identity-schema")
        user = payload.get("user", payload)
        if not isinstance(user, dict) or not user.get("id"):
            raise SafeError("api-identity-schema")
        operator = user.get("operatorId") or user.get("operator_id")
        return {str(user["id"]), *([str(operator)] if operator else [])}

    def attempts(self):
        rows, seen = [], set()
        # Full pagination includes old in-flight attempts and today's manual submissions.
        # Duplicate pages or schema errors close the submission gate.
        for page in range(1, 10001):
            payload = self.get_json(f"/challenges/{CHALLENGE}/attempts?limit=100&page={page}")
            batch = rows_from_payload(payload)
            total = None
            if isinstance(payload, dict):
                meta = payload.get("pagination") or payload
                total = meta.get("total", meta.get("totalCount"))
                if total is not None and (not isinstance(total, int) or total < 0):
                    raise SafeError("api-pagination-schema")
            if not batch:
                if total is not None and len(rows) < total:
                    raise SafeError("api-pagination-incomplete")
                return rows
            ids = [attempt_id(a) for a in batch]
            if len(set(ids)) != len(ids) or seen.intersection(ids):
                raise SafeError("api-pagination-unstable")
            seen.update(ids)
            rows.extend(batch)
            if len(batch) < 100 and (total is None or len(rows) >= total):
                return rows
        raise SafeError("api-pagination-limit")

    def validate(self, outputs, trace):
        self.command([sys.executable, str(self.root / "scripts/check_submission.py"),
                      str(outputs / "submission.json"), "--final", "--trace", str(trace)])

    def submit(self, outputs, trace, dry_run=False):
        args = ["submit", "--challenge-id", CHALLENGE, "--outputs", str(outputs), "--trace", str(trace)]
        if dry_run:
            args.append("--dry-run")
        out = self.cli(args, outputs.parent)
        if dry_run:
            return None
        try:
            return attempt_id(json.loads(out))
        except (ValueError, TypeError):
            raise SafeError("submit-response-invalid") from None

    def status(self, aid):
        try:
            result = json.loads(self.cli(["status", "--attempt-id", aid]))
            if attempt_id(result) != aid:
                raise SafeError("status-id-mismatch")
            return result
        except (ValueError, TypeError):
            raise SafeError("status-response-invalid") from None


class Daemon:
    def __init__(self, root, backend, max_in_flight=1, daily_limit=2, clock=now_utc):
        if max_in_flight < 1 or not 1 <= daily_limit <= 2:
            raise SafeError("invalid-limits")
        self.root, self.backend, self.clock = root, backend, clock
        self.queue = root / "submission/queue"
        self.ledger_path = root / "data/submissions.json"
        self.summary = root / "notes/submissions.md"
        self.events = root / "logs/submit_daemon.events"
        self.max_in_flight, self.daily_limit = max_in_flight, daily_limit
        self.records = []
        self.backend.pause_check = self.checkpoint

    def stopped(self):
        return os.path.lexists(self.queue / "STOP")

    def checkpoint(self):
        if self.stopped():
            raise Paused("stop-file")

    def save(self):
        atomic_json(self.ledger_path, self.records)

    def event(self, text):
        line = clean(text)
        append_once(self.events, line, line)

    def transition(self, record, status, detail=None):
        entry = {"at": timestamp(self.clock()), "status": status}
        if detail:
            entry["detail"] = clean(detail)
        last = record["status_history"][-1] if record["status_history"] else {}
        if (last.get("status"), last.get("detail")) != (status, entry.get("detail")):
            record["status_history"].append(entry)
        record["status"] = status
        self.save()

    def load(self):
        try:
            self.records = json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else []
            if not isinstance(self.records, list):
                raise ValueError()
            names, ids = set(), set()
            for record in self.records:
                name = record["slug"]
                if not ITEM_RE.fullmatch(name) or name in names or not isinstance(record["status_history"], list):
                    raise ValueError()
                names.add(name)
                if record.get("status") not in FINAL | UNCERTAIN | {"awaiting-approval", "ready", "submitted", "running"}:
                    raise ValueError()
                if record["status"] in {"submitted", "running"} and not record.get("attempt_id"):
                    raise ValueError()
                if record.get("submitted_at") and not record.get("attempt_id") and record["status"] not in UNCERTAIN:
                    raise ValueError()
                if record.get("attempt_id"):
                    aid = attempt_id({"id": record["attempt_id"]})
                    if aid in ids:
                        raise ValueError()
                    ids.add(aid)
                if record.get("submitted_at"):
                    parse_time(record["submitted_at"])
                if record["status"] in UNCERTAIN or record.get("attempt_id"):
                    if not record.get("submitted_at"):
                        raise ValueError()
        except (OSError, ValueError, KeyError, TypeError):
            raise SafeError("ledger-invalid") from None

    def terminal_output(self, record):
        status, slug = record["status"], record["slug"]
        card = record.get("final_scorecard") or {}
        stress = card.get("stress") or {}
        def cell(v):
            return str(clean(v) if v is not None else "—").replace("|", "\\|")
        marker = f"<!-- submission:{slug} -->"
        if not self.summary.exists():
            self.summary.parent.mkdir(parents=True, exist_ok=True)
            self.summary.write_text("# Official submission results\n\n"
                                    "| Item | Attempt | Status | AIME26 | GPQA | Gate | N@SLO | TPOT mean | TPOT p95 | Completed UTC |\n"
                                    "|---|---|---|---|---|---|---|---|---|---|\n")
        aime = (card.get("aime26") or {}).get("points")
        gpqa = (card.get("gpqa") or {}).get("points")
        row = "| " + " | ".join(map(cell, [slug + " " + marker, record.get("attempt_id"), status, aime, gpqa,
                       card.get("gate_passed"), stress.get("n_at_slo"), stress.get("tpot_mean"),
                       stress.get("tpot_p95"), record.get("completed_at")])) + " |"
        append_once(self.summary, marker, row)
        self.event(f"RESULT {slug} n_at_slo={stress.get('n_at_slo')} tpot={stress.get('tpot_mean')} "
                   f"aime={aime} gpqa={gpqa} status={status} attempt={record.get('attempt_id')}")

    def discover(self):
        known = {r["slug"]: r for r in self.records}
        for item in sorted(self.queue.iterdir(), key=lambda p: (int(p.name.split('-')[0]) if ITEM_RE.fullmatch(p.name) else -1, p.name)):
            if not ITEM_RE.fullmatch(item.name) or not item.is_dir():
                continue
            if item.name not in known:
                record = {"slug": item.name, "approved_by": None, "submitted_at": None,
                          "attempt_id": None, "status": "awaiting-approval", "status_history": [],
                          "final_scorecard": None}
                self.records.append(record)
                self.transition(record, "awaiting-approval")

    def poll(self, api_rows):
        by_id = {attempt_id(a): a for a in api_rows}
        for record in self.records:
            if record["status"] in FINAL or not record.get("attempt_id"):
                continue
            self.checkpoint()
            aid = record["attempt_id"]
            try:
                cli_row = self.backend.status(aid)
            except Paused:
                raise
            except SafeError:
                self.event(f"POLL_ERROR {record['slug']} attempt={aid} source=cli")
                continue
            api_row = by_id.get(aid)
            if api_row is None:
                self.event(f"POLL_WAIT {record['slug']} attempt={aid} source=api-missing")
                continue
            self.checkpoint()
            state = api_state(api_row)
            detail = {"api_status": api_row.get("status"), "exec_status": api_row.get("execStatus"),
                      "scoring_state": (api_row.get("scoringState") or {}).get("state"),
                      "cli_status": cli_row.get("status")}
            if state in FINAL and api_state(cli_row) == state:
                # Status detail can contain a fuller scorecard than the public listing.
                result_row = cli_row if cli_row.get("scorecard") else api_row
                card = scorecard(result_row)
                if state == "succeeded" and not result_row.get("scorecard"):
                    self.event(f"POLL_WAIT {record['slug']} attempt={aid} source=scorecard-missing")
                    continue
                record["final_scorecard"] = card
                record["completed_at"] = timestamp(self.clock())
                self.transition(record, state, detail)
                self.terminal_output(record)
            else:
                self.transition(record, "running", detail)

    def guard(self, api_rows, owners):
        active_ids, local_today = set(), set()
        today = self.clock().astimezone(SHANGHAI).date()
        for r in self.records:
            if r["status"] in UNCERTAIN:
                return "unresolved-submission"
            if r.get("submitted_at") and parse_time(r["submitted_at"]).astimezone(SHANGHAI).date() == today:
                local_today.add(r.get("attempt_id") or ("reservation:" + r["slug"]))
            if r.get("attempt_id") and r["status"] not in FINAL:
                if r.get("owner_ids") and not set(r["owner_ids"]).intersection(owners):
                    return "account-changed"
                active_ids.add(r["attempt_id"])
        for a in api_rows:
            if not a.get("authorId"):
                return "api-owner-missing"
            if not {str(a.get("authorId")), str(a.get("operatorId"))}.intersection(owners):
                continue
            aid = attempt_id(a)
            if api_state(a) not in FINAL:
                active_ids.add(aid)
            try:
                created = parse_time(a.get("createdAt"))
            except SafeError:
                return "api-created-at-missing"
            if created.astimezone(SHANGHAI).date() == today:
                local_today.add(aid)
        if len(local_today) >= self.daily_limit:
            return "daily-limit"
        if len(active_ids) >= self.max_in_flight:
            return "in-flight-limit"
        return None

    def process_item(self, record, api_rows, owners):
        item = self.queue / record["slug"]
        try:
            self.checkpoint()
            submission, trace, approval = item_snapshot(item)
        except Paused:
            raise
        except SafeError as e:
            self.transition(record, "awaiting-approval", str(e))
            return
        record.update(approved_by=approval["approved_by"], approval=approval)
        reason = self.guard(api_rows, owners)
        if reason:
            self.transition(record, "ready", reason)
            self.event(f"WAIT {record['slug']} reason={reason}")
            return
        # Fresh private snapshot: outputs contains exactly submission.json, never
        # APPROVED/notes/trace. Both prechecks and submission use these same bytes.
        with tempfile.TemporaryDirectory(prefix="submit-", dir=self.queue) as stage:
            stage = Path(stage)
            outputs = stage / "outputs"
            outputs.mkdir()
            (outputs / "submission.json").write_bytes(submission)
            trace_path = stage / "stub-trace.jsonl"
            trace_path.write_bytes(trace)
            try:
                self.checkpoint()
                self.backend.validate(outputs, trace_path)
                self.checkpoint()
                self.backend.submit(outputs, trace_path, dry_run=True)
            except Paused:
                raise
            except SafeError as e:
                record["completed_at"] = timestamp(self.clock())
                self.transition(record, "failed-precheck", str(e))
                self.event(f"FAILED_PRECHECK {record['slug']} reason={e}")
                self.terminal_output(record)
                return
            self.checkpoint()
            # Refresh API quota immediately before the irreversible call.
            refreshed = self.backend.attempts()
            reason = self.guard(refreshed, owners)
            if reason:
                self.transition(record, "ready", reason)
                return
            try:
                if item_snapshot(item) != (submission, trace, approval):
                    raise SafeError("approval-content-changed")
            except SafeError as e:
                self.transition(record, "awaiting-approval", str(e))
                return
            self.checkpoint()
            record["submitted_at"] = timestamp(self.clock())
            record["owner_ids"] = sorted(owners)
            # Write-ahead reservation is durable BEFORE spawning playground. On
            # crash/timeout never retry automatically: CLI has no idempotency key.
            self.transition(record, "submitting")
            self.checkpoint()
            try:
                aid = self.backend.submit(outputs, trace_path)
                if aid in {r.get("attempt_id") for r in self.records}:
                    raise SafeError("duplicate-attempt-id")
                record["attempt_id"] = aid
                self.transition(record, "submitted")
                self.event(f"SUBMITTED {record['slug']} attempt={aid}")
            except Paused:
                raise
            except SafeError as e:
                self.transition(record, "submission-unknown", str(e))
                self.event(f"SUBMISSION_UNKNOWN {record['slug']} reason={e}")

    def tick(self):
        if self.stopped():
            return  # STOP pauses reads, polls, prechecks, submits, and ledger writes.
        with exclusive(self.queue / ".daemon.lock"):
            self.checkpoint()
            self.load()
            self.discover()
            for r in self.records:
                self.checkpoint()
                if r.get("attempt_id"):
                    self.event(f"SUBMITTED {r['slug']} attempt={r['attempt_id']}")
                if r["status"] in FINAL:
                    self.terminal_output(r)  # repair a crash between ledger and summary
                elif r["status"] == "submitting":
                    self.transition(r, "submission-unknown", "interrupted-submit")
                    self.event(f"SUBMISSION_UNKNOWN {r['slug']} reason=interrupted-submit")
            pending = [r for r in self.records if r["status"] not in FINAL and not r.get("submitted_at")]
            active = [r for r in self.records if r.get("attempt_id") and r["status"] not in FINAL]
            # An empty/unapproved queue needs no credentials or network.
            if not active and not any((self.queue / r["slug"] / "APPROVED").is_file() for r in pending):
                return
            self.checkpoint()
            owners = self.backend.owner_ids()
            self.checkpoint()
            api_rows = self.backend.attempts()
            self.poll(api_rows)
            for r in pending:
                self.process_item(r, api_rows, owners)


def create_queue_item(root, candidate, slug, trace, notes=None):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", slug):
        raise SafeError("invalid-slug")
    submission = candidate.read_bytes()
    trace_bytes = trace.read_bytes()
    try:
        obj = json.loads(submission)
        if not isinstance(obj, dict) or set(obj) != {"image", "command", "env", "model_name"}:
            raise ValueError()
    except ValueError:
        raise SafeError("candidate-needs-four-fields") from None
    queue = root / "submission/queue"
    with exclusive(queue / ".enqueue.lock"):
        numbers = [int(p.name.split("-")[0]) for p in queue.iterdir() if ITEM_RE.fullmatch(p.name)]
        ledger = root / "data/submissions.json"
        if ledger.exists():
            for r in json.loads(ledger.read_text()):
                numbers.append(int(r["slug"].split("-")[0]))
        item = queue / f"{max(numbers, default=0) + 1:02d}-{slug}"
        with tempfile.TemporaryDirectory(prefix=".enqueue-", dir=queue) as tmp:
            stage = Path(tmp) / "item"
            stage.mkdir()
            (stage / "submission.json").write_bytes(submission)
            (stage / "stub-trace.jsonl").write_bytes(trace_bytes)
            description = notes.read_text() if notes else "Pending user review. No approval has been granted.\n"
            (stage / "notes.md").write_text(description + f"\nQueue item: {item.name}\n"
                f"submission_sha256: {sha(submission)}\ntrace_sha256: {sha(trace_bytes)}\n")
            os.rename(stage, item)
    return item


class MockBackend:
    """No sockets or subprocesses; used only by the isolated demonstration."""
    def __init__(self, clock):
        self.rows, self.clock = [], clock

    def owner_ids(self):
        return {"mock-user"}

    def attempts(self):
        return self.rows.copy()

    def validate(self, outputs, trace):
        assert sorted(p.name for p in outputs.iterdir()) == ["submission.json"]

    def submit(self, outputs, trace, dry_run=False):
        if dry_run:
            return
        self.rows.append({"id": "mock-1", "authorId": "mock-user", "createdAt": timestamp(self.clock()),
                          "status": "submitted"})
        return "mock-1"

    def status(self, aid):
        return next(a for a in self.rows if str(a["id"]) == aid)


def demo():
    """Synthetic approval is injected in memory; never create an APPROVED file."""
    from unittest.mock import patch
    with tempfile.TemporaryDirectory(prefix="submission-mock-") as tmp:
        root = Path(tmp)
        item = root / "submission/queue/01-mock"
        item.mkdir(parents=True)
        (item / "submission.json").write_text(json.dumps({"image": "mock", "command": "mock", "env": {}, "model_name": "mock"}))
        (item / "stub-trace.jsonl").write_text('{"type":"session_start"}\n{"role":"user"}\n')
        backend = MockBackend(now_utc)
        daemon = Daemon(root, backend)
        is_file = Path.is_file
        def mock_approval(item, submission, trace):
            return {"approved_by": "MOCK ONLY", "approved_at": timestamp(now_utc()), "what": item.name,
                    "submission_sha256": sha(submission), "trace_sha256": sha(trace)}
        # Freeze approval timestamp so both snapshot checks are identical.
        approval = mock_approval(item, (item / "submission.json").read_bytes(), (item / "stub-trace.jsonl").read_bytes())
        with patch.object(sys.modules[__name__], "read_approval", return_value=approval), \
             patch.object(Path, "is_file", lambda p: p.name == "APPROVED" or is_file(p)):
            daemon.tick()
            backend.rows[0].update(status="scored", scorecard={"scorewheel_raw_result": {
                "aime26": {"points": 97.7}, "gpqa-diamond": {"points": 96.8}, "gate_passed": True,
                "stress": dict(zip(STRESS_FIELDS, [14, .021, .033, 12., 8., 4., 2., "PASS", None]))}})
            daemon.tick()
        print("MOCK ONLY: isolated simulation, no network/credentials/production writes")
        print(daemon.events.read_text(), end="")
        print(json.dumps(daemon.records[0]["final_scorecard"], indent=2))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=float, default=600)
    ap.add_argument("--dry-run", action="store_true", help="isolated mock simulation; never contact Playground")
    ap.add_argument("--max-in-flight", type=int, default=1)
    ap.add_argument("--daily-limit", type=int, choices=(1, 2), default=2)
    ap.add_argument("--command-timeout", type=float, default=180)
    ap.add_argument("--enqueue", type=Path, metavar="CANDIDATE")
    ap.add_argument("--slug")
    ap.add_argument("--trace", type=Path)
    ap.add_argument("--notes", type=Path)
    args = ap.parse_args(argv)
    if args.dry_run:
        return demo()
    root = args.root.resolve()
    try:
        if args.enqueue:
            if not args.slug:
                ap.error("--enqueue requires --slug")
            print(create_queue_item(root, args.enqueue, args.slug,
                                    args.trace or root / "submission/stub-trace.jsonl", args.notes))
            return 0
        if not all(math.isfinite(x) and x > 0 for x in (args.interval, args.command_timeout)):
            ap.error("interval and timeout must be positive")
        daemon = Daemon(root, Backend(root, args.command_timeout), args.max_in_flight, args.daily_limit)
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        while True:
            try:
                daemon.tick()
            except Paused:
                pass
            except SafeError as e:
                # No exception repr/tracebacks or CLI output: may contain credentials.
                print("submit-daemon: " + str(e), file=sys.stderr, flush=True)
                if not daemon.stopped():
                    daemon.event("DAEMON_ERROR reason=" + str(e))
                if args.once:
                    return 1
            if args.once:
                return 0
            deadline = time.monotonic() + args.interval
            while time.monotonic() < deadline:
                time.sleep(min(1, max(0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        return 0
    except Exception:
        print("submit-daemon: local-state-or-configuration-error", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
