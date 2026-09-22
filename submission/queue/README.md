# Official submission queue

Each item is `NN-slug/`, with `submission.json`, `stub-trace.jsonl` (a symlink to the shared trace is also supported), and `notes.md`. The queue helper copies these files and **never creates `APPROVED`**. Only the user, or the coordinator recording the user's explicit approval, may supply that file.

```bash
scripts/queue_submission.sh submission/candidate-01.json candidate-01
```

This only queues a snapshot. The current candidate has a placeholder image; `--final` rejects it until replaced with a real digest. Queue the final candidate as a new item after review. Queue numbers are allocated under a lock, also considering existing ledger records; they are never reused after an item has entered the ledger.

The approval format is five text lines (values below describe the format, not an approval):

```text
approved_by: <the user who explicitly approved this submission>
approved_at: <ISO 8601 time with timezone, e.g. 2026-09-22T16:30:00+08:00>
what: <exact directory name, e.g. 01-candidate-01>
submission_sha256: <SHA256 of the exact submission.json bytes>
trace_sha256: <SHA256 of the exact resolved trace bytes>
```

The helper includes both hashes in `notes.md`; independently verify them when approving. Approval covers that exact item and content. The daemon rereads approval after prechecks and copies submission/trace into a private snapshot. `--outputs` contains **only `submission.json`**. Replacing content or removing approval prevents submission. A symlink named `APPROVED` is rejected.

`STOP` in this directory pauses all daemon work, including result polling. It is checked between external operations, including before the actual submit. A request already accepted by Playground cannot be undone by this file. Default polling interval is 600 seconds; a paused daemon resumes at its next tick after `STOP` is removed.

Complete commands, limits, ledger semantics, and recovery: [experiments Tools / T22](../../notes/experiments.md#t22--unattended-submission-queue-and-daemon-codex-w9-2026-09-22-utc).
