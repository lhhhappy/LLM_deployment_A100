# scripts/pod — working in the L2 8×A100 pod (run from the GPU box, e.g. in tmux)

- `pexec '<cmd>'`  run a command in the pod;  `pstatus`  GPUs / engines / queue / running job tail
- `ppush <dest> <paths...>`  copy files in (bohr exec has no stdin: base64 chunks as arguments, sha256-checked)
- `podq init|submit <job.sh>|ls|log <job>|pause|resume|cancel <job>`  job queue; worker runs inside the pod
- Jobs (`jobs/*.sh`) source `lib.sh`: `prepare_src <name> <patches...>` copies the pristine base package to
  `/tmp/ax/src/<name>` and applies repo patches (the base in `/sgl-workspace` is never edited), then
  `start_engine <args>` launches the submission-style server on :30000 (model load ≈21 min).
- Service lifecycle: never stop/delete the service without the user's approval. `podq pause` only stops
  picking new jobs.
