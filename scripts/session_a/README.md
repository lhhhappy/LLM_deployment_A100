# Session A runner (T23 / W10; T31 / W14 daemon contract)

This is automation for **Claude to launch after `lh-arena-sess-a` is running**.
The implementation and validation were CPU only. No Trisol service was entered,
started, stopped or modified while developing it. CLI `exec --help` and
`delete --help` were the only Trisol CLI calls.

```bash
# Read the entire adaptive plan without network, subprocesses or file writes:
scripts/session_a/run_session_a.sh --dry-run --minutes 240

# Later, Claude supplies the approved wall-clock budget explicitly:
scripts/session_a/run_session_a.sh --minutes "$APPROVED_SESSION_MINUTES"

# Local verification; no services, SSH, downloads or GPU work:
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover \
  -s scripts/session_a -p test_session_a.py -v
```

## Daemon CLI (T31)

The runner now accepts the daemon's complete argv. `--service-id` / `--service`
accept either an ID or a name: installed `bohr trisol inference exec --help`
documents `<id|name>`; the runner passes the value directly to that resolver.
`--budget-minutes` is an alias of `--minutes`.

```bash
# Offline illustration, using the real candidate inputs (no service operations):
bash scripts/session_a/run_session_a.sh --dry-run --service-id SERVICE_ID \
  --profiles submission/candidate-b0-d0-baseline.json submission/candidate-b1-d1.json \
             submission/candidate-b2-spf.json submission/candidate-b3-spf-d1.json \
  --matrix spf_d1v12,d1v12 --ladder-mode fast --hint 14 --max-n 30 \
  --budget-minutes 240 --run-id example --out runs/example
```

- `--profiles PATH...`: freeze and transfer every candidate's original bytes.
  Match each matrix config by its candidate's schedule policy and D1 environment
  switch, independent of path names or list order. `d1v12`/`spf_d1v12` use the
  corresponding D1/SPF+D1 profile plus patch 004; HRRN may derive from the
  baseline profile. A single profile is a shared base for ablations. Multiple
  ambiguous/missing matches fail before remote work. `profile-plan.json` records
  the mapping, and pod engine receipts bind the selected file's SHA256.
- `--matrix baseline,spf,d1,spf_d1,hrrn,d1v12,spf_d1v12`: run exactly the supplied
  configurations. Baseline runs first if present; other configs preserve input
  order. Explicitly requested incompatible patches fail instead of silently
  dropping a config. Without `--matrix`, retain the legacy baseline/SPF/SPF+D1/D1
  sequence and optional `--hrrn` (and its unavailable-patch skips).
- `--ladder-mode official-climb|fast|levels`: official traversal begins at N10
  and moves ±4; fast uses the existing exponential bracket/bisection search
  from `--hint`; levels runs `--levels 6,10,18` in exactly that order, even after
  an SLO failure. A baseline always attempts N6/N10 calibration first, then
  follows the selected mode without repeating observed rungs. With baseline,
  other configs run P/P+4 only when an adjacent passing/failing bracket proves
  P. Without baseline, every listed config runs its own requested ladder.
- `--max-n` bounds all rungs (default 30); `--max-levels` caps unique ladder
  measurements per config, including baseline calibration. Reaching a cap
  without resolving a search is reported as incomplete. Explicit levels can
  complete without establishing a critical P. Stock remains only a paired CAP
  reference and never becomes an extra measured matrix configuration.
- `--out ROOT`: local artifacts go directly under ROOT. The daemon may have
  already created ROOT/profiles; other existing execution artifacts require
  recovery or a new run. GPU copies go to `/sjtu/linhang/arena/runs/RUN_ID/`.
  Without `--out`, retain the legacy `runs/session_a/RUN_ID` local/GPU layout.
- `--collect-only --run-id ORIGINAL ... --budget-minutes 0`: pull artifacts from
  the deterministic original pod run directory, verify hashes, rebuild the
  manifest, and mirror them. No transfer, inspection, jobs, engine start/stop,
  or service lifecycle actions. Recovery has an independent 120-second ceiling
  (`--collect-seconds`), additionally limited by the daemon's subprocess timeout.
  It verifies existing local hashes before using incremental export. Engine
  cleanup remains the normal runner finalization/watchdog's responsibility.

Every measured config gets inspection eligibility, IF/CAP checks, warmup, strict
JSON flush and per-level sync. `session_result.json` contains `run_id` and
`results[]` with profile SHA256, matrix, N and raw/run paths relative to `--out`.
It is rebuilt from durable per-level receipts, so recovery works even if a crash
lost the controller's aggregate. `candidate_exact` remains false and `p0` empty:
private-port/patch ablations and small CAP smoke do not establish full submission
eligibility. The daemon can rescore these artifacts but cannot auto-promote them.

T31 verification is CPU/mock only: `evidence/T31/`. Installed CLI help was read;
no real Trisol service was created, entered, or deleted.

`--minutes` (or `--budget-minutes`) is mandatory for an experiment launch; there is no implicit quota commitment.
The dry-run defaults to 480 minutes only to show a complete example. A 240-minute
budget can be insufficient. The example N6/10/14/18 plus all three matrix pairs
has about **550 minutes of conservative step time boxes**, before transfer,
sync and a possible best-config restart. These are planner assumptions, not
measured durations: actual completed steps release unused time. No automation
can guarantee a full matrix if real inference does not fit the approved budget.

The driver spends the available time on a complete batch, rather than exiting
after inspection. N6 and N10 are always attempted after the baseline gates pass.
Additional climb steps reserve two priority matrix pairs and final CAP time.
It admits both rungs of a matrix configuration together. Budget exhaustion,
non-monotone results, and max-N boundaries are explicitly incomplete; it never
invents a critical P or treats a tool failure as an SLO failure.

## Default sequence (explicit matrix/ladder flags override the traversal below)

1. Prepare the selected public AIME/GPQA questions on the controller; package
   `s1-dev/`, `cases/`, all `patches/*.patch`, `candidate-01.json`, requested
   scripts plus their helper imports, and the R4/R7/patch reference files.
   Transfer a gzip tar through noninteractive exec **arguments**, 48 KiB raw /
   64 KiB base64 per argument, eight chunks per exec by default. Every chunk,
   the whole archive, and extracted files are SHA-256 checked. The current
   bundle is about 45 MB compressed / 911 chunks / 114 exec batches. Use
   `--transfer-batch-chunks 1` if the CLI transport imposes a smaller total
   request limit; allow more `--transfer-minutes` for that slower mode.
   `-i` is an interactive terminal, `-t` is team, and one-shot stdin forwarding
   is not documented by this CLI. Neither flag is used for transfer.
2. Inspect the installed package **before starting any engine**: pip show,
   version/commit, pip dependency versions, package path, source hashes and
   diffs, DSA/indexer/sm80 source dispatch evidence, nvidia-smi, free, df,
   `/mnt/models` existence/size, and bounded public Internet HEAD probes.
   Dry-run 000/001/002 with `patch --dry-run -p3 --fuzz=0` in a scratch copy;
   actually apply each successful predecessor in that scratch copy before
   testing the next. Missing 002 is a documented skip. D0 failure aborts.
   Source dispatch evidence is not proof of the executed GPU kernel.
3. Start pristine stock for **capability smoke only**, using the same two
   questions per subject (or `--cap-count 3`). Stop it, start **D0-only FCFS**,
   explicitly remove the inherited/candidate D1 environment switch, and run
   preflight + harness self-tests + extended IF checks + paired CAP immediately.
   These are IF-01..08/10..12 and TL-03; IF-09 is inapplicable to TP8/DP1.
   A failed prerequisite aborts before a ladder; the evidence names the failed
   check for a concrete fix. The runner does not silently rewrite engine code
   or relax a check to continue. Each measured variant also gets its own preflight/CAP gate. It requires the harness dependencies already
   installed and does not rely on pod Internet/package installation.
4. Run the unmodified harness shape warmup (16 chains), strictly flush, measure
   N6 calibration and N10 held-out, then +4 on pass / -4 on fail until adjacent
   rungs. Prior observations are reused on descent. `dev+tpot` requires the
   unchanged ten dev gates and TPOT p95 ≤0.10. Estimated formal gates are
   saved separately; these are not official platform scores.
5. At P and P+4 run, in order, SPF, SPF+D1, D1; optional `--hrrn` last.
   SPF needs the 000+001+002 stack with D1 off; SPF+D1 enables it; D1 uses
   000+001 FCFS; HRRN uses 000. Each configuration gets a new engine, original
   harness warmup, and strict flush before **each** measured rung, including
   the run_dev internal post-preflight flush guarded by `ladder_search.py`.
   All other candidate arguments stay fixed, apart from the private port and
   metrics enabled for all configurations. Patch 004 is applied only for d1v12/spf_d1v12 (after 000/001/002);
   patch 003 is never applied by this runner.
6. Select the best measured config by highest observed passing rung, then
   TPOT mean (baseline wins an exact tie), and run paired CAP once more.
   Finalization independently attempts pull, stop owned jobs/engines, and
   pull again; failed copying cannot suppress engine cleanup. A pod-side
   engine watchdog and per-job supervisors enforce deadlines if the local
   controller disappears. The HTTP hang service on port 8000 is retained;
   engines use port 30000 and have PID/start-time ownership checks.

CAP runs send no output budget or thinking overrides. Truncation, missing final
answers, zero correct in either suite, or losing more than one correct answer
against stock aborts. Raw T19 reports remain `passed=false` for a small,
single-server sample. The session pairs sequential receipts by question IDs and
source hash; these smoke checks **do not pass CAP-01/02's full registry samples
or prove the official >90 ability gate**. D1 full-logit correctness still belongs
to the separate L2/T8 tests; copying `logits_check.py` does not claim those ran.

## Offline inputs and monitoring

Default CAP preparation uses the existing loader's public sources and cache
under `/sjtu/linhang/arena/cache/session_a` on the controller. To prepare without
controller Internet at launch, supply `--cap-file /path/questions.json`, with:

```json
{"aime":{"questions":[{"id":"...","kind":"aime","question":"...","answer":"..."}],"source":{"sha256":"..."}},
 "gpqa":{"questions":[{"id":"...","kind":"gpqa","question":"...","answer":"..."}],"source":{"sha256":"..."}}}
```

Each suite must contain exactly `--cap-count` **real public questions**, with
actual source provenance; the one-row schema example above is not runnable.
Pod network detection never gates offline inference when all dependencies and
questions are present. Missing dependencies/mounts are reported and stop the
run. Nothing downloads model weights or modifies `/mnt/models`.

Each step appends a timestamped line to `logs/session_a.events`. Watch:

```bash
tail -f logs/session_a.events
```

After every action, every rung (including failures), and every 120 seconds
during long actions, the runner pulls a consistent changed-file snapshot,
checks SHA-256, and copies it with another checksum check to **both**:

- `runs/session_a/<UTC timestamp>/`
- `GPU:/sjtu/linhang/arena/runs/session_a/<UTC timestamp>/`

GPU copying uses the repository SSH config/proxy helper and writes only inside
that arena path. A copy failure stops further measurements, preserving the
local copy and entering cleanup. Per-rung artifacts contain original harness
raw/run/report/summary files, all ten gates, TPOT and estimated gates, complete
engine logs and level-specific slices, ten-second Prometheus snapshots, and
queue/forward/cache/KV/Mamba occupancy availability summaries. Missing metrics
are explicitly unavailable, never fabricated as zero. Final receipts include
engine stop and nvidia-smi. `summary.json` includes observed boundaries, omitted
configs, errors, and artifact locations.

Periodic sync protects completed checkpoints; an externally deleted pod can
still lose data written since its last successful snapshot. Claude must wait
for final verified sync before deleting. The runner prints but never executes:

```bash
bohr trisol inference delete lh-arena-sess-a --team arena --yes
```

Useful controls: `--max-n 30`, `--final-minutes 10`, `--transfer-minutes 20`,
`--preflight-minutes 20`, `--cap-minutes 20`, `--time-factor 1.25`,
`--sync-seconds 120`, `--dry-run-outcomes PASS,PASS,PASS,FAIL`.
The estimator uses `scripts/plan_8gpu_session.py` directly and saves its complete
assumptions plus the actual step/rung time boxes in every live run. Live exit 0
means the requested ladder/matrix finished and all final copies succeeded; exit 2 means
failure or an incomplete batch. Use a fresh run directory; no stale-score resume.
