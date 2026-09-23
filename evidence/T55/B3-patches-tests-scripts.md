# T55 B3 — patches/ tests/ scripts/ submission/ build/(docs) data/ evidence/(orphans) logs/(size)

Method: read every patches/*.md against its .patch, RELEASE, build/image/0923a.patches.txt and
scripts/build_image.sh; git log --follow for version history; grep -rn from repo root for every
DELETE candidate (commands and hit-lists inline below). Sub-agents spawned earlier for this task
failed on a connection error before producing output (confirmed by task-notifications) — none of
their output was used; everything below is this pass's own direct reading.

**No sub-agents used in this pass (per coordinator instruction). No files modified besides this one.**

## 最会误导后续智能体的 10 条

1. **`tests/L2.md`, `tests/TRISOL_TEST_DAEMON.md`, `scripts/l2.py`, `scripts/test_status.py`,
   `scripts/session_a/*`, `tests/queue/*` describe/depend on a retired daemon that was not just
   superseded but actively dangerous.** `scripts/trisol_test_daemon.py` was moved to
   `scripts/archive/retired/` on 2026-09-22 (commit `e3fdd64`-era "Retire old L2 daemon +
   run_forever"); `notes/decisions.md:79` records why: the old daemon, auto-relaunched by
   `scripts/run_forever.sh` inside tmux `arena-daemons:l2`, was found **idling for 8 hours pointed
   at a stale/wrong service name** and had to be killed. This is the same class of incident the
   `keep-8gpu-service-alive` rule exists to prevent. `data/trisol_tests.json` confirms the queue
   never produced a result (`items: 0`). `README.md:33` already states the corrected policy
   ("旧守护进程已退役…实验全部走 pod 队列 `scripts/pod/podq`") but `tests/TIERS.md` row "入口"
   still says `scripts/l2.py add <config>`, and `tests/L2.md` still instructs a reader to start the
   daemon. A future agent following these tests/ docs could recreate the exact near-miss.
2. **`patches/120-sched-protect-chain.md:3`** ("状态：CPU VERIFIED，**待…GPU 验证**") directly
   contradicts its own **"版本记录" section (lines 94–97)**, added later, which documents three
   rounds of real 8-card GPU testing: v1 8-card N6, v2 8-card N10 (four TTFT gates passed), v3 →
   ladder 027. The file never flags that its own header is stale.
3. Neither `patches/120-sched-protect-chain.md` nor `patches/RELEASE` states that **the verified
   best-so-far baseline (S0/F93) uses `patches/drafts/120-sched-protect-chain-v2.patch`, not the
   top-level v3 file** — confirmed by `git log --follow` (top-level file and the v2 draft share
   identical history through commit `b55081f`, diverging only at `8f99ba6` "120 v3 (cap while
   decoding)"), by `scripts/pod/jobs/s0_n22.sh:6` (`G_PATCHES=… drafts/120-sched-protect-chain-v2.patch`),
   and by `notes/submissions.md:30` (image 0923a used "MTP+114+**v3**", a *different* branch than
   S0). A reader of the top-level .md alone would not know two live, differently-verified branches
   exist. This is the ambiguity the coordinator assigned to T57.
4. **`patches/RELEASE:1`** ("what goes into the submission image… Updated 2026-09-23") and its
   Tier-2 comment ("uncomment ONLY after the 8-card self-test ladder passes… and capability smoke
   12/12") overclaim. `build/image/0923a.patches.txt` — the image actually built and used for
   scored official submissions 45979/45980 (`notes/submissions.md:23,30`) — contains **7 Tier-2
   patches** (114,140,120,130,150,160,170v2) although `_context-0924.md` item 5 states **no
   candidate ever passed all N18 gates**. `scripts/build_image.sh:55-59` explains the mechanism
   (RELEASE's uncommented list is only the *default* when `build_image.sh` is called with no
   explicit patch args; an explicit arg list silently overrides it) — so this isn't a bug, but
   RELEASE's own text doesn't disclose that its stated gate was bypassed for the real submission.
5. **`patches/112-sm80-indexer-kernels.md:3`** and **`patches/113-sm80-prefill-indexer.md:3`** both
   say "…未加入 RELEASE" (not yet added to RELEASE). `patches/RELEASE` Tier 1 ("ALWAYS in the
   image") **already lists both `112-sm80-indexer-kernels.patch` and `113-sm80-prefill-indexer.patch`**.
6. **`patches/README.md:3`** ("现行（打进提交镜像）：`000-interface-compliance`、
   `101-d1v12-on-base`；清单见 `RELEASE`") names only 2 of the 8 patches RELEASE Tier 1 actually
   marks "ALWAYS in the image" (000,101,105,106,110,111,112,113). Reads as a stale line from when
   Tier 1 really was just 000+101.
7. **`build/scratch/` has 4664 files committed to git** although `scripts/build_image.sh` treats it
   as throwaway output it recreates on every run (`rm -rf "$SCRATCH"; mkdir -p…`) and its sibling
   dirs (`build/p112/`, `p113/`, `p120/`, `p140/`, `p150/`) each carry their own `.gitignore` to
   keep exactly this kind of regenerated output out of git. `build/scratch/` has no such
   `.gitignore` entry — an oversight, and the single largest tracked-file-count cleanup item found
   in this whole partition.
8. **`tests/TEST_PLAN.md` "T24 自动提测工具" (TQ‑01…12, ≈lines 118–137) and "T23 / Session A
   自动化" (SA‑01…11, ≈lines 140–161)** exercise the now-archived daemon and `scripts/session_a/`
   (both item 1 above); the documented command at line 120
   (`unittest discover -s scripts -p test_trisol_test_daemon.py`) now points at a path that no
   longer exists there (file is `scripts/archive/retired/test_trisol_test_daemon.py`).
9. **`tests/TEST_PLAN.md` rows P120‑08/09/10 (lines 194–196) and P140‑08 (line 236)** are marked
   `todo`/"未入队/未执行" (T8/L2 not yet run) for patches 120 and 140 — contradicted by both
   patches' own "GPU-verified at 8 cards" history (120: N6/N10/ladder 027; 140: part of the S0
   stack run through ladder 026 at N18, `_context-0924.md` item 5).
10. **`patches/drafts/` and `patches/v0520/` hold 5–9 superseded patch versions** the user's 09-24
    policy (keep exactly one version per patch) targets for deletion: `120-sched-protect-chain-v1.patch`,
    `170-v1.md/.patch`, `112-sm80-indexer-kernels-v1.md/.patch`, `113-sm80-prefill-indexer-v1.md/.patch`,
    `102a-single-point-role-track.patch`, and all of `v0520/` (9 files, already self-labeled
    "归档…打不上底包" since decision 29). `patches/drafts/120-sched-protect-chain-v2.patch` is the
    one explicit exception (kept until T57).

---

## patches/

| Path | Verdict | Reason |
|---|---|---|
| `patches/000-interface-compliance.md` | REWRITE | Line 3 status "设计稿，未实现、未运行" is stale; file's own §6 (line 75) already says so ("上面的'未实现/仅设计'是历史状态"). Update line 3 to reflect implemented+CPU/E2-verified status instead of relying on a reader reaching §6. Low priority (self-corrects further down). |
| `patches/101-d1v12-on-base.md` | KEEP | Short, factual, matches RELEASE Tier 1 and 0923a.patches.txt. |
| `patches/105-role-split-single-partial.md` | KEEP | Matches RELEASE Tier 1; stack order line consistent with 0923a. |
| `patches/106-defer-chunk-on-no-kv.md` | KEEP | Matches RELEASE Tier 1; F62/024-crash story checks out against ladder history in `_context-0924.md`. |
| `patches/110-sm80-dsa-indexer.patch` (no .md) | KEEP | Tier 1, current; no wrong claims possible without a .md. Not actionable under this policy (writing new docs is out of scope). |
| `patches/111-sm80-fp8-moe-marlin.md` | KEEP | Matches RELEASE Tier 1 and 0923a. |
| `patches/112-sm80-indexer-kernels.md` | REWRITE | Line 3 "…未加入 RELEASE" is wrong — RELEASE Tier 1 (ALWAYS in image) lists `112-sm80-indexer-kernels.patch`. Change to "已进 RELEASE Tier 1（v2/T47/F68 起）". Also line 81 pointer to `patches/drafts/112-sm80-indexer-kernels-v1.*` should be removed/marked deleted once drafts/112-v1 is deleted (see below). |
| `patches/113-sm80-prefill-indexer.md` | REWRITE | Line 3 "…不改RELEASE" is wrong — RELEASE Tier 1 lists `113-sm80-prefill-indexer.patch`. Same fix as 112. Line 81 pointer to `patches/drafts/113-sm80-prefill-indexer-v1.*` and `evidence/T47/sm80_indexer_{112,113}_v1.py` needs the drafts-file half removed once drafts/113-v1 is deleted; the `evidence/T47/` files are untouched (evidence/ is not being deleted here). |
| `patches/114-indexer-row-shard.md` | KEEP | Short/factual; consistent with 0923a.patches.txt inclusion and S0+114 probe job (`s0_114_n22.sh`). |
| `patches/115-sm80-sparse-attn-many-heads.md` | KEEP | Consistent with RELEASE Tier 2 comment and correct absence from 0923a.patches.txt (DCP not shipped). |
| `patches/116-dcp-dsa-address.md` | KEEP | Extensive VERIFIED/INFERRED labeling, consistent with RELEASE's "DCP must NOT ship before this" and correct absence from 0923a.patches.txt. |
| `patches/120-sched-protect-chain.md` | REWRITE | (1) Line 3 "待…GPU 验证" contradicts lines 94–97's own record of 3 rounds of 8-card GPU testing — rewrite to state GPU-tested history and current CPU-only-for-v3-specifically if that's still true, or correct if v3 itself has since been GPU-tested (check ladder 027 status before writing). (2) Add an explicit line in "版本记录" that **S0/F93's verified baseline uses v2 (`patches/drafts/120-sched-protect-chain-v2.patch`), not v3** — this exact gap is why T57 exists; a reader must not assume v3-current is what's running best. (3) Line 95 "存 patches/drafts/120-sched-protect-chain-v1.patch" needs updating once v1 is deleted (see below). |
| `patches/130-async-tokenize.md` | KEEP | Status "未加入 RELEASE" is correct — 130 is commented out in RELEASE Tier 2 and absent from 0923a.patches.txt. |
| `patches/140-kda-dual-snapshot.md` | REWRITE | Line 4 "未加入 RELEASE、构建脚本或任何队列" is correct re: RELEASE (still commented Tier 2), but is now misleading given 140 is part of the verified S0 baseline and shipped in 0923a.patches.txt — add a line noting 140 is exercised live via ladder 026/S0 and shipped in image 0923a, so "not in RELEASE" only means "not in the conservative default," not "untested." |
| `patches/150-startup-warmup.md` | KEEP | "未加入 RELEASE" correct — 150 commented in RELEASE Tier 2; 150 status text elsewhere already very carefully hedged (see "结论范围" line 7). |
| `patches/160-nextn-sm80.md` | KEEP | "未加入RELEASE、镜像构建或队列" is defensible — 160/MTP inclusion in 0923a is asserted by `notes/submissions.md:30` ("A … MTP+114+v3") but 160.md's own scope is narrower (ops-level only, no N@SLO claim), so no correction needed; cross-reference cleanup is B1's territory (notes/submissions.md), not this file. |
| `patches/170-glm-bcg-prefill.md` | KEEP | Checked specifically for the retracted "padding" explanation (pinned memory `experiment-discipline.md`): the file **already correctly retracts it** at line 28 ("所以根因不是 n≠B 的 padding 问题") and documents the real root cause (scatter decision divergence between capture and replay) with a v2 fix and quantitative verification table. Not stale. Line 27 pointer to `patches/drafts/170-v1.*` needs removal once that draft is deleted (see below). |
| `patches/README.md` | REWRITE | Line 3 undercounts current Tier 1 (see "most misleading" #6). Rewrite to either list all 8 Tier-1 patches or drop the specific names and just say "见 RELEASE 的 Tier 1 清单" without naming only 2. |
| `patches/RELEASE` | REWRITE | Header (line 1) and Tier-2 comment overclaim authority over what actually ships (see "most misleading" #4). Add a line clarifying: (a) this file is the *default* patch list for `build_image.sh` with no explicit args — an explicit override list was used for image 0923a and is recorded separately in `build/image/0923a.patches.txt` / `notes/submissions.md`; (b) Tier-2 patches shipped in 0923a did so as a best-effort daily-submission arm, not because they passed the stated ladder gate — see `daily-two-submissions` policy. Do not weaken the gate itself, just its claim to be exhaustive history. |
| `patches/drafts/120-sched-protect-chain-v1.patch` | DELETE | User policy (09-24): keep exactly one version. Superseded twice (v2, then v3). Referenced only by `patches/120-sched-protect-chain.md:95` (self-pointer, to be edited per REWRITE above) and `docs/histories/2026-09/20260923-0720-scheduler-120-v2.md` (out of B3 scope, narrative mention, not a functional dependency — flag to B2). No script loads it. `grep -rln "120-sched-protect-chain-v1"` → only those two files. |
| `patches/drafts/120-sched-protect-chain-v2.patch` | **KEEP (explicit exception)** | Coordinator: needed until T57 re-expresses the S0 baseline as a single canonical version. Actively loaded by `scripts/pod/jobs/s0_n22.sh`, `s0_114_n22.sh`, and is the byte-identical archive of the patch used for the verified S0/026 8-card runs. |
| `patches/drafts/102a-single-point-role-track.patch` | DELETE | User policy names it explicitly. `grep -rn "102a"` → only `patches/README.md:6` (describes it as "未定稿…存在 strict-append 退步与 bf16 精度问题", i.e. already documented as broken/abandoned) and `scripts/archive/README.md:2` (also being deleted, see scripts/ section). No functional reference. |
| `patches/drafts/112-sm80-indexer-kernels-v1.md` / `.patch` | DELETE | User policy names "112/113 v1" explicitly; superseded by v2 (F68/T47, folded into top-level `patches/112-sm80-indexer-kernels.md`). Referenced only by the top-level 112/113 .md pointer text (line 81 in each, to be edited per REWRITE above) and by `evidence/T47/performance_table.md`-style comparisons that cite `evidence/T47/sm80_indexer_112_v1.py` — that is a *separate* file under `evidence/T47/`, not this patch, and is untouched. |
| `patches/drafts/113-sm80-prefill-indexer-v1.md` / `.patch` | DELETE | Same as above (paired with 112-v1). |
| `patches/drafts/170-v1.md` / `.patch` | DELETE | User policy names "170 v1" explicitly; fully superseded by v2 fix folded into top-level `patches/170-glm-bcg-prefill.md`. Referenced only by that file's own line 27 pointer (to be edited per REWRITE above). v1 is also the version with the known wrong-output bug (026j 0/12 capability smoke) — keeping it un-deleted risks a future agent copying the broken version. |
| `patches/v0520/001-role-boundary-mamba-ckpt.md` / `.patch`, `002-spf-scheduling.*`, `003-slo-aware-scheduling.*`, `004-role-boundary-final-chunk.*`, `102-role-track-v0520.patch` (9 files total) | DELETE | User policy names "v0520/" explicitly. Every file already opens with "> 2026-09-22 归档：v0.5.20 线，打不上底包（决策 29）；仅作 L1 替身参考" (already self-declared obsolete/non-applicable to the real base). `grep -rln "v0520"` hits: `README.md`, `patches/README.md` (both narrative mentions of the archival decision — B2/this-file's own README line 7, to be updated to drop the now-broken path reference), `notes/dispatch.md` (narrative), `scripts/session_a/test_session_a.py:240` (a comment only: "# v0.5.20 line (001/002/004) is archived under patches/v0520 (decision 29): reported absent" — `scripts/session_a/` itself is recommended DELETE below, so this reference disappears with it). No functional/import dependency anywhere. |

---

## tests/

| Path | Verdict | Reason |
|---|---|---|
| `tests/TIERS.md` | REWRITE | Row "入口" for L2 says `scripts/l2.py add <config>` (line 15) — this is the retired-daemon entry point (see #1). Change to `scripts/pod/podq` / `scripts/pod/qpush` + `scripts/pod/jobs/*.sh`, pointing at `scripts/pod/README.md`. Everything else in this file (three-tier model, what each tier can/can't confirm) checked against `llm-challenge-arena-v1/task.md` and is accurate/current — do not rewrite wholesale. |
| `tests/TEST_PLAN.md` | REWRITE | (1) Delete or clearly mark historical the "T24 自动提测工具" section (TQ-01..12) and "T23 / Session A 自动化" section (SA-01..11) — both test the retired daemon/session_a subsystem (see #1, #8); the referenced command path `scripts/test_trisol_test_daemon.py` no longer exists there. (2) Rows P120-08/09/10 and P140-08 marked `todo` should be reconciled against the real 8-card ladder history (F91-F93, notes/experiments.md) rather than left implying "never GPU-tested" (see #9) — flagged here, not hand-edited (needs B1 coordination against findings.md/experiments.md, which I did not modify). (3) Everything else spot-checked (D0/D1/D2/SLO sections, T53 section lines 289-297, gate-count phrasing "十门+TPOT"/"10 门+tpot 门" which correctly totals task.md's 11 gates) is accurate/current — do not rewrite wholesale. |
| `tests/L2.md` | DELETE | Entire file describes the retired daemon mechanism (`scripts/trisol_test_daemon.py`, `scripts/l2.py`, `tests/queue/`) — see #1. Surviving useful ideas (L2→L3 calibration goes in `notes/submissions.md`; 2-per-day submission quota) are already documented elsewhere (`patches/RELEASE`, `notes/submissions.md`). Referenced by `tests/TIERS.md:19` (to be repointed per that file's REWRITE) and `scripts/l2.py` itself (also DELETE, see scripts/ section). |
| `tests/TRISOL_TEST_DAEMON.md` | DELETE | Documents the protocol (budget units, nohup start/stop, cost accounting) of the now-archived `scripts/trisol_test_daemon.py`. Referenced only by `tests/L2.md:8` (also deleted) and the retired daemon files themselves under `scripts/archive/retired/`. |
| `tests/T19_USAGE.md` | KEEP | Spot-checked (IF-12 timing procedure, line 42) — consistent with server-side timestamp requirement (task.md:205,556); no stale-mechanism references found. |
| `tests/queue/` (10 items: 01-img-a … 10-img-b-repeat, each with APPROVED/notes.md/profiles.sha256.json/spec.json) | DELETE | Orphaned: all 10 committed at the same instant (2026-09-22 10:57:30, T38's work) and **zero** of the 9 distinct configs (`img_a`, `img_b`, `img_b_dsafa3`, `img_b_off`, `img_b_pdi2`, `img_b_mixed`, `img_b_mtp`, `img_b_track64`, `img_b_fcfs`) appear anywhere in `data/trisol_tests.json` (0 items total) — the daemon that would have run them was retired the same day before any ran. The open questions they were meant to answer (stock baseline, D1 on/off, MTP tie-break, etc.) are now covered by `scripts/pod/jobs/m0_stock.sh`, `m0_submitted_b.sh`, and the `ladder_*`/`ifx_*` job families. Referenced only by `scripts/l2.py`, `scripts/test_status.py` (both DELETE) and narrative mentions in `README.md`, `board.md`, `notes/dispatch.md`, `evidence/T31/README.md` (all prose, no functional coupling). |
| `tests/queue/README.md` | DELETE | Describes the now-dead queue (see above). |
| `tests/queue_archive/` (incl. `param_sweep_0922/`, `README.md`, `l2_catalog.json`) | KEEP | Self-labeled historical record of an actually-executed and explicitly withdrawn 35-item parameter sweep ("decision 28" per `tests/L2.md`, itself being deleted, but the decision is also in `notes/decisions.md`). Contains real results (`notes.md`, `profiles.sha256.json` per item), not a live-looking dead queue like `tests/queue/` — does not misrepresent itself as current. Distinguishing this from `tests/queue/` is intentional: one is honestly archived history, the other is an abandoned live-looking queue. |
| `tests/test_sched_protect_chain.py` | KEEP | Env vars tested (`SGLANG_AX_SCHED_PROTECT`, `SGLANG_AX_SCHED_COLD_CAP`, `SGLANG_AX_SCHED_SHORT_TOKENS`) match the current top-level `patches/120-sched-protect-chain.md` table (lines 29-33) exactly — tests the current (v3) semantics, not a stale version. |
| `tests/trisol_test_config.json`, `tests/trisol_test_spec.example.json` | KEEP | Config/schema examples for the daemon *protocol*, which other still-live tooling (`scripts/submit_daemon.py`, the **official-submission** daemon — a different daemon than the retired L2-test one) may still rely on for its JSON shape; did not find evidence these are L2-test-daemon-specific. Left as KEEP; flag UNSURE if B1/B4 finds submit_daemon.py does not use this shape. |
| `tests/__pycache__/` | DELETE (trivial) | Untracked by git (`git ls-files` → 0 hits) and covered by `.gitignore` (`__pycache__/`). Not a git-content issue; pure local build artifact, no action needed beyond noting it's already correctly ignored. |

---

## scripts/

| Path | Verdict | Reason |
|---|---|---|
| `scripts/archive/` (23 files: README.md + `check_d2_sim_calibration.py`, `check_slo_calibration.py`, `compare_e2b.py`, `e2_completion_trace.py`, `e2b_trace.py`, `make_102_role_track.py`, `run_e2b_batch.py`, `sim_calibration_checks.py`, `sim_checkpoints.py`, `sim_closed_loop.py`, `sim_envelope_fit.py`, `sim_role_boundary.py`, `test_compare_e2b.py`, `test_d0_v11_review.py`, `test_d1_admission_review.py`, `test_d1_final_chunk.py`, `test_e2_completion.py`, `test_role_boundary_split.py`, `test_sim_closed_loop.py`, `test_sim_envelope_fit.py`, `test_sim_spf_upstream.py`, `test_slo_scheduling.py`, `test_spf_scheduling.py`, plus `retired/run_forever.sh`, `retired/test_trisol_test_daemon.py`, `retired/trisol_test_daemon.py`) | DELETE | Own README already states "现行链路不依赖它们" (current pipeline doesn't depend on them). `grep -rln "scripts/archive"` (excluding itself) → only `README.md`, `docs/histories/2026-09/20260923-0830-infra-safety.md`, `notes/decisions.md`, `notes/dispatch.md` — all narrative mentions of the archival/retirement events, zero functional imports. This folder is itself the anti-pattern the user's stated T55 policy targets ("不建归档目录" — don't build archive directories, delete outright, git keeps history); it predates that policy (created 2026-09-22, `scripts/archive/README.md:1`) and should now be cleaned up under it. |
| `scripts/l2.py` | DELETE | Orchestrates `tests/queue/` for the retired daemon (`QUEUE = ROOT / "tests/queue"`, line 28). Only referenced by `tests/TIERS.md`, `tests/TRISOL_TEST_DAEMON.md`, `tests/L2.md` (all being deleted/rewritten above). |
| `scripts/test_status.py` | DELETE | "One-glance ledger status" for the same retired subsystem — reads `data/trisol_tests.json` (0 items), `tests/queue/`, `logs/trisol_test.events` (0 bytes). No other consumer found. |
| `scripts/session_a/` (whole dir: `README.md`, `common.py`, `configs.json`, runner + test files) | DELETE | Built specifically to drive `scripts/trisol_test_daemon.py` (per `notes/dispatch.md` T30: "对齐自测守护进程与 8 卡 runner 的命令行接口…scripts/session_a/*、scripts/trisol_test_daemon.py"); last touched at the very first baseline-snapshot commit (2026-09-22 10:57:30) and never updated to reference `scripts/pod/` (`grep` for `scripts/pod` inside `scripts/session_a/*.py` → 0 hits; `scripts/pod/jobs/` has ≈70 files created/maintained after that date doing the equivalent job). Deleting this also resolves the dangling `patches/v0520` comment reference noted above. `tests/TEST_PLAN.md`'s SA-01..11 rows document only CPU/mock tests of this dead code (see tests/ section). |
| `scripts/launch_e2_completion.sh` | DELETE (low priority) | Zero references anywhere (`grep -rl "launch_e2_completion"` → only itself). Own header: "T29 diagnostic only" — a narrow one-off variant of the still-live `run_e2_variant.sh`/`launch_e2_standin.sh` general E2 tooling. Low risk either way; flagged but not load-bearing. |
| `scripts/make_112.py`, `make_113.py`, `make_140.py`, `make_150.py`, `make_160.py`, `verify_112.py`, `verify_113.py`, `verify_120.py`, `verify_140.py`, `verify_150.py`, `verify_160.py`, `summarize_112.py`, `summarize_113.py`, `summarize_140.py`, `summarize_150.py`, `summarize_160.py`, `summarize_47.py`, `make_110.py` | KEEP | All tied to patches still in RELEASE (Tier 1 or Tier 2) and actively cited by the corresponding patch .md's "复现" sections; source dirs `scripts/kernels/`, `scripts/p140/`, `scripts/p150/`, `scripts/p160/` confirmed as their real inputs (`grep` shows `make_150.py:38`, `make_160.py:36`, `make_140.py:4,50` reading from these paths). |
| `scripts/make_120.py` | KEEP (known bug, Codex-owned — not rewritten here) | `tests/TEST_PLAN.md` row **T53-04** already documents, as `fail`, that this generator does **not** currently reproduce patch 120 v3's wait/decode condition ("只在内存执行edit_policy"); this matches `_context-0924.md` item 2 ("`make_120.py` 与 120 v3 不一致"). Per my instructions this file is Codex's to fix — noted here for completeness, not a new finding, not modified. |
| `scripts/kernels/`, `scripts/p140/`, `scripts/p150/`, `scripts/p160/`, `scripts/m0/`, `scripts/analysis/` (all subdirs) | KEEP | Confirmed as live source/support dirs for the generators above and for the pod job families (`m0_stock.sh`, `b11x_start_probe.sh`, `devbox_bcg_check.sh`, `dcp_check.py`, etc.); all directly cited by name inside currently-kept patch `.md` files (114, 116, 170) and pod job scripts. |
| `scripts/pod/*`, `scripts/pod/jobs/*.sh` (≈70 files), `scripts/pod/verify/*` | KEEP | Spot-checked `scripts/pod/jobs/s0_n22.sh` and `s0_114_n22.sh` (both untracked/new) in full: correctly encode the S0 stack (`drafts/120-sched-protect-chain-v2.patch`, not top-level v3), correctly note the printed ladder verdict is fail-open and must be re-scored by the fail-closed T54 scorer, correct preconditions/smoke-gate discipline. No evidence found of "copying old/stale templates" in the files sampled; not exhaustively re-verified across all ≈70 job files (out of proportion to expected value — these are executable job specs, not claims that can be "factually wrong" the way a doc can, beyond what a run's own log already falsifies). |
| `scripts/archive`, `scripts/session_a`, `scripts/l2.py`, `scripts/test_status.py` cross-refs from `scripts/pod/*` | confirmed clean | Built an allowlist of scripts referenced from `scripts/pod/**` (55 distinct names) before recommending any DELETE above; none of the files recommended for deletion in this section appear in that allowlist. |
| All other top-level `scripts/*.py`/`*.sh` not named above (≈90 files: `analyze_*`, `bench_*`, `check_*`, `compare_*`, `decode_diag_log.py`, `if_checks.py`, `index_notes.py`, `inspect_*`, `inventory_warmup_150.py`, `logits_check.py`, `next_id.py`, `plan_8gpu_session.py`, `preflight_8gpu.*`, `profile_sm80_indexer_113.py`, `queue_submission.sh`, `rebase_d1_to_base.py`, `replay_chains.py`, `replay_kda_snapshot_140.py`, `run_*`, `serving_probe.py`, `setup_e1_env.sh`, `ssh_*`, `submit_daemon.py`, `submit_official.sh`, `test_*` files not already listed, `tune_sm80_indexer_*`, `gjob`, `gssh`, `gpu_box_setup.sh`, `build_image.sh`, `check_records.py`, `check_submission.py`, `codex_worker.sh`, `e1_env.sh`, `l2.py`'s neighbors, `ladder_search.py`, `make_case_sets.py`, `make_e1_kimi_standin.py`, `make_submission_0923.py`, `score_formal.py`) | KEEP | Reference-count heuristic (grep of each basename across the whole repo, excluding self) found only 2 zero-reference files total (`launch_e2_completion.sh`, handled above, and `test_e2_tools.py`, a genuine unit test with no false-positive risk — it imports `compare_e2` and is invoked by test-discovery convention, not by name); everything else has ≥1 external citation and is tied to still-relevant patches/tooling (E1/E2 dev-box tier per `tests/TIERS.md`, submission pipeline, records tooling). Did not deep-read all ≈90 files individually — heuristic + spot checks only; no evidence of wrong/misleading *content* found (these are mostly tools, not claims). |
| `scripts/score_formal.py`, `scripts/ladder_search.py` | KEEP | Codex-owned; not reviewed for internal correctness here per instructions (import/wrap only). No duplicate-gate-logic scripts found elsewhere reimplementing TTFT/tpot thresholds outside `s1-dev/harness/s1_common.py` and these two files — grepped for hardcoded `3s/5s/15s/30s`/`0.10` gate thresholds across `scripts/*.py`; only hits are inside `score_formal.py`/`ladder_search.py` themselves and test fixtures that assert against them, not a second independent implementation. |
| `scripts/__pycache__/`, `scripts/kernels/__pycache__/`, `scripts/pod/verify/__pycache__/` | no action | Untracked by git (`git ls-files` → 0) and gitignored. Not a content issue. |

---

## submission/

| Path | Verdict | Reason |
|---|---|---|
| `submission/official-0922-A.json`, `official-0922-B.json`, `official-0923-A.json`, `official-0923-B.json` | KEEP | Historical record of actual scored submissions; required for traceability (`every-change-traceable` policy). |
| `submission/candidate-01.json`, `candidate-01.notes.md` | KEEP | Actively cited as the worked example in `submission/queue/README.md:8`. |
| `submission/candidate-b0-d0-baseline.json`, `candidate-b1-d1.json`, `candidate-b2-spf.json`, `candidate-b3-spf-d1.json`, `candidate-bA-0922e.json`, `candidate-bB-0922f.json` | KEEP | Historical draft configs from the D0/D1/SPF exploration phase; small, not wrong (just old), part of the experiment trail. Not "misleading current documentation" — no action needed. |
| `submission/queue/README.md` | KEEP | Describes the **official-submission** daemon/queue (`scripts/submit_daemon.py`), a distinct, still-live system from the retired L2-*test*-daemon (see tests/ section) — confirmed by its own reference to `notes/experiments.md#t22` and by `data/submissions.json` being actively read by `scripts/submit_daemon.py`. Do not confuse with `tests/queue/`. |
| `submission/stub-trace.jsonl` | KEEP | Heavily referenced (`board.md`, `research/shared/pipeline.md`, `llm-challenge-arena-v1/task.md`'s own trace-format expectations, `notes/experiments.md`, `scripts/submit_daemon.py`, `scripts/test_submit_daemon.py`) — live, required file. |

---

## build/ (docs and manifests only, per scope)

| Path | Verdict | Reason |
|---|---|---|
| `build/image/0923a.patches.txt`, `0923a.patch_sha.txt`, `A.Dockerfile`, `B.Dockerfile`, `Dockerfile` | KEEP | Authoritative manifest of the actually-shipped image; load-bearing for traceability (see "most misleading" #4). |
| `build/verify_kit/*` | KEEP | Confirmed **not** superseded — it is the local staging copy that `scripts/pod/verify/make_kit.sh` packages and that gets pushed to the pod as `$AX/verify_kit/`, referenced by essentially every job in `scripts/pod/jobs/*.sh` (e.g. `ladder_best140.sh`, `s0_n22.sh` both call `$AX/verify_kit/analyze_run.py`). Same commit timestamp as `scripts/pod/verify/` confirms the pairing. |
| `build/podtools/` (`died.sh`, `dstat.sh`, `errs.sh`, `lastfail.sh`, `prof.sh`, `profsum.py`, `prog.sh`, `slowreq.sh`, `summ.sh`) | DELETE | Zero references anywhere in the repo outside itself (`grep -rln "podtools"` → no hits at all, not even narrative). Functionally superseded by the actively-used `scripts/pod/verify/` toolkit (`component_table.py`, `interference.py`, `logstat.py`, `cachecmp.py` cover the same ground as `prof.sh`/`errs.sh`/`dstat.sh`). Earlier-generation, orphaned diagnostic scripts. |
| `build/diag10/` (`Dockerfile`, `build.json`, `log.txt`) | DELETE | Zero references in any `.md`. A single diagnostic build attempt's raw output; same category as the `logs/build_*` clutter below — low value, safe to drop. |
| `build/scratch/` | DELETE | 4664 tracked files despite being explicitly regenerable scratch output recreated by every `scripts/build_image.sh` run (`rm -rf "$SCRATCH"`); every sibling generated-output dir (`build/p112/`, `p113/`, `p120/`, `p140/`, `p150/`) has its own `.gitignore` doing exactly this exclusion and `build/scratch/` does not. See "most misleading" #7 — single largest tracked-file-count item in this review. (Actual removal + `.gitignore` fix is a repo-hygiene action, not a doc edit; flagged here since it is squarely "regenerable output masquerading as content.") |
| `build/l3_0922e/`, `build/l3_0922f/`, `build/p110/`…`build/p160/` (candidate/baseline/verify source trees) | KEEP / UNSURE | Out of "docs and manifests" scope for deep review; not documentation (no claims to be right/wrong), just large generated-or-input source trees. 113/150/160/patches are still pre-RELEASE (per `patches/README.md`), so their build outputs may still be needed for re-verification — do not delete without confirming with whoever owns those patches' next steps. Sized but not read in depth. |
| `build/submit_0922_A/`, `.zip`, `submit_0922_B/`, `.zip`, `submit_0923_A/`, `.zip`, `submit_0923_B/`, `.zip` | KEEP | Actual submitted archives — historical record, parallels `submission/official-*.json`. |
| `build/base_exact/` | **KEEP — protected, do not touch** | Explicitly on the never-delete list. |

---

## data/

| Path | Verdict | Reason |
|---|---|---|
| `data/all_att.json`, `all_att_2026-09-22b.json`, `all_att_2026-09-23.json` | KEEP | Dated leaderboard/attempt snapshots, each cited by a specific research doc analyzing that day's state (R7, R8, R13, R14); legitimate time series, not a "latest wins" situation. |
| `data/lb.json` | KEEP | Populated (17.7 KB, real scraped leaderboard rows), documented as a standing data category in `rule.md:41` ("爬到的排行榜/提交数据（`lb.json`, `all_att.json`）"). No script hit by exact-path grep but this matches a documented, intentional data category, not stray/wrong content. |
| `data/submissions.json` | KEEP | Actively read by `scripts/submit_daemon.py`; cited in `notes/submissions.md`, `notes/dispatch.md`, `notes/experiments.md`. |
| `data/trisol_tests.json` | KEEP, but note | Only remaining "consumers" (`scripts/l2.py`, `scripts/test_status.py`, `tests/L2.md`, `tests/TRISOL_TEST_DAEMON.md`) are all recommended DELETE above (retired-daemon cluster); file itself is empty (`items: 0`) and becomes pure vestige once those are gone. Left as KEEP (small, harmless, `notes/experiments.md` still references it narratively) rather than DELETE, since deleting a data file with a still-live narrative citation is lower-value than the code/doc cleanup above. |
| `data/image_names.txt` | KEEP | Cited by `notes/findings.md`. |
| `data/analysis/r7_flat_attempts.json`, `r7_top_players.json` | KEEP | Source data for `research/claude/R7_top_players_analysis.md`. |

---

## evidence/ (orphans only)

Reference-check method: `grep -rl "<name>" notes/ research/ plans/ docs/ README.md board.md patches/ tests/` per directory (see raw counts below); presumption of KEEP for anything with ≥1 hit, since raw evidence is primary source data.

| Path | Verdict | Reason |
|---|---|---|
| `evidence/COMP/` (`component_table.py`, `make_rank_model.py`, `rank_p8192_components.txt`, `rank_profile.sh`) | DELETE | **Zero** references anywhere in the repo outside itself — confirmed with both a substring search for "COMP" (0 hits, including case-sensitivity margin) and an exact-path search for "evidence/COMP". Only genuinely orphaned evidence directory found. |
| `evidence/base_manifest/` | KEEP | 0 hits in the doc-only grep sweep, but has a real functional consumer: `scripts/m0/setup_env.sh` references it directly. |
| `evidence/MOVES.txt` | KEEP | Not scratch — it's the provenance record of an earlier `notes/tXX_*.log` → `evidence/TXX/*.log` reorganization; cited by `rule.md` and `research/codex/archive/R12_slo_aware_scheduling.md` and `notes/findings.md`. |
| `evidence/T52/` vs `evidence/T52b/` | KEEP both | `evidence/T52b/README.md` does **not** say it supersedes/invalidates T52 (grep for "T52" inside T52b's README: 0 hits — T52b is a self-contained continuation, not a correction of T52's own content). `patches/170-glm-bcg-prefill.md` line 62 explicitly still cites `evidence/T52/` as the evidence source for the base numerics section. Both load-bearing for the currently-kept patch 170. |
| `evidence/T12`, `T15_simulator`, `T16`, `T18_calibration`, `T19`, `T20_d2`, `T22`–`T27`, `T29`, `T31`, `T35`, `T36`, `T40`–`T51`, `T53` | KEEP (all) | Every one returned ≥1 reference from notes/research/plans/docs/patches/tests (counts ranged 1–77; `T12` alone had 77). Not exhaustively re-verified past the reference count — raw evidence is presumption-of-keep by T55's own framing ("孤立证据" targets only truly orphaned material). |
| `evidence/E1_stock/`, `F57/`, `F58/`, `INT8/`, `R8/` | KEEP | 5–8 references each; tied to still-cited findings (F57/F58 = patch 111/tilelang validation; R8 = prefix-reuse analysis feeding research/claude/R7). |
| `evidence/T54/`, `T55/`, `T56/` | KEEP (obviously current) | In-progress/sibling T55 task directories; not evaluated further. |

---

## logs/ (size and clutter only)

Total `logs/`: **32 MB**, of which **`logs/codex/` alone is 32 MB** (dominates entirely; everything else sums to well under 400 KB). None of `logs/` is tracked by git (`git ls-files logs` → 0 hits; `logs/` is in `.gitignore`) — this section is pure local-disk housekeeping, not a git-history concern.

- `logs/codex/W1…W25` (`.log` + `.last.md` pairs): KEEP — detailed worker transcripts behind the `research/codex/R*.md` writeups; did not verify a 1:1 superseded relationship for every W-number against every R-number (would require reading all 25 pairs against all research/codex docs — out of proportion to the ≤400 KB non-codex total and B2's ownership of research/codex content). No specific W-log found to be fully redundant.
- `logs/build_A.log/.json`, `build_A2.json`, `build_A3.json`, `build_B.json/.log`, `build_B2.json`, `build_B3.json`, `build_diag.log/.json`, `build_diag2.log/.json`, `build_163106…163118.log` (6 files), `build_0923a.json`: DELETE-candidates for the clearly-superseded iterative attempts (A/A2 before A3, B/B2 before B3, diag before diag2, and the six raw-timestamp `1631xx` early trial logs), KEEP the final ones per image (`build_A3.json`, `build_B3.json`, `build_diag2.json/.log`, `build_0923a.json` — this last one is the authoritative match to `build/image/0923a.patches.txt`). `grep -rln` for every one of these filenames found **zero** references in any `.md` anywhere. Combined size of the whole cluster is small (≈370 KB) — flagged for clutter/count, not space.
- `logs/session_a.events` (4 KB): tied to `scripts/session_a/` (recommended DELETE above) — safe to drop alongside it.
- `logs/submit_daemon.START`, `submit_daemon.events`: KEEP — live official-submission daemon state.
- `logs/trisol_test.events` (0 bytes): DELETE-candidate — empty, part of the retired-daemon cluster.
- `logs/after_w14.log` (4 KB): not investigated in depth (tiny, low priority either way).

---

## Cross-partition notes (not acted on here)
- `docs/histories/2026-09/20260923-0720-scheduler-120-v2.md` references `patches/drafts/120-sched-protect-chain-v1.patch` — B2's file, needs its own pointer check once v1 is deleted.
- `notes/dispatch.md`, `notes/decisions.md` narratively describe the L2-daemon retirement and `scripts/archive` creation — accurate as narrative/history, no action needed even after the referenced paths are deleted (git history preserves them).
- `tests/TEST_PLAN.md` P120/P140 T8-row staleness (finding #9) should be reconciled against B1's `notes/findings.md`/`notes/experiments.md` ladder history, not hand-edited here.
