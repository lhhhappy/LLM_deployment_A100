# T19 live checks and CPU tests (W6)

These tools use an **already authorized, running, isolated server**. They never
launch an engine, allocate Trisol, build an image or submit. Flush probes clear
the whole supplied server, so run them without unrelated clients. D0 JSON flush,
metrics timestamps and incremental streaming must be enabled where tested.
`S1_API_KEY` authenticates the first server; `S1_OTHER_API_KEY` the second.
Reports omit prompts, response text, raw remote errors and credentials.

All three scripts support `--dry-run`: no server/data HTTP, harness import,
installation, dataset cache or output-file writes. Python stdlib is sufficient
except the unchanged harness `Renderer`, used for IF-06 and dev-pair rendering;
it needs the harness requirements already installed. Imports cannot create
bytecode in `s1-dev`. These are diagnostics, not replacements for `run_dev.py`.

## Interface checks

```bash
python3 -B scripts/if_checks.py --dry-run
python3 -B scripts/if_checks.py --base-url http://127.0.0.1:8000 \
  --same-host --flush-wait 300 --timeout 900 --out runs/if_checks.json

# L2 random weights: test mechanisms without claiming GLM ability/tokenization.
python3 -B scripts/if_checks.py --base-url http://127.0.0.1:8000 \
  --cases IF-04,IF-05,IF-08,IF-10,IF-11,IF-12 --same-host \
  --flush-wait 300 --timeout 900 --out runs/if_l2.json

# Optional preflight integration; existing quick preflight stays available.
scripts/preflight_8gpu.sh --extended-if --same-host --timeout 900 \
  --flush-server-wait 300 --out runs/preflight_extended.json
```

| Case | What the tool actually checks |
|---|---|
| IF-02 | Original AIME-style arithmetic prompt, `model=default`; nonempty `content`, final marker, nonempty `reasoning_content`. Default omits output budget. `--chat-budget N` optionally requests a known budget; `finish_reason=length` passes only if usage equals N and the final answer is still in content. Unknown/default server cap cannot justify a length finish. |
| IF-04 | All of 1, 2, 240, 4096 with `ignore_eos=true`; exact final token count, valid monotonically increasing counters, framed SSE `[DONE]`, first **positive-token** and final timestamps. Run independently for MTP off/on on T8; this script does not toggle server flags. |
| IF-05 | At least two nonempty chunks; per-chunk output ID count matches the increase in `completion_tokens` when IDs are available. Concatenated text/IDs must equal a fresh cold nonstream greedy reference. Cumulative prefixes fail. This avoids treating a legitimate repeated word as a protocol error. Nondeterministic reference outputs also fail and require investigation; a one-chunk stream is blocked. |
| IF-06 | Seed 19, 20 unique random serving dev requests, unmodified `Renderer.render` including tools/history. Both local renderer token count and server `prompt_tokens` must equal frozen `glm_tokens`. No context cap, truncation or resampling of long requests. Requires GLM tokenizer/context capacity, so random L2 does not establish this case. |
| IF-08 | Fresh long SSE, first token observed while in flight; zero-timeout flush must be 400 + false. A second flush with positive timeout starts before completion and must wait, return 200 + true; re-probe reports zero cached tokens. Default `--long-tokens 4096`; increase if stream completes before overlap can be tested. Cross-socket terminal delivery permits 50 ms, not additional generation. |
| IF-10 | Unknown `X-S1-T19-Unknown` and `X-S1-Future-Field` on both generate and chat; HTTP 200, valid generate SSE and chat choices. |
| IF-11 | Complete a request twice with the same explicit rid; disconnect the third stream after a nonfinal token, then reuse the exact rid. Only duplicate-rid 400/409 can retry during `--cleanup-timeout` (default 5 s); no flush, abort or normal-completion wait masks the disconnect behavior. Returned IDs must match. |
| IF-12 | `--same-host` explicitly confirms a direct client using the same OS clock. Wall-clock send time is taken after encoding the request and immediately before `urlopen`; receive minus send must be in [0, 50 ms]. First positive-token prefill timestamp must be at/after receive. `ttft_source=server` is derived by the same timestamp criterion as the harness, not an invented server response field. Remote/proxied timing cannot pass this case. |

Exit 0 means **every selected case passed**. Missing preconditions are `blocked`,
exit 1, never a pass. Reports preserve per-case results and continue independent
cases after a failure. A long stream is drained on cleanup; the tools never issue
`abort_all`. Long output budgets here are controlled interface probes only.

## IF-09 DP flush

VERIFIED from read-only source: `GenerateReqInput.routed_dp_rank` is in
`src/sglang/python/sglang/srt/managers/io_struct.py:286`, dispatch uses it in
`data_parallel_controller.py:752`, and `meta_info.dp_rank` is returned in
`tokenizer_manager.py:2460`. Thus controllable routing can be checked automatically:

```bash
# Supply ALL configured ranks, not an arbitrary subset. This is for DP size 2.
python3 -B scripts/if_checks.py --cases IF-09 --dp-ranks 0,1 \
  --base-url http://127.0.0.1:8000 --flush-wait 300 --timeout 900 \
  --out runs/if_dp2.json
```

Each listed rank takes a turn as the busy rank; zero-timeout aggregate flush must
fail even when rank 0 is idle. Then every rank is populated and a global flush is
followed by one probe per rank, all requiring `cached_tokens=0`. Every response
must echo the requested rank. Match the complete list to the launch configuration.

If a deployed gateway/version cannot control and verify DP routing, keep IF-09
todo/blocked and use this documented procedure after arranging rank-pinned worker
access on **our own** server:

1. Record configured DP ranks and prove the route to each using scheduler logs or
   per-worker endpoints; unknown `X-S1-*` header acceptance is not routing proof.
2. Populate a distinct long prompt twice per rank; record a positive second hit.
3. Keep exactly one rank busy with long generation and call the aggregate frontend
   `/flush_cache?timeout=0`: require HTTP 400 and `{"success":false}`. Repeat with
   each rank busy, especially a nonzero rank.
4. Let the stream finish (or use a waiting aggregate flush). Require successful
   JSON flush, then replay each rank's prompt on that same rank and require zero
   cached tokens, including host-tier caches if enabled.
5. Save rank mapping, response counters, worker evidence and launch flags. Without
   proven per-rank control/observability, do not mark IF-09 pass.

## D1-07 CPU method tests

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover \
  -s scripts -p test_role_boundary_split.py -v
```

The test compiles the actual `_PrefillAdmission` and
`PrefillAdder._maybe_role_boundary_split` AST from `build/d1/b/.../schedule_policy.py`,
and the actual `mamba_checkpoint_grid` helper from the read-only source. It checks
method AST parity with the patch. It does not translate the method into another
implementation or import torch. Fakes cover only request/adder/cache fields,
configured model grid and boundary IDs. Fifteen tests include all requested skips,
scan-window edge, last boundary, grid LCM and a normal split; the request's sampling
budget stays 240 while the partial admission's budget becomes 0. This establishes
method behavior, not scheduler integration or state restoration.

## D1-04 normalized logprob comparison

```bash
python3 -B scripts/logits_check.py --dry-run
# Pair index 0 means requests 0 and 1 of that original dev chain.
python3 -B scripts/logits_check.py --base-url http://127.0.0.1:8000 \
  --chain-id CHAIN_ID --pair-index 0 --calibrate --out runs/cold_noise.json
python3 -B scripts/logits_check.py --base-url http://127.0.0.1:8000 \
  --chain-id CHAIN_ID --pair-index 0 --expected-cached-tokens B \
  --out runs/d1_pair.json
# Alternatively: --pair-file /sjtu/linhang/arena/runs/pair.json
# Format: {"previous":"full rendered prompt", "next":"full rendered prompt"}
```

Select at least **three distinct adjacent pairs from `cases/reminder_heavy.json`**
for D1-04. `CHAIN_ID`/`B` above are placeholders, not runnable fixture data. B is
the independently determined aligned role boundary checkpoint depth; require the
warm `cached_tokens` to equal B and corroborate D1 `taken`/scheduler diagnostics.
Without `--expected-cached-tokens`, the tool requires a positive warm hit and
reports `checkpoint_depth_verified=false`: an ordinary cached chunk is insufficient
evidence that D1's boundary checkpoint was restored.

The tool flushes before every cold run and every previous→next sequence, checks
cold counts are zero, warms using the **previous full prompt**, then measures next.
The default previous decode is 32 tokens (`--previous-tokens` can reproduce the
chosen mechanism probe); the next decode is greedy 32 with ignore_eos. Fresh
discovery runs form the union of all observed first-token top-k IDs; fresh cold
twice and warm runs request logprobs for those same IDs. This handles rank crossings
without silently comparing only the intersection. Calibration-only makes four cold
runs (two discovery, two final measurement), no warm sequence. Normal mode makes
six next-prompt runs and two previous-prompt runs per server.

VERIFIED parameter names in `io_struct.py:226`: `return_logprob`,
`logprob_start_len=-1`, `top_logprobs_num`, `token_ids_logprob`,
`return_text_in_logprobs=false`. Response fields checked against
`tokenizer_manager.py:2759`: `output_token_logprobs`, `output_top_logprobs`,
`output_token_ids_logprobs`. The API exposes **normalized logprobs, not full raw
logits**. Max absolute error only covers the observed union; this limitation is
in every report. Default tolerance is exactly 2× measured cold/cold error. An
explicit `--tolerance-floor` is recorded and should need separate justification.
Cold repeats and cold/warm must have identical 32 token IDs.

With `--other-url`, both servers are measured independently, then cold and warm
outputs are compared across servers using the larger calibrated tolerance. Use
`--expected-cached-tokens` for the first server and
`--other-expected-cached-tokens` for the second; stock may restore a different depth.
Both must support D0 JSON flush. Greedy instability fails instead of being hidden
by increased numeric tolerance. Full-logit/KDA internal-state equality still needs
an instrumented engine; this API diagnostic cannot establish that stronger claim.

## CAP-01/02 spot checks and sources

```bash
python3 -B scripts/cap_spot_check.py --dry-run
python3 -B scripts/cap_spot_check.py --base-url http://127.0.0.1:8000 \
  --candidate-url http://127.0.0.1:8001 --suite aime --count 10 \
  --out runs/cap_aime.json
python3 -B scripts/cap_spot_check.py --base-url http://127.0.0.1:8000 \
  --candidate-url http://127.0.0.1:8001 --suite gpqa --count 20 \
  --out runs/cap_gpqa.json
```

Always model=default, temperature=0, nonstream chat; **no max_tokens,
max_completion_tokens, reasoning/thinking override or output clamp**. Both servers
see the same full question and deterministic option permutation; server order
alternates. Only content is used for answer extraction, never reasoning_content.
An explicit final marker or a bare whole-response answer is needed. Empty content,
unextractable answer or length finish fails the interface check. Candidate correct
count must be at least stock−1. A one-server run is diagnostic only, exits 1 and
reports `comparison_available=false`. A sample below 10 AIME/20 GPQA also exits 1
with `registry_sample_complete=false`. This is not the official >90 ability gate.

Public source probes on 2026-09-22 (metadata and content read in memory; no full
datasets saved in the repo):

`evidence/T19/public_sources.json` records HTTP status, byte sizes, source hashes and
successful in-memory parsing of 10 AIME / 20 GPQA questions using the actual loader.

- [AIME 2024 dataset](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024): 30 rows,
  source metadata SHA `8d88b2876a82a080e2f172cc9b25d0d9d2cb4792`. The
  [hf-mirror dataset API](https://hf-mirror.com/api/datasets/Maxwell-Jia/AIME_2024)
  returns 308 to HF, then 200 here; helper handles Python 3.10's missing 308 support.
  Runtime reads JSON rows from HF datasets-server; exact response SHA256 is recorded
  because that viewer endpoint is not revision-pinned. Metadata SHA is provenance,
  not a claim that the viewer snapshot is pinned to it.
- [Original GPQA repository](https://huggingface.co/datasets/Idavidrein/gpqa) reports
  `gated=auto`. [OpenAI simple-evals](https://github.com/openai/simple-evals/blob/main/gpqa_eval.py)
  explicitly uses the [public Diamond CSV](https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv),
  reachable with HTTP 200 (1,373,492 bytes). Runtime fetches that public source and
  records SHA256, sample IDs and option permutations. No gated authentication is attempted.
- Runtime downloads are confined to `/sjtu/linhang/arena/cache/t19` (or another
  subdirectory of that cache); none are vendored. No automatic package installs.
  Existing cache files are reused, with their current hashes reported.
- If network/data access later fails, `--suite tiny` explicitly uses two short
  rewordings of public AIME 2024 I-2 (answer 25) and II-11 (answer 601), cited to the
  AIME source above. It is labeled `AIME-smoke-only`, cannot satisfy CAP-01/02 and
  never silently substitutes for GPQA.

## CPU HTTP tests

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover \
  -s scripts -p test_t19_tools.py -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts python3 -B -m unittest \
  test_t16_tools.PreflightTests -v
python3 scripts/check_records.py
```

Real local HTTP mock sockets exercise SSE framing, concurrent flush waits,
disconnect cleanup, all script CLIs/dry runs and malformed/incorrect responses.
Renderer/dataset inputs are lightweight fixtures for these CPU tests; only a
future live GLM run can pass IF-06, D1-04 or CAP rows in the registry.
