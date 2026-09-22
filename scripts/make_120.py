#!/usr/bin/env python3
"""Rebuild 120 on a COPY of base_exact + 000,101,110,111; no GPU/network.

Creates build/p120/{baseline,candidate}/sglang and a deterministic -p3 patch.
Every edit has an exact single anchor. Read-only inputs are never modified.
"""
from pathlib import Path
import difflib
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'build/p120'
PATCHES = ['000-interface-compliance', '101-d1v12-on-base',
           '110-sm80-dsa-indexer', '111-sm80-fp8-moe-marlin']
FILES = ['srt/managers/schedule_policy.py', 'srt/managers/scheduler.py']


def replace(s, old, new):
    assert s.count(old) == 1, (old[:100], s.count(old))
    return s.replace(old, new, 1)


def edit_policy(s):
    s = replace(s, 'class AddReqResult(Enum):', '''@lru_cache(maxsize=1)
def _ax_sched_protect_config():
    """120: process-start knobs; off does not parse or validate tuning knobs."""
    if os.environ.get("SGLANG_AX_SCHED_PROTECT", "1") == "0":
        return None
    cap = int(os.environ.get("SGLANG_AX_SCHED_COLD_CAP", "2048"))
    short = int(os.environ.get("SGLANG_AX_SCHED_SHORT_TOKENS", "4096"))
    if cap <= 0 or short <= 0:
        raise ValueError("AX scheduler token limits must be positive")
    return cap, short


class AddReqResult(Enum):''')
    s = replace(s, '        prefill_tile_block_m: int = 64,\n',
                '        prefill_tile_block_m: int = 64,\n        ax_protect: Optional[tuple] = None,\n')
    s = replace(s, '        self.page_size = page_size\n', '''        self.page_size = page_size
        # (aligned cold cap, short-hit threshold, checkpoint/alignment grid).
        # Only the normal TP scheduler opts in; other callers retain stock.
        self.ax_protect = ax_protect
        self.ax_continuation = None
''')
    s = replace(s, '    def add_chunked_req(self, req: Req):\n', '''    def _ax_short_hit(self, req: Req) -> bool:
        return (
            len(req.prefix_indices) > 0
            and not req.needs_host_load_back()
            and 0 < len(req.full_untruncated_fill_ids) - len(req.prefix_indices)
            <= self.ax_protect[1]
        )

    def add_chunked_req(self, req: Req):
''')
    s = replace(s, '        # A mid-chunk rank prefills this pass regardless of the delayer\n', '''        if self.ax_protect is not None:
            self.ax_continuation = req
            cap, _, grid = self.ax_protect
            _rem_tokens = min(_rem_tokens, cap, self.rem_input_tokens)
            # Preserve checkpoint, KV-page and DSA/deterministic alignment.
            # Below one grid unit retain the native resource-limited progress;
            # the no-starvation bound assumes room for at least one unit.
            if _rem_tokens >= grid:
                _rem_tokens = _rem_tokens // grid * grid

        # A mid-chunk rank prefills this pass regardless of the delayer
''')
    s = replace(s, '        # TODO support cp with multiple requests\n', '''        if self.ax_protect is not None and (
            self.ax_continuation is not None or self.new_chunked_req is not None
        ):
            # Reserve/charge the active chunk FIRST (including its request row).
            # Spend the rest only on complete device-cache short hits, in LPM
            # order. No host reload, second partial, or budget overshoot.
            needed = self.ceil_paged_tokens(
                len(req.full_untruncated_fill_ids) - len(req.prefix_indices)
            )
            if not self._ax_short_hit(req) or needed > min(
                self.rem_chunk_tokens, self.rem_input_tokens
            ):
                return AddReqResult.OTHER

        # TODO support cp with multiple requests
''')
    s = replace(s, '            if self.dllm_config is not None:\n                if self.rem_dllm_tokens <= 0:\n', '''            if self.ax_protect is not None:
                cap, _, grid = self.ax_protect
                chunk_tokens_limit = min(chunk_tokens_limit, self.rem_input_tokens)
                if not self._ax_short_hit(req):
                    chunk_tokens_limit = min(chunk_tokens_limit, cap)
                if input_tokens > chunk_tokens_limit:
                    chunk_tokens_limit = chunk_tokens_limit // grid * grid

            if self.dllm_config is not None:
                if self.rem_dllm_tokens <= 0:
''')
    s = replace(s, '                # [ax] at most one partial prefill per round while the role split is on\n', '''                if self.ax_protect is not None and (
                    has_chunked_req or self.new_chunked_req is not None
                ):
                    return AddReqResult.OTHER
                # [ax] at most one partial prefill per round while the role split is on
''')
    return s


def edit_scheduler(s):
    s = replace(s, '    SchedulePolicy,\n', '    SchedulePolicy,\n    _ax_sched_protect_config,\n')
    # mamba_checkpoint_grid is shared with 101 rather than a duplicated constant.
    s = replace(s, '    def _should_defer_prefill(self) -> bool:\n', '''    def _ax_sched_protect_enabled(self) -> bool:
        # 120 is scoped to ordinary TP serving. Specialized schedulers keep
        # their collective/cadence/admission contracts (including mixed mode).
        return (
            _ax_sched_protect_config() is not None
            and self.chunked_prefill_size is not None
            and self.ps.pp_size == 1
            and not self.require_mlp_sync
            and self.disaggregation_mode == DisaggregationMode.NULL
            and self.dllm_config is None
            and not self.is_mixed_chunk
            and not self.is_hybrid_swa
            and not self.enable_hisparse
            and not self.enable_hierarchical_cache
            and not get_memory().enable_flexkv
            and not self.enable_lora
            and not self.enable_priority_preemption
            and self.prefill_delayer is None
        )

    def _ax_sched_protect_limits(self, chunk_size):
        if not self._ax_sched_protect_enabled():
            return None
        from sglang.srt.runtime_context import mamba_checkpoint_grid

        cap, short = _ax_sched_protect_config()
        grid = math.lcm(self.page_size, self.truncation_align_size or 1)
        if self.tree_cache.supports_mamba():
            grid = math.lcm(grid, mamba_checkpoint_grid(self.tree_cache.page_size))
        budget = min(chunk_size, self.max_prefill_tokens)
        if budget < grid:
            return None
        # A user cap below one grid unit is rounded UP to permit progress.
        cap = max(grid, cap // grid * grid)
        return min(cap, budget // grid * grid), short, grid

    def _ax_should_decode(self, running_batch: ScheduleBatch) -> bool:
        if not self._ax_sched_protect_enabled() or self.prefill_decode_interval:
            return False
        due = getattr(self, "_ax_decode_due", False)
        self._ax_decode_due = False
        # Called AFTER last extend is merged, so newly completed short requests
        # count as running. With no decoders, immediately continue cold prefill.
        return due and not running_batch.is_empty() and not running_batch.is_prefill_only

    def _should_defer_prefill(self) -> bool:
''')
    s = replace(s, '        elif self._should_defer_prefill():\n', '''        elif self._should_defer_prefill() or self._ax_should_decode(running_batch):
''')
    s = replace(s, '        self._arm_prefill_decode_interval(ret)\n', '''        self._arm_prefill_decode_interval(ret)
        if self._ax_sched_protect_enabled() and self.prefill_decode_interval == 0:
            self._ax_decode_due = ret is not None and ret.forward_mode.is_extend()
''')
    s = replace(s, '            prefill_tile_block_m=prefill_tile_block_m,\n', '''            prefill_tile_block_m=prefill_tile_block_m,
            ax_protect=self._ax_sched_protect_limits(chunked_prefill_size),
''')
    s = replace(s, '                        req.kv.mamba_pool_idx = None\n                break\n', '''                        req.kv.mamba_pool_idx = None
                if (
                    adder.ax_protect is not None
                    and not added
                    and res == AddReqResult.OTHER
                    and (adder.ax_continuation is not None or adder.new_chunked_req is not None)
                ):
                    # A long/non-fitting waiter must not hide a short hit. Keep
                    # the native rejection cleanup above, and keep LPM order.
                    continue
                break
''')
    return s


def main():
    WORK.mkdir(exist_ok=True)
    baseline = WORK / 'baseline/sglang'
    candidate = WORK / 'candidate/sglang'
    # Only replace directories owned by this reproducible generator.
    for name in ('baseline', 'candidate'):
        path = WORK / name
        if path.exists():
            shutil.rmtree(path)
    shutil.copytree(ROOT / 'build/base_exact/sglang', baseline)
    logs = []
    for name in PATCHES:
        proc = subprocess.run(
            ['patch', '-p3', '--fuzz=0', '--batch', '-i', str(ROOT / 'patches' / (name + '.patch'))],
            cwd=baseline, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        logs.append(name + '\n' + proc.stdout)
        if proc.returncode:
            raise RuntimeError(logs[-1])
    shutil.copytree(baseline, candidate)
    patch = []
    for rel, edit in zip(FILES, (edit_policy, edit_scheduler)):
        original = (baseline / rel).read_text()
        modified = edit(original)
        (candidate / rel).write_text(modified)
        patch.extend(difflib.unified_diff(original.splitlines(True), modified.splitlines(True),
                     fromfile='a/python/sglang/' + rel, tofile='b/python/sglang/' + rel))
    target = ROOT / 'patches/120-sched-protect-chain.patch'
    target.write_text(''.join(patch))
    evidence = ROOT / 'evidence/T41'
    evidence.mkdir(exist_ok=True)
    (evidence / 'baseline_apply.log').write_text('\n'.join(logs))
    receipt = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in [ROOT / 'patches' / (n + '.patch') for n in PATCHES] + [target]}
    (evidence / 'patch_sha256.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'Generated {target.relative_to(ROOT)} ({len(patch)} lines)')


if __name__ == '__main__':
    main()
