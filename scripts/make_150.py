#!/usr/bin/env python3
"""Generate 150 from the exact read-only base plus the complete assigned stack."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'build/p150'
PATCHES = ['000-interface-compliance', '101-d1v12-on-base',
           '105-role-split-single-partial', '110-sm80-dsa-indexer',
           '111-sm80-fp8-moe-marlin', '112-sm80-indexer-kernels',
           '113-sm80-prefill-indexer', '140-kda-dual-snapshot',
           '120-sched-protect-chain', '130-async-tokenize']
FILES = ['srt/entrypoints/ax_shapes.py', 'srt/entrypoints/warmup.py',
         'srt/managers/io_struct.py', 'srt/managers/tokenizer_control_mixin.py',
         'srt/managers/scheduler_components/flush_wrapper.py', 'srt/managers/scheduler.py',
         'srt/observability/request_metrics_exporter.py']


def main():
    before, after = WORK / 'baseline/sglang', WORK / 'candidate/sglang'
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / '.gitignore').write_text('*\n')
    for dest in (before, after):
        if dest.exists():
            shutil.rmtree(dest)
    shutil.copytree(ROOT / 'build/base_exact/sglang', before)
    logs = []
    for name in PATCHES:
        p = ROOT / 'patches' / (name + '.patch')
        r = subprocess.run(['patch', '-p3', '--fuzz=0', '--batch', '--no-backup-if-mismatch', '-i', str(p)],
                           cwd=before, text=True, capture_output=True, check=True)
        logs.append(r.stdout)
    shutil.copytree(before, after)
    (after / FILES[0]).write_bytes((ROOT / 'scripts/p150/ax_shapes.py').read_bytes())
    def edit(rel, old, new):
        p = after / rel
        s = p.read_text()
        assert s.count(old) == 1, (rel, old)
        p.write_text(s.replace(old, new))
    p = after / FILES[1]
    p.write_text(p.read_text() + '\n\n@warmup("ax_shapes")\nasync def ax_shapes(disaggregation_mode: str, tokenizer_manager: TokenizerManager):\n    from sglang.srt.entrypoints.ax_shapes import run\n\n    await run(disaggregation_mode, tokenizer_manager)\n')
    edit(FILES[2], 'class FlushCacheReqInput(BaseReq, kw_only=True):\n    timeout_s: Optional[float] = None',
         'class FlushCacheReqInput(BaseReq, kw_only=True):\n    timeout_s: Optional[float] = None\n    verify_empty: bool = False')
    edit(FILES[3], 'self: TokenizerManager, timeout_s: Optional[float] = None\n    ) -> FlushCacheReqOutput:',
         'self: TokenizerManager, timeout_s: Optional[float] = None,\n        verify_empty: bool = False,\n    ) -> FlushCacheReqOutput:')
    edit(FILES[3], 'FlushCacheReqInput(timeout_s=timeout_s)',
         'FlushCacheReqInput(timeout_s=timeout_s, verify_empty=verify_empty)')
    edit(FILES[4], '        ipc_channels: SchedulerIpcChannels,',
         '        ipc_channels: SchedulerIpcChannels,\n        verify_empty: Optional[Callable[[bool], None]] = None,')
    edit(FILES[4], '        self._flush_cache = flush_cache',
         '        self._flush_cache = flush_cache\n        self._verify_empty = verify_empty')
    insert = '''    def _flush(self, req: FlushCacheReqInput) -> FlushCacheReqOutput:
        success = self._flush_cache()
        if req.verify_empty:
            try:
                if self._verify_empty is None:
                    raise RuntimeError("No pool verifier installed")
                self._verify_empty(success)
            except Exception as exc:
                logging.exception("Verified cache flush failed")
                return FlushCacheReqOutput(success=False, message=str(exc))
        return FlushCacheReqOutput(success=success)

'''
    edit(FILES[4], '    def handle(self, recv_req:', insert + '    def handle(self, recv_req:')
    p = after / FILES[4]
    p.write_text(p.read_text().replace('FlushCacheReqOutput(success=self._flush_cache())', 'self._flush(recv_req)'))
    edit(FILES[4], '            success = self._flush_cache()\n            self._pending = None',
         '            result = self._flush(pending_req)\n            self._pending = None')
    edit(FILES[4], '                FlushCacheReqOutput(success=success), pending_req',
         '                result, pending_req')
    edit(FILES[5], '        self.flush_wrapper = SchedulerFlushWrapper(',
         '        from sglang.srt.entrypoints.ax_shapes import verify_empty as ax_verify_empty\n\n        self.flush_wrapper = SchedulerFlushWrapper(')
    edit(FILES[5], '            flush_cache=self.flush_cache,\n            is_fully_idle=self.is_fully_idle,\n            ipc_channels=self.ipc_channels,',
         '            flush_cache=self.flush_cache,\n            is_fully_idle=self.is_fully_idle,\n            ipc_channels=self.ipc_channels,\n            verify_empty=lambda success: ax_verify_empty(self, success),')
    edit(FILES[6], '        for exporter in self._exporters:\n            await exporter.write_record(obj, out_dict)',
         '        if not obj.log_metrics:\n            return\n        for exporter in self._exporters:\n            await exporter.write_record(obj, out_dict)')
    diff = []
    for rel in FILES:
        old = (before / rel).read_text().splitlines(True) if (before / rel).exists() else []
        diff.extend(difflib.unified_diff(old, (after / rel).read_text().splitlines(True),
                    'a/python/sglang/' + rel if old else '/dev/null', 'b/python/sglang/' + rel))
    patch = ROOT / 'patches/150-startup-warmup.patch'
    patch.write_text(''.join(diff))
    ev = ROOT / 'evidence/T46'
    ev.mkdir(parents=True, exist_ok=True)
    (ev / 'generate_apply.log').write_text(''.join(logs))
    (ev / 'generate_receipt.json').write_text(json.dumps({
        'baseline_patches': PATCHES, 'patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest(),
        'files': FILES}, indent=2) + '\n')
    print('Generated', patch.relative_to(ROOT))


if __name__ == '__main__':
    main()
