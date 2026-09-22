#!/usr/bin/env python3
"""Generate 160 from the exact read-only base plus the complete assigned stack."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'build/p160'
PATCHES = ['000-interface-compliance', '101-d1v12-on-base',
           '105-role-split-single-partial', '110-sm80-dsa-indexer',
           '111-sm80-fp8-moe-marlin', '112-sm80-indexer-kernels',
           '113-sm80-prefill-indexer', '140-kda-dual-snapshot',
           '120-sched-protect-chain', '130-async-tokenize', '150-startup-warmup']
FILES = ['srt/arg_groups/ax_mtp_sm80.py', 'srt/arg_groups/speculative_hook.py',
         'srt/managers/scheduler_components/metrics_reporter.py']


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
    (after / FILES[0]).write_bytes((ROOT / 'scripts/p160/ax_mtp_sm80.py').read_bytes())
    def edit(rel, old, new):
        p = after / rel
        s = p.read_text()
        assert s.count(old) == 1, (rel, old)
        p.write_text(s.replace(old, new))
    edit(FILES[1], '    if "trtllm_mha" in attention_backends_of(resolved_view(server_args)):',
         '    from sglang.srt.arg_groups.ax_mtp_sm80 import configure\n\n'
         '    configure(server_args, model_arch)\n\n'
         '    if "trtllm_mha" in attention_backends_of(resolved_view(server_args)):')
    edit(FILES[2], '            spec_accept_length = self.spec_num_accept_tokens / self.spec_num_forward_ct',
         '            # 160: exact window weights; rounded accept len alone cannot be aggregated.\n'
         '            msg += (f"spec tokens: {self.spec_num_accept_tokens}, "\n'
         '                    f"spec rounds: {self.spec_num_forward_ct}, ")\n'
         '            spec_accept_length = self.spec_num_accept_tokens / self.spec_num_forward_ct')
    diff = []
    for rel in FILES:
        old = (before / rel).read_text().splitlines(True) if (before / rel).exists() else []
        diff.extend(difflib.unified_diff(old, (after / rel).read_text().splitlines(True),
                    'a/python/sglang/' + rel if old else '/dev/null', 'b/python/sglang/' + rel))
    patch = ROOT / 'patches/160-nextn-sm80.patch'
    patch.write_text(''.join(diff))
    ev = ROOT / 'evidence/T48'
    ev.mkdir(parents=True, exist_ok=True)
    (ev / 'generate_apply.log').write_text(''.join(logs))
    (ev / 'generate_receipt.json').write_text(json.dumps({
        'baseline_patches': PATCHES, 'patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest(),
        'files': FILES}, indent=2) + '\n')
    print('Generated', patch.relative_to(ROOT))


if __name__ == '__main__':
    main()
