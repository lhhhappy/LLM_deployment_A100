#!/usr/bin/env python3
"""Generate 140 from read-only base_exact + 000/101/105/110/111/112, fuzz=0.

Support source files live in scripts/p140/. No image builds or service actions.
"""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'build/p140'
PATCHES = ['000-interface-compliance', '101-d1v12-on-base',
           '105-role-split-single-partial', '110-sm80-dsa-indexer',
           '111-sm80-fp8-moe-marlin', '112-sm80-indexer-kernels']
CHANGED = []


def edit(rel, old, new, count=1):
    path = WORK / 'candidate/sglang' / rel
    s = path.read_text()
    assert s.count(old) == count, (rel, old[:100], s.count(old), count)
    path.write_text(s.replace(old, new))
    if rel not in CHANGED:
        CHANGED.append(rel)


def main():
    CHANGED.clear()
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / '.gitignore').write_text('*\n')
    before, after = WORK / 'baseline/sglang', WORK / 'candidate/sglang'
    for path in (before, after):
        if path.exists():
            shutil.rmtree(path)
    shutil.copytree(ROOT / 'build/base_exact/sglang', before)
    logs = []
    for patch in PATCHES:
        r = subprocess.run(['patch', '-p3', '--fuzz=0', '--batch', '--no-backup-if-mismatch',
                            '-i', str(ROOT / 'patches' / (patch + '.patch'))],
                           cwd=before, check=True, text=True, capture_output=True)
        logs.append(r.stdout)
    shutil.copytree(before, after)
    for src, rel in [
        ('kda_dual_snapshot.py', 'srt/mem_cache/kda_dual_snapshot.py'),
        ('kda_snapshot.py', 'kernels/ops/attention/fla/kda_snapshot.py'),
    ]:
        (after / rel).write_bytes((ROOT / 'scripts/p140' / src).read_bytes())
        CHANGED.append(rel)

    p = 'srt/mem_cache/unified_radix_cache.py'
    edit(p, '        self.reset()\n        logger.info(', '''        from sglang.srt.mem_cache.kda_dual_snapshot import configure

        self.ax_kda_dual_snapshot = configure(self, params)
        self.reset()
        logger.info(''')
    # Both finished and unfinished insert paths, before row rematch/cleanup.
    for indent in ('            ', '        '):
        edit(p, '\n' + indent + 'result = self.insert(insert_params)\n',
             '\n' + indent + 'result = self.insert(insert_params)\n' + indent + '''if self.ax_kda_dual_snapshot:
''' + indent + '''    from sglang.srt.mem_cache.kda_dual_snapshot import commit

''' + indent + '''    commit(self, req, insert_params, result)
''')

    p = 'srt/managers/schedule_policy.py'
    # Keep 120's hunk contexts intact: only change the helper entry and the
    # role-id helper. The latter disables 101 AND 105 admission guards when on.
    edit(p, '    raw = os.environ.get("SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS", "").strip()', '''    if os.environ.get("SGLANG_AX_KDA_DUAL_SNAPSHOT", "0") == "1":
        return frozenset()
    raw = os.environ.get("SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS", "").strip()''')

    p = 'srt/managers/schedule_batch.py'
    edit(p, '    # Deferred COW: source mamba pool index from radix cache node', '''    # 140: one optional slot owned by this request until commit/discard.
    ax_kda_snapshot_slot: Optional[torch.Tensor] = None
    ax_kda_snapshot_depth: Optional[int] = None
    ax_kda_role_depth: int = 0
    # Deferred COW: source mamba pool index from radix cache node''')
    edit(p, '    mamba_track_seqlens: torch.Tensor = None  # shape: [b], int64', '''    mamba_track_seqlens: torch.Tensor = None  # shape: [b], int64
    ax_kda_dual_snapshot_batch: bool = False
    ax_kda_snapshot_offsets: Optional[torch.Tensor] = None  # [b, 2], relative tokens
    ax_kda_snapshot_slots: Optional[torch.Tensor] = None  # [b, 2], physical slots''')
    edit(p, '        mamba_track_seqlens_cpu = []\n', '''        mamba_track_seqlens_cpu = []
        if getattr(self.tree_cache, "ax_kda_dual_snapshot", False):
            from sglang.srt.mem_cache.kda_dual_snapshot import batch_supported

            self.ax_kda_dual_snapshot_batch = batch_supported(
                self, mamba_checkpoint_grid(self.tree_cache.page_size)
            )
''')
    edit(p, '            if req.mamba_branching_seqlen is not None:\n', '''            if req.mamba_branching_seqlen is not None and not self.ax_kda_dual_snapshot_batch:
''')
    edit(p, '        # Collect mamba init info for deferred ops on forward stream\n', '''        if self.ax_kda_dual_snapshot_batch:
            from sglang.srt.mem_cache.kda_dual_snapshot import prepare

            prepare(self, mamba_track_indices_cpu, mamba_track_mask_cpu,
                    mamba_checkpoint_grid(self.tree_cache.page_size))

        # Collect mamba init info for deferred ops on forward stream
''')
    # A reused running batch must not carry extend-only destinations into decode.
    edit(p, '        self.mamba_track_seqlens = None\n', '''        self.mamba_track_seqlens = None
        self.ax_kda_snapshot_offsets = None
        self.ax_kda_snapshot_slots = None
        self.ax_kda_dual_snapshot_batch = False
''', count=2)
    p = 'srt/model_executor/forward_batch_info.py'
    edit(p, '    mamba_track_seqlens: Optional[torch.Tensor] = None  # shape: [b], int64', '''    mamba_track_seqlens: Optional[torch.Tensor] = None  # shape: [b], int64
    ax_kda_snapshot_offsets: Optional[torch.Tensor] = None
    ax_kda_snapshot_slots: Optional[torch.Tensor] = None''')
    edit(p, '            mamba_track_seqlens=batch.mamba_track_seqlens,', '''            mamba_track_seqlens=batch.mamba_track_seqlens,
            ax_kda_snapshot_offsets=batch.ax_kda_snapshot_offsets,
            ax_kda_snapshot_slots=batch.ax_kda_snapshot_slots,''')

    edit(p, '        if self.mamba_track_seqlens is not None:\n', '''        if self.ax_kda_snapshot_offsets is not None:
            self.ax_kda_snapshot_offsets = self._pad_tensor_to_size(
                self.ax_kda_snapshot_offsets, bs, value=-1
            )
            self.ax_kda_snapshot_slots = self._pad_tensor_to_size(
                self.ax_kda_snapshot_slots, bs, value=-1
            )
        if self.mamba_track_seqlens is not None:
''')

    p = 'srt/mem_cache/memory_pool.py'
    edit(p, '        mamba_index = req.kv.mamba_pool_idx\n        assert mamba_index is not None, "double free? mamba_index is None"', '''        if req.kv.ax_kda_snapshot_slot is not None:
            from sglang.srt.mem_cache.kda_dual_snapshot import discard

            discard(self, req)
        mamba_index = req.kv.mamba_pool_idx
        assert mamba_index is not None, "double free? mamba_index is None"''')

    p = 'srt/mem_cache/unified_cache/components/mamba_component.py'
    edit(p, '        assert params.mamba_value is not None\n', '''        assert params.mamba_value is not None
        if getattr(self.cache, "ax_kda_dual_snapshot", False):
            # Role wins permanently at this exact depth; split parents start
            # untagged, while the old child's depth and tag stay unchanged.
            node.ax_kda_role = getattr(node, "ax_kda_role", False) or getattr(params, "ax_kda_role", False)
            node.ax_kda_tail = not node.ax_kda_role and (
                getattr(node, "ax_kda_tail", False) or getattr(params, "ax_kda_tail", False)
            )
''')
    edit(p, '        for node in reversed(holders):\n            if excess <= 0 or node is tail:\n                break', '''        ordered = list(reversed(holders))
        if getattr(self.cache, "ax_kda_dual_snapshot", False):
            ordered.sort(key=lambda n: (getattr(n, "ax_kda_role", False),
                                        not getattr(n, "ax_kda_tail", False)))
        for node in ordered:
            if excess <= 0:
                break
            if node is tail:
                continue''')
    edit(p, '        enabled = self.tree_core.enable_session_radix_cache\n        if self._evict_device_cursor', '''        enabled = self.tree_core.enable_session_radix_cache
        if getattr(self.cache, "ax_kda_dual_snapshot", False):
            from sglang.srt.mem_cache.kda_dual_snapshot import eviction_candidate

            self._evict_device_cursor = eviction_candidate(lru, ct)
        if self._evict_device_cursor''')
    edit(p, '        if is_finished:\n            if cache_len is None:', '''        if getattr(self.cache, "ax_kda_dual_snapshot", False):
            role = req.kv.ax_kda_role_depth
            insert_params.ax_kda_role = bool(role and cache_len == role)
            insert_params.ax_kda_tail = bool(role and cache_len and cache_len > role)

        if is_finished:
            if cache_len is None:''')

    p = 'srt/mem_cache/unified_cache/components/full_component.py'
    edit(p, '    def _evict_device_start(self, request_cnt: int) -> None:\n', '''    def _ax_eviction_priority(self, node):
        key = self.session_ref_eviction_strategy(node)
        if getattr(self.cache, "ax_kda_dual_snapshot", False):
            return (not getattr(node, "ax_kda_tail", False), key)
        return key

    def _evict_device_start(self, request_cnt: int) -> None:
''')
    edit(p, '(self.session_ref_eviction_strategy(n), n)\n            for n in self.tree_core.evictable_device_leaves', '(self._ax_eviction_priority(n), n)\n            for n in self.tree_core.evictable_device_leaves')
    edit(p, '(self.session_ref_eviction_strategy(lv.parent), lv.parent)',
         '(self._ax_eviction_priority(lv.parent), lv.parent)')

    p = 'srt/layers/attention/linear/kda_backend.py'
    edit(p, '        kernel = self.extend_kernel\n', '''        kernel = self.extend_kernel
        if kwargs.get("snapshot_offsets") is not None:
            kernel = self.triton_kernel
''')
    edit(p, '        if self.forward_metadata.has_mamba_track_mask:\n            # Snapshot the conv', '''        snapshot_offsets = forward_batch.ax_kda_snapshot_offsets
        snapshot_slots = forward_batch.ax_kda_snapshot_slots
        if snapshot_offsets is not None:
            from sglang.kernels.ops.attention.fla.kda_snapshot import store_conv

            assert ssm_states.dtype == torch.float32
            store_conv(mixed_qkv, mamba_cache_params.conv[0], query_start_loc,
                       snapshot_offsets, snapshot_slots)
        if self.forward_metadata.has_mamba_track_mask and snapshot_offsets is None:
            # Snapshot the conv''')
    edit(p, '        track_ssm = self.forward_metadata.has_mamba_track_mask\n',
         '        track_ssm = self.forward_metadata.has_mamba_track_mask and snapshot_offsets is None\n')
    edit(p, '            return_intermediate_states=track_ssm,\n', '''            return_intermediate_states=track_ssm,
            snapshot_offsets=snapshot_offsets,
            snapshot_slots=snapshot_slots,
''')
    p = 'srt/layers/attention/linear/kernels/kda_triton.py'
    edit(p, '            output_intermediate_states=return_intermediate_states,\n', '''            output_intermediate_states=return_intermediate_states,
            snapshot_offsets=kwargs.get("snapshot_offsets"),
            snapshot_slots=kwargs.get("snapshot_slots"),
''')
    p = 'kernels/ops/attention/fla/kda.py'
    edit(p, 'from typing import Optional\n', 'import os\nfrom typing import Optional\n')
    edit(p, '    _small_grid = _B * _NT_pr * _H_pr <= 256\n', '''    _small_grid = _B * _NT_pr * _H_pr <= 256
    # 140: prefix length must not change intra-chunk arithmetic. The fused
    # small-grid variant rounds differently, so a long extend and its short
    # prefix otherwise disagree even with an exact fp32 state exporter.
    if os.environ.get("SGLANG_AX_KDA_DUAL_SNAPSHOT", "0") == "1":
        _small_grid = False
''')
    edit(p, '    output_intermediate_states: bool = False,\n):', '''    output_intermediate_states: bool = False,
    snapshot_offsets: Optional[torch.Tensor] = None,
    snapshot_slots: Optional[torch.Tensor] = None,
):''')
    edit(p, '        use_exp2=True,\n', '''        use_exp2=True,
        snapshot_offsets=snapshot_offsets,
        snapshot_slots=snapshot_slots,
''')
    edit(p, '        output_intermediate_states=output_intermediate_states,\n', '''        output_intermediate_states=output_intermediate_states,
        snapshot_offsets=kwargs.get("snapshot_offsets"),
        snapshot_slots=kwargs.get("snapshot_slots"),
''')
    p = 'kernels/ops/attention/fla/chunk_delta_h.py'
    edit(p, '    stride_init_state,\n    cu_seqlens,', '''    stride_init_state,
    snapshot_offsets,
    snapshot_slots,
    cu_seqlens,''')
    edit(p, '    USE_EXP2: tl.constexpr,\n):', '''    USE_EXP2: tl.constexpr,
    EXPORT_SNAPSHOTS: tl.constexpr,
):''')
    # Direct FP32 stores AFTER completing the requested chunk, before any bf16 h store.
    stores = '''
        # 140: do not round through h. The two destinations never alias an
        # active request slot; every layer writes its own pool view.
        if EXPORT_SNAPSHOTS:
            for snap in tl.static_range(2):
                depth = tl.load(snapshot_offsets + i_n * 2 + snap)
                dst = tl.load(snapshot_slots + i_n * 2 + snap).to(tl.int64)
                if valid_state and dst >= 0 and depth == (i_t + 1) * BT and depth <= T:
                    target = initial_state + dst * stride_init_state + i_h * V * K
'''
    for i in range(1, 5):
        indent = '                    '
        if i > 1:
            stores += indent + f'if K > {(i - 1) * 64}:\n'
            indent += '    '
        stores += indent + f'p_snap = tl.make_block_ptr(target, (V, K), (K, 1), (i_v * BV, {(i - 1)*64}), (BV, 64), (1, 0))\n'
        stores += indent + f'tl.store(p_snap, b_h{i}, boundary_check=(0, 1))\n'
    edit(p, '    # epilogue\n', stores + '\n    # epilogue\n')
    edit(p, '    use_exp2: bool = False,\n)', '''    use_exp2: bool = False,
    snapshot_offsets: Optional[torch.Tensor] = None,
    snapshot_slots: Optional[torch.Tensor] = None,
)''')
    edit(p, '    h = k.new_empty(B, NT, H, V, K)\n', '''    if snapshot_offsets is not None:
        assert initial_state is not None and initial_state.dtype == torch.float32
        assert snapshot_offsets.shape == snapshot_slots.shape == (N, 2)
        assert snapshot_offsets.is_contiguous() and snapshot_slots.is_contiguous()
    h = k.new_empty(B, NT, H, V, K)
''')
    edit(p, '        cu_seqlens=cu_seqlens,\n', '''        snapshot_offsets=snapshot_offsets,
        snapshot_slots=snapshot_slots,
        cu_seqlens=cu_seqlens,
''')
    edit(p, '        USE_EXP2=use_exp2,\n', '''        USE_EXP2=use_exp2,
        EXPORT_SNAPSHOTS=snapshot_offsets is not None,
''')

    diff = []
    for rel in sorted(CHANGED):
        old = (before / rel).read_text().splitlines(True) if (before / rel).exists() else []
        diff.extend(difflib.unified_diff(old, (after / rel).read_text().splitlines(True),
                    'a/python/sglang/' + rel if old else '/dev/null', 'b/python/sglang/' + rel))
    patch = ROOT / 'patches/140-kda-dual-snapshot.patch'
    patch.write_text(''.join(diff))
    ev = ROOT / 'evidence/T45'
    ev.mkdir(parents=True, exist_ok=True)
    (ev / 'generate_apply.log').write_text(''.join(logs))
    (ev / 'generate_receipt.json').write_text(json.dumps({
        'baseline_patches': PATCHES, 'changed': sorted(CHANGED),
        'patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest(),
    }, indent=2) + '\n')
    print('Generated', patch.relative_to(ROOT))


if __name__ == '__main__':
    main()
