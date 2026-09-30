# Sourced before any engine imports. Opt-in RAM workspace also owns runtime writes.
# Do not change HOME or the model/engine configuration.
if [ -n "${AX_WORKSPACE_ROOT:-}" ]; then
  python3 -B - "$AX" "$AX_WORKSPACE_ROOT" <<'PY' || return 1
import json, pathlib, shutil, subprocess, sys, tempfile
ax = pathlib.Path(sys.argv[1]).resolve()
root = pathlib.Path(sys.argv[2]).resolve()
assert ax == root / 'ax', 'workspace has not been bound to the requested RAM directory'
mounts = []
for line in pathlib.Path('/proc/self/mountinfo').read_text().splitlines():
    left, right = line.split(' - ', 1)
    fields = left.split()
    mount = pathlib.Path(fields[4].replace('\\040', ' '))
    if mount == ax or mount in ax.parents:
        mounts.append((len(mount.parts), mount, right.split()[0], fields[5]))
_, mount, fs, flags = max(mounts)
assert fs == 'tmpfs' and 'noexec' not in flags.split(','), 'RAM workspace must allow compiled code'
available = shutil.disk_usage(ax).free
cg = pathlib.Path('/sys/fs/cgroup')
limit = cg/'memory.max' if (cg/'memory.max').exists() else cg/'memory/memory.limit_in_bytes'
used = cg/'memory.current' if (cg/'memory.current').exists() else cg/'memory/memory.usage_in_bytes'
headroom = int(limit.read_text()) - int(used.read_text())
assert min(available, headroom) >= 64 * 1024**3, 'less than 64 GiB runtime headroom'
with tempfile.TemporaryDirectory(prefix='exec-check-', dir=ax) as temp:
    exe = pathlib.Path(temp)/'true'
    shutil.copy2('/bin/true', exe)
    subprocess.run([str(exe)], check=True, timeout=5)
# Some third-party versions ignore XDG/cache envs. Bind their conventional
# cache paths as well, preserving the small fresh-image directories intact.
aliases = []
for name in ('.cache', '.triton', '.nv', '.tilelang'):
    source = pathlib.Path('/root')/name
    target = ax/'cache'/'legacy-root'/name[1:]
    if source.is_symlink():
        assert source.resolve() == target, f'foreign cache symlink: {source}'
    else:
        backup = source.with_name(source.name+'.before-ram')
        assert not backup.exists(), f'cache backup already exists: {backup}'
        if source.exists():
            size = int(subprocess.check_output(['du', '-sx', '-B1', str(source)], timeout=10).split()[0])
            assert size <= 512 * 1024**2, f'cache needs separate migration budget: {source}'
            shutil.copytree(source, target, symlinks=True, dirs_exist_ok=True)
            source.rename(backup)
        else:
            target.mkdir(parents=True, exist_ok=True)
        source.symlink_to(target, target_is_directory=True)
    aliases.append(dict(path=str(source), target=str(target)))
print('STORAGE_READY '+json.dumps(dict(workspace=str(ax), mount=str(mount), filesystem=fs,
                                     flags=flags, tmpfs_free_bytes=available,
                                     cgroup_headroom_bytes=headroom, exec_check=True,
                                     cache_aliases=aliases)), flush=True)
PY
  # JIT dependency filtering compares resolved files against its build directory.
  # Passing /tmp/ax here makes staging cuda.cu look external through the symlink.
  _ax_runtime_path=$(readlink -f "$AX") || return 1
  export TMPDIR="$_ax_runtime_path/tmp" TMP="$_ax_runtime_path/tmp" TEMP="$_ax_runtime_path/tmp"
  export XDG_CACHE_HOME="$_ax_runtime_path/cache" PYTHONPYCACHEPREFIX="$_ax_runtime_path/cache/pycache"
  export SGLANG_CACHE_DIR="$_ax_runtime_path/cache/sglang" SGLANG_JIT_CACHE_DIR="$_ax_runtime_path/cache/sglang/jit"
  export TRITON_CACHE_DIR="$_ax_runtime_path/cache/triton" TORCHINDUCTOR_CACHE_DIR="$_ax_runtime_path/cache/inductor"
  export TORCH_EXTENSIONS_DIR="$_ax_runtime_path/cache/torch_extensions" CUDA_CACHE_PATH="$_ax_runtime_path/cache/cuda"
  export FLASHINFER_WORKSPACE_BASE="$_ax_runtime_path/cache/flashinfer" TILELANG_CACHE_DIR="$_ax_runtime_path/cache/tilelang"
  export HF_HOME="$_ax_runtime_path/cache/huggingface" HF_HUB_CACHE="$_ax_runtime_path/cache/huggingface/hub"
  export PIP_CACHE_DIR="$_ax_runtime_path/cache/pip" UV_CACHE_DIR="$_ax_runtime_path/cache/uv"
  export NUMBA_CACHE_DIR="$_ax_runtime_path/cache/numba" VLLM_CACHE_ROOT="$_ax_runtime_path/cache/vllm"
  export HUMMING_CACHE_DIR="$_ax_runtime_path/cache/humming" HUMMING_TMP_DIR="$_ax_runtime_path/tmp/humming"
  mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HUMMING_CACHE_DIR" "$HUMMING_TMP_DIR" || return 1
  ulimit -c 0
  # One short receipt. The engine inherits these paths; no change to algorithms.
  env | LC_ALL=C sort | grep -E '^(TMPDIR|TMP|TEMP|XDG_CACHE_HOME|PYTHONPYCACHEPREFIX|SGLANG_CACHE_DIR|SGLANG_JIT_CACHE_DIR|TRITON_CACHE_DIR|TORCHINDUCTOR_CACHE_DIR|TORCH_EXTENSIONS_DIR|CUDA_CACHE_PATH|FLASHINFER_WORKSPACE_BASE|TILELANG_CACHE_DIR|HF_HOME|HF_HUB_CACHE|PIP_CACHE_DIR|UV_CACHE_DIR|NUMBA_CACHE_DIR|VLLM_CACHE_ROOT|HUMMING_CACHE_DIR|HUMMING_TMP_DIR)='
fi
