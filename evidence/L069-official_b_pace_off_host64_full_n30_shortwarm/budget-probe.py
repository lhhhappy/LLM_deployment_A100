"""CPU parameter screen; execute source sizing helpers, no large allocation.

Run from the repository root. Geometry matches 068 logs. Device byte-ratio
proxy is chosen inside the observed 068 rounded-capacity interval; it is not
an independently measured device footprint. Exact new capacities need startup.
"""
import ast
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace as NS

out = Path(__file__).parent
source = Path('engine/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py')
names = {'_split_hicache_size', '_carve_declared_sidecars', '_root_config'}
tree = ast.parse(source.read_text())
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
assert len(nodes) == 3
module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
env = {'logger': logging.getLogger('probe')}
exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), env)

old_tokens = 1451904
bpt = 12288 + 1584
kv_share = (old_tokens - 32) * bpt / 1e9
pools = (NS(get_kv_size_bytes=lambda: kv_share), NS(get_kv_size_bytes=lambda: 32-kv_share))
kv = NS(layer_num=11, kv_cache_dim=512, store_dtype=NS(itemsize=2))
configs = (
    NS(decl=NS(is_layout_root=True), packed_draft_device_pools=(object(),)),
    NS(decl=NS(is_layout_root=False, owned_device_layers=None, device_pool=NS(layer_num=11), storage_info=NS(bytes_per_token_per_layer=132)), packed_draft_device_pools=(object(),)),
)
result = {}
for budget in (32, 64):
    shares = env['_split_hicache_size'](budget, pools)
    carved = env['_carve_declared_sidecars'](shares[0], kv_pool=kv, configs=configs)
    full_tokens = (int(carved * 1e9 // 12288)//64+1)*64
    result[budget] = dict(kv_sidecar_gb=shares[0], mamba_gb=shares[1], kv_only_gb=carved, full_tokens_proxy=full_tokens)
assert result[32]['full_tokens_proxy'] == old_tokens
assert all(result[64][k] == 2*result[32][k] for k in ('kv_sidecar_gb','mamba_gb','kv_only_gb'))
old = Path('scripts/pod/jobs/official_b_pace_off_full_n30_shortwarm.sh').read_text()
new = Path('scripts/pod/jobs/official_b_pace_off_host64_full_n30_shortwarm.sh').read_text()
strip = lambda s: '\n'.join(l for l in s.splitlines() if l.strip() and not l.lstrip().startswith('#'))
assert strip(new) == strip(old).replace('--hicache-size 32','--hicache-size 64')
memory = json.loads(Path('evidence/L068-official_b_pace_off_full_n30_shortwarm/host-memory-audit.txt').read_text().splitlines()[0])
limit = int(memory['cgroup']['/sys/fs/cgroup/memory/memory.limit_in_bytes'])
current = int(memory['cgroup']['/sys/fs/cgroup/memory/memory.usage_in_bytes'])
reserve = 64*2**30
assert current + 64*8*10**9 + reserve < limit
assert 64*8*10**9 + reserve < memory['meminfo']['MemAvailable']
receipt = dict(status='PASS_CPU_PARAMETER_SCREEN', source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              single_job_variable='--hicache-size 32 -> 64 decimal GB/rank', tp=8,
              incremental_host_budget_bytes=32*8*10**9, total_new_host_budget_bytes=64*8*10**9,
              cgroup_limit_bytes=limit, cgroup_current_bytes=current, conservative_reserve_bytes=reserve,
              source_helper_probe=result, expected_full_host_tokens_inclusive=[2*old_tokens-64,2*old_tokens],
              limitations='Source sizing helpers with pool proxies; no allocation or performance measurement. Memory is a snapshot. Startup must verify both host pools and unchanged device KV=1397760, Mamba=418.')
(out/'budget-probe.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
