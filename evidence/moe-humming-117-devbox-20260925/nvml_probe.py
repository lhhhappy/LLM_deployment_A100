"""Where Humming's first-call cost goes: its tuning table with and without an NVML handle held open."""
import json, time
import pynvml, torch
from humming import dtypes
from humming.schema import HummingWeightSchema
from humming.transform import prepare_layer_config
from humming.tune import get_heuristics_config
from humming.utils import device as hdev

def cfg(n, k):
    return prepare_layer_config(shape_n=n, shape_k=k, num_experts=289, torch_dtype=torch.bfloat16,
                                weight_schema=HummingWeightSchema(b_dtype=dtypes.float8e4m3, weight_scale_group_size=128))

def table(held):
    get_heuristics_config.cache_clear()
    if held:
        pynvml.nvmlInit()
    t = time.perf_counter()
    out = [get_heuristics_config(cfg(n, k), gemm_type="indexed") for n, k in ((512, 4096), (4096, 256))]
    dt = time.perf_counter() - t
    if held:
        pynvml.nvmlShutdown()
    return dt, json.dumps(out)

t = time.perf_counter(); bw = hdev.calculate_gpu_bandwidth(); tops = hdev.estimate_tensorcore_max_tops()
one = time.perf_counter() - t
print(json.dumps(dict(event="single_pair_s", s=round(one, 4), bandwidth_GBps=bw, tops=tops)), flush=True)
cold, a = table(False)
warm, b = table(True)
print(json.dumps(dict(event="table", released_s=round(cold, 2), held_s=round(warm, 2), identical=a == b,
                      entries=[len(x) for x in json.loads(a)])), flush=True)
