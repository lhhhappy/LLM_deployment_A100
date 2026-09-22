#!/usr/bin/env python3
"""T48 argument resolution only: no worker, model weights, server or collective."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,sys.argv[1])
from sglang.srt.server_args import ServerArgs
from sglang.srt.arg_groups.overrides import resolved_view
os.environ['SGLANG_AX_KDA_DUAL_SNAPSHOT']='1'
os.environ['SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS']='154827,154829'
a=ServerArgs(model_path=sys.argv[2],speculative_draft_model_path=sys.argv[2],
             speculative_algorithm='NEXTN',speculative_num_steps=3,speculative_eagle_topk=1,
             speculative_num_draft_tokens=4,max_running_requests=32,tp_size=8,
             page_size=64,mamba_radix_cache_strategy='extra_buffer',
             dsa_prefill_backend='fa3',dsa_decode_backend='fa3')
a.resolve_once()
r=resolved_view(a)
fields=['speculative_algorithm','max_running_requests','speculative_num_steps','speculative_num_draft_tokens',
        'dsa_prefill_backend','dsa_decode_backend','linear_attn_verify_backend','kv_cache_dtype']
out={k:getattr(r,k) for k in fields}
assert out['dsa_prefill_backend']==out['dsa_decode_backend']=='tilelang',out
assert out['speculative_algorithm']=='EAGLE',out
assert os.environ['SGLANG_AX_KDA_DUAL_SNAPSHOT']=='0'
assert os.environ['SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS']==''
print(json.dumps(out,sort_keys=True))

from sglang.srt.configs.model_config import ModelConfig,get_dsa_mtp_topk_width
from sglang.srt.models.glm5_next_nextn import Glm5NextForConditionalGenerationNextN
from sglang.srt.layers.quantization.fp8 import Fp8Config
draft=ModelConfig.from_server_args(a,is_draft_model=True)
assert draft.hf_config.architectures==['Glm5NextForConditionalGenerationNextN']
assert draft.hf_text_config.linear_attn_config is None
assert draft.hf_config.index_share_for_mtp_iteration is True
assert get_dsa_mtp_topk_width(draft.hf_config)==2051
mapper=Glm5NextForConditionalGenerationNextN.get_hf_to_sglang_mapper(draft.hf_config)
quant=Fp8Config.from_config(draft.hf_config.quantization_config)
quant.apply_weight_name_mapper(mapper)
assert 'model.decoder.self_attn.indexer.wq_b' in quant.ignored_layers
assert 'model.decoder.self_attn.kv_b_proj' in quant.ignored_layers
assert 'model.decoder.mlp.experts' not in quant.ignored_layers
print(json.dumps(dict(draft_arch=draft.hf_config.architectures,linear_attn_config=None,
                     index_share=True,seed_width=2051,ignored_remap=True)))
