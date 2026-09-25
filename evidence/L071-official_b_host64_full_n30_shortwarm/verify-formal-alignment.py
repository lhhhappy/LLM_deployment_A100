#!/usr/bin/env python3
"""Compare the adopted live process receipt to the uploaded 0925a config."""
import hashlib
import json
from pathlib import Path
import re
import shlex

here = Path(__file__).resolve().parent
root = here.parents[1]
formal = json.loads((root/'evidence/submission-0925a/submission.json').read_text())
live = json.loads((here/'adopt-verification.json').read_text())
job = (here/'preload.sh').read_text()
environment = dict(s.split('=',1) for s in shlex.split(re.search(r'^G_ENV="([^"]+)"',job,re.M)[1]))
environment.update(SGLANG_OPT_USE_TOPK_V2='0', SGLANG_OPT_DEEPGEMM_HC_PRENORM='0',
                   SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS='154827,154829')
observed = json.loads((here/'precision-snapshot.json').read_text())['environments']
assert observed['image_process'] == {'NCCL_VERSION':'2.28.3-1'}
assert observed['engine'] == {**observed['image_process'],**environment}
assert hashlib.sha256(json.dumps(observed['engine'],sort_keys=True).encode()).hexdigest() == live['environment_sha256']
normalized_argv = list(live['argv'])
assert normalized_argv[normalized_argv.index('--port')+1] == '30000'
normalized_argv[normalized_argv.index('--port')+1] = '8000'
assert normalized_argv == shlex.split(formal['command']), 'unexpected command difference'
differences = {k:dict(formal=formal['env'].get(k),live=environment.get(k))
               for k in set(formal['env'])|set(environment) if formal['env'].get(k)!=environment.get(k)}
assert differences == {'SGLANG_AX_PACE_TPOT':dict(formal='0',live='0.085')}, differences
assert live['commit'] == '759a6ebb8e31723519ad5daf438e26e24b32501a'
result = dict(status='VERIFIED', engine_commit=live['commit'], pid=live['pid'],
              formal_image=formal['image'], formal_attempt=46251,
              command_equal_except_port=True, mechanism_environment_differences=differences,
              inherited_image_environment=observed['image_process'],
              intended_experiment='071 enables mechanism 122 against 069/0925a; not an exact replay of 46251',
              tf32_matmul='not enabled by either command; frozen source default False; 069 log also False',
              unchanged=['MTP','host64','GPU memory fraction 0.87','model/parser/backends','token/output contract'],
              runtime_difference='071 RAM cache location and cold startup differ; startup repair preserves binaries',
              evaluation_boundary='same scorer rules; local frozen dataset/preheat are not the formal workload')
(here/'formal-alignment.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
