#!/usr/bin/env python3
"""Offline candidate audit: submission vs 46676, reviewed N30 job and image.

Never builds, uploads, creates attempts or reads credentials. Fails on an
unexpected candidate value. Inputs and a full key-by-key diff are reviewable.
"""
import argparse
import base64
import gzip
import hashlib
import json
import lzma
import re
import shlex
import subprocess
from pathlib import Path


def assignment(source, key):
    rows = [r for r in source.splitlines() if r.startswith(key + '=')]
    assert len(rows) == 1, key
    values = shlex.split(rows[0].split('=', 1)[1])
    assert len(values) == 1, key
    return values[0]


def flags(command):
    args = shlex.split(command)
    result, prefix = {}, []
    i = 0
    while i < len(args):
        key = args[i]
        if not key.startswith('--'):
            assert not result, f'unexpected positional argument: {key}'
            prefix.append(key)
            i += 1
            continue
        assert key not in result, f'duplicate flag: {key}'
        value = None
        if i + 1 < len(args) and not args[i + 1].startswith('--'):
            value = args[i + 1]
            i += 1
        result[key] = value
        i += 1
    return prefix, result


def validate_candidate(candidate, baseline, job, image_url):
    assert set(candidate) == {'image', 'command', 'env', 'model_name'}
    assert candidate['image'] == image_url
    assert candidate['model_name'] == 'default'
    head, actual = flags(candidate['command'])
    base_head, expected = flags(baseline['command'])
    assert head == base_head == ['python3', '-m', 'sglang.launch_server']
    # Deliberate command changes; every untouched argument must remain identical.
    removed = ('--speculative-algorithm', '--speculative-draft-model-path',
               '--speculative-num-steps', '--speculative-eagle-topk', '--speculative-num-draft-tokens')
    for key in removed:
        assert key in expected
        del expected[key]
    expected.update({'--chunked-prefill-size': '16384', '--max-mamba-cache-size': '400',
                     '--dcp-size': '1', '--log-level-http': 'warning'})
    assert actual == expected, 'unexpected command difference from 46676'
    _, job_args = flags(assignment(job, 'G_ARGS'))
    assert all(actual.get(k) == v and k in actual for k, v in job_args.items()), 'N30 arguments differ'
    assert actual['--max-running-requests'] == '32' and actual['--cuda-graph-max-bs'] == '32'
    job_env = dict(item.split('=', 1) for item in shlex.split(assignment(job, 'G_ENV')))
    env = {**baseline['env'], **job_env,
           'SGLANG_AX_PREFIX_TRACE_S': '0', 'SGLANG_AX_PREFIX_TRACE_ROUNDS': '0',
           'SGLANG_AX_DCP_LOCAL_EXTEND': '0'}
    assert candidate['env'] == env, 'unexplained env difference from baseline + N30 job'
    must = {'SGLANG_AX_PREFIX_PRODUCER': '1', 'SGLANG_AX_DEADLINE_CHAIN_FIRST': '1',
            'SGLANG_AX_DEADLINE_FREEZE_CLASS': '1', 'SGLANG_AX_CHAIN_RISK_INTERVAL': '1',
            'SGLANG_AX_CHAIN_RISK_CHUNK': '0', 'SGLANG_AX_DEADLINE_LOAD': '1.05',
            'SGLANG_AX_SCHED_COLD_CAP': '16384', 'SGLANG_AX_BACKLOG_COLD_CAP': '16384',
            'SGLANG_AX_SCHED_SHORT_TOKENS': '2048', 'SGLANG_AX_SCHED_COLD_CAP_MAX': '0',
            'SGLANG_AX_DSA_SPARSE_TRITON': '0', 'SGLANG_AX_DEADLINE_FAMILY': '0',
            'SGLANG_AX_BACKLOG_MAX_SLOW': '80', 'SGLANG_AX_DEADLINE_TIERS': '1'}
    assert all(env.get(k) == v for k, v in must.items()), 'candidate performance contract changed'
    expected_mechs = dict(p.split('=', 1) for p in assignment(job, 'G_EXPECT').split())
    for k, v in {'132': 'on', '131': 'on', '131_sync': 'rank0', '131_chunk': 'auto',
                 '128p': 'on', '126': 'off', '118': 'off', 'dcp': '1', 'spec': '-'}.items():
        assert expected_mechs.get(k) == v, f'G_EXPECT missing/wrong: {k}'
    return actual, job_args, job_env




def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evidence', type=Path, required=True)
    args = ap.parse_args()
    out = args.evidence
    candidate = json.loads((out / 'submission.json').read_text())
    baseline = json.loads((out / 'baseline-46676.json').read_text())
    job = (out / 'n30-job-reviewed.sh').read_text()
    image = json.loads((out / 'build-receipt.json').read_text())
    assert image['ok'] and image['data']['status'] == 2
    actual, job_args, job_env = validate_candidate(candidate, baseline, job, image['data']['imageUrl'])
    commit = assignment(job, 'G_COMMIT')
    docker = (out / 'Dockerfile').read_text()
    assert len(docker.encode()) <= 65536
    assert 'FROM registry.dp.tech/dptech/dp/native/prod-4727808/4650601/lh-img:0925a\n' in docker
    assert f'echo {commit} > /opt/ax/engine_commit;' in docker
    detail = json.loads((out / 'image-detail.json').read_text())
    assert detail['ok'] and detail['data']['status'] == 2
    assert detail['data']['url'] == candidate['image']
    platform_docker = base64.b64decode(detail['data']['dockerFile'])
    # The image catalog removes exactly the final newline. Permit no other edit.
    assert platform_docker in (docker.encode(), docker.encode().removesuffix(b'\n'))
    build_log = json.loads((out / 'build-log-retry.json').read_text())
    assert build_log['ok']
    emitted = re.findall(r'^.*ENGINE_SOURCE_VERIFIED (\d+) ([0-9a-f]{64})\s*$',
                         build_log['data']['log'], re.M)
    assert len(emitted) == 1, 'missing/ambiguous in-image source verification'
    encoded = re.search(r"^    echo '([^']+)' \| .* > /tmp/ax/engine.diff;", docker, re.M).group(1)
    payload = (lzma.decompress(base64.b85decode(encoded)) if 'base64.b85decode' in docker
               else gzip.decompress(base64.b64decode(encoded)))
    expected_diff = subprocess.check_output(['git', 'diff', '--no-color', '--binary',
                                            'image-lh-img-0925a', commit, '--', 'engine/sglang'])
    assert payload == expected_diff
    patched_paths = sorted(r[len('+++ b/engine/sglang/'):] for r in payload.decode().splitlines()
                           if r.startswith('+++ b/engine/sglang/'))
    digest = hashlib.sha256()
    for path in patched_paths:
        digest.update(path.encode() + b'\0')
        digest.update(subprocess.check_output(['git','show',commit+':engine/sglang/'+path]))
        digest.update(b'\0')
    assert emitted == [(str(len(patched_paths)), digest.hexdigest())]
    old_flags = flags(baseline['command'])[1]
    rows = []
    for field, old, new in [('command', old_flags, actual), ('env', baseline['env'], candidate['env'])]:
        for key in sorted(set(old) | set(new)):
            rows.append(dict(field=field, key=key, before=old.get(key, '<unset>'),
                             after=new.get(key, '<unset>'), changed=key not in old or key not in new or old.get(key) != new.get(key)))
    result = dict(status='PASS', official_upload=False, engine_commit=commit, image=image['data']['imageUrl'],
                  image_build_id=image['data']['id'], payload_equals_engine_diff=True,
                  platform_dockerfile_matches=True, platform_removed_final_newline=platform_docker != docker.encode(),
                  in_image_source_verified={'files':len(patched_paths),'sha256':digest.hexdigest()},
                  runtime_receipt_status='PENDING_TP8; expected-mechanisms.txt is an expectation, not an observed log',
                  candidate_sha256=hashlib.sha256((out/'submission.json').read_bytes()).hexdigest(),
                  formal_vs_job={'performance_args_match': True,
                    'explicit_default_args': {'--dcp-size': '1'},
                    'logging_args': {'--log-level-http': 'warning'},
                    'explicit_default_env': {'SGLANG_AX_DCP_LOCAL_EXTEND': '0'},
                    'logging_env': {k: {'job': job_env[k], 'formal': candidate['env'][k]}
                                   for k in ('SGLANG_AX_PREFIX_TRACE_S','SGLANG_AX_PREFIX_TRACE_ROUNDS')}},
                  diff=rows)
    (out/'config-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    md = ['# 46676 → chain-max 16k / Mamba400：逐项配置核对', '',
          '**离线检查 PASS；正式未上传。TP8 候选机制行和 N30 结果尚待运行。**', '',
          f'引擎 `{commit}`；镜像 `{candidate["image"]}`，构建 `{image["data"]["id"]}`。', '',
          '与复核过的 N30 job：性能参数一致；DCP=1 与本地续算关闭显式固定。正式关闭逐请求前缀决策日志、HTTP access INFO，保留启动机制行、30 秒摘要和错误；未改正文、thinking 或输出预算。', '',
          '`SHORT_TOKENS=2048` 是准入阈值，126 关闭，**不是保证预留 2048 token**。', '',
          '| 类别 | 字段 | 46676 原始值 | 候选值 | 变化 |',
          '|---|---|---|---|---|']
    for row in rows:
        def fmt(x):
            return '`'+('flag present' if x is None else str(x))+'`'
        md.append('| '+ ' | '.join([row['field'], '`'+row['key']+'`', fmt(row['before']), fmt(row['after']),
                                  '变更' if row['changed'] else '相同'])+' |')
    md += ['', '未设置不是推测为零：46676 未显式钉 Mamba 池和 chunk；其 8k chunk 等有效默认值应以对应运行日志为准。去 MTP 后 101 角色边界机制恢复，旧 MTP 路径会清除此 env；这也属于候选组合差异。', '',
           '文件：`submission.json` 是唯一候选配置；`baseline-46676.json` 是已上传 46676 配置副本；`n30-job-reviewed.sh` 是 fable 已复核任务快照；`expected-mechanisms.txt` 仅是待核验期望。']
    (out/'config-audit.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({k:result[k] for k in ('status','official_upload','image_build_id','payload_equals_engine_diff')}))


if __name__ == '__main__':
    main()
