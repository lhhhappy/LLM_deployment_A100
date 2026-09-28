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


def validate_candidate(candidate, baseline, job, image_url, *, pinned=True, final_variant=False):
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
    expected.update({'--chunked-prefill-size': '16384',
                     '--dcp-size': '1', '--log-level-http': 'warning'})
    if pinned:
        expected['--max-mamba-cache-size'] = '400'
    if final_variant:
        del expected['--dcp-size']
        expected.update({'--max-running-requests': '48', '--cuda-graph-max-bs': '48'})
    assert actual == expected, 'unexpected command difference from 46676'
    _, job_args = flags(assignment(job, 'G_ARGS'))
    capacity_keys = ('--max-running-requests', '--cuda-graph-max-bs')
    for k, v in job_args.items():
        assert k in actual and actual[k] == v, f'N30 argument differs: {k}'
    assert all(actual[k] == ('48' if final_variant else '32') for k in capacity_keys)
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
    if final_variant:
        must.update({'SGLANG_AX_DEADLINE_MAX_WAIT_S': '600',
                     'SGLANG_AX_DEADLINE_MAX_WAIT_WARM_S': '120',
                     'IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD': '4096'})
    assert all(env.get(k) == v for k, v in must.items()), 'candidate performance contract changed'
    expected_mechs = dict(p.split('=', 1) for p in assignment(job, 'G_EXPECT').split())
    for k, v in {'132': 'on', '131': 'on', '131_sync': 'rank0', '131_chunk': 'auto',
                 '128p': 'on', '126': 'off', '118': 'off', 'dcp': '1', 'spec': '-'}.items():
        assert expected_mechs.get(k) == v, f'G_EXPECT missing/wrong: {k}'
    return actual, job_args, job_env




def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evidence', type=Path, required=True)
    variants = ap.add_mutually_exclusive_group()
    variants.add_argument('--unpinned-from', type=Path,
                    help='Reuse the reviewed image receipt; require removal of only the 400-slot flag')
    variants.add_argument('--final-from', type=Path,
                    help='Derive FINAL from NOPIN: cold600/warm120, hold4096, running/graph48, default DCP1')
    variants.add_argument('--pin-final-from', type=Path,
                    help='Reuse FINAL and its image receipt; add only the 400-slot flag')
    args = ap.parse_args()
    out = args.evidence
    candidate = json.loads((out / 'submission.json').read_text())
    baseline = json.loads((out / 'baseline-46676.json').read_text())
    job = (out / 'n30-job-reviewed.sh').read_text()
    prior_dir = args.pin_final_from or args.final_from or args.unpinned_from
    final_variant = bool(args.final_from or args.pin_final_from)
    reviewed = json.loads((prior_dir / 'config-audit.json').read_text()) if prior_dir else None
    image_dir = (Path(reviewed['image_verification_source']) if final_variant
                 else args.unpinned_from or out)
    image = json.loads((image_dir / 'build-receipt.json').read_text())
    assert image['ok'] and image['data']['status'] == 2
    actual, job_args, job_env = validate_candidate(
        candidate, baseline, job, image['data']['imageUrl'],
        pinned=prior_dir is None or bool(args.pin_final_from), final_variant=final_variant)
    commit = assignment(job, 'G_COMMIT')
    if prior_dir:
        previous = (prior_dir / 'submission.json').read_bytes()
        if args.pin_final_from:
            anchor = b' --cuda-graph-max-bs 48'
            assert previous.count(anchor) == 1 and b'--max-mamba-cache-size' not in previous
            assert (out / 'submission.json').read_bytes() == previous.replace(
                anchor, anchor + b' --max-mamba-cache-size 400'), 'extra edit beyond pinning FINAL pool'
        elif args.final_from:
            expected = json.loads(previous)
            for before, after in ((' --max-running-requests 32', ' --max-running-requests 48'),
                                  (' --cuda-graph-max-bs 32', ' --cuda-graph-max-bs 48'),
                                  (' --dcp-size 1', '')):
                assert expected['command'].count(before) == 1
                expected['command'] = expected['command'].replace(before, after)
            expected['env'].update({'SGLANG_AX_DEADLINE_MAX_WAIT_S': '600',
                                    'SGLANG_AX_DEADLINE_MAX_WAIT_WARM_S': '120',
                                    'IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD': '4096'})
            assert candidate == expected, 'extra edit beyond the final configuration decision'
        else:
            flag = b' --max-mamba-cache-size 400'
            assert previous.count(flag) == 1
            assert (out / 'submission.json').read_bytes() == previous.replace(flag, b''), 'extra edit beyond removing pool pin'
        assert reviewed['status'] == 'PASS' and reviewed['engine_commit'] == commit
        assert reviewed['image'] == candidate['image']
        image_checks = {k: reviewed[k] for k in (
            'payload_equals_engine_diff', 'platform_dockerfile_matches',
            'platform_removed_final_newline', 'in_image_source_verified')}
    else:
        image_checks = verify_image(out, candidate, commit)
    old_flags = flags(baseline['command'])[1]
    rows = []
    for field, old, new in [('command', old_flags, actual), ('env', baseline['env'], candidate['env'])]:
        for key in sorted(set(old) | set(new)):
            rows.append(dict(field=field, key=key, before=old.get(key, '<unset>'),
                             after=new.get(key, '<unset>'), changed=key not in old or key not in new or old.get(key) != new.get(key)))
    result = dict(status='PASS', official_upload=False, engine_commit=commit, image=image['data']['imageUrl'],
                  image_build_id=image['data']['id'], **image_checks,
                  image_verification_source=str(image_dir),
                  only_removed_pool_pin=bool(args.unpinned_from),
                  only_added_pool_pin=bool(args.pin_final_from),
                  final_variant=final_variant,
                  runtime_receipt_status='OFFLINE_ONLY; runtime evidence is recorded separately; expected-mechanisms.txt is not an observed log',
                  candidate_sha256=hashlib.sha256((out/'submission.json').read_bytes()).hexdigest(),
                  formal_vs_job={'performance_args_match': True,
                    'validation_boundary': ('N30 with running/graph48 can validate startup/capture and shared mechanisms; N34+ performance still requires its own run'
                                            if final_variant else 'matching performance configuration'),
                    'explicit_default_args': {} if final_variant else {'--dcp-size': '1'},
                    'implicit_default_args': {'--dcp-size': '1'} if final_variant else {},
                    'logging_args': {'--log-level-http': 'warning'},
                    'explicit_default_env': {'SGLANG_AX_DCP_LOCAL_EXTEND': '0'},
                    'logging_env': {k: {'job': job_env[k], 'formal': candidate['env'][k]}
                                   for k in ('SGLANG_AX_PREFIX_TRACE_S','SGLANG_AX_PREFIX_TRACE_ROUNDS')}},
                  diff=rows)
    (out/'config-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    pool = 'FINAL + Mamba400 / running48' if args.pin_final_from else 'FINAL：不钉池 / running48' if args.final_from else '不钉 Mamba 池' if args.unpinned_from else 'Mamba400'
    comparison = ('与已更新的 eznb 性能配置一致：cold600、warm120、hold4096、running/graph=48；DCP两边均未传参数，源码默认1。'
                  if args.final_from else '与复核过的 N30 job：性能参数一致；DCP=1 与本地续算关闭显式固定。')
    if args.pin_final_from:
        comparison = '与 eznc 性能配置一致：FINAL + Mamba400；cold600、warm120、hold4096、running/graph=48不变，DCP默认1。'
    md = [f'# 46676 → chain-max 16k / {pool}：逐项配置核对', '',
          '**离线检查 PASS；本工具不执行上传。TP8 实测与正式状态见本目录 README，机制期望不代替运行收据。**', '',
          f'引擎 `{commit}`；镜像 `{candidate["image"]}`，构建 `{image["data"]["id"]}`。', '',
          comparison + '正式关闭逐请求前缀决策日志、HTTP access INFO，保留启动机制行、30 秒摘要和错误；未改正文、thinking 或输出预算。', '',
          '`SHORT_TOKENS=2048` 是准入阈值，126 关闭，**不是保证预留 2048 token**。', '']
    if args.unpinned_from:
        md += ['相对 400 版逐字节只删除 ` --max-mamba-cache-size 400`；其余配置不变。复用已审镜像收据，没有重建镜像或重复源文件核验。', '']
    if args.pin_final_from:
        md += ['相对 47043 的 FINAL 配置逐字节仅在 `--cuda-graph-max-bs 48` 后加入 ` --max-mamba-cache-size 400`；镜像、引擎、env 及其余命令保持原样。124m 计数代码属于另一个实验引擎，不包含在本镜像中。', '']
    if args.final_from:
        md += ['**与 NOPIN 版的最终差异**：cold 饥饿上限600秒，warm显式120秒（不显式设置会继承600）；原生前缀hold阈值4096；running/cuda graph上限32→48；去掉显式DCP1。其余不变，复用已审镜像。', '',
               '**并发与验证边界**：running上限扩展的直接准入收益出现在实际并发超过32时（下一评测档N34及以上）。但48档CUDA graph捕获和静态缓冲在启动时建立，不能说整个改动只影响N34以上。eznb已同步使用48/48，可验证这套启动配置与捕获；N30结果仍不能代替N34及以上的性能验证。没有实测数据时不量化显存增量。', '']
    md += ['| 类别 | 字段 | 46676 原始值 | 候选值 | 变化 |',
           '|---|---|---|---|---|']
    for row in rows:
        def fmt(x):
            return '`'+('flag present' if x is None else str(x))+'`'
        md.append('| '+ ' | '.join([row['field'], '`'+row['key']+'`', fmt(row['before']), fmt(row['after']),
                                  '变更' if row['changed'] else '相同'])+' |')
    md += ['', '未设置不是推测为零：46676 未显式钉 Mamba 池和 chunk；其 8k chunk 等有效默认值应以对应运行日志为准。去 MTP 后 101 角色边界机制恢复，旧 MTP 路径会清除此 env；这也属于候选组合差异。', '',
           '文件：`submission.json` 是当前配置；`baseline-46676.json` 是已上传 46676 配置副本；`n30-job-reviewed.sh` 是 fable 已复核任务快照；`expected-mechanisms.txt` 仅是待核验期望。']
    (out/'config-audit.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({k:result[k] for k in ('status','official_upload','image_build_id','payload_equals_engine_diff','only_removed_pool_pin')}))


def verify_image(out, candidate, commit):
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
    return dict(payload_equals_engine_diff=True, platform_dockerfile_matches=True,
                platform_removed_final_newline=platform_docker != docker.encode(),
                in_image_source_verified={'files':len(patched_paths),'sha256':digest.hexdigest()})


if __name__ == '__main__':
    main()
