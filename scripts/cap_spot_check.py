#!/usr/bin/env python3
"""CAP-01/02 small public AIME/GPQA comparison, never an official ability score.

Runtime downloads ONLY into /sjtu/linhang/arena/cache (no datasets in the repo).
Sources: https://huggingface.co/datasets/Maxwell-Jia/AIME_2024
https://github.com/openai/simple-evals/blob/main/gpqa_eval.py (public Diamond CSV)
Original GPQA HF repository is gated: https://huggingface.co/datasets/Idavidrein/gpqa
Both servers receive identical questions/permutations via /v1/chat/completions,
model=default. NO max_tokens/max_completion_tokens or thinking override is sent.
Answers are extracted ONLY from content. A length finish is flagged as truncation.
--suite tiny is an explicit offline AIME smoke fallback, NOT CAP-01/02 coverage.
--suite gpqa failure is reported; no silent substitution with math questions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import re
import urllib.parse
import urllib.request

from serving_probe import CheckFailed, client_for, require, write_report

CACHE_ROOT = Path('/sjtu/linhang/arena/cache')
AIME_DATASET = 'Maxwell-Jia/AIME_2024'
AIME_SOURCE = 'https://huggingface.co/datasets/' + AIME_DATASET
AIME_ROWS = ('https://datasets-server.huggingface.co/rows?dataset=Maxwell-Jia%2FAIME_2024'
             '&config=default&split=train&offset=0&length=30')
GPQA_CSV = 'https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv'
GPQA_SOURCE = 'https://github.com/openai/simple-evals/blob/main/gpqa_eval.py'
# Two short, reworded public AIME 2024 problems; source/known answer retained.
# Full public corpora (including solutions) are never embedded in this repository.
TINY = [
    {'id': '2024-I-2', 'kind': 'aime', 'answer': '25', 'source': AIME_SOURCE,
     'question': r'For real x,y>1, log_x(y^x)=log_y(x^(4y))=10. Determine xy.'},
    {'id': '2024-II-11', 'kind': 'aime', 'answer': '601', 'source': AIME_SOURCE,
     'question': 'Count ordered triples of nonnegative integers (a,b,c) with a+b+c=300 and '
                 'a^2*b+a^2*c+b^2*a+b^2*c+c^2*a+c^2*b=6000000.'},
]


class Redirect308(urllib.request.HTTPRedirectHandler):
    # Python 3.10 urllib lacks 308; hf-mirror currently redirects to HF with 308.
    def http_error_308(self, request, fp, code, message, headers):
        # All dataset fetches are GET, so 302 and 308 preserve the same method.
        return self.http_error_302(request, fp, 302, message, headers)


def fetch(url, cache_dir, name):
    cache_dir = Path(cache_dir).resolve()
    root = CACHE_ROOT.resolve()
    require(cache_dir == root or root in cache_dir.parents, 'dataset cache must remain under /sjtu/linhang/arena/cache')
    path = cache_dir / name
    if path.exists():
        data = path.read_bytes()
    else:
        try:
            opener = urllib.request.build_opener(Redirect308())
            with opener.open(url, timeout=60) as response:
                data = response.read(10 * 1024 * 1024 + 1)
            require(len(data) <= 10 * 1024 * 1024, 'dataset response exceeds expected size')
        except CheckFailed:
            raise
        except Exception:
            raise CheckFailed('public dataset fetch failed; use --suite tiny explicitly for AIME smoke only') from None
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data, hashlib.sha256(data).hexdigest()


def load_questions(suite, count, seed=19, cache_dir=CACHE_ROOT/'t19'):
    rng = random.Random(seed)
    if suite == 'tiny':
        require(count <= len(TINY), 'tiny suite contains only two questions')
        return [dict(q) for q in TINY[:count]], {'source': AIME_SOURCE, 'fallback': True}
    if suite == 'aime':
        # Query mirror metadata as requested; digest both metadata and the exact
        # viewer response because the viewer API is not revision-pinned.
        metadata, meta_hash = fetch('https://hf-mirror.com/api/datasets/' + AIME_DATASET,
                                    cache_dir, 'aime_metadata.json')
        meta = json.loads(metadata)
        raw, digest = fetch(AIME_ROWS, cache_dir, 'aime2024_rows.json')
        parsed = json.loads(raw)
        require(not any(r.get('truncated_cells') for r in parsed['rows']), 'dataset viewer truncated a cell')
        rows = [r['row'] for r in parsed['rows']]
        require(len(rows) == 30, 'AIME source must contain all 30 questions')
        require(0 < count <= len(rows), 'AIME sample size out of range')
        chosen = rng.sample(sorted(rows, key=lambda r: r['ID']), count)
        questions = [{'id': r['ID'], 'question': r['Problem'], 'answer': str(int(r['Answer'])),
                      'kind': 'aime', 'source': AIME_SOURCE} for r in chosen]
        provenance = {'source': AIME_SOURCE, 'download_url': AIME_ROWS, 'sha256': digest,
                      'metadata_sha256': meta_hash, 'metadata_revision': meta.get('sha'),
                      'note': 'viewer snapshot SHA256 is authoritative; not revision-pinned'}
    else:
        raw, digest = fetch(GPQA_CSV, cache_dir, 'gpqa_diamond.csv')
        rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
        require(0 < count <= len(rows), 'GPQA sample size out of range')
        chosen = rng.sample(sorted(rows, key=lambda r: r['Record ID']), count)
        questions = []
        for r in chosen:
            choices = [r['Correct Answer']] + [r['Incorrect Answer ' + str(i)] for i in (1, 2, 3)]
            order = list(range(4))
            rng.shuffle(order)
            questions.append({'id': str(r['Record ID']), 'kind': 'gpqa', 'source': GPQA_SOURCE,
                'question': r['Question'] + '\n' + '\n'.join(f'{"ABCD"[i]}. {choices[j]}' for i, j in enumerate(order)),
                'answer': 'ABCD'[order.index(0)], 'permutation': order})
        provenance = {'source': GPQA_SOURCE, 'download_url': GPQA_CSV, 'sha256': digest,
                      'note': 'public simple-evals Diamond mirror; original HF repo gated'}
    require(len({q['id'] for q in questions}) == len(questions), 'duplicate question IDs')
    return questions, provenance


def extract_answer(content, kind):
    if not isinstance(content, str):
        return None
    # Do not pick an arbitrary number/letter from a derivation. Prefer the last
    # explicit final marker; accept a bare answer only when it is the whole text.
    value = r'[A-D]' if kind == 'gpqa' else r'\d{1,3}'
    pattern = (r'\\boxed\s*\{\s*(' + value + r')\s*\}|'
               r'(?:final\s+answer|answer)\s*[:=]\s*\$?\(?\s*(' + value + r')\b')
    matches = list(re.finditer(pattern, content, re.I))
    if matches:
        answer = next(x for x in matches[-1].groups() if x is not None)
    else:
        match = re.fullmatch(r'\s*\$?\(?(' + value + r')\)?\$?[.!]?\s*', content, re.I)
        answer = match.group(1) if match else None
    if answer is None:
        return None
    return answer.upper() if kind == 'gpqa' else str(int(answer))


def evaluate(client, question):
    suffix = ('Give the final option as "Answer: A" (or B, C, D).' if question['kind'] == 'gpqa'
              else r'Give the final integer answer as \boxed{n}.')
    # Intentionally OMIT every output-budget/think-control field.
    body = {'model': 'default', 'stream': False, 'temperature': 0,
            'messages': [{'role': 'user', 'content': question['question'] + '\n' + suffix}]}
    reply = client.json('/v1/chat/completions', body)
    require(reply.get('model') == 'default', 'chat response model mismatch')
    choices = reply.get('choices')
    require(isinstance(choices, list) and bool(choices), 'chat response choices missing')
    choice = choices[0]
    content = choice.get('message', {}).get('content')
    answer = extract_answer(content, question['kind'])
    stop = choice.get('finish_reason')
    return {'answer': answer, 'correct': answer == question['answer'],
            'content_nonempty': isinstance(content, str) and bool(content.strip()),
            'answer_in_content': answer is not None, 'finish_reason': stop,
            'completion_tokens': reply.get('usage', {}).get('completion_tokens'),
            'interface_ok': answer is not None and stop == 'stop'}


def compare(clients, questions, suite, provenance, seed):
    rows = []
    for index, question in enumerate(questions):
        row = {'id': question['id'], 'expected_answer': question['answer'], 'results': {}}
        # Alternate server order to reduce a systematic warm/load order effect.
        order = list(enumerate(clients))
        if index % 2:
            order.reverse()
        for i, client in order:
            label = 'stock' if i == 0 else 'candidate'
            try:
                row['results'][label] = evaluate(client, question)
            except CheckFailed as e:
                row['results'][label] = {'correct': False, 'interface_ok': False, 'reason': str(e)}
        if 'permutation' in question:
            row['option_permutation'] = question['permutation']
        rows.append(row)
    labels = ('stock', 'candidate')[:len(clients)]
    scores = {label: sum(row['results'][label]['correct'] for row in rows) for label in labels}
    interface_ok = all(row['results'][label]['interface_ok'] for row in rows for label in labels)
    comparative = len(clients) == 2
    passed = interface_ok and comparative and scores['candidate'] >= scores['stock'] - 1
    case = {'aime': 'CAP-01', 'gpqa': 'CAP-02', 'tiny': 'AIME-smoke-only'}[suite]
    full_sample = len(questions) >= {'aime': 10, 'gpqa': 20, 'tiny': 3}[suite]
    return {'case': case, 'suite': suite, 'seed': seed, 'count': len(rows), 'source': provenance,
            'scores': scores, 'rows': rows, 'comparison_available': comparative,
            'spot_check_passed': passed, 'registry_sample_complete': full_sample,
            'passed': passed and full_sample,
            'note': 'small diagnostic sample; cannot establish official >90 ability gate; no output clamp sent'}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--base-url', default='http://127.0.0.1:8000', help='stock server')
    p.add_argument('--candidate-url')
    p.add_argument('--suite', choices=('aime', 'gpqa', 'tiny'), default='aime')
    p.add_argument('--count', type=int)
    p.add_argument('--seed', type=int, default=19)
    p.add_argument('--cache-dir', type=Path, default=CACHE_ROOT/'t19')
    p.add_argument('--timeout', type=float, default=3600)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    count = a.count if a.count is not None else {'aime': 10, 'gpqa': 20, 'tiny': 2}[a.suite]
    if count <= 0 or a.timeout <= 0:
        p.error('count and timeout must be positive')
    if a.dry_run:
        print(json.dumps({'dry_run': True, 'suite': a.suite, 'count': count,
            'sources': [AIME_SOURCE, GPQA_SOURCE], 'cache': str(a.cache_dir),
            'request': {'model': 'default', 'max_tokens': 'OMITTED', 'thinking_override': 'OMITTED'},
            'note': 'no server HTTP, dataset downloads, cache/output writes'}, indent=2))
        return 0
    try:
        questions, provenance = load_questions(a.suite, count, a.seed, a.cache_dir)
        clients = [client_for(a.base_url, a.timeout)]
        if a.candidate_url:
            clients.append(client_for(a.candidate_url, a.timeout, 'S1_OTHER_API_KEY'))
        report = compare(clients, questions, a.suite, provenance, a.seed)
    except CheckFailed as e:
        report = {'passed': False, 'reason': str(e)}
    except Exception:
        report = {'passed': False, 'reason': 'dataset/local/HTTP failure; details suppressed'}
    write_report(report, a.out)
    # Tiny/single-server runs are useful diagnostics but deliberately non-passing
    # for the registry. Their explicit spot_check_passed remains in the report.
    return 0 if report.get('passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
