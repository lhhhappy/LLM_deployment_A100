#!/usr/bin/env python3
"""Enumerate JIT constexpr/specialization and autotune keys from the actual stack.

Static inventory != observed service reachability. Includes helper JIT functions
and kernels for unselected backends so missing coverage stays explicit.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'build/p150/candidate/sglang'
EV=ROOT/'evidence/T46'


def main():
    rows=[]
    for path in sorted(SOURCE.rglob('*.py')):
        source=path.read_text(encoding='utf-8-sig'); tree=ast.parse(source)
        def expr(node): return ast.unparse(node)
        aliases={}
        for n in ast.walk(tree):
            if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call):
                call=n.value
                if isinstance(call.func,ast.Call) and 'autotune' in expr(call.func.func) and call.args:
                    aliases[expr(call.args[0])] = [expr(x.value) for x in call.func.keywords if x.arg=='key']
        for n in ast.walk(tree):
            if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)): continue
            decorators=[expr(d) for d in n.decorator_list]
            if not any('triton.jit' in d or 'triton.autotune' in d for d in decorators): continue
            auto=[]; dns=[]; heuristics=[]
            for d in n.decorator_list:
                if isinstance(d,ast.Call):
                    for kw in d.keywords:
                        if kw.arg=='key' and 'autotune' in expr(d.func): auto.append(expr(kw.value))
                        if kw.arg=='do_not_specialize': dns.append(expr(kw.value))
                    if 'heuristics' in expr(d.func): heuristics.extend(expr(a) for a in d.args)
            rel=str(path.relative_to(SOURCE))
            focus=('/attention/fla/' in rel or '/ops/mamba/' in rel or rel.endswith('sm80_indexer_kernels.py'))
            rows.append(dict(file=rel,line=n.lineno,function=n.name,
                autotune_keys=auto+aliases.get(n.name,[]),
                constexpr=[a.arg for a in n.args.args if a.annotation is not None and 'constexpr' in expr(a.annotation)],
                do_not_specialize=dns,heuristics=heuristics,
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                scope='KDA/indexer/conv candidate' if focus else 'other backend/helper; no service coverage claim'))
    (EV/'jit_inventory.json').write_text(json.dumps(rows,indent=2)+'\n')
    focus=[r for r in rows if r['scope']=='KDA/indexer/conv candidate']
    lines=['# Static kernel inventory, exact 150 candidate','',
           'Every row also specializes on pointer dtypes, constexpr values and runtime scalar alignment/equality unless excluded. No row here certifies service execution.', '',
           '| Source/function | Autotune key | JIT constexpr | do_not_specialize |',
           '|---|---|---|---|']
    for r in focus:
        lines.append('| '+f"{r['file']}:{r['line']} `{r['function']}`"+' | '+('; '.join(r['autotune_keys']) or 'none')+' | '+', '.join(r['constexpr'])+' | '+'; '.join(r['do_not_specialize'])+' |')
    (EV/'kernel_keys.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'all_jit_functions':len(rows),'focused_jit_functions':len(focus),
                      'focused_autotuned':sum(bool(r['autotune_keys']) for r in focus)}))


if __name__=='__main__': main()
