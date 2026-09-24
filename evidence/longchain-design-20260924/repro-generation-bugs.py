"""CPU-only review probes; never edits source/frozen artifacts."""
from pathlib import Path
import argparse, collections, copy, gzip, json, sys, tempfile
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'scripts/analysis'))
sys.path.insert(0, str(REPO/'s1-dev/harness'))
import longchain as lc
from longchain_check import check_dataset
import s1_common as common
if '--baseline' in sys.argv:
    import importlib.util
    for name, rel in [('review_old_generator','longchain.py'),('review_old_checker','longchain_check.py')]:
        spec=importlib.util.spec_from_file_location(name, REPO/'evidence/longchain-audit/frozen-candidate/code/scripts/analysis'/rel)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        if name=='review_old_generator': lc=module
        else: check_dataset=module.check_dataset

def write_lines(path, rows):
    with open(path, 'w') as f:
        for row in rows: lc.write_line(f, row)

def fixture(root, renderer):
    root.mkdir(); (root/'bodies').mkdir(); (root/'samples').mkdir()
    body={'req_id':'biomaster:canon:a','system':'','tools':[], 'messages':[{'role':'user','content':'Compare the two results.'}]}
    n=len(renderer.tokenizer.encode(renderer.render(body),add_special_tokens=False))
    row={'pack':'biomaster','view':'canon','session_id':'s','chain_id':'c','chain_index':0,'logical_call_id':'a',
         'dispatch_offset_ms':0,'phase':'session_start','glm_tokens':n,'glm_lcp_with_prev':0,'uncached_expected':n,
         'in_serving_load':True,'max_output_i':10,'replay_gap_ms':0,'gap_valid':True,'body_ref':'bodies/f.jsonl.gz'}
    source={'n_requests':1,'sum_glm_tokens':n,'sum_uncached_expected':n,'max_output_i_sum':10,'phases':{'session_start':1}}
    chain=lc.chain_record(source,[row],0)
    prov={'req_id':body['req_id'],'source_req_id':body['req_id'],'kind':'original','previous_req_id':None,
          'prompt_tokens':n,'lcp_tokens':0,'added_tokens':n,'removed_tokens':0,'body_sha256':lc.digest(body)}
    write_lines(root/'requests.jsonl',[row]); write_lines(root/'chains.jsonl',[chain]);write_lines(root/'provenance.jsonl',[prov])
    write_lines(root/'samples/f.jsonl',[{'chain_id':'c'}])
    with gzip.open(root/'bodies/f.jsonl.gz','wt') as f: lc.write_line(f,body)
    index,_,_=common.load_index(str(root)); common.freeze_cohort(str(root),'f',index,1,str(root/'cohort.json'))
    manifest={'generator':lc.VERSION,'status':'BUILT_UNVALIDATED','set':'f','seed':1,'max_context_tokens':1048576,
              'assumptions':[],'actual_output_sum':10,'chain_summaries':[{'chain_id':'c','source_prompt_sum':n,'source_output_sum':10,'output_sum':10}],
              'artifacts':{str(p.relative_to(root)):lc.file_digest(p) for p in root.rglob('*') if p.is_file()}}
    lc.json_dump(root/'manifest.json',manifest)
    return row

def polish(root,dest,edits):
    lc.polish(argparse.Namespace(root=str(root),out=str(dest),edits=str(edits),set=None,
              harness_dir=str(REPO/'s1-dev/harness'),tok_dir=str(REPO/'s1-dev/glm_tok')))

def check(root):
    r=check_dataset(str(root),str(REPO/'s1-dev/harness'),str(REPO/'s1-dev/glm_tok'))
    return {'status':r['status'],'errors':r['errors']}

out={}
block=[{'role':'assistant','tool_calls':[{'id':'a','function':{'name':'calculate','arguments':'{}'}}]},
       {'role':'tool','tool_call_id':'a','content':'data'}]
out['short_id_rename']=lc.rename_new_calls(block,'new')
block=[{'role':'assistant','tool_calls':[{'id':'abcdefgh','function':{'name':'Read'}},{'id':'new','function':{'name':'Read'}}]},
       {'role':'tool','tool_call_id':'abcdefgh','content':'ok'},{'role':'tool','tool_call_id':'new','content':'ok'}]
out['cascading_rename']=lc.rename_new_calls(block,'new')
out['alternate_id_rename']=lc.rename_new_calls([{'role':'assistant','tool_calls':[{'tool_call_id':'old','function':{'name':'Read'}}]}, {'role':'tool','tool_call_id':'old','content':'ok'}],'new')
if '--baseline' not in sys.argv:
    out['structure_path_rename']=lc.rename_new_calls([
      {'role':'assistant','tool_calls':[{'id':'assistant','type':'function','function':{
          'name':'assistant','arguments':{'name':'assistant','type':'assistant','role':'assistant'}}}]},
      {'role':'tool','tool_call_id':'assistant','content':{'name':'assistant'}}], 'scoped')

# Tokenization is irrelevant to lineage/integrity probes. This test double is
# intentionally isolated; these results are not a real GLM rendering receipt.
class CharacterRenderer:
    def __init__(self, *_): self.tokenizer=self
    def render(self, body): return ''.join(m.get('content','') for m in body['messages'])
    def encode(self, text, **_): return list(map(ord,text))
common.Renderer=CharacterRenderer
lc.implementation_receipt=lambda *_: {'test_double':'CharacterRenderer'}
renderer=CharacterRenderer()
with tempfile.TemporaryDirectory(prefix='longchain-review-') as tmp:
    tmp=Path(tmp); base=tmp/'base'; fixture(base,renderer)
    out['fixture']=check(base)
    edits=tmp/'edits.jsonl'
    def edit(old,new):
        write_lines(edits,[{'scope':'chain_seed_query','chain_id':'c','req_id':'biomaster:canon:a','message_index':0,'expected_content':old,'replacement_content':new}])
    edit('Compare the two results.','Compare both results with uncertainty estimates.')
    first=tmp/'first'; polish(base,first,edits);out['first_polish']=check(first)
    edit('Compare both results with uncertainty estimates.','Compare both results and explain uncertainty.')
    second=tmp/'second';polish(first,second,edits);out['second_polish']=check(second)
    bad=tmp/'bad'; row=fixture(bad,renderer);row['max_output_i']=777;write_lines(bad/'requests.jsonl',[row])
    out['tampered_parent']=check(bad)
    edit('Compare the two results.','Compare both results with uncertainty estimates.')
    washed=tmp/'washed'
    try:
        polish(bad,washed,edits);out['polished_tamper']=check(washed)
        out['polished_tamper']['actual_request_budget']=list(lc.read_jsonl(washed/'requests.jsonl'))[0]['max_output_i']
        out['polished_tamper']['manifest_output_sum']=json.loads((washed/'manifest.json').read_text())['actual_output_sum']
    except ValueError as exc:
        out['polished_tamper']={'rejected':str(exc)}
if '--baseline' not in sys.argv:
    with tempfile.TemporaryDirectory(prefix='longchain-build-review-') as tmp:
        tmp=Path(tmp);source=tmp/'source';source.mkdir();(source/'bodies').mkdir()
        source_rows=[]; source_chains=[]; source_bodies=[]
        for ci in range(2):
            cid=f'c{ci}';messages=[{'role':'user','content':f'Q{ci}'}]
            bodies=[{'req_id':f'biomaster:canon:{ci}-0','system':'','tools':[],'messages':copy.deepcopy(messages)}]
            messages.append({'role':'assistant','content':'abc'})
            bodies.append({'req_id':f'biomaster:canon:{ci}-1','system':'','tools':[],'messages':copy.deepcopy(messages)})
            rows=[]
            for i,body in enumerate(bodies):
                n=len(renderer.render(body)); row={'pack':'biomaster','view':'canon','session_id':'shared-source-session',
                    'chain_id':cid,'chain_index':ci,'logical_call_id':f'{ci}-{i}','dispatch_offset_ms':10+50*i,
                    'end_offset_ms':50+50*i,'phase':'session_start' if i==0 else 'intra','glm_tokens':n,
                    'glm_lcp_with_prev':0 if i==0 else 2,'uncached_expected':n if i==0 else n-2,
                    'in_serving_load':True,'max_output_i':10,'replay_gap_ms':5,'gap_valid':True,
                    'net_think_ms':5,'tool_union_ms':0,'body_ref':'bodies/source.jsonl.gz','sys_tools_hash':'family'}
                rows.append(row)
            chain=lc.chain_record({'n_requests':4,'sum_glm_tokens':26,'sum_uncached_expected':11,
                 'max_output_i_sum':40,'phases':{'session_start':1,'intra':3}},rows,1)
            chain.update(n_requests=4,sum_glm_tokens=26,sum_uncached_expected=11,max_output_i_sum=40,phases={'session_start':1,'intra':3})
            chain.pop('source_chain_targets')
            source_rows.extend(rows);source_chains.append(chain);source_bodies.extend(bodies)
        write_lines(source/'requests.jsonl',source_rows);write_lines(source/'chains.jsonl',source_chains)
        with gzip.open(source/'bodies/source.jsonl.gz','wt') as f:
            for b in source_bodies:lc.write_line(f,b)
        target=tmp/'built'
        lc.build(argparse.Namespace(source_root=str(source),out=str(target),harness_dir=str(REPO/'s1-dev/harness'),
                  tok_dir=str(REPO/'s1-dev/glm_tok'),chains=2,seed=123,set='probe',max_context_tokens=1000))
        idx,_,_=common.load_index(str(target))
        generated=list(lc.read_jsonl(target/'requests.jsonl')); prov=list(lc.read_jsonl(target/'provenance.jsonl'))
        out['identity_time_build']={'checker':check(target),'rows':len(generated),
             'sessions_by_chain':{cid:sorted({r['session_id'] for r in generated if r['chain_id']==cid}) for cid in ('c0','c1')},
             'offsets_by_chain':{cid:[r['dispatch_offset_ms'] for r in generated if r['chain_id']==cid] for cid in ('c0','c1')},
             'source_session_ids':sorted({p['source_session_id'] for p in prov}),
             'synthetic_offset_origin':sorted({p['dispatch_offset_origin'] for p in prov if p['kind']=='synthetic'}),
             'cohort_requests':json.loads((target/'cohort.json').read_text())['n_requests']}
result=Path(__file__).with_name('repro-generation-bugs-baseline.json' if '--baseline' in sys.argv else 'repro-generation-bugs.json');lc.json_dump(result,out); print(json.dumps(out,indent=2))
