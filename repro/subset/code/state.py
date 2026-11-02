#!/usr/bin/env python3
"""Persistent EvoMAS knowledge; all shared state writes occur after whole-task evaluation."""
from __future__ import annotations
import argparse,asyncio,hashlib,json,os,re,shutil,sys,tempfile
from pathlib import Path
import yaml

TASKS={'bbo':'bbo_noisy_continuous','inventory':'bbo_simopt_inventory',
       'graph':'johnson1991_leighton_graph_coloring','qec':'qec_decoder_arena'}
BBO=['bbo_read_context','bbo_read_public_data','bbo_read_prior_feedback',
     'bbo_read_prior_trace','bbo_read_history','bbo_write_candidate',
     'bbo_preflight_candidate','bbo_probe_candidate']
SUBSET=['subset_read_context','subset_read_public','subset_read_history',
        'subset_write_candidate','subset_preflight']
TO_BBO=dict(zip(SUBSET,[BBO[0],BBO[1],BBO[4],BBO[5],BBO[6]]))
TO_SUBSET={x:SUBSET[0] for x in BBO[:1]}
TO_SUBSET.update({BBO[1]:SUBSET[1],BBO[2]:SUBSET[2],BBO[3]:SUBSET[2],
                  BBO[4]:SUBSET[2],BBO[5]:SUBSET[3],BBO[6]:SUBSET[4],BBO[7]:SUBSET[4]})

def atom(path,obj):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(tmp,path)

def adapt(text,task):
 c=yaml.safe_load(text)
 if not isinstance(c,dict) or not isinstance(c.get('agents'),dict):raise ValueError('Invalid MAS YAML')
 mapping=TO_BBO if task=='bbo' else TO_SUBSET
 allowed=set(BBO if task=='bbo' else SUBSET)
 for a in c['agents'].values():
  tools=[]
  for t in a.get('tools',[]):
   mapped=mapping.get(t,t)
   if mapped in allowed and mapped not in tools:tools.append(mapped)
  a['tools']=tools
  if task=='bbo' and 'bbo_read_history' not in tools:a['tools'].append('bbo_read_history')
  if a.get('model_id') not in ('openai:deepseek-v4-flash-s1','openai:deepseek-v4-flash-s2'):
   a['model_id']='openai:deepseek-v4-flash-s1'
 c.pop('meta',None)
 return yaml.safe_dump(c,sort_keys=False,allow_unicode=True)

def initialize(root,state):
 if state.exists():raise FileExistsError(state)
 state.mkdir(parents=True);(state/'pool').mkdir()
 seeds=root/'repro/subset/seeds'
 for name in ('majority_vote.yaml','smoa.yaml'):
  shutil.copyfile(seeds/name,state/'pool'/name)
 atom(state/'memory.json',[])
 atom(state/'manifest.json',{'schema':'evomas-rsi-task-memory-v1','runs':[],
                              'initial_seed_pool':str(seeds.resolve())})

def export(state,task,dest):
 if dest.exists():raise FileExistsError(dest)
 if task not in TASKS:raise ValueError(task)
 memory=json.loads((state/'memory.json').read_text())
 manifest=json.loads((state/'manifest.json').read_text())
 dest.mkdir(parents=True);(dest/'working_pool').mkdir()
 for p in sorted((state/'pool').glob('*.yaml')):
  (dest/'working_pool'/p.name).write_text(adapt(p.read_text(),task),encoding='utf8')
 atom(dest/'memory.json',memory)
 atom(dest/'manifest.json',{'schema':manifest['schema'],
  'runs':[{'task':x['task'],'config_sha256':x['config_sha256']} for x in manifest['runs']]})
 atom(dest/'provenance.json',{'state':str(state.resolve()),'task':task,
  'state_sha256':hashlib.sha256((state/'manifest.json').read_bytes()).hexdigest()})

def summary_of(run):
 s=json.loads((run/'summary.json').read_text())
 task=s.get('task','bbo')
 if task not in TASKS:raise ValueError('Unknown task '+str(task))
 if task=='bbo':
  if s.get('completed_solver_versions')!=s.get('solver_limit'):raise ValueError('Incomplete BBO')
 else:
  if s.get('status')!='completed' or s.get('completed_versions')!=6:raise ValueError('Incomplete subset run')
 if s.get('hidden_status')!='completed' or not isinstance(s.get('hidden_score'),(int,float)):
  raise ValueError('Hidden score required before committing knowledge')
 winner=run/'best_visible_solver.py'
 if hashlib.sha256(winner.read_bytes()).hexdigest()!=s['best_visible']['sha256']:
  raise ValueError('Winner SHA mismatch')
 return task,s

def consolidate(root,task,run,s):
 # Only PUBLIC execution observations enter the transferable experience.
 if task=='bbo':
  operations=s.get('operations',[])
  trace=(run/'evolution_trace.txt').read_text(encoding='utf8')[:7000]
  candidates=[{'label':c.get('label'),'score':c.get('visible_score'),'valid':c.get('contract_valid')}
              for c in s.get('candidates',[])]
  config=run/'best_config.yaml'
 else:
  operations=s.get('operations',[])
  trace=json.dumps(operations,ensure_ascii=False)[:7000]
  candidates=[{'label':c.get('label'),'score':c.get('visible',{}).get('metric'),
               'valid':c.get('valid')} for c in s['candidates']]
  config=run/'best_reward_config.yaml'
 if not config.is_file():raise FileNotFoundError(config)
 payload={'task':task,'public_candidates':candidates,'operations':operations,'trace':trace,
          'best_config_excerpt':config.read_text(encoding='utf8')[:9000]}
 prompt=('Summarize this ONE completed RSI-Exam research task into one transferable EvoMAS '
         'experience. Identify which prompt, model assignment, tools, or DAG coordination '
         'helped and which failed. Distinguish observed evidence from hypotheses. Use only '
         'the PUBLIC observations supplied; do not invent sealed labels or numerical scores. '
         'Return 100-240 words of plain text.\n'+json.dumps(payload,ensure_ascii=False)[:19000])
 sys.path.insert(0,str(root/'repro/subset/EvoMAS-openai'))
 from src.models import get_model
 model=get_model('openai:deepseek-v4-flash-s1',temperature=0,max_tokens=1800)
 return str(asyncio.run(model(prompt)))[:5000],config

def commit(root,state,run):
 import fcntl
 task,s=summary_of(run)
 with (state/'lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  manifest=json.loads((state/'manifest.json').read_text())
  if str(run.resolve()) in {x['run'] for x in manifest['runs']}:raise ValueError('Run already committed')
  analysis,config=consolidate(root,task,run,s)
  raw=config.read_text(encoding='utf8');digest=hashlib.sha256(raw.encode()).hexdigest()
  cfg=yaml.safe_load(raw)
  cfg['name']=f'evolved_{task}_{digest[:12]}'
  cfg['description']=f'Public-reward best MAS from RSI-Exam {TASKS[task]}; adapt for new tasks'
  cfg['successful_tasks']=[{'q':'RSI-Exam '+TASKS[task],
     'notes':'Best configuration on public cases; do not compare raw metrics across tasks'}]
  raw=yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True)
  out=state/'pool'/f'evolved_{task}_{digest[:12]}.yaml'
  if out.exists() and out.read_text()!=raw:raise RuntimeError('Pool hash collision')
  out.write_text(raw,encoding='utf8')
  memory=json.loads((state/'memory.json').read_text())
  candidates=s.get('candidates',[])
  def score(c):return c.get('visible',{}).get('metric') if task!='bbo' else c.get('visible_score')
  values=[score(c) for c in candidates if isinstance(score(c),(int,float))]
  first=float(values[0]) if values else 0.0
  best=(s['best_visible'].get('visible',{}).get('metric') if task!='bbo'
        else s['best_visible'].get('visible_score'))
  memory_query=('RSI-Exam Query-Budgeted Noisy Black-Box Optimization bbo_noisy_continuous'
                if task=='bbo' else 'RSI-Exam '+TASKS[task])
  memory.append({'query':memory_query, 'action':'consolidate',
    'config_changes':f'Best MAS {digest[:12]}; {len(operations_of(s))} evolution operations',
    'old_accuracy':first,'new_accuracy':float(best or 0),'success':bool(best is not None and best>first),
    'analysis':analysis})
  atom(state/'memory.json',memory)
  manifest['runs'].append({'task':task,'run':str(run.resolve()),'config_sha256':digest,
                           'public_winner_sha256':s['best_visible']['sha256'],
                           'hidden_score':s['hidden_score']})
  atom(state/'manifest.json',manifest)
  print(json.dumps(manifest['runs'][-1],ensure_ascii=False))

def operations_of(s):return s.get('operations',[])

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
 for cmd in ('init','export','commit','merge'):
  sp=sub.add_parser(cmd);sp.add_argument('--root',type=Path,required=True);sp.add_argument('--state',type=Path,required=True)
  if cmd=='export':sp.add_argument('--task',choices=TASKS,required=True);sp.add_argument('--dest',type=Path,required=True)
  if cmd=='commit':sp.add_argument('--run',type=Path,required=True)
  if cmd=='merge':
   sp.add_argument('--mode',choices=('original','joint','parallel'),required=True)
   sp.add_argument('--seed',type=int,required=True)
 a=p.parse_args();root=a.root.resolve();state=a.state.resolve()
 if a.command=='init':initialize(root,state)
 elif a.command=='export':export(state,a.task,a.dest)
 elif a.command=='commit':commit(root,state,a.run.resolve())
 else:
  ready=[]
  for task in TASKS:
   matches=[path for path in (root/'repro/subset/outputs').glob(
       f'{task}_{a.mode}_seed{a.seed}_*') if path.is_dir() and (path/'summary.json').is_file()]
   if len(matches)!=1:raise ValueError(f'{task}: expected one completed run, found {len(matches)}')
   run=matches[0].resolve()
   summary_of(run)
   ready.append(run)
  for run in ready:
   manifest=json.loads((state/'manifest.json').read_text())
   if str(run) not in {item['run'] for item in manifest['runs']}:
    commit(root,state,run)
if __name__=='__main__':main()
