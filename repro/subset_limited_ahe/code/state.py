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
        'subset_write_candidate','subset_preflight','subset_public_python']
TO_BBO=dict(zip(SUBSET,[BBO[0],BBO[1],BBO[4],BBO[5],BBO[6]]))
TO_SUBSET={x:SUBSET[0] for x in BBO[:1]}
TO_SUBSET.update({BBO[1]:SUBSET[1],BBO[2]:SUBSET[2],BBO[3]:SUBSET[2],
                  BBO[4]:SUBSET[2],BBO[5]:SUBSET[3],BBO[6]:SUBSET[4],BBO[7]:SUBSET[4]})

def atom(path,obj):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(tmp,path)

def adapt(text,task,source_task=None):
 c=yaml.safe_load(text)
 if not isinstance(c,dict) or not isinstance(c.get('agents'),dict):raise ValueError('Invalid MAS YAML')
 mapping=TO_BBO if task=='bbo' else TO_SUBSET
 allowed=set(BBO if task=='bbo' else SUBSET)
 for agent_id,a in c['agents'].items():
  if source_task is None and task in ('graph','qec'):
   # Explicit roles survive task adaptation; every seed agent has a distinct job.
   graph_roles={
    'worker1':'Propose one efficient TabuCol-style solver with incremental conflict counts and bounded per-graph search.',
    'worker2':'Propose a distinct restart or perturbation policy; compare per-graph conflicts on public instances.',
    'worker3':'Audit color range, return lengths, deadline guards, and fallbacks on the largest public graphs.',
    'proc_l1_1':'Propose a fast incremental conflict-minimizing search with a concrete bounded update rule.',
    'proc_l1_2':'Propose an alternative initialization and restart strategy with explicit large-graph complexity.',
    'judge_l1':'Compare both proposals by public conflicts, 45-second feasibility, and implementation risk; select a direction.',
    'proc_l2_1':'Refine the selected search implementation; address a measured conflict or throughput bottleneck.',
    'proc_l2_2':'Independently challenge the selected algorithm on deadlines, color validity, and relabelled graphs.',
    'judge_l2':'Compare the two revisions using public per-graph conflicts and correctness; state unresolved risks.',
    'aggregator':'Synthesize one complete solve(data) returning all requested colorings, integers in [0,target_k); selfcheck within 45 seconds per graph.'}
   qec_roles={
    'worker1':'Investigate correlation-aware decoding using the public DEM and labeled development data; test a concrete hypothesis.',
    'worker2':'Investigate dev-calibrated weights or color-pair decoding; avoid memorizing public evaluation labels.',
    'worker3':'Audit decode output shape and boolean dtype for d5, d5_Z, d7, d9 and per-setting runtime.',
    'proc_l1_1':'Propose a complete correlation-aware decoder using allowed libraries and public development labels.',
    'proc_l1_2':'Propose an independent calibration or color-pair method and its generalization risk at d7 and d9.',
    'judge_l1':'Compare both decoder proposals on four public setting scores, valid return dtype, and time budgets.',
    'proc_l2_1':'Revise the selected decoder, keeping the complete decode(setting,dets) interface.',
    'proc_l2_2':'Challenge d7/d9 behavior, boolean [n_shots,n_observables] shape, and unseen-setting robustness.',
    'judge_l2':'Select the revision with strongest four-setting public evidence; list failures explicitly.',
    'aggregator':'Produce one complete decode(setting,dets) returning a boolean [n_shots,n_observables] array for every setting; check all four public settings.'}
   focus=(graph_roles if task=='graph' else qec_roles).get(agent_id)
   if focus is None:
    focus=f'As {agent_id}, develop a distinct {task} solver proposal and validate it against the official task contract.'
   a['prompt']=focus+' Use only imports and public assets allowed by the current official task.'
  if source_task in TASKS and source_task!=task:
   # Carry the MAS roles/topology/model knowledge, but not another task's solver instructions.
   role=str(a.get('role') or a.get('id') or 'worker')[:80]
   a['prompt']=(f'Act as {role} on the CURRENT RSI-Exam task. Follow its official '
                'instruction, cooperate through this MAS topology, and produce a '
                'valid task-specific solver. Do not reuse a previous task solver.')
  tools=[]
  for t in a.get('tools',[]):
   mapped=mapping.get(t,t)
   if mapped in allowed and mapped not in tools:tools.append(mapped)
  if source_task is None and task in ('graph','qec') and agent_id!='aggregator':
   tools=[tool for tool in tools if tool not in ('subset_write_candidate','subset_preflight')]
  a['tools']=tools
  if task=='bbo' and 'bbo_read_history' not in tools:a['tools'].append('bbo_read_history')
  if a.get('model_id') not in ('openai:deepseek-v4-flash-s1','openai:deepseek-v4-flash-s2',
                                'openai:GLM-5.3-W4A8','openai:GLM-5.3-Flash'):
   a['model_id']='openai:deepseek-v4-flash-s1'
 c.pop('meta',None)
 return yaml.safe_dump(c,sort_keys=False,allow_unicode=True)

def initialize(root,state):
 if state.exists():raise FileExistsError(state)
 state.mkdir(parents=True);(state/'pool').mkdir()
 seeds=root/'repro/subset_limited_ahe/seeds'
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
  source_task=next((name for name in TASKS if p.stem.startswith('evolved_'+name+'_')),None)
  if p.name not in ('majority_vote.yaml','smoa.yaml') and source_task != task:continue
  (dest/'working_pool'/p.name).write_text(adapt(p.read_text(),task,source_task),encoding='utf8')
 memory=[item for item in memory if item.get('task')==task or
         (not item.get('task') and TASKS[task] in item.get('query',''))]
 # The state keeps task routing metadata; EvoMAS ActionExperience accepts only
 # its seven original fields when loading the per-run memory snapshot.
 experience_fields=('query','action','config_changes','old_accuracy',
                    'new_accuracy','success','analysis')
 atom(dest/'memory.json',[{key:item[key] for key in experience_fields} for item in memory])
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
  expected=s.get('max_versions')
  if (expected not in (4,6) or s.get('status')!='completed' or
      s.get('completed_versions')!=expected or len(s.get('candidates',[]))!=expected):
   raise ValueError('Incomplete subset run or unsupported version count')
 if s.get('hidden_status')!='completed' or not isinstance(s.get('hidden_score'),(int,float)):
  raise ValueError('Hidden score required before committing knowledge')
 if task=='bbo':
  diagnostic=Path(s.get('final_result',{}).get('diagnostic_path') or '')
  if diagnostic.is_file():
   detail=json.loads(diagnostic.read_text())
   if (detail.get('correctness') is False and detail.get('num_evals')==0
       and any('grader failed:' in str(error) for error in detail.get('errors') or [])):
    raise ValueError('Trusted BBO grader failed before solver evaluation; do not commit')
 winner=run/'best_visible_solver.py'
 if hashlib.sha256(winner.read_bytes()).hexdigest()!=s['best_visible']['sha256']:
  raise ValueError('Winner SHA mismatch')
 return task,s

def consolidate(root, task, run, s):
 """Deterministic public-only task memory; hypotheses remain labelled as such."""
 config=(run/'best_visible_config.yaml' if task=='bbo' else
         run/'evaluations'/f"{s['best_visible']['label']}.yaml")
 if not config.is_file():raise FileNotFoundError(config)
 candidates=s.get('candidates',[])
 def score(item):
  return (item.get('visible_score') if task=='bbo' else
          item.get('visible',{}).get('metric'))
 parents={x['label']:score(x) for x in candidates if x.get('label') in ('parent_00','parent_01')}
 best=(s['best_visible'].get('visible_score') if task=='bbo' else
       s['best_visible'].get('visible',{}).get('metric'))
 decision_path=run/'decision_evidence.jsonl'
 rows=[json.loads(line) for line in decision_path.read_text().splitlines()] if decision_path.is_file() else []
 parent_best=max((value for value in parents.values() if isinstance(value,(int,float))),default=None)
 peak=parent_best
 evidence=[]
 for row in rows:
  visible=row.get('child_public') or {}
  score_now=(row.get('child_score') if task=='bbo' else visible.get('metric'))
  if not isinstance(score_now,(int,float)) or not visible.get('valid',True):
   outcome='invalid'
  elif not row.get('effective_change') or peak is None or score_now==peak:
   outcome='unresolved'
  elif score_now>peak:
   outcome='worked'
  else:
   outcome='regressed'
  setting_errors=[v.get('setting') for v in visible.get('settings',[]) if v.get('error')]
  decision=row.get('model_decision') or {}
  evidence.append({'step':row.get('label'),
    'component':(row.get('changed_components') or [None])[0],
    'public_score':score_now,'best_before':peak,'outcome':outcome,
    'per_setting_errors':setting_errors,
    'hypothesis_unverified':str(decision.get('root_cause_hypothesis') or '')[:250]})
  if isinstance(score_now,(int,float)) and visible.get('valid',True):
   peak=max(score_now,peak) if peak is not None else score_now
 analysis=json.dumps({'task':task,'basis':'public selfcheck only',
    'public_metric':'-conflicts, higher is better' if task=='graph' else 'higher is better',
    'parent_scores':parents,'selected_public_winner':s['best_visible']['label'],
    'winner_public_score':best,'iteration_evidence':evidence,
    'warning':'Hypotheses are agent claims, not established causes; ties are not improvements.'},
    ensure_ascii=False,default=str)
 return analysis,config

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
  cfg['description']=f'Public-score best MAS from RSI-Exam {TASKS[task]}; use for this task only'
  cfg['successful_tasks']=[{'q':'RSI-Exam '+TASKS[task],
     'notes':'Best configuration on public cases; do not compare raw metrics across tasks'}]
  raw=yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True)
  out=state/'pool'/f'evolved_{task}_{digest[:12]}.yaml'
  if out.exists() and out.read_text()!=raw:raise RuntimeError('Pool hash collision')
  # Persist only a distinct, validated improvement over the parents and the
  # historical public best under the same task image and instruction.
  parent_texts=[(run/'evaluations'/f'parent_{i:02d}.yaml').read_text(encoding='utf8')
                for i in range(2) if (run/'evaluations'/f'parent_{i:02d}.yaml').is_file()]
  if task=='bbo':
   # The BBO pipeline validates the executable MAS before writing best_config.
   if not isinstance(yaml.safe_load(raw),dict):raise ValueError('Invalid BBO MAS')
  else:
   sys.path.insert(0,str(root/'repro/subset_limited_ahe/EvoMAS-openai'))
   sys.path.insert(0,str(root/'repro/subset_limited_ahe/code'))
   from subset_runtime import validate_yaml
   validate_yaml(raw)
  candidates=s.get('candidates',[])
  def score(c):return c.get('visible',{}).get('metric') if task!='bbo' else c.get('visible_score')
  parent_scores=[score(c) for c in candidates if c.get('label') in ('parent_00','parent_01')
                 and isinstance(score(c),(int,float)) and
                 (c.get('valid',True) if task!='bbo' else c.get('contract_valid',True))]
  best=(s['best_visible'].get('visible',{}).get('metric') if task!='bbo'
        else s['best_visible'].get('visible_score'))
  improved=(s['best_visible'].get('label','').startswith('step_') and
            isinstance(best,(int,float)) and bool(parent_scores) and
            best>max(parent_scores))
  sys.path.insert(0,str(root/'repro/subset_limited_ahe/code'))
  from trajectory_evidence import crossover_change
  distinct=all(crossover_change(x,config.read_text(encoding='utf8'))
               for x in parent_texts)
  # A partial QEC selfcheck can improve its mean while two settings still crash.
  # Preserve that observation in memory, but do not promote its YAML to the pool.
  qec_settings=s['best_visible'].get('visible',{}).get('settings',[]) if task=='qec' else []
  complete=(task!='qec' or len(qec_settings)==4 and
            all(not item.get('error') for item in qec_settings))
  memory=json.loads((state/'memory.json').read_text())
  if len(memory)!=len(manifest['runs']):
   raise ValueError('Persistent memory and manifest are out of sync')
  historical=[]
  for past,record in zip(manifest['runs'],memory):
   if past.get('task')!=task or record.get('task')!=task:continue
   previous=Path(past['run'])/'summary.json'
   if previous.is_file():
    old=json.loads(previous.read_text())
    if any(old.get(key)!=s.get(key) for key in ('public_image_id','instruction_sha256')):
     continue
   elif any(record.get(key)!=s.get(key) for key in ('public_image_id','instruction_sha256')):
    continue
   value=record.get('new_accuracy')
   if isinstance(value,(int,float)) and not isinstance(value,bool):
    historical.append(float(value))
  historical_best=max(historical,default=None)
  historical_improved=(isinstance(best,(int,float)) and
                       (historical_best is None or best>historical_best+1e-9))
  accepted=bool(improved and distinct and complete and historical_improved)
  if accepted:out.write_text(raw,encoding='utf8')
  analysis_data=json.loads(analysis)
  analysis_data['historical_public_best']=historical_best
  analysis_data['archived_yaml']=accepted
  analysis=json.dumps(analysis_data,ensure_ascii=False,default=str)
  first=float(max(parent_scores)) if parent_scores else 0.0
  memory_query=('RSI-Exam Query-Budgeted Noisy Black-Box Optimization bbo_noisy_continuous'
                if task=='bbo' else 'RSI-Exam '+TASKS[task])
  memory.append({'query':memory_query, 'action':'consolidate',
    'config_changes':f'Best MAS {digest[:12]}; {len(operations_of(s))} evolution operations',
    'old_accuracy':first,'new_accuracy':float(best or 0),
    'success':accepted,'task':task,
    'public_image_id':s.get('public_image_id'),
    'instruction_sha256':s.get('instruction_sha256'),
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
 for cmd in ('init','export','commit','merge','replay','reconcile'):
  sp=sub.add_parser(cmd);sp.add_argument('--root',type=Path,required=True);sp.add_argument('--state',type=Path,required=True)
  if cmd=='export':sp.add_argument('--task',choices=TASKS,required=True);sp.add_argument('--dest',type=Path,required=True)
  if cmd in ('commit','reconcile'):sp.add_argument('--run',type=Path,required=True)
  if cmd=='replay':sp.add_argument('--source-state',type=Path,required=True)
  if cmd=='merge':
   sp.add_argument('--mode',choices=('original',),required=True)
   sp.add_argument('--seed',type=int,required=True)
 a=p.parse_args();root=a.root.resolve();state=a.state.resolve()
 if a.command=='init':initialize(root,state)
 elif a.command=='export':export(state,a.task,a.dest)
 elif a.command=='commit':commit(root,state,a.run.resolve())
 elif a.command=='reconcile':
  import fcntl
  run=a.run.resolve();task,s=summary_of(run)
  with (state/'lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   manifest=json.loads((state/'manifest.json').read_text())
   hits=[item for item in manifest['runs'] if item['run']==str(run)]
   if len(hits)!=1 or hits[0]['task']!=task:
    raise ValueError('Run must already be committed exactly once to this state')
   hits[0]['hidden_score']=s['hidden_score']
   hits[0]['hidden_status']='completed'
   atom(state/'manifest.json',manifest)
   print(json.dumps({'task':task,'run':str(run),'hidden_score':s['hidden_score']}))
 elif a.command=='replay':
  source=a.source_state.resolve()
  if source==state:raise ValueError('Replay source and destination must be distinct')
  runs=json.loads((source/'manifest.json').read_text())['runs']
  for item in runs:
   run=Path(item['run']).resolve()
   if not (run/'summary.json').is_file():raise FileNotFoundError(run)
   current=json.loads((state/'manifest.json').read_text())['runs']
   if str(run) not in {x['run'] for x in current}:commit(root,state,run)
 else:
  ready=[]
  for task in TASKS:
   matches=[path for path in (root/'repro/subset_limited_ahe/outputs').glob(
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
