#!/usr/bin/env python3
"""EvoMAS subset research: official starter per task; native MAS DAG and meta operators.

A research process only. It never mounts tests/heldout or directly reads hidden labels.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import shutil
import sys
import time
from pathlib import Path
import yaml
from subset_visible import NAMES, evaluate, static_gate

MODELS=['openai:deepseek-v4-flash-s1','openai:deepseek-v4-flash-s2']
TOOLS=['subset_read_context','subset_read_public','subset_read_history',
       'subset_write_candidate','subset_preflight','subset_public_python']

def seed_yaml(text):
 config=yaml.safe_load(text)
 for a in config['agents'].values():
  a['tools']=TOOLS.copy()
  if a['model_id'] not in MODELS:a['model_id']=MODELS[0]
 return yaml.safe_dump(config,sort_keys=False)

def inherited_yaml(text):
 config=yaml.safe_load(text)
 if not isinstance(config,dict):raise ValueError('Invalid YAML mapping')
 mapping={'bbo_read_context':'subset_read_context','bbo_read_public_data':'subset_read_public',
          'bbo_read_prior_feedback':'subset_read_history','bbo_read_prior_trace':'subset_read_history',
          'bbo_read_history':'subset_read_history','bbo_write_candidate':'subset_write_candidate',
          'bbo_preflight_candidate':'subset_preflight','bbo_probe_candidate':'subset_preflight'}
 for a in config['agents'].values():
  a['tools']=list(dict.fromkeys(mapping.get(x,x) for x in a.get('tools',[])
                                  if mapping.get(x,x) in TOOLS))
  if a['model_id'] not in MODELS:a['model_id']=MODELS[0]
 config.pop('meta',None)
 config['description']='Prior evolved task configuration; revalidate under current task'
 return yaml.safe_dump(config,sort_keys=False)

def score_for_reward(task,result):
 if not result['valid']:return -10000.0
 value=float(result['metric'])
 if task=='graph':return 100.0/(1.0+max(0,result['effective_conflicts'])/100.0)
 if task=='qec':return 100.0*max(0.0,min(1.0,value/2.5))
 return 100.0*max(0.0,min(1.0,value))

def main():
 p=argparse.ArgumentParser()
 p.add_argument('task',choices=NAMES)
 p.add_argument('--project-root',type=Path,required=True)
 p.add_argument('--task-root',type=Path,required=True)
 p.add_argument('--seed-pool',type=Path,required=True)
 p.add_argument('--output-dir',type=Path,required=True)
 p.add_argument('--carry-from',type=Path,help='Legacy one-run transfer; use frozen state snapshots for formal runs')
 p.add_argument('--initial-memory',type=Path,help='Frozen persistent memory snapshot')
 p.add_argument('--mutation-mode',choices=['original','joint','parallel'],default='original')
 p.add_argument('--seed',type=int,default=43)
 p.add_argument('--max-versions',type=int,default=6)
 p.add_argument('--mutation-probability',type=float,default=.7)
 p.add_argument('--dry-run',action='store_true')
 a=p.parse_args()
 if a.output_dir.exists():p.error('Output already exists; never overwrite a research run')
 if a.max_versions<3 or not 0<=a.mutation_probability<=1:p.error('Invalid limits')
 if not (a.task_root/'environment/methods/main/solver.py').is_file():p.error('Official starter absent')
 if not (a.task_root/'instruction.md').is_file():p.error('Official instruction absent')
 if not a.dry_run and not (a.project_root/'repro/subset/EvoMAS-openai/src/tools/subset_generation.py').is_file():
  p.error('Isolated EvoMAS fork absent')
 if a.carry_from and not (a.carry_from/'summary.json').is_file():p.error('Only a completed prior run may be inherited')
 if a.carry_from:
  previous=json.loads((a.carry_from/'summary.json').read_text())
  legacy_bbo=(previous.get('completed_solver_versions')==previous.get('solver_limit')
              and previous.get('best_visible',{}).get('selfcheck_passed') is True)
  if (previous.get('status')!='completed' and not legacy_bbo) or not (a.carry_from/'working_pool').is_dir():
   p.error('Previous run incomplete')
 started=time.monotonic();rng=random.Random(a.seed)
 a.output_dir.mkdir(parents=True)
 pool=a.output_dir/'working_pool';pool.mkdir()
 evo=a.project_root/'repro/subset/EvoMAS-openai'
 sys.path.insert(0,str(evo))
 sys.path.insert(0,str(Path(__file__).resolve().parent))
 from src.meta_model.metamodel import MetaModel
 from src.meta_model.reward import compute_reward
 from src.meta_model.pool import ConfigurationPool,add_to_pool_if_better
 from src.tools.subset_generation import CATALOG_TEXT
 from mutation_ablation import install
 from subset_runtime import validate_yaml,execute
 install(MetaModel)
 os.environ['EVOMAS_ALLOWED_MODELS']=' '.join(MODELS)
 initial_names=[];rejected=[]
 for path in sorted(a.seed_pool.glob('*.yaml')):
  text=(seed_yaml(path.read_text(encoding='utf8')) if path.stem in ('majority_vote','smoa')
        else inherited_yaml(path.read_text(encoding='utf8')))
  validate_yaml(text)
  (pool/path.name).write_text(text,encoding='utf8')
  if path.stem in ('majority_vote','smoa'):initial_names.append(path.name)
 if len(initial_names)!=2:raise ValueError('Exactly majority_vote and smoa must exist in initial seed pool')
 if a.carry_from:
  for path in sorted((a.carry_from/'working_pool').glob('*.yaml')):
   if path.name in initial_names:continue
   try:
    text=inherited_yaml(path.read_text(encoding='utf8'))
    validate_yaml(text)
    (pool/path.name).write_text(text,encoding='utf8')
   except Exception as exc:rejected.append({'file':path.name,'reason':str(exc)[:500]})
  old=a.carry_from/'memory.json'
  if old.is_file():
   records=json.loads(old.read_text())
   if not isinstance(records,list):raise ValueError('EvoMAS memory must be a list')
   (a.output_dir/'memory.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
 if a.initial_memory:
  if a.carry_from:raise ValueError('Use only one memory input')
  records=json.loads(a.initial_memory.read_text())
  if not isinstance(records,list):raise ValueError('Memory must be a list')
  (a.output_dir/'memory.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
 if a.dry_run:
  class Fake:
   def select(self,*args,**kwargs):return [str(pool/n) for n in initial_names]
   def generate(self,path,*args,**kw):return Path(path).read_text()
   def mutate(self,config,*args,**kw):return config
   def crossover(self,c1,*args,**kw):return c1
   def update_memory(self,**kw):pass
  meta=Fake()
 else:
  meta=MetaModel(model_id=MODELS[0],memory_path=str(a.output_dir/'memory.json'),
                 memory_evolution=True)
  meta.bbo_tool_policy=CATALOG_TEXT+'\nFor this task, keep only tools from this catalog.'
  meta.mutation_mode=a.mutation_mode
  meta.mutation_trace_path=a.output_dir/'mutation_trace.jsonl'
 task_query=f'RSI-Exam {NAMES[a.task]}: '+(a.task_root/'instruction.md').read_text()[:250]
 task_description=(a.task_root/'instruction.md').read_text()[:9000]
 def valid_or_parent(proposed,parent,label):
  try:validate_yaml(proposed);return proposed
  except Exception as exc:
   (a.output_dir/f'{label}.rejected.yaml').write_text(str(proposed))
   (a.output_dir/f'{label}.rejected_reason.txt').write_text(str(exc))
   return parent
 try:parents=meta.select(task_query,task_description,str(pool),k=2)
 except Exception:parents=[str(pool/n) for n in initial_names]
 if len(set(map(str,parents)))!=2:parents=[str(pool/n) for n in initial_names]
 # Provenance: only two parent MAS configurations; no previous solver source.
 selected_paths=[Path(x) for x in parents]
 for path in selected_paths:
  if path.parent.resolve()!=pool.resolve():raise ValueError('Parent outside current task pool')
 configs=[];metrics=[];candidates=[];history=[];operations=[]
 def evaluate_one(label,config):
  version=a.output_dir/'evaluations'/label;version.mkdir(parents=True)
  (a.output_dir/'evaluations'/f'{label}.yaml').write_text(config)
  t0=time.monotonic();runtime={};result={}
  try:
   runtime=execute(a.output_dir/'evaluations'/f'{label}.yaml',a.task_root,a.task,
                   version,history,dry_run=a.dry_run)
   solver=version/'candidate_solver.py'
   result=({'valid':True,'metric':0.01,'raw_conflicts':100} if a.dry_run else
           evaluate(a.task,a.task_root,solver))
  except Exception as exc:
   (version/'candidate_solver.py').write_text('# Invalid candidate: see failure.json\n') if not (version/'candidate_solver.py').exists() else None
   result={'valid':False,'error_type':type(exc).__name__,'error':str(exc)[-1600:]}
  (version/'visible.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
  solver=version/'candidate_solver.py';digest=hashlib.sha256(solver.read_bytes()).hexdigest()
  meta_info=runtime.get('metadata',{})
  item={'label':label,'sha256':digest,'visible':result,
        'candidate_path':str(solver),'elapsed':time.monotonic()-t0,
        'valid':result['valid']}
  candidates.append(item)
  # Agent-reported total tokens are not a complete billing record; retain as a proxy.
  tokens=int(meta_info.get('total_tokens') or
             (meta_info.get('input_tokens',0)+meta_info.get('output_tokens',0)))
  metric={'accuracy':score_for_reward(a.task,result),'total_time':item['elapsed'],
          'total_tokens':tokens,'visible_result':result}
  configs.append(config);metrics.append(metric)
  history.append({'label':label,'metric':result.get('metric'),'valid':result['valid'],
                  'error':result.get('error'),'source':solver.read_text(encoding='utf8')[:100000]})
  (a.output_dir/'candidate_manifest.json').write_text(json.dumps(candidates,ensure_ascii=False,indent=2))
  return metric
 for i,path in enumerate(selected_paths):
  seed=path.read_text(encoding='utf8')
  try:proposed=meta.generate(str(path),[{'query':task_query}],task_description,model_list=MODELS)
  except Exception:proposed=seed
  evaluate_one(f'parent_{i:02d}',valid_or_parent(proposed,seed,f'parent_{i:02d}'))
 if not candidates:raise RuntimeError('No version evaluated')
 while len(candidates)<a.max_versions:
  step=len(candidates)-1
  rewards=[compute_reward(m,beta=1e-6,cost_weight='both') for m in metrics]
  best=max(range(len(rewards)),key=rewards.__getitem__)
  use_mutation=len(configs)<2 or rng.random()<a.mutation_probability
  action='mutation' if use_mutation else 'crossover'
  label=f'step_{step:02d}_{action}'
  feedback=json.dumps(history[-6:],ensure_ascii=False)[-14000:]
  try:
   if use_mutation: proposed=meta.mutate(configs[best],metrics[best],feedback,model_list=MODELS)
   else:
    ranked=sorted(range(len(rewards)),key=rewards.__getitem__,reverse=True)
    proposed=meta.crossover(configs[ranked[0]],configs[ranked[1]],metrics[ranked[0]],
                            metrics[ranked[1]],model_list=MODELS)
  except Exception:proposed=configs[best]
  config=valid_or_parent(proposed,configs[best],label)
  child=evaluate_one(label,config)
  new_reward=compute_reward(child,beta=1e-6,cost_weight='both')
  operations.append({'label':label,'action':action,'parent_reward':rewards[best],
                     'reward':new_reward,'improved':new_reward>rewards[best]})
  # The whole task is consolidated once by state.py after public and hidden evaluation.
  if use_mutation and a.mutation_mode!='original' and not a.dry_run:
   with (a.output_dir/'mutation_trace.jsonl').open('a') as f:
    f.write(json.dumps({'event':'outcome','label':label,'visible_score':child['visible_result'].get('metric'),
       'parent_visible_score':metrics[best]['visible_result'].get('metric'),
       'improved_reward':new_reward>rewards[best]})+'\n')
 rewards=[compute_reward(m,beta=1e-6,cost_weight='both') for m in metrics]
 best_config=max(range(len(rewards)),key=rewards.__getitem__)
 (a.output_dir/'best_reward_config.yaml').write_text(configs[best_config],encoding='utf8')
 if metrics[best_config]['visible_result']['valid']:
  pool_manager=ConfigurationPool(str(pool))
  accepted=add_to_pool_if_better(pool_manager,configs[best_config],metrics[best_config],
      metrics[:min(2,len(selected_paths))],task_query,beta=1e-6,cost_weight='both',threshold=.05)
  if accepted:
   try:validate_yaml(Path(accepted).read_text())
   except Exception as exc:
    (a.output_dir/'rejected_pool_configs').mkdir(exist_ok=True)
    shutil.move(accepted,str(a.output_dir/'rejected_pool_configs'/Path(accepted).name))
    rejected.append({'file':str(accepted),'reason':str(exc)})
 eligible=[c for c in candidates if c['valid'] and math.isfinite(c['visible']['metric'])]
 if not eligible:raise RuntimeError('No valid public candidate; do not submit hidden')
 winner=max(eligible,key=lambda c:c['visible']['metric'])
 shutil.copyfile(winner['candidate_path'],a.output_dir/'best_visible_solver.py')
 if hashlib.sha256((a.output_dir/'best_visible_solver.py').read_bytes()).hexdigest()!=winner['sha256']:
  raise RuntimeError('Winner digest mismatch')
 image_id=None
 if not a.dry_run:
  image='rsi-subset-'+NAMES[a.task]+':public'
  inspect=subprocess.run(['docker','image','inspect','--format','{{.Id}}',image],
                         capture_output=True,text=True,check=True)
  image_id=inspect.stdout.strip()
 summary={'status':'completed','task':a.task,'mutation_mode':a.mutation_mode,
          'public_image_id':image_id,
          'instruction_sha256':hashlib.sha256((a.task_root/'instruction.md').read_bytes()).hexdigest(),
          'best_visible':winner,'hidden_status':'not_run','hidden_score':None,
          'completed_versions':len(candidates),'operations':operations,
          'working_pool':str(pool),'rejected_inherited_pool_configs':rejected,
          'research_seconds':time.monotonic()-started,'meta_model':MODELS[0],
          'agent_max_steps':50 if a.task=='qec' else 45,
          'model_request_timeout_sec':float(os.environ.get('BBO_OPENAI_TIMEOUT','120')),
          'model_request_retries':int(os.environ.get('BBO_OPENAI_MAX_RETRIES','1')),
          'model_pool':MODELS,'seed':a.seed,'max_versions':a.max_versions,
          'mutation_probability':a.mutation_probability,
          'starter_sha256':hashlib.sha256((a.task_root/'environment/methods/main/solver.py').read_bytes()).hexdigest(),
          'carry_from':str(a.carry_from) if a.carry_from else None,
          'initial_memory':str(a.initial_memory) if a.initial_memory else None,
          'memory_granularity':'one_consolidation_per_task_after_run',
          'candidate_count':len(candidates),'candidates':candidates}
 (a.output_dir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 print(json.dumps({'output_dir':str(a.output_dir),'best':winner['label'],
                   'visible':winner['visible'],'versions':len(candidates)},ensure_ascii=False))
if __name__=='__main__':main()
