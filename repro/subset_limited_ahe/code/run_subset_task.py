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
from skill_memory import render

MODELS=['openai:'+name.strip() for name in os.environ.get('MODEL_POOL',
        'deepseek-v4-flash-s1,deepseek-v4-flash-s2').split(',')]
from trajectory_evidence import (mutation_component, crossover_change, crossover_hybrid, append as append_evidence,
    context as evidence_context, public_rank)
TOOLS=['subset_read_context','subset_read_public','subset_read_history',
       'subset_write_candidate','subset_preflight','subset_public_python']

def constrain_tools(config):
 agents=config['agents'];reports=config['topology']['reports_to']
 terminal=[aid for aid in agents if not reports.get(aid)]
 if len(terminal)!=1:raise ValueError('Expected exactly one terminal candidate writer')
 restricted={'subset_write_candidate','subset_preflight','subset_public_python'}
 for aid,agent in agents.items():
  assigned=list(dict.fromkeys(agent.get('tools',[])))
  if aid==terminal[0]:
   agent['tools']=list(dict.fromkeys(assigned+[
       'subset_write_candidate','subset_preflight','subset_public_python']))
  else:
   agent['tools']=[tool for tool in assigned if tool not in restricted]
 return config

def seed_yaml(text):
 config=yaml.safe_load(text)
 config=constrain_tools(config)
 for a in config['agents'].values():
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
 constrain_tools(config)
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
 p.add_argument('--mutation-mode',choices=['original'],default='original')
 p.add_argument('--seed',type=int,default=43)
 p.add_argument('--max-versions',type=int,default=6)
 p.add_argument('--mutation-probability',type=float,default=.7)
 p.add_argument('--dry-run',action='store_true')
 a=p.parse_args()
 if a.output_dir.exists():p.error('Output already exists; never overwrite a research run')
 if a.max_versions<3 or not 0<=a.mutation_probability<=1:p.error('Invalid limits')
 if not (a.task_root/'environment/methods/main/solver.py').is_file():p.error('Official starter absent')
 if not (a.task_root/'instruction.md').is_file():p.error('Official instruction absent')
 if not a.dry_run and not (a.project_root/'repro/subset_limited_ahe/EvoMAS-openai/src/tools/subset_generation.py').is_file():
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
 evo=a.project_root/'repro/subset_limited_ahe/EvoMAS-openai'
 sys.path.insert(0,str(evo))
 sys.path.insert(0,str(Path(__file__).resolve().parent))
 from src.meta_model.metamodel import MetaModel
 from src.meta_model.reward import compute_reward
 from src.meta_model.pool import ConfigurationPool,add_to_pool_if_better
 from src.tools.subset_generation import CATALOG_TEXT
 from subset_runtime import validate_yaml,execute
 snapshot=a.seed_pool.parent
 skills=json.loads((snapshot/'skills.json').read_text()) if (snapshot/'skills.json').is_file() else []
 role_notes=json.loads((snapshot/'role_memory.json').read_text()) if (snapshot/'role_memory.json').is_file() else []
 shared_skills=render(skills)
 os.environ['EVOMAS_ALLOWED_MODELS']=' '.join(MODELS)
 initial_names=[];rejected=[]
 for path in sorted(a.seed_pool.glob('*.yaml')):
  if path.stem not in ('majority_vote','smoa') and not path.stem.startswith(f'evolved_{a.task}_'):
   continue
  text=(seed_yaml(path.read_text(encoding='utf8')) if path.stem in ('majority_vote','smoa')
        else inherited_yaml(path.read_text(encoding='utf8')))
  validate_yaml(text)
  (pool/path.name).write_text(text,encoding='utf8')
  if path.stem in ('majority_vote','smoa'):initial_names.append(path.name)
 if len(initial_names)!=2:raise ValueError('Exactly majority_vote and smoa must exist in initial seed pool')
 if a.carry_from:
  for path in sorted((a.carry_from/'working_pool').glob('*.yaml')):
   if path.name in initial_names or not path.stem.startswith(f'evolved_{a.task}_'):continue
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
   def mutate_multi(self,config,*args,**kw):return config
   def crossover(self,c1,*args,**kw):return c1
   def update_memory(self,**kw):pass
  meta=Fake()
 else:
  meta=MetaModel(model_id='openai:'+os.environ.get('META_MODEL','deepseek-v4-flash-s1'),memory_path=str(a.output_dir/'memory.json'),
                 memory_evolution=True)
  meta.bbo_tool_policy=(CATALOG_TEXT+'\nFor this task, keep only tools from this catalog. '
   'Only the single terminal agent (no outgoing reports_to edges) may receive '
   'subset_write_candidate, subset_preflight, or subset_public_python. '
   'Other agents must return proposals in their reports. Preserve one terminal agent. '
   'A prompt or model_id mutation must preserve every tool assignment, topology, '
   'agent ID, and every unrelated YAML field exactly. Do not repeat an edit that '
   'already regressed or failed without new public evidence for a distinct mechanism.')
  meta.mutation_decision_path=a.output_dir/'mutation_rationales.jsonl'
  meta.multi_trace_path=a.output_dir/'multi_trace.jsonl'
 task_query=f'RSI-Exam {NAMES[a.task]}: '+(a.task_root/'instruction.md').read_text()[:250]
 task_description=(a.task_root/'instruction.md').read_text()[:9000]
 task_description+='\n\nCross-task conditional research procedures (public-only):\n'+shared_skills[:3000]
 if not a.dry_run:
  meta.bbo_tool_policy+='\nCross-task conditional research procedures:\n'+shared_skills[:2500]
 def valid_or_parent(proposed,parent,label):
  try:validate_yaml(proposed);return proposed
  except Exception as exc:
   (a.output_dir/f'{label}.rejected.yaml').write_text(str(proposed))
   (a.output_dir/f'{label}.rejected_reason.txt').write_text(str(exc))
   return parent
 # Select from this task's pool; use the two seeds on selection failure.
 fallback=[str(pool/n) for n in initial_names]
 try:parents=meta.select(task_query,task_description,str(pool),k=2)
 except Exception:parents=fallback
 if (len(parents)!=2 or len(set(map(str,parents)))!=2 or
     any(Path(x).resolve().parent!=pool.resolve() or not Path(x).is_file() for x in parents)):
  parents=fallback
 # Provenance: only two parent MAS configurations; no previous solver source.
 selected_paths=[Path(x) for x in parents]
 for path in selected_paths:
  if path.parent.resolve()!=pool.resolve():raise ValueError('Parent outside current task pool')
 configs=[];metrics=[];candidates=[];history=[];operations=[]
 def evaluate_one(label,config,invalid_reason=None):
  version=a.output_dir/'evaluations'/label;version.mkdir(parents=True)
  (a.output_dir/'evaluations'/f'{label}.yaml').write_text(config)
  t0=time.monotonic();runtime={};result={}
  try:
   if invalid_reason is not None:
    raise ValueError('MAS configuration rejected before execution: '+invalid_reason)
   runtime=execute(a.output_dir/'evaluations'/f'{label}.yaml',a.task_root,a.task,
                   version,history,dry_run=a.dry_run,role_notes=role_notes,shared_skills=shared_skills)
   solver=version/'candidate_solver.py'
   if not a.dry_run:
    try:static_gate(a.task,solver.read_text(encoding='utf8'))
    except (SyntaxError,ValueError) as gate_error:
     failed_source=solver.read_text(encoding='utf8')
     (version/'rejected_candidate.py').write_text(failed_source,encoding='utf8')
     repair_history=history+[{'label':label+'_syntax_repair','metric':None,'valid':False,
      'error':f'Correct this candidate before public selfcheck: {type(gate_error).__name__}: {gate_error}',
      'source':failed_source[:100000]}]
     runtime=execute(a.output_dir/'evaluations'/f'{label}.yaml',a.task_root,a.task,
                     version,repair_history,dry_run=False,role_notes=role_notes,shared_skills=shared_skills)
     solver=version/'candidate_solver.py'
     static_gate(a.task,solver.read_text(encoding='utf8'))
     (version/'syntax_repair_success.txt').write_text(str(gate_error)[:500])
   try:
    result=({'valid':True,'metric':0.01,'raw_conflicts':100} if a.dry_run else
            evaluate(a.task,a.task_root,solver))
   except (ValueError,subprocess.TimeoutExpired) as check_error:
    result={'valid':False,'error_type':type(check_error).__name__,
            'error':str(check_error)[-1600:]}
   if not a.dry_run and not result.get('valid'):
    # A public failure is diagnostic evidence. Give the same candidate slot
    # one bounded repair attempt; never consume another evolution version.
    first_result=result
    first_source=solver.read_text(encoding='utf8')
    (version/'public_repair_initial.json').write_text(json.dumps(first_result,ensure_ascii=False,default=str))
    (version/'public_repair_initial.py').write_text(first_source,encoding='utf8')
    repair_history=history+[{'label':label+'_public_repair','metric':first_result.get('metric'),
      'valid':False,'error':'Public selfcheck failed: '+json.dumps(first_result,ensure_ascii=False,default=str)[-2400:]
      +'. Repair this task solver contract; keep official budgets and interfaces.',
      'source':first_source[:100000]}]
    try:
     repaired_runtime=execute(a.output_dir/'evaluations'/f'{label}.yaml',a.task_root,a.task,
       version,repair_history,dry_run=False,role_notes=role_notes,shared_skills=shared_skills)
     repaired_source=solver.read_text(encoding='utf8')
     static_gate(a.task,repaired_source)
     repaired_result=evaluate(a.task,a.task_root,solver)
     if not repaired_result.get('valid'):
      raise ValueError('Public repair remained invalid: '+str(repaired_result.get('error'))[:400])
     runtime=repaired_runtime;result=repaired_result
     (version/'public_repair_success.txt').write_text('Same-slot public repair passed')
    except Exception as repair_error:
     (version/'public_repair_failure.txt').write_text(str(repair_error)[-800:])
     solver.write_text(first_source,encoding='utf8')
     result=first_result
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
                  'error':result.get('error'),'source':solver.read_text(encoding='utf8')[:16000]})
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
  best=max(range(len(rewards)),key=lambda i:public_rank(a.task,metrics[i]['visible_result'],rewards[i]))
  use_mutation=len(configs)<2 or rng.random()<a.mutation_probability
  action='mutation' if use_mutation else 'crossover'
  label=f'step_{step:02d}_{action}'
  evidence=[{'label':h['label'],'public_metric':h['metric'],'valid':h['valid'],
             'error':h['error'],'source_sha256':candidates[i]['sha256']}
            for i,h in enumerate(history[-6:],start=max(0,len(history)-6))]
  prior=a.output_dir/'decision_evidence.jsonl'
  prior_rows=[json.loads(x) for x in prior.read_text().splitlines()[-4:]] if prior.is_file() else []
  concise=[{'label':x['label'],'action':x['action'],
            'selected_parent':x.get('selected_parent'),
            'hypothesis':(x.get('model_decision') or {}).get('root_cause_hypothesis'),
            'intended_change':(x.get('model_decision') or {}).get('intended_change'),
            'changed_components':x.get('changed_components'),
            'parent_public':x.get('parent_public'),'child_public':x.get('child_public'),
            'insight':x.get('insight')} for x in prior_rows]
  current_public=metrics[best]['visible_result']
  current_public_summary={key:current_public.get(key) for key in ('metric','valid','error','failed_graphs')}
  if a.task=='qec':
   current_public_summary['settings']=[{key:item.get(key) for key in ('setting','score','ler','error')}
                                        for item in current_public.get('settings',[])]
  elif a.task=='graph':
   current_public_summary['per_graph_tail']=str(current_public.get('stdout_tail') or '')[-1400:]
  feedback=json.dumps({'observations':evidence,'prior_decisions':concise,
        'cross_task_skills':shared_skills[:2600],
        'normalized_diagnoses':evidence_context(a.output_dir),
        'current_parent_label':candidates[best]['label'],
        'current_parent_public':current_public_summary,
        'lineage_rule':'Only the current parent YAML is inherited. Invalid or regressed children are observations, not edits in the current parent. Diagnose the current parent using its actual YAML and public result.'},ensure_ascii=False,default=str)[-14000:]
  reason_path=a.output_dir/'mutation_rationales.jsonl'
  reason_count=len(reason_path.read_text().splitlines()) if reason_path.is_file() else 0
  try:
   if use_mutation: proposed=(meta.mutate_multi if a.mutation_mode=='multi' else meta.mutate)(configs[best],metrics[best],feedback,model_list=MODELS)
   else:
    ranked=sorted(range(len(rewards)),key=lambda i:public_rank(a.task,metrics[i]['visible_result'],rewards[i]),reverse=True)
    proposed=meta.crossover(configs[ranked[0]],configs[ranked[1]],metrics[ranked[0]],
                            metrics[ranked[1]],model_list=MODELS)
  except Exception:proposed=configs[best]
  config=valid_or_parent(proposed,configs[best],label)
  before=yaml.safe_load(configs[best]);after=yaml.safe_load(config)
  changed=[field for field in ('prompt','model_id','tools') if any(
    before['agents'].get(k,{}).get(field)!=after['agents'].get(k,{}).get(field)
    for k in set(before['agents'])|set(after['agents']))]
  if before.get('topology')!=after.get('topology') or set(before['agents'])!=set(after['agents']):
   changed.append('topology')
  crossover_duplicate=(not use_mutation and any(
   not crossover_change(configs[i],config) for i in ranked[:2]))
  scope_error=None
  if use_mutation:
   try:mutation_component(configs[best],config)
   except (ValueError,yaml.YAMLError) as exc:scope_error=str(exc)
  needs_repair=(use_mutation and scope_error is not None) or crossover_duplicate or (
   (a.output_dir/f'{label}.rejected_reason.txt').is_file())
  invalid_reason=None
  if needs_repair and not a.dry_run:
   reason=(a.output_dir/f'{label}.rejected_reason.txt')
   explanation=(reason.read_text()[:500] if reason.is_file() else
    'Crossover made no agent component change relative to one parent' if crossover_duplicate else
    scope_error or f'Mutation changed {changed}, expected prompt or model_id only')
   if use_mutation and getattr(meta,'last_mutation_validation_error',None):
    explanation+='; parser detail: '+meta.last_mutation_validation_error[:500]
   repair_prompt=(feedback+'\nPrevious YAML draft was rejected before any MAS execution: '+explanation+
                  ('\nReturn complete YAML changing only prompt OR model_id. Preserve all other fields.' if use_mutation else
                   '\nReturn one complete valid YAML with at least one prompt, model_id, tools, or topology change relative to EACH parent. Changes to name, description, or metadata alone do not count.'))
   try:
    if use_mutation: repaired=(meta.mutate_multi if a.mutation_mode=='multi' else meta.mutate)(configs[best],metrics[best],repair_prompt,model_list=MODELS)
    else:
     left=dict(metrics[ranked[0]],repair_instruction=repair_prompt)
     right=dict(metrics[ranked[1]],repair_instruction=repair_prompt)
     repaired=meta.crossover(configs[ranked[0]],configs[ranked[1]],left,right,model_list=MODELS)
    validate_yaml(repaired)
    repaired_before=yaml.safe_load(configs[best]);repaired_after=yaml.safe_load(repaired)
    repaired_changed=[field for field in ('prompt','model_id','tools') if any(
     repaired_before['agents'].get(k,{}).get(field)!=repaired_after['agents'].get(k,{}).get(field)
     for k in set(repaired_before['agents'])|set(repaired_after['agents']))]
    if repaired_before.get('topology')!=repaired_after.get('topology') or set(repaired_before['agents'])!=set(repaired_after['agents']):
     repaired_changed.append('topology')
    if use_mutation:mutation_component(configs[best],repaired)
    if not use_mutation and any(not crossover_change(configs[i],repaired) for i in ranked[:2]):
     raise ValueError('Crossover repair changed no agent component relative to a parent')
    config=repaired;changed=repaired_changed
    (a.output_dir/f'{label}.repair_success.txt').write_text(explanation)
   except Exception as exc:
    hybrid=None
    if not use_mutation:
     try:
      hybrid=crossover_hybrid(configs[ranked[0]],configs[ranked[1]])
      if hybrid is not None:
       validate_yaml(hybrid)
       if any(not crossover_change(configs[i],hybrid) for i in ranked[:2]):
        raise ValueError('Hybrid did not change both parents')
       config=hybrid
       changed=['prompt']
       reason.unlink(missing_ok=True)
       (a.output_dir/f'{label}.hybrid_fallback.txt').write_text(
        'Both model crossover drafts failed; combined distinct prompts from both '
        'parents under the better public parent topology.\n'+explanation)
     except Exception as hybrid_error:
      exc=ValueError(str(exc)[:300]+'; hybrid fallback: '+str(hybrid_error)[:300])
      hybrid=None
    if hybrid is None:
     reason.write_text(explanation+'\nRepair failed: '+str(exc)[:500])
     invalid_reason=reason.read_text()[:1000]
     config=configs[best];changed=[]
  elif use_mutation and scope_error is not None:
   (a.output_dir/f'{label}.rejected_reason.txt').write_text(
    f'Original mutation must change exactly one component; observed {changed}')
   invalid_reason=f'Original mutation must change exactly one component; observed {changed}'
   config=configs[best];changed=[]
  child=evaluate_one(label,config,invalid_reason=invalid_reason)
  new_reward=compute_reward(child,beta=1e-6,cost_weight='both')
  parent_visible=metrics[best]['visible_result']
  child_visible=child['visible_result']
  parent_metric=parent_visible.get('metric')
  child_metric=child_visible.get('metric')
  public_improved=(parent_visible.get('valid') and child_visible.get('valid') and
                   isinstance(parent_metric,(int,float)) and isinstance(child_metric,(int,float)) and
                   math.isfinite(parent_metric) and math.isfinite(child_metric) and
                   child_metric>parent_metric+1e-9)
  operations.append({'label':label,'action':action,'parent_reward':rewards[best],
                     'reward':new_reward,'reward_improved':new_reward>rewards[best],
                     'improved':bool(public_improved),
                     'changed_components':changed,'effective_change':bool(changed)})
  declared=None;decision=None
  if use_mutation and reason_path.is_file():
   reason_rows=reason_path.read_text().splitlines()
   if len(reason_rows)>reason_count:
    recorded=json.loads(reason_rows[-1]);declared=recorded.get('declared_rationale')
    decision=recorded.get('decision')
  parent_public=metrics[best]['visible_result'];child_public=child['visible_result']
  parent_score=score_for_reward(a.task,parent_public)
  child_score=score_for_reward(a.task,child_public)
  actual=('invalid' if not child_public['valid'] else
          'increase' if child_score>parent_score+1e-9 else
          'decrease' if child_score<parent_score-1e-9 else 'same')
  predicted=decision.get('predicted_direction') if isinstance(decision,dict) else None
  verdict=(actual==predicted if predicted in ('increase','decrease','same')
           and actual!='invalid' else None)
  insight=('No effective MAS change; rerun variation is not causal evidence.' if not changed else
           'Public evaluation invalid; repair the failure before judging the hypothesis.' if actual=='invalid' else
           'Public score improved; hypothesis remains provisional.' if actual=='increase' else
           'Public score declined or tied; revise the hypothesis before repeating this edit.')
  with (a.output_dir/'decision_evidence.jsonl').open('a') as f:
   f.write(json.dumps({'label':label,'model_declared_rationale':declared,'model_decision':decision,
    'action':action,'selected_parent':candidates[best]['label'],
    'parent_public':metrics[best]['visible_result'],'observations':evidence,
    'changed_components':changed,
    'child_public':child['visible_result'],'reward_delta':new_reward-rewards[best],
    'effective_change':bool(changed),'actual_direction':actual,'prediction_met':verdict,
    'insight':insight},ensure_ascii=False,default=str)+'\n')
  append_evidence(a.output_dir,a.task,json.loads((a.output_dir/'decision_evidence.jsonl').read_text().splitlines()[-1]))
  # The whole task is consolidated once by state.py after public and hidden evaluation.
 rewards=[compute_reward(m,beta=1e-6,cost_weight='both') for m in metrics]
 best_config=max(range(len(rewards)),key=rewards.__getitem__)  # Original reward archive; parent selection is public-first.
 (a.output_dir/'best_reward_config.yaml').write_text(configs[best_config],encoding='utf8')
 if (metrics[best_config]['visible_result']['valid'] and
     configs[best_config] not in configs[:len(selected_paths)] and
     rewards[best_config]>max(rewards[:len(selected_paths)])):
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
 winner=max(eligible,key=lambda c:public_rank(a.task,c['visible'],0.0))
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
          'agent_max_steps':int(os.environ.get('SUBSET_AGENT_MAX_STEPS','24')),
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
