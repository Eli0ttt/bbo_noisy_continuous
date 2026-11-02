"""Task-aware adapter at EvoMAS's native DAG/SmolagentsRunner boundary."""
from __future__ import annotations
import ast
import hashlib
import json
import re
import sys
from pathlib import Path
import yaml
from subset_visible import static_gate

MODELS={'openai:deepseek-v4-flash-s1','openai:deepseek-v4-flash-s2'}
TOOLS={'subset_read_context','subset_read_public','subset_read_history',
       'subset_write_candidate','subset_preflight'}

def validate_yaml(source):
 from src.mas.spec import MasSpec
 spec=MasSpec(**yaml.safe_load(source))
 if spec.backend!='smolagents':raise ValueError('Requires native smolagents backend')
 agents=spec.agents
 if not agents:raise ValueError('Empty agents')
 levels=spec.get_execution_levels()
 if sorted(a for level in levels for a in level)!=sorted(agents):raise ValueError('Cyclic/missing agent')
 graph=spec.topology.reports_to
 if any(a not in agents or any(t not in agents for t in ts) for a,ts in graph.items()):
  raise ValueError('Unknown agent in DAG')
 terminal=[a for a in agents if not graph.get(a)]
 if len(terminal)!=1 or levels[-1]!=terminal:raise ValueError('Need exactly one terminal agent')
 for aid,a in agents.items():
  if a.id!=aid or a.agent_type!='CodeAgent' or a.model_id not in MODELS:
   raise ValueError(f'Invalid role/model for {aid}')
  if set(a.tools)-TOOLS:raise ValueError(f'Unknown task tool for {aid}')
  if not a.prompt.strip():raise ValueError(f'Empty prompt for {aid}')
 return spec,terminal[0]

def extract(source,task):
 blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',source,re.I|re.S)
 for part in sorted(blocks,key=len,reverse=True)+[source]:
  try:static_gate(task,part.strip())
  except (SyntaxError,ValueError):
   try:tree=ast.parse(part)
   except SyntaxError:continue
   embedded=[n.value.value for n in tree.body if isinstance(n,ast.Assign)
             and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
             and n.targets[0].id.startswith('solver_code')
             and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)]
   valid=[]
   for candidate in embedded:
    try:static_gate(task,candidate);valid.append(candidate)
    except (SyntaxError,ValueError):pass
   if len(valid)==1:return valid[0].strip()+'\n'
   continue
  return part.strip()+'\n'
 raise ValueError('No complete candidate for this task in terminal report')

def role_hint(agent,task):
 output={'inventory':'Optimizer with __init__(dim, lower, upper, budget, seed, rng), ask(n), tell(X,y,metadata=None), 4D [0,1]^4, 48 noisy costs; stdlib + NumPy 2.2.6 only, self-contained solver.py.',
         'graph':'solve(data) returning {"colorings": [[int,...], ...]}; fixed target_k, fewer monochromatic edges, each graph <45 seconds; stdlib + NumPy, optional official gc_lib.',
         'qec':'decode(setting,dets) returning a boolean [n_shots,n_observables] array; per-setting decode limit 600/900s; stdlib + NumPy + SciPy + Stim + PyMatching only; dev labels allowed, eval truth withheld.'}[task]
 prompt=agent.prompt if agent.prompt not in ('worker','smoa_aggregator','aggregator','judge','team_worker','processor','debater_initial','debater_round','reviewer_create','reviewer_review','reviewer_revise') else ''
 return ('This is the RSI-Exam '+task+' task. Produce a complete solver.py, not a fixed answer. '
         +output+' Preserve official interface and generalize to sealed cases. '
         'Research tools are available only when assigned in your YAML. '
         'subset_read_context(item="instruction.md" or "candidate.py"); '
         'subset_read_public(item="summary"); subset_read_history(item="index" or label); '
         'subset_write_candidate(source=complete_code); subset_preflight(item="candidate.py") '
         'after writing. Interpreter restrictions: do not use open/exec/importlib there; '
         'put solver code inside a string tool argument or fenced final answer. '
         'Use earlier reports from EvoMAS DAG context, not other agents\' scratch files. '
         + ('Evolved prompt guidance: '+prompt if prompt else ''))

def execute(config_path,task_root,task,output_dir,history,dry_run=False):
 from src.agents.spec import AgentResult
 from src.mas import runtime as runtime_mod
 from src.agents.runners.smolagents import SmolagentsRunner
 from src.tools.subset_generation import CATALOG
 spec,terminal=validate_yaml(config_path.read_text(encoding='utf8'))
 output_dir.mkdir(parents=True,exist_ok=True)
 workspace=output_dir/'tool_workspace';workspace.mkdir(exist_ok=True)
 starter=(task_root/'environment/methods/main/solver.py').read_text(encoding='utf8')
 instruction=(task_root/'instruction.md').read_text(encoding='utf8')
 (workspace/'instruction.md').write_text(instruction)
 (workspace/'history.json').write_text(json.dumps(history,ensure_ascii=False))
 for aid in spec.agents:
  p=workspace/aid;p.mkdir(exist_ok=True)
  (p/'candidate.py').write_text(starter)
 # One official starter per new task/session; previous-run solver never appears here.
 prompt=(f'Official RSI-Exam {task} research instruction:\n{instruction}\n'
         'The starter below is the current task baseline, not a preceding task output.\n'
         f'```python\n{starter}\n```\n'
         'All earlier candidate results listed next are from THIS session only:\n'
         +json.dumps([{k:v for k,v in h.items() if k!='source'} for h in history],ensure_ascii=False)[-16000:])
 failures=[]
 class RoleRunner:
  def __init__(self):
   self.delegate=SmolagentsRunner() if not dry_run else None
   if self.delegate:
    self.delegate.subset_tool_root=output_dir
    self.delegate.subset_task_root=task_root
    self.delegate.subset_task_name=task
  def run(self,agent,text,context=None):
   if dry_run:
    content=('```python\n'+starter+'\n```') if agent.id==terminal else f'Mock report by {agent.id}'
    result=AgentResult(agent_id=agent.id,content=content,success=True)
   else:result=self.delegate.run(agent,role_hint(agent,task)+'\n'+text,context)
   (output_dir/f'agent_{agent.id}.json').write_text(json.dumps({
    'content':result.content,'success':result.success,'error':result.error,
    'metadata':result.metadata},ensure_ascii=False,default=str))
   if not result.success:failures.append({'agent':agent.id,'error':result.error})
   return result
 old=runtime_mod._get_runner_class
 runtime_mod._get_runner_class=lambda backend:RoleRunner if backend=='smolagents' else old(backend)
 try:final,metadata=runtime_mod.MasRuntime(spec).run(prompt)
 finally:runtime_mod._get_runner_class=old
 (output_dir/'aggregator_response.txt').write_text(final)
 if failures or final.startswith('Error:'):raise RuntimeError(f'MAS failed: {str(failures)[:700]}')
 try:source=extract(final,task);origin='terminal_report'
 except ValueError:
  # Accept only terminal's own file after an audited successful write.
  audit=output_dir/'tool_audit.jsonl'
  rows=[json.loads(x) for x in audit.read_text().splitlines()] if audit.is_file() else []
  if not any(x.get('agent')==terminal and x.get('tool')=='subset_write_candidate'
             and x.get('success') for x in rows):raise
  source=extract((workspace/terminal/'candidate.py').read_text(),task)
  origin='terminal_writer'
 target=output_dir/'candidate_solver.py'
 target.write_text(source,encoding='utf8')
 result={'metadata':metadata,'candidate_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
         'source':origin,'terminal':terminal,
         'configured_tools':{k:list(v.tools) for k,v in spec.agents.items()}}
 (output_dir/'mas_summary.json').write_text(json.dumps(result,indent=2,default=str))
 return result
