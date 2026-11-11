"""Research-only, task-scoped tools. Never imported by a submitted solver."""
from __future__ import annotations
import json
import os
import re
import subprocess
import threading
from pathlib import Path
from smolagents import Tool

CATALOG = {
 'subset_read_context': 'Read the public instruction or your own candidate.py.',
 'subset_read_public': 'Read a bounded schema and counts of official public data only.',
 'subset_read_history': 'Read current research session candidate feedback by exact label.',
 'subset_write_candidate': 'Write complete Python source to your own candidate.py.',
 'subset_preflight': 'Check your written candidate against this task interface and import allow-list.',
 'subset_public_python': 'Run a short Python experiment in this task\'s public, network-isolated Docker image; imports follow that task image, not the research host.'}
CATALOG_TEXT = 'Task generation tools; any subset may be assigned in evolved YAML:\n' + '\n'.join(
 f'- {k}: {v}' for k,v in CATALOG.items())
LOCK=threading.Lock()
PUBLIC_PROBES={}  # One bounded budget per candidate MAS execution root.

def audit(root,agent,tool,ok):
 with LOCK:
  with (root/'tool_audit.jsonl').open('a',encoding='utf8') as f:
   f.write(json.dumps({'agent':agent,'tool':tool,'success':bool(ok)})+'\n')

class Base(Tool):
 output_type='string'
 def __init__(self,root,agent,task_root,task):
  if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',agent):raise ValueError('unsafe agent ID')
  self.root=Path(root).resolve();self.agent=agent;self.task_root=Path(task_root).resolve();self.task=task
  self.scratch=self.root/'tool_workspace'/agent
  self.scratch.mkdir(parents=True,exist_ok=True)
  super().__init__()
 def record(self,ok): audit(self.root,self.agent,self.name,ok)

class ReadContext(Base):
 name='subset_read_context';description=CATALOG[name]
 inputs={'item':{'type':'string','description':'Exactly instruction.md or candidate.py'}}
 def forward(self,item:str)->str:
  path={'instruction.md':self.root/'tool_workspace/instruction.md',
        'candidate.py':self.scratch/'candidate.py'}.get(item)
  ok=path is not None and path.is_file() and not path.is_symlink();self.record(ok)
  return path.read_text(encoding='utf8')[:24000] if ok else 'DENIED: item unavailable'

class ReadPublic(Base):
 name='subset_read_public';description=CATALOG[name]
 inputs={'item':{'type':'string','description':'Exactly summary'}}
 def forward(self,item:str)->str:
  if item!='summary':self.record(False);return 'DENIED: choose summary'
  if self.task=='qec':
   base=self.task_root/'environment/settings_visible'
   result={'settings':{p.parent.name:{k:v for k,v in json.loads(p.read_text()).items()
                         if k in ('distance','rounds','n_detectors','n_observables','n_dev','n_eval','decode_budget_sec')}
                       for p in sorted(base.glob('*/meta.json'))}}
  else:
   data=json.loads((self.task_root/'environment/data/visible.json').read_text())
   cases=data.get('instances',[])
   result={'n_instances':len(cases),'fields':sorted(cases[0]) if cases else [],
           'budget':data.get('budget')}
  self.record(True);return json.dumps(result,ensure_ascii=False)[:10000]

class ReadHistory(Base):
 name='subset_read_history';description=CATALOG[name]
 inputs={'item':{'type':'string','description':'Exactly index or a label returned by index'}}
 def forward(self,item:str)->str:
  path=self.root/'tool_workspace/history.json'
  entries=json.loads(path.read_text()) if path.is_file() else []
  if item=='index':self.record(True);return json.dumps([{k:v for k,v in e.items() if k!='source'} for e in entries])
  found=next((x for x in entries if x['label']==item),None)
  self.record(found is not None)
  return json.dumps(found,ensure_ascii=False)[:24000] if found else 'DENIED: unknown current-session label'

class WriteCandidate(Base):
 name='subset_write_candidate';description=CATALOG[name]
 inputs={'source':{'type':'string','description':'Complete candidate.py Python source, up to 100000 characters'}}
 def forward(self,source:str)->str:
  if not isinstance(source,str) or not source.strip() or len(source)>100000:
   self.record(False);return 'DENIED: source is empty or too long'
  path=self.scratch/'candidate.py'
  if path.is_symlink():self.record(False);return 'DENIED: symbolic link'
  fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC|getattr(os,'O_NOFOLLOW',0),0o600)
  with os.fdopen(fd,'w',encoding='utf8') as f:f.write(source)
  self.record(True);return 'WROTE: your candidate.py'

class Preflight(Base):
 name='subset_preflight';description=CATALOG[name]
 inputs={'item':{'type':'string','description':'Exactly candidate.py after writing it'}}
 def forward(self,item:str)->str:
  path=self.scratch/'candidate.py'
  if item!='candidate.py' or path.is_symlink() or not path.is_file():
   self.record(False);return 'DENIED: write your candidate.py first'
  try:
   from subset_visible import static_gate
   static_gate(self.task,path.read_text(encoding='utf8'))
   self.record(True);return 'PASS: syntax, task entrypoint and allowed imports; official runtime selfcheck still required'
  except Exception as exc:
   self.record(False);return f'FAIL: {type(exc).__name__}: {exc}'

class PublicPython(Base):
 name='subset_public_python';description=CATALOG[name]
 inputs={'source':{'type':'string','description':'Python snippet up to 12000 characters using only this task\'s public image; print the result'}}
 def forward(self,source:str)->str:
  if not isinstance(source,str) or not source.strip() or len(source)>12000:
   self.record(False);return 'DENIED: nonempty source must be at most 12000 characters'
  images={'inventory':'bbo_simopt_inventory',
          'graph':'johnson1991_leighton_graph_coloring','qec':'qec_decoder_arena'}
  if self.task not in images:
   self.record(False);return 'DENIED: unknown task'
  with LOCK:
   used=PUBLIC_PROBES.get(str(self.root),0)
   if used<4:PUBLIC_PROBES[str(self.root)]=used+1
  if used>=4:
   self.record(False);return 'DENIED: public Python probe limit (4 per candidate); finish your proposal'
  cmd=['docker','run','--rm','-i','--network','none','--read-only',
       '--security-opt','no-new-privileges','--tmpfs','/tmp:rw,nosuid,size=256m',
       '--cpus','2','--memory','4g','--pids-limit','128',
       '-e','PYTHONDONTWRITEBYTECODE=1','--workdir',
       '/app/methods/main' if self.task=='graph' else '/app',
       'rsi-subset-'+images[self.task]+':public','python','-']
  try:
   run=subprocess.run(cmd,input=source,text=True,capture_output=True,timeout=35)
   output=(run.stdout+'\n'+run.stderr)[-12000:]
   self.record(run.returncode==0)
   return f'PUBLIC_IMAGE_EXIT={run.returncode}\n{output}'
  except subprocess.TimeoutExpired:
   self.record(False);return 'PUBLIC_IMAGE_TIMEOUT: 35 seconds'
  except OSError as exc:
   self.record(False);return f'PUBLIC_IMAGE_ERROR: {type(exc).__name__}: {exc}'

FACTORIES={x.name:x for x in (ReadContext,ReadPublic,ReadHistory,WriteCandidate,Preflight,PublicPython)}
def make_tool(name,root,agent,task_root,task):
 return FACTORIES[name](root,agent,task_root,task)
