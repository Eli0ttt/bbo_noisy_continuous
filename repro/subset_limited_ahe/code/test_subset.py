"""No API / no Docker regression against attached task starters and seed YAML."""
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path
import yaml

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import subset_visible as v
import run_subset_task as r
import subset_runtime as native
for task,name in v.NAMES.items():
 p=ROOT/'tasks'/name/'environment/methods/main/solver.py'
 if not p.is_file():continue
 v.static_gate(task,p.read_text())
 src=f'```python\n{p.read_text()}\n```'
 assert native.extract(src,task).strip()==p.read_text().strip()
seed=ROOT/'review/repro/subset_ablation/pool/upstream_bbeh_s2_full_two/majority_vote.yaml'
if seed.is_file():
 text=r.seed_yaml(seed.read_text())
 cfg=yaml.safe_load(text)
 assert all(set(a['tools'])==set(r.TOOLS) for a in cfg['agents'].values())
 again=r.inherited_yaml(seed.read_text())
 assert 'subset_write_candidate' in yaml.safe_load(again)['agents']['worker1']['tools']
mod=types.ModuleType('smolagents');mod.Tool=type('Tool',(),{'__init__':lambda self:None})
sys.modules['smolagents']=mod
spec=importlib.util.spec_from_file_location('subset_generation',HERE.parent/'overlay/src/tools/subset_generation.py')
tools=importlib.util.module_from_spec(spec);spec.loader.exec_module(tools)
with tempfile.TemporaryDirectory() as td:
 task_root=ROOT/'tasks/bbo_simopt_inventory'
 context=tools.make_tool('subset_read_context',Path(td),'worker',task_root,'inventory')
 assert context.forward('candidate.py').startswith('DENIED')
 writer=tools.make_tool('subset_write_candidate',Path(td),'worker',task_root,'inventory')
 source=(task_root/'environment/methods/main/solver.py').read_text()
 assert writer.forward(source).startswith('WROTE')
 assert context.forward('candidate.py')==source[:100000]
 assert tools.make_tool('subset_preflight',Path(td),'worker',task_root,'inventory').forward('candidate.py').startswith('PASS')
 assert tools.make_tool('subset_read_public',Path(td),'worker',task_root,'inventory').forward('summary').startswith('{')
print('PASS: three task interfaces, seed translation, task-scoped tools')
