#!/usr/bin/env python3
"""Public-only evaluation of a task-specific candidate in its official image.

This script never opens tests/ or heldout/ and never modifies the task checkout.
"""
from __future__ import annotations
import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

NAMES = {'inventory':'bbo_simopt_inventory',
         'graph':'johnson1991_leighton_graph_coloring',
         'qec':'qec_decoder_arena'}
ALLOWED = {'inventory':{'numpy'}, 'graph':{'numpy','gc_lib'},
           'qec':{'numpy','scipy','stim','pymatching'}}
CAPS = {'inventory':(2,'12g',None), 'graph':(2,'12g',None),
        'qec':(8,'16g',1000)}

def static_gate(task, source):
    tree=ast.parse(source)
    classes=[x for x in tree.body if isinstance(x,ast.ClassDef)]
    funcs=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))]
    if task=='inventory':
        optimizer=next((x for x in classes if x.name=='Optimizer'),None)
        if optimizer is None: raise ValueError('Optimizer class missing')
        found={x.name:x for x in optimizer.body if isinstance(x,ast.FunctionDef)}
        for fn,required in {'__init__':['self','dim','lower','upper','budget','seed','rng'],
                            'ask':['self','n'],'tell':['self','X','y']}.items():
            if fn not in found or [x.arg for x in found[fn].args.args][:len(required)] != required:
                raise ValueError(f'Optimizer.{fn} signature missing or wrong')
    else:
        name='solve' if task=='graph' else 'decode'
        fn=next((x for x in funcs if x.name==name),None)
        required=['data'] if task=='graph' else ['setting','dets']
        if fn is None or [x.arg for x in fn.args.args][:len(required)] != required:
            raise ValueError(f'{name}{tuple(required)} entrypoint missing')
    if task=='inventory':
        for node in ast.walk(tree):
            if (isinstance(node,ast.Attribute) and node.attr in ('erf','math') and
                isinstance(node.value,ast.Name) and node.value.id in ('np','numpy')):
                raise ValueError('NumPy 2.2.6 has no np.erf/np.math; use stdlib math.erf or a tested approximation')
    for node in ast.walk(tree):
        names=[]
        if isinstance(node,ast.Import): names=[a.name for a in node.names]
        if isinstance(node,ast.ImportFrom): names=[node.module or '']
        for name in names:
            head=name.split('.')[0]
            if head and head not in sys.stdlib_module_names and head not in ALLOWED[task]:
                raise ValueError(f'forbidden import {name}')

def command(task, root, solver, runner, setting=None):
    cpus,mem,_=CAPS[task]
    image=f'rsi-subset-{NAMES[task]}:public'
    cmd=['docker','run','--rm','--network','none','--read-only',
         '--tmpfs','/tmp:rw,nosuid,size=1g','--cpus',str(cpus),
         '--memory',mem,'--pids-limit','512','-e','PYTHONDONTWRITEBYTECODE=1',
         '--mount',f'type=bind,src={solver.resolve()},dst=/app/methods/main/solver.py,readonly']
    if task=='qec':
        cmd += ['--mount',f'type=bind,src={runner.resolve()},dst=/runner/qec_one.py,readonly']
    cmd += [image,'python']
    if task=='qec': cmd += ['/runner/qec_one.py',setting]
    else: cmd += ['/app/selfcheck.py']
    return cmd

def evaluate(task,root,solver):
    static_gate(task,solver.read_text(encoding='utf-8'))
    image=f'rsi-subset-{NAMES[task]}:public'
    subprocess.run(['docker','image','inspect',image],check=True,stdout=subprocess.DEVNULL)
    _,_,limit=CAPS[task]
    runner=Path(__file__).with_name('qec_one.py')
    if task=='qec':
        rows=[]
        for name in ('color_d5','color_d5_Z','color_d7','color_d9'):
            meta=json.loads((root/'environment'/'settings_visible'/name/'meta.json').read_text())
            budget=float(meta['decode_budget_sec'])
            try:
                proc=subprocess.run(command(task,root,solver,runner,name),capture_output=True,text=True,timeout=budget+90)
                if proc.returncode: raise ValueError(proc.stderr[-1800:])
                rows.append(json.loads(proc.stdout.strip().splitlines()[-1]))
            except (subprocess.TimeoutExpired,ValueError,IndexError,KeyError,json.JSONDecodeError) as exc:
                rows.append({'setting':name,'score':0.0,'error':str(exc)[-1000:]})
        return {'valid':len(rows)==4 and all(not x.get('error') for x in rows),
                'task':task,'metric':sum(x['score'] for x in rows)/len(rows),
                'metric_direction':'higher','settings':rows}
    proc=subprocess.run(command(task,root,solver,runner),capture_output=True,text=True,timeout=limit)
    if proc.returncode:
        raise ValueError(f'official visible selfcheck exit={proc.returncode}: {proc.stderr[-1800:]}')
    text=proc.stdout
    if task=='inventory':
        found=re.search(r'visible raw_metric\s*=\s*([\d.eE+-]+); mapped reward\s*=\s*([\d.eE+-]+)',text)
        if not found:raise ValueError(f'official score missing: {text[-1000:]}')
        return {'valid':True,'task':task,'raw_metric':float(found[1]),'metric':float(found[2]),
                'metric_direction':'higher','stdout_tail':text[-1600:]}
    totals=re.search(r'total conflicts over\s+(\d+)/(\d+) solved graph\(s\)\s*=\s*(\d+)',text)
    if not totals:raise ValueError(f'official graph diagnostics missing: {text[-1000:]}')
    ok,all_count,conf=map(int,totals.groups())
    failed=all_count-ok
    # A killed or invalid graph forfeits that instance only under the official rule.
    # Rank completed cases ahead of missing cases; this is a research surrogate.
    effective=conf+1000000*failed
    return {'valid':True,'task':task,'raw_conflicts':conf,
            'solved_graphs':ok,'total_graphs':all_count,'failed_graphs':failed,
            'effective_conflicts':effective,
            'metric':-effective,'metric_direction':'research surrogate; lower failed/conflicts is better',
            'stdout_tail':text[-2200:]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('task',choices=NAMES)
    p.add_argument('task_root',type=Path)
    p.add_argument('solver',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    if args.output.exists():p.error('Refusing to overwrite output')
    if not (args.task_root/'environment'/'Dockerfile').is_file():p.error('official task root missing')
    if not args.solver.is_file():p.error('candidate solver missing')
    try: result=evaluate(args.task,args.task_root,args.solver)
    except Exception as exc:
        result={'valid':False,'task':args.task,'error_type':type(exc).__name__,
                'error':str(exc)[-2000:]}
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))
    return 0 if result['valid'] else 1

if __name__=='__main__':raise SystemExit(main())
