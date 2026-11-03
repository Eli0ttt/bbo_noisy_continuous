#!/usr/bin/env python3
"""One sealed Harbor submission for a completed public run, with exact source SHA."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path

TASKS={'bbo':'bbo_noisy_continuous','inventory':'bbo_simopt_inventory',
       'graph':'johnson1991_leighton_graph_coloring','qec':'qec_decoder_arena'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
 p.add_argument('--run',type=Path,required=True);p.add_argument('--task',choices=TASKS,required=True)
 p.add_argument('--harbor-bin',type=Path,default=Path.home()/'.local/share/uv/tools/harbor/bin/harbor')
 a=p.parse_args();root=a.root.resolve();run=a.run.resolve();summary=run/'summary.json'
 s=json.loads(summary.read_text());winner=run/'best_visible_solver.py'
 if s.get('hidden_status')!='not_run' or s.get('hidden_job'):raise ValueError('Hidden already attempted; inspect prior job')
 if a.task=='bbo':
  if s.get('completed_solver_versions')!=s.get('solver_limit'):raise ValueError('Incomplete BBO')
  task_root=root/'dsh_rsi_runs/bbo_noisy_continuous_prebuilt'
 else:
  if s.get('task')!=a.task or s.get('status')!='completed' or s.get('completed_versions')!=6:
   raise ValueError('Incomplete or mismatched subset run')
  task_root=root/'rsi_exam_tasks'/TASKS[a.task]
 if not (task_root/'task.toml').is_file() or not a.harbor_bin.is_file():raise FileNotFoundError('Task or Harbor unavailable')
 digest=hashlib.sha256(winner.read_bytes()).hexdigest()
 if digest!=s['best_visible']['sha256']:raise ValueError('Selected source SHA mismatch')
 if a.task!='bbo' and (not s['best_visible'].get('valid') or
    not any(c['sha256']==digest and c['valid'] for c in s['candidates'])):
  raise ValueError('Winner not in valid public candidate manifest')
 if len(winner.read_bytes())>100000:raise ValueError('Candidate source exceeds research cap')
 name=f'evomas-{a.task}-{digest[:12]}-{time.time_ns()}'
 jobs=run/'harbor_jobs';jobs.mkdir(exist_ok=True)
 job=jobs/name
 mounts=json.dumps([{'type':'bind','source':str(winner),'target':'/opt/evomas-selected/solver.py',
                     'read_only':True}])
 cmd=[str(a.harbor_bin),'run','--yes','--env','docker','--job-name',name,
      '--jobs-dir',str(jobs),'--n-concurrent','1','--n-attempts','1','--max-retries','0',
      '--path',str(task_root),'--agent','dsh_rsi_agent.evo_submission_agent:EvoSubmissionAgent',
      '--model','deepseek-v4-flash-s1','--mounts',mounts,
      '--agent-kwarg','solver_sha256='+digest]
 env=os.environ.copy();env['PYTHONPATH']=str(root)+os.pathsep+env.get('PYTHONPATH','')
 s['hidden_status']='running';s['hidden_job']=str(job);summary.write_text(json.dumps(s,indent=2,ensure_ascii=False)+'\n')
 log=run/(name+'.hidden.log');start=time.monotonic()
 with log.open('w') as stream:r=subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT)
 s['hidden_seconds']=time.monotonic()-start;s['hidden_log']=str(log)
 if r.returncode:
  s['hidden_status']='failed';s['hidden_return_code']=r.returncode
  summary.write_text(json.dumps(s,indent=2,ensure_ascii=False)+'\n')
  raise RuntimeError('Harbor failed; inspect '+str(log))
 files=list(job.glob('*/result.json'))
 if len(files)!=1:raise RuntimeError('Harbor result count is not one; inspect '+str(job))
 result=json.loads(files[0].read_text());value=((result.get('verifier_result') or {}).get('rewards') or {}).get('reward')
 if not isinstance(value,(int,float)):raise RuntimeError('No numeric official reward; inspect '+str(files[0]))
 s['hidden_status']='completed';s['hidden_score']=float(value)
 # The official metric and the platform's mapped reward are distinct for some tasks.
 metric_names={'bbo':('oracle_normalized_auc70_final30','higher'),
               'inventory':('mean_log_cost_progress_auc70_final30','higher'),
               'graph':('total_conflicts','lower'),
               'qec':('mean_setting_score','higher')}
 label,direction=metric_names[a.task]
 metric=None
 diagnostic=files[0].parent/'verifier'/('score_details.json' if a.task=='qec' else 'grade_debug.json')
 if diagnostic.is_file():
  try:
   detail=json.loads(diagnostic.read_text())
   metric=detail.get('metric') if a.task in ('qec','graph') else detail.get('reward')
  except (OSError,ValueError):pass
 s['final_result']={'official_metric_name':label,'official_metric_direction':direction,
                    'official_metric':metric,'platform_reward':float(value),
                    'diagnostic_path':str(diagnostic) if diagnostic.is_file() else None}
 summary.write_text(json.dumps(s,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps({'task':a.task,'final_result':s['final_result'],
                   'job':str(job),'source_sha256':digest}))
if __name__=='__main__':main()
