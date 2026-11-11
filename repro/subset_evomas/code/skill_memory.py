"""Public-only reusable research procedures and task-local role notes.

A skill is a conditional experiment procedure, not a copied solver or a claim
that one successful rollout proves transfer. Hidden scores are never ingested.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

RULES = {
 'configuration_contract': (
  'When a proposed MAS YAML fails syntax, DAG, model, tool, or semantic-difference checks.',
  ['Keep the last valid parent; isolate the failing field using the validator error.',
   'Repair one draft within the same candidate slot; require a real agent-level difference.'],
  'Parse YAML, validate the executable DAG and allow-lists, then run public selfcheck; record any fallback.'),
 'solver_contract': (
  'When a solver candidate fails an interface, shape, import, or per-setting runtime check.',
  ['Reproduce the failing public call with the exact task API and input shape.',
   'Fix the smallest contract error, then run every required public setting before scoring.'],
  'Require a complete valid public selfcheck; do not treat a numeric partial score as success.'),
 'timeout_diagnosis': (
  'When an agent request or a public runtime times out.',
  ['Identify the stage, model ID, tool, and agent role from the trace.',
   'Separate a recovered upstream agent timeout from a failed terminal candidate.'],
  'Re-run only a public check and record validity and latency; a timeout alone does not prove an algorithm flaw.'),
 'controlled_change': (
  'When a candidate improves the current public best after one documented configuration change.',
  ['Record the parent, component, predicted direction, and actual changed fields.',
   'Test the complete child with the same public selfcheck and retain the previous best for rollback.'],
  'Compare task-level public score and validity to the previous best; repeat on another task before claiming transfer.'),
 'regression_rollback': (
  'When a changed configuration scores below the current public best.',
  ['Keep the previously valid best as the next parent.',
   'Record the failed hypothesis and choose a distinct measured weakness before repeating the edit.'],
  'Use the same public metric and setting coverage; do not infer causality from a single rollout.'),
 'no_effect': (
  'When a proposed YAML changes no effective agent field, or a public score remains tied.',
  ['Check semantic differences in prompt, model_id, tools, and topology.',
   'If the attempt is a crossover, require a valid combination distinct from both parents.'],
  'Confirm a real change and complete public selfcheck; a tied or invalid candidate is not a demonstrated gain.'),
}

def _rule(row):
    failure = row.get('observed_failure') or ''
    verdict = row.get('verdict')
    if failure == 'mas_yaml_invalid': return 'configuration_contract'
    if failure in ('solver_contract_or_selfcheck_invalid','solver_static_contract',
                   'solver_runtime_or_selfcheck_invalid','per_setting_interface_or_runtime'): return 'solver_contract'
    if 'timeout' in failure: return 'timeout_diagnosis'
    if verdict == 'worked': return 'controlled_change'
    if verdict == 'regressed': return 'regression_rollback'
    if verdict == 'unresolved' and row.get('action'): return 'no_effect'
    return None

def add_run(skills, roles, task, run):
    """Idempotent consolidation using only public trace and agent reports."""
    run=Path(run); skills=list(skills); roles=list(roles)
    index={x['id']:x for x in skills}
    path=run/'trajectory_evidence.jsonl'
    if not path.is_file():return skills,roles
    for line in path.read_text(encoding='utf8').splitlines():
        try: row=json.loads(line)
        except json.JSONDecodeError:continue
        kind=_rule(row)
        if not kind:continue
        component=row.get('component') or 'any'
        key=kind+(':'+component if kind=='controlled_change' else '')
        ident=hashlib.sha256(key.encode()).hexdigest()[:16]
        if ident not in index:
            condition,steps,verify=RULES[kind]
            record={'id':ident,'kind':kind,'component':component if kind=='controlled_change' else None,
                    'condition':condition,'steps':steps,'verification':verify,
                    'status':'provisional','evidence':[]}
            skills.append(record);index[ident]=record
        evidence={'task':task,'run':str(run),'label':row.get('label'),
                  'verdict':row.get('verdict'),'failure':row.get('observed_failure'),
                  'public_score':row.get('public_child_score'),
                  'parent_public_score':row.get('public_parent_score'),
                  'hypothesis':str(row.get('root_cause_hypothesis') or '')[:220],
                  'intended_change':str(row.get('intended_change') or '')[:220]}
        record=index[ident]
        if not any(x['run']==evidence['run'] and x['label']==evidence['label'] for x in record['evidence']):
            record['evidence'].append(evidence)
        worked={x['task'] for x in record['evidence'] if x['verdict']=='worked'}
        record['status']='corroborated_public' if len(worked)>=2 else 'provisional'
    # Role notes retain a short attributed observation, never a full rollout.
    for directory in sorted((run/'evaluations').glob('*')) if (run/'evaluations').is_dir() else []:
        if not directory.is_dir() or not directory.name.startswith(('parent_','step_')):continue
        visible=directory/'visible.json'
        if visible.is_file():
            try: outcome=json.loads(visible.read_text())
            except json.JSONDecodeError:outcome={}
        else:outcome={}
        config_file=directory.parent/(directory.name+'.yaml')
        agent_specs={}
        if config_file.is_file():
            try:
                import yaml
                agent_specs=yaml.safe_load(config_file.read_text()).get('agents',{})
            except (ValueError, AttributeError, yaml.YAMLError):pass
        for agent in sorted(directory.glob('agent_*.json')):
            try: report=json.loads(agent.read_text(encoding='utf8'))
            except json.JSONDecodeError:continue
            aid=agent.stem[len('agent_'):]
            # Avoid propagating unbounded or unrelated text into another task.
            excerpt=str(report.get('content') or '')[:350]
            spec=agent_specs.get(aid,{})
            note={'task':task,'agent_id':aid,'role':str(spec.get('role') or spec.get('metadata',{}).get('role_description') or aid.split('_')[0])[:100],
                  'run':str(run),'label':directory.name,'public_metric':outcome.get('metric'),
                  'public_valid':outcome.get('valid'), 'agent_succeeded':report.get('success'),
                  'research_note_unverified':excerpt}
            if not any(x['run']==note['run'] and x['label']==note['label'] and
                       x['agent_id']==aid for x in roles):roles.append(note)
    # Bounded task-local working memory: recent two notes per role and task.
    recent=[]; counts={}
    for x in reversed(roles):
        key=(x['task'],x['agent_id']);counts[key]=counts.get(key,0)+1
        if counts[key]<=2:recent.append(x)
    return skills,list(reversed(recent))

def render(skills,limit=5):
    ranked=sorted(skills,key=lambda s:(s['status']=='corroborated_public',
        len({x['task'] for x in s['evidence']}),len(s['evidence']),s['id']),reverse=True)[:limit]
    if not ranked:return 'No cross-task research skills established yet.'
    result=['Shared conditional research procedures (public evidence only; provisional rules are not proven):']
    for s in ranked:
        sources=', '.join(sorted({e['task'] for e in s['evidence']}))
        result.append(f"[{s['id']}] {s['status']} / observed on {sources}. When: {s['condition']} "
                      + 'Steps: '+'; '.join(s['steps'])+' Verify: '+s['verification'])
    return '\n'.join(result)[:4800]

def role_context(notes,agent_id):
    matches=[x for x in notes if x.get('agent_id')==agent_id][-2:]
    return '\n'.join(f"Prior task-local observation ({x['label']}, valid={x['public_valid']}, "
                     f"score={x['public_metric']}): {x['research_note_unverified']}"
                     for x in matches)[:900]
