"""Public-only task-local evidence: compare edits to the best prior selfcheck."""
from __future__ import annotations
import hashlib
import json
import math
import copy
from pathlib import Path
import yaml


def crossover_change(parent: str, child: str) -> bool:
    """Ignore YAML descriptions and reward metadata when checking recombination."""
    old, new = yaml.safe_load(parent), yaml.safe_load(child)
    if not isinstance(old, dict) or not isinstance(new, dict):
        return False
    if old.get('topology') != new.get('topology'):
        return True
    agents_old, agents_new = old.get('agents', {}), new.get('agents', {})
    if not isinstance(agents_old, dict) or not isinstance(agents_new, dict):
        return False
    if set(agents_old) != set(agents_new):
        return True
    return any(agents_old[aid].get(field) != agents_new[aid].get(field)
               for aid in agents_old for field in ('prompt', 'model_id', 'tools'))


def crossover_hybrid(first: str, second: str) -> str | None:
    """Last-resort recombination of distinct same-task parent prompts.

    Keep the first parent's valid topology, model IDs, and tools. Incorporate
    one distinct role instruction from the second parent. Return None when
    no genuine two-parent hybrid can be made.
    """
    a, b = yaml.safe_load(first), yaml.safe_load(second)
    if not isinstance(a, dict) or not isinstance(b, dict):
        return None
    aa, bb = a.get('agents'), b.get('agents')
    if not isinstance(aa, dict) or not isinstance(bb, dict):
        return None
    for aid in sorted(aa.keys() & bb.keys()):
        left, right = aa[aid].get('prompt'), bb[aid].get('prompt')
        if not isinstance(left, str) or not isinstance(right, str) or left == right:
            continue
        out = copy.deepcopy(a)
        out['agents'][aid]['prompt'] = (
            left.rstrip() + '\nAssess this alternative from the second parent using '
            'the public task evidence: ' + right[:600])
        candidate = yaml.safe_dump(out, sort_keys=False, allow_unicode=True)
        if crossover_change(first, candidate) and crossover_change(second, candidate):
            return candidate
    return None


def mutation_component(parent: str, child: str) -> str:
    before, after = yaml.safe_load(parent), yaml.safe_load(child)
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError('Mutation must return complete YAML')
    if set(before) != set(after) or set(before.get('agents', {})) != set(after.get('agents', {})):
        raise ValueError('Mutation cannot change YAML keys or agent identities')
    if any(before[k] != after[k] for k in before if k not in ('agents', 'topology')):
        raise ValueError('Mutation cannot change root metadata')
    changed = {'topology'} if before.get('topology') != after.get('topology') else set()
    for aid in before['agents']:
        old, new = before['agents'][aid], after['agents'][aid]
        if set(old) != set(new):
            raise ValueError('Mutation cannot change agent fields')
        changed.update(k for k in old if old[k] != new[k])
    if changed not in ({'prompt'}, {'model_id'}, {'tools'}, {'topology'}):
        raise ValueError(f'Mutation must change exactly one of prompt, model_id, tools, topology; got {sorted(changed)}')
    if changed == {'prompt'}:
        old_prompts={a['prompt'] for a in before['agents'].values()}
        new_prompts={a['prompt'] for a in after['agents'].values()}
        if len(new_prompts) < len(old_prompts):
            raise ValueError('Prompt mutation collapsed distinct agent roles; preserve prompt diversity')
    return next(iter(changed))


def public_rank(task: str, result: dict, reward: float = 0.0) -> tuple:
    """Task performance first; use successful QEC settings and cost only on ties."""
    metric = result.get('metric')
    if not result.get('valid') or not isinstance(metric, (int, float)) or not math.isfinite(metric):
        return (0, float('-inf'), 0, reward)
    good = sum(not item.get('error') for item in result.get('settings', [])) if task == 'qec' else 0
    return (1, float(metric), good, reward)


def _read(path: Path, limit: int = 800) -> str | None:
    return path.read_text(encoding='utf8', errors='replace')[:limit] if path.is_file() else None


def context(run: Path, limit: int = 5000) -> str:
    path = run/'trajectory_evidence.jsonl'
    if not path.is_file():
        return 'No prior iteration evidence in this task run.'
    rows = [json.loads(line) for line in path.read_text().splitlines()[-4:]]
    brief = [{key: row.get(key) for key in ('label', 'component', 'crossover_hybrid_fallback', 'observed_failure',
              'public_setting_errors', 'root_cause_hypothesis', 'public_child_score',
              'best_public_before', 'verdict', 'prediction_met')} for row in rows]
    return json.dumps(brief, ensure_ascii=False, default=str)[-limit:]


def append(run: Path, task: str, row: dict) -> dict:
    label = row['label']
    directory = run/'evaluations'/label
    failures = _read(directory/'agent_failures.json', 1200)
    reject = (_read(run/'evaluations'/f'{label}.fallback_reason.txt') or
              _read(run/f'{label}.rejected_reason.txt'))
    hybrid_fallback = ((run/'evaluations'/f'{label}.hybrid_fallback.txt').is_file() or
                       (run/f'{label}.hybrid_fallback.txt').is_file())
    audit = directory/'tool_audit.jsonl'
    tool_failures = []
    if audit.is_file():
        for line in audit.read_text(encoding='utf8', errors='replace').splitlines():
            try: event = json.loads(line)
            except json.JSONDecodeError: continue
            if event.get('event') == 'called' and event.get('success') is False:
                tool_failures.append({'agent':event.get('agent'), 'tool':event.get('tool'),
                                      'error':str(event.get('error') or '')[:120]})
    child = row.get('child_public') or {}
    if not child and (directory/'visible.json').is_file():
        child = json.loads((directory/'visible.json').read_text())
    parent = row.get('parent_public') or {}
    def score(visible, direct):
        if isinstance(direct, (int,float)) and math.isfinite(direct):return float(direct)
        value = visible.get('metric')
        return float(value) if isinstance(value,(int,float)) and math.isfinite(value) else None
    ps, cs = score(parent,row.get('parent_score')), score(child,row.get('child_score'))
    settings = [{'setting':v.get('setting'), 'score':v.get('score'),
                 'error':str(v.get('error') or '')[-350:]}
                for v in child.get('settings', [])]
    setting_errors = [v for v in settings if v['error']]
    details = ' '.join(str(x or '') for x in (failures, reject, child.get('error'),
                                              setting_errors[:3],tool_failures[:3]))
    lower = details.lower()
    if 'timed out' in lower or 'timeout' in lower: failure = 'api_or_runtime_timeout'
    elif setting_errors: failure = 'per_setting_interface_or_runtime'
    elif reject: failure = 'mas_yaml_invalid'
    elif 'syntax' in lower or 'import' in lower: failure = 'solver_static_contract'
    elif not child.get('valid',cs is not None): failure = 'solver_runtime_or_selfcheck_invalid'
    elif cs is not None and ps is not None and cs < ps-1e-9:
        failure = 'public_performance_regression'
    else: failure = None
    prior_path=run/'trajectory_evidence.jsonl'
    prior=[json.loads(line) for line in prior_path.read_text().splitlines()] if prior_path.is_file() else []
    baseline=[]
    manifest=run/'candidate_manifest.json'
    contract_valid=None
    if manifest.is_file():
        for c in json.loads(manifest.read_text()):
            if c.get('label') == label and 'contract_valid' in c:
                contract_valid=bool(c['contract_valid']) and bool(c.get('selfcheck_passed', True))
            if c.get('label') not in ('parent_00','parent_01'):continue
            v=c.get('visible',{})
            value=score(v,c.get('visible_score'))
            if value is not None and v.get('valid',c.get('contract_valid',True)):
                baseline.append(value)
    previous=[x['public_child_score'] for x in prior if x.get('public_child_score') is not None
              and x.get('public_valid')]
    peak=max(baseline+previous+[ps] if ps is not None else baseline+previous, default=None)
    valid=bool(child.get('valid',cs is not None)) and cs is not None and contract_valid is not False
    if contract_valid is False:
        failure='solver_contract_or_selfcheck_invalid'
    elif valid and failure == 'api_or_runtime_timeout':
        failure='agent_timeout_recovered_by_valid_candidate'
    changed=bool(row.get('effective_change'))
    if not valid: verdict='invalid'
    elif not changed: verdict='unresolved'
    elif peak is not None and cs > peak+1e-9: verdict='worked'
    elif peak is not None and cs < peak-1e-9: verdict='regressed'
    else: verdict='unresolved'
    decision=row.get('model_decision') or {}
    predicted=decision.get('predicted_direction')
    # The task's public metric is -conflicts: increasing it means fewer conflicts.
    # A bare 'decrease' for graph may mean raw conflicts, so do not guess its target.
    if task == 'graph' and predicted == 'decrease':
        prediction_met=None; prediction_issue='ambiguous_graph_metric_direction'
    elif valid and predicted in ('increase','decrease','same') and cs is not None and ps is not None:
        actual=('increase' if cs>ps+1e-9 else 'decrease' if cs<ps-1e-9 else 'same')
        prediction_met=(predicted==actual);prediction_issue=None
    else: prediction_met=None;prediction_issue='missing_or_unscorable_prediction'
    source=directory/'candidate_solver.py'
    digest=hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
    record={'task':task,'label':label,'action':row.get('action'),
       'crossover_hybrid_fallback':hybrid_fallback,
       'component':(row.get('changed_components') or [None])[0],
       'observed_failure':failure,'diagnostic_excerpt':details[:800],
       'failed_tool_calls':len(tool_failures),'failed_tool_examples':tool_failures[:3],
       'public_setting_errors':setting_errors,
       'root_cause_hypothesis':decision.get('root_cause_hypothesis'),
       'observed_evidence':decision.get('observed_evidence'),
       'intended_change':decision.get('intended_change'),
       'predicted_effect':decision.get('predicted_effect'),
       'predicted_direction':predicted,'decision_source':decision.get('source','json') if decision else None,
       'public_parent_score':ps,'public_child_score':cs,'best_public_before':peak,
       'public_valid':valid,'contract_valid':contract_valid,'candidate_sha256':digest,
       'prediction_met':prediction_met,'prediction_issue':prediction_issue,
       'verdict':verdict,'artifact_dir':f'evaluations/{label}'}
    with prior_path.open('a',encoding='utf8') as stream:
        stream.write(json.dumps(record,ensure_ascii=False,default=str)+'\n')
    records=prior+[record]
    (run/'worked_regressed.json').write_text(json.dumps({
       'task':task,'worked':[x for x in records if x['verdict']=='worked'],
       'regressed':[x for x in records if x['verdict']=='regressed'],
       'invalid':[x for x in records if x['verdict']=='invalid'],
       'unresolved':[x for x in records if x['verdict']=='unresolved'],
       'note':'Public-only; improvement requires a new best score. Hypotheses are not established causes.'},
       ensure_ascii=False,indent=2),encoding='utf8')
    return record
