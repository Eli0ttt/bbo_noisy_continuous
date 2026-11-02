"""Isolated EvoMAS mutation operators. These run on the research host only."""
import asyncio
import inspect
import json
import logging
import re
from pathlib import Path

import yaml
from src.models.model import get_model
from src.meta_model.metamodel import extract_yaml_from_response, validate_mas_config

LOG = logging.getLogger(__name__)
COMPONENTS = ('prompt', 'model_id', 'tools', 'topology')


def _section(config, component):
    if component == 'topology':
        return config.get('topology')
    return {key: agent.get(component) for key, agent in config['agents'].items()}


def _changed_only(original, proposed, component):
    """A specialist may only edit its designated component; merge is deterministic."""
    a, b = yaml.safe_load(original), yaml.safe_load(proposed)
    if not isinstance(b, dict) or set(a['agents']) != set(b.get('agents', {})):
        raise ValueError('Agent identities changed')
    for key in a:
        if key not in ('agents', 'topology') and b.get(key) != a[key]:
            raise ValueError(f'Unexpected root edit: {key}')
    if component != 'topology' and b.get('topology') != a.get('topology'):
        raise ValueError('Unexpected topology edit')
    if component == 'topology':
        if b.get('topology') is None:
            raise ValueError('Missing topology')
        for key in a['agents']:
            if b['agents'][key] != a['agents'][key]:
                raise ValueError('Unexpected agent edit')
    else:
        for key, agent in a['agents'].items():
            if not isinstance(b['agents'][key], dict):
                raise ValueError('Malformed agent')
            for field, value in agent.items():
                if field != component and b['agents'][key].get(field) != value:
                    raise ValueError(f'Unexpected edit: {key}.{field}')
            if set(b['agents'][key]) - set(agent) - {component}:
                raise ValueError('Unexpected agent field')
    return b


def _merge(original, proposals):
    merged = yaml.safe_load(original)
    for component in COMPONENTS:
        b = _changed_only(original, proposals[component], component)
        if component == 'topology':
            merged['topology'] = b['topology']
        else:
            for key in merged['agents']:
                if component in b['agents'][key]:
                    merged['agents'][key][component] = b['agents'][key][component]
                else:
                    merged['agents'][key].pop(component, None)
    return yaml.safe_dump(merged, sort_keys=False, allow_unicode=True)


async def _call(model, prompt):
    if inspect.iscoroutinefunction(model.__call__):
        result = await model(prompt)
    else:
        result = await asyncio.to_thread(model, prompt)
        if inspect.isawaitable(result):
            result = await result
    return str(result)


def _run(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError('EvoMAS synchronous pipeline cannot be called inside an active event loop')


def _candidate(text):
    return extract_yaml_from_response(text)


def _valid(text, model_list):
    if not validate_mas_config(text):
        raise ValueError('Invalid MAS schema')
    config = yaml.safe_load(text)
    if model_list and any(a.get('model_id') not in model_list for a in config['agents'].values()):
        raise ValueError('Model ID outside fixed pool')
    return text


def _base_prompt(self, config, logs, observations, models):
    memory = self.get_memory_context(max_experiences=3) if len(self.memory) else ''
    history_path = getattr(self, 'mutation_trace_path', None)
    prior = ''
    if history_path and Path(history_path).is_file():
        prior = '\n'.join(Path(history_path).read_text(encoding='utf-8').splitlines()[-16:])
    return (f'Prior mutation role assignments, validation, and observed outcomes:\n{prior[:5000]}\n'
            f'Current complete MAS YAML:\n```yaml\n{config}\n```\n'
            f'Current performance and errors:\n{str(logs)[:9000]}\n'
            f'Visible feedback and prior attempts:\n{observations[:18000]}\n'
            f'EvoMAS action memory:\n{memory[:5000]}\n'
            f'Allowed agent model IDs: {models}.\n'
            f'Tool catalog and contract:\n{getattr(self, "bbo_tool_policy", "")[:6000]}\n'
            'Keep the same agent IDs, name and backend. Respect DAG and tool availability. '
            'Return one COMPLETE YAML in a ```yaml block. Do not include solver source.')


def mutate(self, config, logs, observations, model_list=None):
    mode = getattr(self, 'mutation_mode', 'original')
    if mode == 'original':
        return self._original_mutate(config, logs, observations, model_list=model_list)
    if mode not in ('joint', 'parallel'):
        raise ValueError(mode)
    models = model_list or []
    context = _base_prompt(self, config, logs, observations, models)
    trace = {'mode': mode, 'assignments': {}, 'status': 'fallback'}
    try:
        if mode == 'joint':
            prompt = (context + '\nExamine each component IN THIS ORDER: prompt -> model_id -> '
                      'tools -> topology. Make coherent changes only where evidence warrants it. '
                      'Output a complete YAML and preserve official task constraints.')
            result = _valid(_candidate(_run(_call(self.model, prompt))), models)
        else:
            async def parallel():
                assignment_request = (context + '\nChoose a model from the allowed IDs for each of '
                    'prompt, model_id, tools, topology. Use prior mutation outcomes in the context '
                    'when assigning specialties. Models may repeat. Respond ONLY JSON object with '
                    'those four keys and exact allowed model IDs.')
                raw = await _call(self.model, assignment_request)
                found = re.search(r'\{.*\}', raw, re.S)
                try:
                    mapping = json.loads(found.group() if found else raw)
                except (ValueError, AttributeError):
                    mapping = {}
                assignments = {c: mapping.get(c) if mapping.get(c) in models else
                               models[i % len(models)] for i, c in enumerate(COMPONENTS)}
                trace['assignments'] = assignments
                async def specialist(component):
                    model = get_model(model_id=assignments[component],
                                      temperature=self.temperature, max_tokens=self.max_tokens)
                    prompt = (context + f'\nChange ONLY {component}. Preserve all other fields '
                              'exactly, including root metadata. Output COMPLETE YAML.')
                    raw = await _call(model, prompt)
                    try:
                        result = _valid(_candidate(raw), models)
                        _changed_only(config, result, component)
                    except ValueError as error:
                        repaired = await _call(self.model, prompt +
                            f'\nFix formatting/contract error: {error}. Preserve only {component} edits. '
                            'Output complete YAML.\nInvalid response:\n' + raw[:14000])
                        result = _valid(_candidate(repaired), models)
                        _changed_only(config, result, component)
                    return result
                parts = await asyncio.gather(*(specialist(c) for c in COMPONENTS),
                                             return_exceptions=True)
                if any(isinstance(v, BaseException) for v in parts):
                    raise ValueError('Specialist failures: ' + ', '.join(
                        f'{c}: {type(v).__name__}: {v}' for c, v in zip(COMPONENTS, parts)
                        if isinstance(v, BaseException)))
                proposals = dict(zip(COMPONENTS, parts))
                draft = _merge(config, proposals)
                combined = (context + '\nFour validated specialist proposals follow. Combine '
                    'their changes into a complete YAML; preserve agent IDs, model allow-list, '
                    'task contract, tools, and DAG. Resolve interactions and formatting.\n' +
                    '\n'.join(f'### {c}\n```yaml\n{proposals[c]}\n```' for c in COMPONENTS) +
                    f'\nDeterministic merge draft:\n```yaml\n{draft}\n```')
                raw = await _call(self.model, combined)
                try:
                    return _valid(_candidate(raw), models)
                except ValueError:
                    repair = await _call(self.model, combined +
                        '\nYour merged YAML was invalid. Repair its schema and format. '
                        'Return a complete valid YAML only.\nInvalid response:\n' + raw[:14000])
                    return _valid(_candidate(repair), models)
            result = _run(parallel())
        trace['status'] = 'valid'
        return result
    except Exception as exc:
        LOG.warning('Mutation %s failed; preserving parent: %s', mode, exc)
        trace['error'] = f'{type(exc).__name__}: {str(exc)[:700]}'
        return config
    finally:
        path = getattr(self, 'mutation_trace_path', None)
        if path:
            with Path(path).open('a', encoding='utf-8') as out:
                out.write(json.dumps(trace, ensure_ascii=False) + '\n')


def install(cls):
    if not hasattr(cls, '_original_mutate'):
        cls._original_mutate = cls.mutate
        cls.mutate = mutate
