#!/usr/bin/env python3
"""Fail closed if an official task checkout lacks required public contract assets."""
import json
import sys
from pathlib import Path

FILES = {
 'bbo': ['instruction.md','environment/Dockerfile','environment/methods/main/solver.py','environment/selfcheck.py','environment/data/visible.json'],
 'inventory': ['instruction.md','environment/Dockerfile','environment/methods/main/solver.py','environment/selfcheck.py','environment/data/visible.json','environment/data/visible_anchors.json'],
 'graph': ['instruction.md','environment/Dockerfile','environment/methods/main/solver.py','environment/methods/main/gc_lib.py','environment/selfcheck.py','environment/data/visible.json'],
 'qec': ['instruction.md','environment/Dockerfile','environment/methods/main/solver.py','environment/selfcheck.py','environment/arena_harness.py', 'environment/settings_visible/color_d5/meta.json','environment/settings_visible/color_d5_Z/meta.json','environment/settings_visible/color_d7/meta.json','environment/settings_visible/color_d9/meta.json'],
}

def check(task, root):
    root = Path(root)
    missing = [name for name in FILES[task] if not (root / name).is_file()]
    if missing:
        raise SystemExit(f'{task}: missing official bundle assets: {missing}')
    if task == 'qec':
        for name in ('color_d5','color_d5_Z','color_d7','color_d9'):
            directory = root/'environment/settings_visible'/name
            for filename in ('model.dem','model.stim','dev.npz','eval.npz'):
                if not (directory/filename).is_file():
                    raise SystemExit(f'{task}: missing {directory/filename}')
            meta = json.loads((directory/'meta.json').read_text())
            if not (0 < meta.get('decode_budget_sec',0) <= 900):
                raise SystemExit('QEC budget metadata unexpected')
    print(f'{task}: public bundle assets found in {root}')

if __name__ == '__main__':
    check(sys.argv[1], sys.argv[2])
