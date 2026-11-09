"""Inside the public QEC image: enforce the official per-setting decode limit."""
import importlib.util
import json
import signal
import sys
import time
import numpy as np
sys.path.insert(0, '/app')
import arena_harness as ah

name=sys.argv[1]
setting=ah.load_setting('/app/settings_visible',name)
budget=float(setting['meta']['decode_budget_sec'])
spec=importlib.util.spec_from_file_location('candidate_solver','/app/methods/main/solver.py')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
class Deadline(Exception): pass
def stop(_sig,_frame): raise Deadline()
signal.signal(signal.SIGALRM,stop)
t0=time.monotonic()
signal.setitimer(signal.ITIMER_REAL,budget)
try:
    prediction=np.asarray(mod.decode(setting,setting['eval_dets']))
    dt=time.monotonic()-t0
    if dt > budget:
        raise Deadline(f"decode took {dt:.3f}s > {budget:.3f}s")
finally:
    signal.setitimer(signal.ITIMER_REAL,0)
if prediction.dtype != np.dtype(bool):
    raise TypeError('decode must return a boolean array')
if prediction.shape != setting['eval_obs'].shape:
    raise ValueError(f'predicted shape {prediction.shape}, expected {setting["eval_obs"].shape}')
ler=ah.ler_of(prediction,setting['eval_obs'])
score=ah.setting_score(ler,setting['meta'],setting['meta']['n_eval'])
print(json.dumps({'setting':name,'ler':float(ler),'score':float(score),'decode_sec':dt,'budget_sec':budget}))
