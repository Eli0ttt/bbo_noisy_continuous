#!/usr/bin/env python3
"""No-API regression check for BBO feedback and the observed counter bugs."""
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ADAPTER = Path(__file__).resolve().parent
UPSTREAM = ADAPTER.parent.parent / "EvoMAS-openai" / "src" / "tools" / "bbo_generation.py"
sys.path.insert(0, str(ADAPTER))

try:
    from smolagents import Tool
except ImportError:
    # Inspect only: the tool runner is not created by this smoke test.
    import types
    sys.modules["smolagents"] = types.SimpleNamespace(Tool=type("Tool", (), {}))

from visible_selfcheck import CONTRACT_PROBE, CandidateSelfcheckError, evaluate_solver
from evomas_bbo_pipeline import build_research_feedback
spec = importlib.util.spec_from_file_location("bbo_generation", UPSTREAM)
tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)

HEADER = '''import numpy as np
class Optimizer:
    def __init__(self, dim, lower, upper, budget, seed, rng):
        self.dim, self.lower, self.upper = dim, lower, upper
        self.budget, self.rng, self.queries_used = budget, rng, 0
    def ask(self, n):
'''
GOOD = HEADER + '''        return self.rng.uniform(self.lower, self.upper, (n, self.dim))
    def tell(self, X, y, metadata=None):
        self.queries_used += len(y)
'''
DOUBLE = HEADER + '''        if self.queries_used >= self.budget:
            return np.zeros((n, self.dim))
        self.queries_used += n
        return self.rng.uniform(self.lower, self.upper, (n, self.dim))
    def tell(self, X, y, metadata=None):
        self.queries_used += len(y)
'''
NEGATIVE = HEADER + '''        if self.queries_used >= 60:
            n_actual = n
            n_explore = max(2, n_actual // 2)
            n_exploit = n_actual - n_explore
            return self.rng.uniform(self.lower, self.upper, (n_exploit, self.dim))
        return self.rng.uniform(self.lower, self.upper, (n, self.dim))
    def tell(self, X, y, metadata=None):
        self.queries_used += len(y)
'''


def probe(source, folder):
    path = folder / "candidate.py"
    path.write_text(source, encoding="utf-8")
    result = subprocess.run([sys.executable, "-c", CONTRACT_PROBE, str(path)],
                            text=True, capture_output=True, timeout=15)
    records = [json.loads(line.removeprefix("PROBE_JSON:"))
               for line in result.stdout.splitlines() if line.startswith("PROBE_JSON:")]
    assert len(records) == 1, result.stdout + result.stderr
    return result.returncode, records[0]


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    assert tools.inspect_candidate(GOOD).startswith("PASS")
    code, outcome = probe(GOOD, root)
    assert code == 0 and outcome["valid"] and not outcome["warnings"]

    assert "both ask and tell" in tools.inspect_candidate(DOUBLE)
    code, outcome = probe(DOUBLE, root)
    assert code == 0 and any("tell received 120" in item for item in outcome["warnings"])
    assert outcome["cases"][0]["unique"] < 120

    assert "n_explore" in tools.inspect_candidate(NEGATIVE)
    code, outcome = probe(NEGATIVE, root)
    assert code != 0 and not outcome["valid"] and "negative dimensions" in outcome["error"]

    evaluations = root / "evaluations"
    candidate = evaluations / "parent_00"
    candidate.mkdir(parents=True)
    (candidate / "candidate_solver.py").write_text(GOOD, encoding="utf-8")
    (candidate / "agent_worker.json").write_text(json.dumps({
        "agent_id": "worker", "success": True, "content": "proposal"}))
    records = [{"label": "parent_00", "visible_score": 0.1,
                "contract_valid": True, "sha256": hashlib.sha256(GOOD.encode()).hexdigest()}]
    metrics = [{"score_anytime": 0.07, "score_final": 0.17,
                "mean_final_latent_objective": 40.0,
                "behavior_probe": {"warnings": ["example warning"]}, "errors": ""}]
    feedback = build_research_feedback(evaluations, records, metrics, 0)
    assert feedback["solver_source"] == GOOD
    assert "example warning" in feedback["summary"]
    assert "visible=0.100000" in feedback["summary"]

    # Check orchestration: synthetic probe is not substituted for official selfcheck.
    calls = []
    def fake_docker(command, **kwargs):
        calls.append(command)
        if command[1] == "cp":
            return subprocess.CompletedProcess(command, 0)
        if "py_compile" in command:
            return subprocess.CompletedProcess(command, 0, "", "")
        if "/app/selfcheck.py" in command:
            payload = {"score": 0.1, "score_anytime": 0.07, "score_final": 0.17,
                       "mean_final_latent_objective": 40.0}
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        return subprocess.CompletedProcess(command, 0,
                'CONTRACT_PASS\nPROBE_JSON:{"valid":true,"cases":[],"warnings":[]}\n', "")
    with patch("visible_selfcheck.subprocess.run", side_effect=fake_docker):
        verified = evaluate_solver(GOOD, root / "mock_score", "visible_container")
    assert verified["score"] == 0.1 and verified["contract_valid"]
    assert any("/app/selfcheck.py" in call for call in calls)

print("PASS: valid, double-count, negative-allocation, visible-feedback, official-call cases")
