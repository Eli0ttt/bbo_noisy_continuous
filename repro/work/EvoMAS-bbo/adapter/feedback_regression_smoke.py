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
from evomas_bbo_pipeline import (build_research_feedback, load_prior_run,
                                 copy_initial_memory, should_restore_basic_tools,
                                 evaluate_config, validated_config,
                                 stage_prior_incumbent, select_best_visible)
from native_bbo_runtime import CandidateExtractionError, extract_candidate
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
    seed_pool = ADAPTER.parent / "pool" / "upstream_bbeh_s2_full_basic"
    if seed_pool.is_dir():
        import yaml
        for seed in seed_pool.glob("*.yaml"):
            config = yaml.safe_load(seed.read_text(encoding="utf-8"))
            assert all(agent.get("tools") for agent in config["agents"].values()), seed
            assert any("bbo_write_candidate" in agent["tools"]
                       for agent in config["agents"].values()), seed
    assert extract_candidate("```python\n" + GOOD + "\n```") == GOOD.strip() + "\n"
    for malformed in ("I will write code later", "```python\nprint('hi')\n```"):
        try:
            extract_candidate(malformed)
        except CandidateExtractionError:
            pass
        else:
            raise AssertionError("missing Optimizer was accepted")
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
    assert feedback["source_contract_valid"]

    (candidate / "evomas_metrics.json").write_text(json.dumps(metrics[0]))
    (root / "best_visible_solver.py").write_text(GOOD)
    (root / "memory.json").write_text("[]")
    (root / "summary.json").write_text(json.dumps({
        "best_visible": records[0], "candidates": records}))
    carried, memory = load_prior_run(root)
    assert carried["source_sha256"] == records[0]["sha256"]
    assert memory == root / "memory.json"
    (candidate / "evomas_metrics.json").write_text(json.dumps({
        "bbo_score": 0.1, "contract_valid": True, "accuracy": 10.0,
        "score_anytime": 0.07, "score_final": 0.17,
        "mean_final_latent_objective": 40.0}))
    next_run = root / "next_run"
    next_run.mkdir()
    next_eval = next_run / "evaluations"
    next_eval.mkdir()
    incumbent = stage_prior_incumbent(root, next_run, next_eval)
    lower = {"label": "new_child", "contract_valid": True, "visible_score": 0.02,
             "candidate_path": str(candidate / "candidate_solver.py"),
             "sha256": records[0]["sha256"]}
    assert select_best_visible([lower], 1, next_run, False, incumbent)["label"] == "continuation_incumbent"
    (next_run / "summary.json").write_text(json.dumps({
        "best_visible": incumbent, "incumbent_provenance": incumbent,
        "candidates": [lower]}))
    reused, _ = load_prior_run(next_run)
    assert reused["source_sha256"] == records[0]["sha256"]
    (root / "memory.json").write_text(json.dumps([{
        "query": "BBO", "action": "mutation", "config_changes": "prompt",
        "old_accuracy": 23.79, "new_accuracy": 15.48, "success": False,
        "analysis": "wrong 2379 percent and unrelated BBEH interpretation"}]))
    assert copy_initial_memory(root / "memory.json", root / "corrected_memory.json") == 1
    normalized = json.loads((root / "corrected_memory.json").read_text())[0]
    assert abs(normalized["old_accuracy"] - 0.2379) < 1e-12
    assert "BBEH" not in normalized["analysis"]
    originals = {"croto.yaml", "smoa.yaml"}
    assert should_restore_basic_tools("basic", originals, root / "croto.yaml")
    assert not should_restore_basic_tools("basic", originals, root / "evolved_01.yaml")
    with patch("evomas_bbo_pipeline.execute_mas", side_effect=CandidateExtractionError("missing code")):
        failed = evaluate_config("agents: {}\n", "invalid", root, root / "instruction.md",
                                 "unused", False, "bbo")
    assert not failed["contract_valid"] and not failed["selfcheck_passed"]
    assert failed["accuracy"] < 0
    assert (root / "invalid" / "aggregator_response.txt").exists() is False
    assert "missing code" in (root / "invalid" / "selfcheck.json").read_text()
    (root / "best_visible_solver.py").write_text(GOOD + "# tampered\n")
    try:
        load_prior_run(root)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered incumbent must be rejected")

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

print("PASS: contract boundaries, public feedback, continuation integrity, official-call cases")
