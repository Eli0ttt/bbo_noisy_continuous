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
from evomas_bbo_pipeline import (build_research_feedback, load_prior_memory,
                                 copy_initial_memory, should_restore_basic_tools,
                                 normalize_basic_pool_seeds,
                                 retarget_unavailable_models,
                                 evaluate_config, validated_config,
                                 sanitize_working_pool, select_best_visible)
from native_bbo_runtime import (CandidateExtractionError, agent_instruction,
                                extract_candidate, find_project_root,
                                canonical_model_id, parse_model_pool,
                                configure_allowed_models)
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
    from types import SimpleNamespace
    judge = SimpleNamespace(role="judge", prompt="judge", tools=["bbo_read_context"])
    instruction = agent_instruction(judge)
    assert "Assigned callable BBO tools for THIS agent: bbo_read_context" in instruction
    assert "bbo_write_candidate(source=" not in instruction
    assert find_project_root() == Path(__file__).resolve().parents[4]
    seed_pool = ADAPTER.parent / "pool" / "upstream_bbeh_s2_full_basic"
    if seed_pool.is_dir():
        import yaml
        for seed in seed_pool.glob("*.yaml"):
            config = yaml.safe_load(seed.read_text(encoding="utf-8"))
            expected = ["bbo_read_context", "bbo_read_public_data",
                        "bbo_read_prior_feedback", "bbo_read_prior_trace",
                        "bbo_write_candidate", "bbo_preflight_candidate",
                        "bbo_probe_candidate"]
            assert all(agent.get("tools") == expected
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
    assert canonical_model_id("GLM-5.3-W4A8") == "openai:GLM-5.3-W4A8"
    assert parse_model_pool("deepseek-v4-flash-s1,GLM-5.3-Flash") == [
        "openai:deepseek-v4-flash-s1", "openai:GLM-5.3-Flash"]
    try:
        parse_model_pool("not-an-available-model")
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported model accepted")
    public_root = root / "public_context"
    public_root.mkdir()
    public = tools.make_tool("bbo_read_public_data", public_root, "worker")
    assert json.loads(public.forward("summary"))["n_instances"] == 12
    assert json.loads(public.forward("instance_0"))["dim"] == 10
    assert public.forward("../hidden").startswith("DENIED")
    prior = tools.make_tool("bbo_read_prior_feedback", public_root, "worker")
    trace_tool = tools.make_tool("bbo_read_prior_trace", public_root, "worker")
    assert prior.forward("summary").startswith("NO_PRIOR_VERSION")
    assert trace_tool.forward("checkpoints").startswith("NO_PRIOR_VERSION")
    (public_root / "tool_workspace" / "prior_feedback.json").write_text(json.dumps({
        "source_label": "step_01", "summary": "public feedback",
        "public_trace": [{"instance": 0, "latent_best_20_60_120": [4,3,2]}]}))
    assert json.loads(prior.forward("summary"))["source_label"] == "step_01"
    assert json.loads(trace_tool.forward("checkpoints"))[0]["instance"] == 0
    tool_root = root / "agent_probe"
    tool_root.mkdir()
    writer = tools.make_tool("bbo_write_candidate", tool_root, "worker")
    checker = tools.make_tool("bbo_probe_candidate", tool_root, "worker", "offline-visible")
    (tool_root / "tool_workspace" / "worker" / "candidate.py").write_text(GOOD)
    assert checker.forward("candidate.py").startswith("DENIED: write your candidate")
    assert writer.forward(GOOD).startswith("WROTE")
    def fake_probe_docker(command, **kwargs):
        if "python" in command:
            payload = {"valid": True, "cases": [{"observed": 120, "unique": 120}],
                       "warnings": []}
            return subprocess.CompletedProcess(command, 0,
                "CONTRACT_PASS\nPROBE_JSON:" + json.dumps(payload) + "\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")
    with patch("subprocess.run", side_effect=fake_probe_docker) as mocked_probe:
        assert "quick synthetic ask/tell" in checker.forward("candidate.py")
        assert any("docker" == call.args[0][0] and "python" in call.args[0]
                   for call in mocked_probe.call_args_list)
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
    assert load_prior_memory(root) == root / "memory.json"
    next_run = root / "next_run"
    next_run.mkdir()
    next_eval = next_run / "evaluations"
    next_eval.mkdir()
    lower = {"label": "new_child", "contract_valid": True, "visible_score": 0.02,
             "candidate_path": str(candidate / "candidate_solver.py"),
             "sha256": records[0]["sha256"]}
    higher = dict(lower, label="new_parent", visible_score=0.03)
    assert select_best_visible([lower, higher], 2, next_run, False)["label"] == "new_parent"
    # An earlier solver is never carried even when its visible score is higher.
    assert (next_run / "best_visible_solver.py").read_text() == GOOD
    (root / "best_visible_solver.py").write_text("tampered old solver")
    assert load_prior_memory(root) == root / "memory.json"
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
    copied = root / "copied"
    copied.mkdir()
    (copied / "croto.yaml").write_text((seed_pool / "croto.yaml").read_text())
    evolved_text = (seed_pool / "croto.yaml").read_text().replace(
        "- bbo_write_candidate\n", "", 1)
    (copied / "evolved_01.yaml").write_text(evolved_text)
    normalize_basic_pool_seeds(copied, "basic", originals)
    assert all(a["tools"] == expected for a in yaml.safe_load(
        (copied / "croto.yaml").read_text())["agents"].values())
    assert (copied / "evolved_01.yaml").read_text() == evolved_text
    active = parse_model_pool("GLM-5.3-W4A8")
    rewritten = retarget_unavailable_models(copied, active)
    assert len(rewritten) == 2
    configure_allowed_models(active)
    assert all(agent["model_id"] == active[0] for agent in yaml.safe_load(
        (copied / "croto.yaml").read_text())["agents"].values())
    configure_allowed_models(["openai:deepseek-v4-flash-s2"])
    with patch("evomas_bbo_pipeline.execute_mas", side_effect=CandidateExtractionError("missing code")):
        failed = evaluate_config("agents: {}\n", "invalid", root, root / "instruction.md",
                                 "unused", False, "bbo")
    assert not failed["contract_valid"] and not failed["selfcheck_passed"]
    assert failed["accuracy"] < 0
    assert (root / "invalid" / "aggregator_response.txt").exists() is False
    assert "missing code" in (root / "invalid" / "selfcheck.json").read_text()
    sys.path.insert(0, str(ADAPTER.parent.parent / "EvoMAS-openai"))
    pool = root / "pool"
    pool.mkdir()
    good_seed = (seed_pool / "majority_vote.yaml").read_text(encoding="utf-8")
    (pool / "valid.yaml").write_text(good_seed)
    (pool / "invalid.yaml").write_text("agents: {}\n")
    (pool / "mas_index.json").write_text(json.dumps({
        "valid": {"path": "/stale/valid.yaml"},
        "invalid": {"path": "/stale/invalid.yaml"}}))
    rejected = sanitize_working_pool(pool, "bbo")
    assert len(rejected) == 1 and rejected[0]["name"] == "invalid.yaml"
    assert (root / "rejected_pool_configs" / "invalid.yaml").is_file()
    index = json.loads((pool / "mas_index.json").read_text())
    assert list(index) == ["valid"] and index["valid"]["path"] == str(pool / "valid.yaml")

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
    def invalid_solver_docker(command, **kwargs):
        if "/app/selfcheck.py" in command:
            return subprocess.CompletedProcess(command, 1, "",
                'Traceback (most recent call last):\n  File "/app/selfcheck.py", line 112, in _validate_points\nValueError: ask returned an out-of-bounds point')
        return fake_docker(command, **kwargs)
    with patch("visible_selfcheck.subprocess.run", side_effect=invalid_solver_docker):
        try:
            evaluate_solver(GOOD, root / "mock_out_of_bounds", "visible_container")
        except CandidateSelfcheckError as exc:
            assert "out-of-bounds point" in str(exc)
        else:
            raise AssertionError("official harness contract failure was not classified as candidate failure")
    (root / "bad_bounds").mkdir()
    (root / "bad_bounds" / "candidate_solver.py").write_text(GOOD)
    with patch("evomas_bbo_pipeline.execute_mas", return_value={"metadata": {}}), \
         patch("visible_selfcheck.evaluate_solver", side_effect=CandidateSelfcheckError("out-of-bounds point")):
        failed = evaluate_config("agents: {}\n", "bad_bounds", root, root / "instruction.md",
                                 "visible_container", False, "bbo")
    assert not failed["selfcheck_passed"] and failed["accuracy"] < 0

print("PASS: contract boundaries, public feedback, continuation integrity, official-call cases")
