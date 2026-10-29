#!/usr/bin/env python3
"""No API or Docker: pool-only warm start and current-session history evidence."""
from __future__ import annotations
import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evomas_bbo_pipeline import clear_inherited_pool_evidence, build_research_feedback

with tempfile.TemporaryDirectory() as root:
    root = Path(root)
    pool = root / "working_pool"
    pool.mkdir()
    (pool / "compact.yaml").write_text(
        'name: compact\ndescription: Evolved configuration with reward 40.1209\n'
        'backend: smolagents\nmeta:\n  reward: 40.1209\n  accuracy: 40.9\n'
        '  total_tokens: 5000\n  added_at: yesterday\n'
        'agents: {worker: {prompt: "search adaptively", tools: [bbo_read_history]}}\n'
        'topology: {reports_to: {worker: []}}\n')
    (pool / "provenance.json").write_text('{"seed_tool_policy":"basic"}')
    (pool / "mas_index.json").write_text('{"compact":{"accuracy":0.9}}')
    (pool / "memory.json").write_text('[{"prior_score":0.9}]')
    (pool / "old_reports").mkdir()
    (pool / "old_reports" / "hidden.json").write_text('{"score":1}')
    clear_inherited_pool_evidence(pool)
    assert sorted(x.name for x in pool.iterdir()) == ["compact.yaml", "provenance.json"]
    import yaml
    entry = yaml.safe_load((pool / "compact.yaml").read_text())
    assert entry["agents"]["worker"]["prompt"] == "search adaptively"
    assert entry["topology"]["reports_to"] == {"worker": []}
    assert not any(k in entry["meta"] for k in ("reward", "accuracy", "total_tokens", "added_at"))
    assert "40.1209" not in entry["description"]
    clear_inherited_pool_evidence(pool)

    evaluations = root / "evaluations"
    one = evaluations / "parent_00"
    one.mkdir(parents=True)
    source = "import numpy as np\nclass Optimizer: pass\n"
    (one / "candidate_solver.py").write_text(source)
    (one / "agent_worker.json").write_text(json.dumps({
        "agent_id": "worker", "success": True,
        "content": "The measured visible feedback suggests smaller batches."}))
    trace = [{"instance": 0, "latent_best_20_60_120": [50, 40, 30]}]
    (one / "public_trace.json").write_text(json.dumps(trace))
    (one / "selfcheck.json").write_text(json.dumps({"score": .1, "score_anytime": .07}))
    candidate = {"label": "parent_00", "sha256": hashlib.sha256(source.encode()).hexdigest(),
                 "visible_score": .1, "contract_valid": True, "selfcheck_passed": True}
    metrics = {"score_anytime": .07, "score_final": .17,
               "mean_final_latent_objective": 40.0,
               "behavior_probe": {}, "errors": ""}
    feedback = build_research_feedback(evaluations, [candidate], [metrics], 0)
    assert feedback["history"][0]["solver_source"] == source
    assert feedback["history"][0]["agent_reports"][0]["agent"] == "worker"
    assert feedback["history"][0]["public_trace"] == trace
    assert feedback["history"][0]["official_visible_selfcheck"]["score"] == .1
print("PASS: pool YAML preserved without old scores/memory; session reports and public trace indexed")
