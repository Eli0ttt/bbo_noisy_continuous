#!/usr/bin/env python3
"""Complete a run that stopped after final MAS wrote a tool-owned candidate."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import yaml
from visible_selfcheck import evaluate_solver
from evomas_bbo_pipeline import select_best_visible, metrics_from_bbo
from native_bbo_runtime import extract_candidate


def recover(run_dir: Path, container: str, previous_log: Path | None = None) -> dict:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "candidate_manifest.json"
    candidates = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_dir = run_dir / "evaluations" / "final_reward_best"
    if len(candidates) != 2 or [x["label"] for x in candidates] != ["parent_00", "parent_01"]:
        raise ValueError("This recovery only handles the interrupted 2-parent, 0-step, 3-version run")
    final = (final_dir / "aggregator_response.txt").read_text(encoding="utf-8")
    if "solver.py" not in final and "candidate.py" not in final:
        raise ValueError("Final response does not identify a solver file")
    owned = final_dir / "tool_workspace" / "aggregator" / "candidate.py"
    if owned.is_symlink() or not owned.is_file():
        raise ValueError("No terminal aggregator-owned regular candidate.py")
    events = [json.loads(line) for line in (final_dir / "tool_audit.jsonl").read_text().splitlines()]
    if not any(x.get("agent") == "aggregator" and x.get("tool") == "bbo_write_candidate"
               and x.get("event") == "called" and x.get("success") is True for x in events):
        raise ValueError("No successful aggregator tool write was audited")
    candidate = extract_candidate(owned.read_text(encoding="utf-8"))
    started = time.monotonic()
    result = evaluate_solver(candidate, final_dir, container)
    runtime_summary = {"metadata": {"total_tokens": 0}}
    metrics = metrics_from_bbo(result, runtime_summary, time.monotonic() - started)
    (final_dir / "evomas_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    final_path = final_dir / "candidate_solver.py"
    cfg = yaml.safe_load((run_dir / "evaluations" / "final_reward_best.yaml").read_text())
    tools = {aid: list(agent.get("tools", [])) for aid, agent in cfg["agents"].items() if agent.get("tools")}
    candidates.append({"label": "final_reward_best", "visible_score": result["score"],
                       "contract_valid": result["contract_valid"], "candidate_path": str(final_path),
                       "sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
                       "configured_tools": tools,
                       "tools_loaded": sum(e["event"] == "loaded" for e in events),
                       "tool_call_attempts": sum(e["event"] == "called" for e in events),
                       "tool_calls": sum(e["event"] == "called" and e["success"] for e in events)})
    manifest_path.write_text(json.dumps(candidates, indent=2, sort_keys=True) + "\n")
    winner = select_best_visible(candidates, 3, run_dir, False)
    research_seconds = None
    if previous_log:
        stamps = [datetime.strptime(row[:23], "%Y-%m-%d %H:%M:%S,%f")
                  for row in previous_log.read_text(errors="replace").splitlines()
                  if len(row) >= 23 and row[4:5] == "-" and row[19:20] == ","
                  and row[:4].isdigit()]
        if len(stamps) >= 2:
            research_seconds = (stamps[-1] - stamps[0]).total_seconds()
    summary = {"recovered": True, "research_wall_seconds": research_seconds,
               "original_run_status": "failed_after_final_MAS",
 "solver_limit": 3, "num_parents": 2, "max_steps": 0,
               "operations": [], "tool_mode": "bbo", "tool_use_observed": any(
                   x.get("configured_tools") and x.get("tool_calls", 0) > 0 for x in candidates),
               "candidates": candidates, "best_visible": winner,
               "final_reward_best_metrics": metrics,
               "recovery_seconds": time.monotonic() - started,
               "hidden_score": None, "hidden_status": "not_run"}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--container", required=True)
    p.add_argument("--previous-log", type=Path)
    p.add_argument("--hidden-task-root", type=Path)
    p.add_argument("--hidden-jobs-dir", type=Path)
    a = p.parse_args()
    if bool(a.hidden_task_root) != bool(a.hidden_jobs_dir):
        raise ValueError("Provide both hidden task and jobs directories")
    summary = recover(a.run_dir, a.container, a.previous_log)
    if a.hidden_task_root:
        from hidden_handoff import submit
        summary = submit(a.run_dir, a.hidden_task_root, a.hidden_jobs_dir,
                         Path.home()/".local/share/uv/tools/harbor/bin/harbor",
                         "deepseek-v4-flash-s2")
    print(json.dumps(summary, indent=2))
