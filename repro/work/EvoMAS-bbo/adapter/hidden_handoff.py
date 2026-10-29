#!/usr/bin/env python3
"""Run selected visible winner once through separate official Harbor verifier."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


def validate_completed_selection(summary: dict) -> None:
    candidates = summary["candidates"]
    completed = summary.get("completed_solver_versions", len(candidates))
    if (not candidates
            or len(candidates) != completed
            or len(candidates) > summary["solver_limit"]
            or any(type(item.get("contract_valid")) is not bool for item in candidates)
            or summary["best_visible"].get("contract_valid") is not True
            or summary["best_visible"].get("selfcheck_passed") is not True
            or summary["best_visible"]["sha256"] not in
               {item.get("sha256") for item in candidates if item.get("contract_valid")}):
        raise ValueError("Completed candidates and the visible winner need valid contract records")


def submit(run_dir: Path, task_root: Path, jobs_dir: Path, harbor_bin: Path, model: str):
    run_dir = run_dir.resolve()
    summary_file = run_dir / "summary.json"
    summary = json.loads(summary_file.read_text())
    if summary.get("hidden_status") == "completed":
        raise RuntimeError("Hidden has already been run for this output directory")
    previous = summary.get("hidden_job")
    if previous:
        for result_path in Path(previous).glob("*/result.json"):
            prior = json.loads(result_path.read_text())
            value = ((prior.get("verifier_result") or {}).get("rewards") or {}).get("reward")
            if isinstance(value, (int, float)):
                raise RuntimeError("Previous Harbor job already has a sealed reward; "
                                   "inspect that result before any retry: " + str(result_path))
    if not task_root.is_dir() or not (task_root / "task.toml").is_file():
        raise FileNotFoundError(f"Harbor task directory missing: {task_root}")
    if not harbor_bin.is_file():
        raise FileNotFoundError(f"Harbor executable missing: {harbor_bin}")
    validate_completed_selection(summary)
    winner = run_dir / "best_visible_solver.py"
    source = winner.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != summary["best_visible"]["sha256"]:
        raise ValueError("Visible winner digest does not match the chosen candidate")
    if len(source) > 100000:
        raise ValueError("Solver exceeds the research-side source limit")
    # Existing DSH jobs directories may be owned by a different account or
    # mounted read-only. Harbor needs to create a child directory itself.
    requested_jobs_dir = jobs_dir.expanduser().resolve()
    try:
        requested_jobs_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".evomas-write-check-", dir=requested_jobs_dir):
            pass
        effective_jobs_dir = requested_jobs_dir
    except (OSError, PermissionError):
        effective_jobs_dir = run_dir / "harbor_jobs"
        effective_jobs_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".evomas-write-check-", dir=effective_jobs_dir):
            pass
    name = "evomas-bbo-" + digest[:12] + "-" + str(time.time_ns())
    job_dir = effective_jobs_dir / name
    if job_dir.exists():
        raise FileExistsError(job_dir)
    mounts = [{"type": "bind", "source": str(winner.resolve()),
               "target": "/opt/evomas-selected/solver.py", "read_only": True}]
    command = [str(harbor_bin), "run", "--yes", "--env", "docker",
               "--job-name", name, "--jobs-dir", str(effective_jobs_dir),
               "--n-concurrent", "1", "--n-attempts", "1", "--max-retries", "0",
               "--path", str(task_root), "--agent",
               "dsh_rsi_agent.evo_submission_agent:EvoSubmissionAgent",
               "--model", model, "--mounts", json.dumps(mounts),
               "--agent-kwarg", "solver_sha256="+digest]
    env = os.environ.copy()
    from native_bbo_runtime import find_project_root
    project = find_project_root()
    env["PYTHONPATH"] = str(project) + os.pathsep + env.get("PYTHONPATH", "")
    # Keep failed attempts in the active-time total while exposing the current
    # attempt separately. Historic summaries may only have hidden_seconds.
    earlier = float(summary.get("hidden_attempt_seconds_total") or 0)
    if summary.get("hidden_status") == "failed" and earlier == 0:
        earlier = float(summary.get("hidden_seconds") or 0)
    summary["hidden_attempt_seconds_total"] = earlier
    summary.pop("hidden_return_code", None)
    summary["hidden_status"] = "running"
    summary["hidden_job"] = str(job_dir)
    summary["hidden_jobs_dir_requested"] = str(requested_jobs_dir)
    summary["hidden_jobs_dir_effective"] = str(effective_jobs_dir)
    summary_file.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    started = time.monotonic()
    log = run_dir / (name + ".hidden.log")
    summary["hidden_log"] = str(log)
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    summary["hidden_seconds"] = time.monotonic() - started
    summary["hidden_attempt_seconds_total"] = earlier + summary["hidden_seconds"]
    if result.returncode:
        summary["hidden_status"] = "failed"
        summary["hidden_return_code"] = result.returncode
        summary["total_runtime_seconds"] = (float(summary.get("research_wall_seconds") or 0)
                                           + float(summary.get("recovery_seconds") or 0)
                                           + summary["hidden_attempt_seconds_total"])
        summary_file.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-35:])
        raise RuntimeError(f"Harbor exit {result.returncode}; {log}\n{tail}")
    trial_results = list(job_dir.glob("*/result.json"))
    if len(trial_results) != 1:
        raise RuntimeError("Expected exactly one official Harbor verifier result")
    trial = json.loads(trial_results[0].read_text())
    verifier = trial.get("verifier_result") or {}
    reward = (verifier.get("rewards") or {}).get("reward")
    if not isinstance(reward, (float, int)):
        raise RuntimeError("Official result contains no numeric verifier reward")
    summary["hidden_score"] = float(reward)
    summary["hidden_status"] = "completed"
    summary.pop("hidden_return_code", None)
    summary["total_runtime_seconds"] = (float(summary.get("research_wall_seconds") or 0) +
                                       float(summary.get("recovery_seconds") or 0) +
                                       summary["hidden_attempt_seconds_total"])
    summary_file.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--task-root", type=Path, required=True)
    p.add_argument("--jobs-dir", type=Path, required=True)
    p.add_argument("--harbor-bin", type=Path, default=Path.home()/".local/share/uv/tools/harbor/bin/harbor")
    p.add_argument("--model", default="deepseek-v4-flash-s2")
    a = p.parse_args()
    print(json.dumps(submit(a.run_dir, a.task_root, a.jobs_dir, a.harbor_bin, a.model), indent=2))
