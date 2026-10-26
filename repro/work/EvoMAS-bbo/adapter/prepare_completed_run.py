#!/usr/bin/env python3
"""Upgrade a completed older EvoMAS BBO run for one trusted hidden handoff."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from evomas_bbo_pipeline import select_best_visible
from visible_selfcheck import CONTRACT_PROBE


def prepare(run_dir: Path, container: str, previous_log: Path | None = None) -> dict:
    run_dir = run_dir.resolve()
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("hidden_status") == "completed":
        raise RuntimeError("Already has completed official hidden verification")
    candidates = json.loads((run_dir / "candidate_manifest.json").read_text())
    if len(candidates) < 1:
        raise ValueError("Missing visible candidate manifest")
    for item in candidates:
        label = item["label"]
        if not label.replace("_", "").isalnum():
            raise ValueError("Invalid candidate label")
        source = run_dir / "evaluations" / label / "candidate_solver.py"
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(source)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"Candidate hash mismatch: {label}")
        remote = "/tmp/bbo_contract_" + uuid.uuid4().hex + ".py"
        subprocess.run(["docker", "cp", str(source), f"{container}:{remote}"], check=True,
                       stdout=subprocess.DEVNULL)
        probe = subprocess.run(["docker", "exec", container, "python", "-c",
                                CONTRACT_PROBE, remote], capture_output=True,
                               text=True, timeout=30, check=False)
        item["contract_valid"] = probe.returncode == 0 and "CONTRACT_PASS" in probe.stdout
        item["contract_issue"] = "" if item["contract_valid"] else probe.stderr[-700:]
        item["candidate_path"] = str(source)
    winner = select_best_visible(candidates, len(candidates), run_dir, False)
    summary["candidates"] = candidates
    summary["solver_limit"] = len(candidates)
    summary["best_visible"] = winner
    summary["hidden_status"] = "not_run"
    summary["hidden_score"] = None
    if previous_log and previous_log.is_file():
        stamps = [datetime.strptime(row[:23], "%Y-%m-%d %H:%M:%S,%f")
                  for row in previous_log.read_text(errors="replace").splitlines()
                  if len(row) >= 23 and row[4:5] == "-" and row[19:20] == ","
                  and row[:4].isdigit()]
        if len(stamps) > 1:
            summary["research_wall_seconds"] = (stamps[-1] - stamps[0]).total_seconds()
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (run_dir / "candidate_manifest.json").write_text(json.dumps(candidates, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--previous-log", type=Path)
    parser.add_argument("--hidden-task-root", type=Path)
    parser.add_argument("--hidden-jobs-dir", type=Path)
    args = parser.parse_args()
    if bool(args.hidden_task_root) != bool(args.hidden_jobs_dir):
        raise ValueError("Provide both hidden task and jobs directories")
    result = prepare(args.run_dir, args.container, args.previous_log)
    if args.hidden_task_root:
        from hidden_handoff import submit
        result = submit(args.run_dir, args.hidden_task_root, args.hidden_jobs_dir,
                        Path.home()/".local/share/uv/tools/harbor/bin/harbor",
                        "deepseek-v4-flash-s2")
    print(json.dumps(result, indent=2))
