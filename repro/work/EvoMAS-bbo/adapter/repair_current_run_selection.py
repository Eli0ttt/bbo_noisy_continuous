#!/usr/bin/env python3
"""Repair a completed run that mistakenly selected an earlier run's solver.

This does not rerun the MAS or score a candidate. It preserves the previous
summary and selected solver for audit, then selects only this run's verified
visible candidates before the official hidden handoff.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def repair(run_dir: Path):
    run_dir = run_dir.resolve()
    summary_file = run_dir / "summary.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if summary.get("hidden_status") != "not_run":
        raise ValueError("Refusing to change a run after hidden verification began")
    candidates = summary["candidates"]
    if len(candidates) != summary["solver_limit"]:
        raise ValueError("Incomplete run candidate manifest")
    valid = []
    for item in candidates:
        path = run_dir / "evaluations" / item["label"] / "candidate_solver.py"
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Candidate source mismatch: {item['label']}")
        if item.get("contract_valid") is True and item.get("selfcheck_passed") is True:
            metric_path = path.parent / "evomas_metrics.json"
            metrics = json.loads(metric_path.read_text(encoding="utf-8"))
            if not metrics.get("contract_valid") or abs(
                    float(metrics["bbo_score"]) - float(item["visible_score"])) > 1e-9:
                raise ValueError(f"Visible score mismatch: {item['label']}")
            valid.append((item, path))
    if not valid:
        raise ValueError("This run has no valid visible solver")
    winner, source = max(valid, key=lambda pair: pair[0]["visible_score"])
    selected = run_dir / "best_visible_solver.py"
    old = summary["best_visible"]
    if old["label"] == winner["label"]:
        if hashlib.sha256(selected.read_bytes()).hexdigest() != winner["sha256"]:
            raise ValueError("Selected solver hash mismatch")
        return winner
    if old["label"] in {item["label"] for item in candidates}:
        raise ValueError("Existing selection belongs to this run; investigate before repair")
    if hashlib.sha256(selected.read_bytes()).hexdigest() != old["sha256"]:
        raise ValueError("Previously selected solver hash mismatch")
    backup_summary = run_dir / "summary.before_current_selection.json"
    backup_solver = run_dir / "best_visible_solver.before_current_selection.py"
    if backup_summary.exists() or backup_solver.exists():
        raise FileExistsError("Partial repair backup already exists; inspect before retry")
    shutil.copy2(summary_file, backup_summary)
    shutil.copy2(selected, backup_solver)
    summary["excluded_prior_incumbent"] = summary.pop("incumbent_provenance", old)
    summary["best_visible"] = winner
    summary["selection_protocol"] = "best_valid_visible_candidate_from_this_run_only"
    # Both replacements stay in the same filesystem; backups make interruption recoverable.
    with tempfile.NamedTemporaryFile(dir=run_dir, prefix=".selected-", delete=False) as stream:
        staged_solver = Path(stream.name)
        stream.write(source.read_bytes())
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=run_dir,
                                     prefix=".summary-", delete=False) as stream:
        staged_summary = Path(stream.name)
        json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    try:
        os.replace(staged_solver, selected)
        os.replace(staged_summary, summary_file)
    finally:
        staged_solver.unlink(missing_ok=True)
        staged_summary.unlink(missing_ok=True)
    return winner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(repair(args.run_dir), ensure_ascii=False, indent=2))
