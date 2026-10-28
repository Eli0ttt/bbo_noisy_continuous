#!/usr/bin/env python3
"""Evaluate a full solver on the official visible selfcheck and local API probe."""
from __future__ import annotations
import json
import subprocess
import uuid
from pathlib import Path

class CandidateSelfcheckError(Exception):
    """Official traceback references the solver."""


CONTRACT_PROBE = r'''
import importlib.util, json, sys
import numpy as np
spec = importlib.util.spec_from_file_location("candidate_contract_probe", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
report = {"valid": True, "cases": [], "warnings": []}
for batches in ([1] * 120, [31, 31, 31, 27], [120], [7, 13, 17, 19, 23, 41]):
    opt = module.Optimizer(10, np.full(10, -5.0), np.full(10, 5.0), 120, 0, np.random.default_rng(0))
    seen = []
    try:
        for n in batches:
            points = np.asarray(opt.ask(n), dtype=float)
            assert points.ndim == 2 and 1 <= len(points) <= n and points.shape[1] == 10
            assert np.isfinite(points).all() and (points >= -5).all() and (points <= 5).all()
            seen.extend(map(tuple, points))
            # Deterministic synthetic observations, independent of coordinates.
            # This tests the interface and state transitions, not optimization.
            y = (np.arange(len(points)) + len(seen)) % 7
            opt.tell(points, y.astype(float))
    except Exception as exc:
        report.update(valid=False, failed_case=list(batches[:8]),
                      failed_after=len(seen), error=type(exc).__name__ + ": " + str(exc))
        print("PROBE_JSON:" + json.dumps(report))
        sys.exit(1)
    unique = len(set(seen))
    report["cases"].append({"requests": len(batches), "observed": len(seen),
                            "unique": unique})
    if len(batches) == 120:
        for attr in ("queries_used", "query_count"):
            value = getattr(opt, attr, None)
            if isinstance(value, (int, np.integer)) and value != len(seen):
                report["warnings"].append(
                    f"internal {attr}={value} but tell received {len(seen)} points")
        if len(seen) >= 30 and len(set(seen[-30:])) == 1:
            report["warnings"].append("last 30 proposals were the same point")
print("CONTRACT_PASS")
print("PROBE_JSON:" + json.dumps(report))
'''


def run_behavior_probe(remote_solver, output_dir, container):
    """Research-side synthetic ask/tell probe; never exposed as an agent tool."""
    try:
        proc = subprocess.run(
            ["docker", "exec", container, "python", "-c", CONTRACT_PROBE, remote_solver],
            text=True, capture_output=True, timeout=60, check=False,
        )
        (output_dir / "contract_probe.log").write_text(proc.stdout + proc.stderr,
                                                         encoding="utf-8")
        record = next((json.loads(line.removeprefix("PROBE_JSON:"))
                       for line in proc.stdout.splitlines()
                       if line.startswith("PROBE_JSON:")), None)
        result = record if isinstance(record, dict) else {
            "valid": False, "error": (proc.stderr or proc.stdout)[-1200:]}
        result["valid"] = bool(result.get("valid") and proc.returncode == 0
                               and "CONTRACT_PASS" in proc.stdout)
    except subprocess.TimeoutExpired:
        result = {"valid": False, "error": "synthetic contract probe timed out"}
        (output_dir / "contract_probe.log").write_text(result["error"], encoding="utf-8")
    (output_dir / "behavior_probe.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_selfcheck_output(stdout):
    required = {"score", "score_anytime", "score_final", "mean_final_latent_objective"}
    for line in stdout.splitlines():
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and required.issubset(result):
            return result
    raise ValueError("official selfcheck did not emit a complete JSON result")


def evaluate_solver(solver_text, output_dir, container):
    output_dir.mkdir(parents=True, exist_ok=True)

    local_solver = output_dir / "candidate_solver.py"
    local_solver.write_text(solver_text, encoding="utf-8")

    remote_solver = f"/tmp/bbo_evo_{uuid.uuid4().hex}.py"

    subprocess.run(
        [
            "docker",
            "cp",
            "-a",
            str(local_solver),
            f"{container}:{remote_solver}",
        ],
        check=True,
    )

    syntax = subprocess.run(
        [
            "docker",
            "exec",
            container,
            "python",
            "-m",
            "py_compile",
            remote_solver,
        ],
        check=False, text=True, capture_output=True,
    )
    if syntax.returncode:
        (output_dir / "selfcheck.stdout.log").write_text(
            syntax.stdout + syntax.stderr, encoding="utf-8")
        raise CandidateSelfcheckError(syntax.stderr or syntax.stdout)

    behavior = run_behavior_probe(remote_solver, output_dir, container)

    proc = subprocess.run(
        [
            "docker",
            "exec",
            container,
            "python",
            "/app/selfcheck.py",
            "--solver",
            remote_solver,
            "--json",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    (output_dir / "selfcheck.stdout.log").write_text(
        proc.stdout + proc.stderr,
        encoding="utf-8",
    )
    if proc.returncode:
        if "Traceback (most recent call last)" in proc.stderr and remote_solver in proc.stderr:
            raise CandidateSelfcheckError(proc.stderr)
        proc.check_returncode()

    result = parse_selfcheck_output(proc.stdout)

    result["contract_valid"] = behavior["valid"]
    result["contract_issue"] = "" if behavior["valid"] else behavior.get("error", "")[-1500:]
    result["behavior_probe"] = behavior

    (output_dir / "selfcheck.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return result

