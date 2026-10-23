from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


CONTAINER = "bbo_noisy_continuous_prebuilt__ootgwrk__env-main-1"


def run_command(args: list[str], output_path: Path) -> subprocess.CompletedProcess:
    result = subprocess.run(
        args,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output_path.write_text(result.stdout, encoding="utf-8")
    return result


def extract_solver(response: str) -> str:
    blocks = re.findall(
        r"```(?:python|py)?\s*(.*?)```",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )

    candidates = [
        block.strip()
        for block in blocks
        if "class Optimizer" in block
        and "def ask" in block
        and "def tell" in block
    ]

    if not candidates:
        raise RuntimeError(
            "No fenced candidate containing class Optimizer, ask, and tell was found"
        )

    return max(candidates, key=len) + "\n"


def main() -> int:
    response_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    output_dir.mkdir(parents=True, exist_ok=True)

    response = response_path.read_text(encoding="utf-8")
    solver = extract_solver(response)

    solver_path = output_dir / "candidate_solver.py"
    solver_path.write_text(solver, encoding="utf-8")

    digest = hashlib.sha256(solver.encode()).hexdigest()
    (output_dir / "candidate_solver.sha256").write_text(
        digest + "\n",
        encoding="utf-8",
    )

    container_solver = (
        f"/tmp/bbo_formal_candidate_{os.getpid()}.py"
    )

    copy_result = run_command(
        [
            "docker",
            "cp",
            str(solver_path),
            f"{CONTAINER}:{container_solver}",
        ],
        output_dir / "docker_cp.log",
    )
    if copy_result.returncode != 0:
        raise RuntimeError("docker cp failed")

    compile_result = run_command(
        [
            "docker",
            "exec",
            CONTAINER,
            "python",
            "-m",
            "py_compile",
            container_solver,
        ],
        output_dir / "py_compile.log",
    )
    if compile_result.returncode != 0:
        raise RuntimeError(
            "Candidate failed py_compile; official selfcheck was not run"
        )

    selfcheck_result = run_command(
        [
            "docker",
            "exec",
            CONTAINER,
            "python",
            "/app/selfcheck.py",
            "--solver",
            container_solver,
            "--json",
        ],
        output_dir / "selfcheck.log",
    )
    if selfcheck_result.returncode != 0:
        raise RuntimeError(
            "Official selfcheck returned a nonzero exit code"
        )

    lines = [
        line.strip()
        for line in selfcheck_result.stdout.splitlines()
        if line.strip()
    ]

    try:
        metrics = json.loads(lines[-1])
    except Exception as exc:
        raise RuntimeError(
            "Could not parse the final JSON line from selfcheck.log"
        ) from exc

    summary = {
        "source_response": str(response_path),
        "candidate_solver": str(solver_path),
        "sha256": digest,
        "container": CONTAINER,
        "metrics": metrics,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
