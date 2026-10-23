#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import yaml

ADAPTER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ADAPTER_DIR))

from bbo_evaluator_smoke import extract_solver, require_env


def call_builder(config_text, instruction, parent_solver, model):
    from openai import OpenAI

    prompt = f"""You are an optimizer implementation agent inside an EvoMAS configuration.

Return exactly one complete self-contained solver.py source file.

Requirements:
- define Optimizer(dim, lower, upper, budget, seed, rng);
- use only Python standard library and NumPy;
- ask(n) must return finite in-bounds points;
- tell(X, y) may use only noisy observations;
- do not read evaluator assets or use network access;
- preserve the parent solver when no safe improvement is justified;
- return only one Python code block and no explanation.

<EvoMASConfiguration>
{config_text}
</EvoMASConfiguration>

<OfficialBBOTask>
{instruction}
</OfficialBBOTask>

<ParentSolver>
{parent_solver}
</ParentSolver>
"""

    client = OpenAI(
        api_key=require_env("OPENAI_API_KEY"),
        base_url=require_env("OPENAI_BASE_URL"),
        timeout=180.0,
        max_retries=0,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=12000,
    )

    return extract_solver(response.choices[0].message.content or "")


class CandidateSelfcheckError(Exception):
    """Official traceback references the solver."""


def evaluate_solver(solver_text, output_dir, container):
    output_dir.mkdir(parents=True, exist_ok=True)

    local_solver = output_dir / "candidate_solver.py"
    local_solver.write_text(solver_text, encoding="utf-8")

    remote_solver = f"/tmp/bbo_evo_{uuid.uuid4().hex}.py"

    subprocess.run(
        [
            "docker",
            "cp",
            str(local_solver),
            f"{container}:{remote_solver}",
        ],
        check=True,
    )

    subprocess.run(
        [
            "docker",
            "exec",
            container,
            "python",
            "-m",
            "py_compile",
            remote_solver,
        ],
        check=True,
    )

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

    result = json.loads(proc.stdout.strip().splitlines()[-1])

    (output_dir / "selfcheck.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-solver", type=Path, required=True)
    parser.add_argument("--instruction", type=Path, required=True)
    parser.add_argument("--evomas-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument(
        "--meta-model",
        default="deepseek-v4-flash-s2",
    )
    parser.add_argument(
        "--worker-model",
        default="deepseek-v4-flash-s2",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.evomas_root))

    from src.meta_model.metamodel import MetaModel

    instruction = args.instruction.read_text(encoding="utf-8")
    parent_solver = args.parent_solver.read_text(encoding="utf-8")

    meta_model = MetaModel(
        model_id=f"openai:{args.meta_model}",
        memory_evolution=False,
        verbose=True,
    )

    selected = meta_model.select(
        task_query="BBO noisy continuous optimization",
        task_description=instruction[:2000],
        pool_dir=str(args.config.parent),
        k=1,
    )

    selected_config_path = Path(selected[0])
    selected_config = selected_config_path.read_text(encoding="utf-8")

    generated_config = meta_model.generate(
        mas_config_path=str(selected_config_path),
        task_samples=[{"query": instruction[:2000]}],
        task_description=instruction[:2000],
        model_list=[f"openai:{args.worker_model}"],
    )

    (args.output_dir / "selected_config.yaml").write_text(
        selected_config,
        encoding="utf-8",
    )

    (args.output_dir / "generated_config.yaml").write_text(
        generated_config,
        encoding="utf-8",
    )

    solver = call_builder(
        generated_config,
        instruction,
        parent_solver,
        args.worker_model,
    )

    result = evaluate_solver(
        solver,
        args.output_dir,
        args.container,
    )

    summary = {
        "selected_config": str(selected_config_path),
        "meta_model": args.meta_model,
        "worker_model": args.worker_model,
        "score": result["score"],
        "score_anytime": result["score_anytime"],
        "score_final": result["score_final"],
        "mean_final_latent_objective": result[
            "mean_final_latent_objective"
        ],
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
