#!/usr/bin/env python3

import argparse
import json
import os
import re
import subprocess
import uuid
from pathlib import Path

import yaml
from openai import OpenAI


def extract_solver(text: str) -> str:
    """Extract a complete Optimizer implementation from a model response."""

    # Prefer fenced Python/code blocks when the model returns them.
    blocks = re.findall(
        r"```(?:python|py)?[^\r\n]*\r?\n(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # If the model returned plain Python without a Markdown code block,
    # fall back to checking the whole response.
    candidates = blocks if blocks else [text]

    for block in candidates:
        if "class Optimizer" not in block:
            continue

        if "def __init__" not in block:
            continue

        candidate = block.strip() + "\n"

        # Validate Python syntax before writing/copying the candidate.
        compile(
            candidate,
            "<candidate_solver>",
            "exec",
        )

        return candidate

    raise ValueError(
        "model response did not contain a complete Optimizer solver"
    )


def require_env(name: str) -> str:
    """Return a required environment variable or fail clearly."""

    value = os.environ.get(name)

    if not value:
        raise RuntimeError(
            f"required environment variable is not set: {name}"
        )

    return value


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--parent-solver",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--instruction",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--container",
        required=True,
    )

    parser.add_argument(
        "--model",
        default="deepseek-v4-flash-s1",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Prepare output directory.
    # ------------------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Read input files.
    # ------------------------------------------------------------------

    config = yaml.safe_load(
        args.config.read_text(
            encoding="utf-8",
        )
    )

    instruction = args.instruction.read_text(
        encoding="utf-8",
    )

    parent_solver = args.parent_solver.read_text(
        encoding="utf-8",
    )

    config_text = yaml.safe_dump(
        config,
        sort_keys=False,
        allow_unicode=True,
    )

    # ------------------------------------------------------------------
    # Build the candidate-generation prompt.
    #
    # Important:
    # We intentionally do NOT use nested Markdown triple-backtick blocks
    # here. XML-like section markers avoid quoting / formatting problems
    # when this source file is copied through Markdown or chat UIs.
    # ------------------------------------------------------------------

    prompt = f"""You are the candidate builder for an EvoMAS configuration.

Generate exactly one complete self-contained Python file named solver.py.

The file must define:

    Optimizer(dim, lower, upper, budget, seed, rng)

Hard requirements:
- use only Python standard library and NumPy;
- ask(n) returns finite points with exactly dim columns;
- every point must stay inside lower and upper;
- tell(X, y) learns only from noisy observations;
- do not read evaluator assets or use network access;
- preserve the parent algorithm when no safe improvement is justified;
- output only one Python code block and no prose outside it.

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

    # ------------------------------------------------------------------
    # Create OpenAI-compatible client.
    # OPENAI_API_KEY and OPENAI_BASE_URL should already point at
    # the DeepSeek-compatible endpoint.
    # ------------------------------------------------------------------

    client = OpenAI(
        api_key=require_env(
            "OPENAI_API_KEY"
        ),
        base_url=require_env(
            "OPENAI_BASE_URL"
        ),
        timeout=180.0,
        max_retries=0,
    )

    # ------------------------------------------------------------------
    # Ask the model to construct a candidate solver.
    # ------------------------------------------------------------------

    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
        max_tokens=12000,
    )

    raw_response = (
        response.choices[0].message.content
        or ""
    )

    # ------------------------------------------------------------------
    # Extract and validate candidate solver source code.
    # ------------------------------------------------------------------

    candidate_solver = extract_solver(
        raw_response
    )

    candidate_path = (
        args.output_dir
        / "candidate_solver.py"
    )

    response_path = (
        args.output_dir
        / "candidate_response.txt"
    )

    candidate_path.write_text(
        candidate_solver,
        encoding="utf-8",
    )

    response_path.write_text(
        raw_response,
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Copy candidate solver into evaluator container.
    # ------------------------------------------------------------------

    remote_path = (
        f"/tmp/bbo_candidate_{uuid.uuid4().hex}.py"
    )

    subprocess.run(
        [
            "docker",
            "cp",
            str(candidate_path),
            f"{args.container}:{remote_path}",
        ],
        check=True,
    )

    # ------------------------------------------------------------------
    # First check: candidate must compile inside the evaluator container.
    # ------------------------------------------------------------------

    compile_result = subprocess.run(
        [
            "docker",
            "exec",
            args.container,
            "python",
            "-m",
            "py_compile",
            remote_path,
        ],
        text=True,
        capture_output=True,
    )

    compile_log = (
        compile_result.stdout
        + compile_result.stderr
    )

    compile_log_path = (
        args.output_dir
        / "py_compile.stdout.log"
    )

    compile_log_path.write_text(
        compile_log,
        encoding="utf-8",
    )

    if compile_result.returncode != 0:
        raise RuntimeError(
            "candidate solver failed py_compile; "
            f"see {compile_log_path}"
        )

    # ------------------------------------------------------------------
    # Second check: run the official BBO selfcheck inside the container.
    # ------------------------------------------------------------------

    selfcheck = subprocess.run(
        [
            "docker",
            "exec",
            args.container,
            "python",
            "/app/selfcheck.py",
            "--solver",
            remote_path,
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    selfcheck_log = (
        selfcheck.stdout
        + selfcheck.stderr
    )

    selfcheck_log_path = (
        args.output_dir
        / "selfcheck.stdout.log"
    )

    selfcheck_log_path.write_text(
        selfcheck_log,
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Parse final JSON result from selfcheck stdout.
    # ------------------------------------------------------------------

    lines = (
        selfcheck.stdout
        .strip()
        .splitlines()
    )

    if not lines:
        raise RuntimeError(
            "selfcheck produced no stdout"
        )

    try:
        result = json.loads(
            lines[-1]
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "could not parse selfcheck JSON; "
            f"see {selfcheck_log_path}"
        ) from exc

    # ------------------------------------------------------------------
    # Verify expected evaluator fields.
    # ------------------------------------------------------------------

    required_result_keys = {
        "score",
        "score_anytime",
        "score_final",
        "mean_final_latent_objective",
    }

    missing_keys = sorted(
        required_result_keys
        - result.keys()
    )

    if missing_keys:
        raise RuntimeError(
            "selfcheck JSON missing required keys: "
            f"{missing_keys}"
        )

    # ------------------------------------------------------------------
    # Save structured selfcheck result.
    # ------------------------------------------------------------------

    selfcheck_json_path = (
        args.output_dir
        / "selfcheck.json"
    )

    selfcheck_json_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Print compact summary for shell / automation use.
    # ------------------------------------------------------------------

    summary = {
        "candidate": str(
            candidate_path
        ),
        "score": result[
            "score"
        ],
        "score_anytime": result[
            "score_anytime"
        ],
        "score_final": result[
            "score_final"
        ],
        "mean_final_latent_objective": result[
            "mean_final_latent_objective"
        ],
    }

    print(
        json.dumps(
            summary,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()