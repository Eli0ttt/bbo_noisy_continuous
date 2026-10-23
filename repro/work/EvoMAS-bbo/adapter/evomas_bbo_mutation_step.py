#!/usr/bin/env python3

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ADAPTER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ADAPTER_DIR))

from evomas_bbo_step0 import evaluate_solver
from evomas_bbo_topology_smoke import build_levels, call_agent


def evaluate_config(config, instruction, parent_solver, output_dir, container):
    output_dir.mkdir(parents=True, exist_ok=True)

    levels = build_levels(config)
    reports = {}

    for level in levels:
        def run_one(agent_id):
            return agent_id, call_agent(
                config["agents"][agent_id],
                instruction,
                parent_solver,
                reports,
            )

        parallel = config.get("execution", {}).get(
            "parallel_workers",
            False,
        )

        if parallel and len(level) > 1:
            with ThreadPoolExecutor(max_workers=len(level)) as pool:
                results = list(pool.map(run_one, level))
        else:
            results = [run_one(agent_id) for agent_id in level]

        for agent_id, (raw_response, solver) in results:
            reports[agent_id] = solver

            (output_dir / f"{agent_id}_response.txt").write_text(
                raw_response,
                encoding="utf-8",
            )

            (output_dir / f"{agent_id}_solver.py").write_text(
                solver,
                encoding="utf-8",
            )

    final_agent = levels[-1][-1]
    result = evaluate_solver(
        reports[final_agent],
        output_dir,
        container,
    )

    return {
        "levels": levels,
        "final_agent": final_agent,
        "result": result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--base-score-json", type=Path, required=True)
    parser.add_argument("--instruction", type=Path, required=True)
    parser.add_argument("--parent-solver", type=Path, required=True)
    parser.add_argument("--evomas-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument(
        "--meta-model",
        default="deepseek-v4-flash-s1",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.evomas_root))

    from src.meta_model.metamodel import MetaModel

    base_config_text = args.base_config.read_text(
        encoding="utf-8",
    )
    base_score = json.loads(
        args.base_score_json.read_text(
            encoding="utf-8",
        )
    )
    instruction = args.instruction.read_text(
        encoding="utf-8",
    )
    parent_solver = args.parent_solver.read_text(
        encoding="utf-8",
    )

    meta_model = MetaModel(
        model_id=f"openai:{args.meta_model}",
        memory_evolution=False,
        verbose=True,
    )

    observations = (
        "The previous topology-aware BBO evaluation completed successfully.\n"
        f"Official score: {base_score['score']}\n"
        f"Anytime score: {base_score['score_anytime']}\n"
        f"Final score: {base_score['score_final']}\n"
        "The candidate did not exceed the current floor baseline."
    )

    execution_logs = {
        "accuracy": float(base_score["score"]) * 100.0,
        "errors": [],
        "bbo_score": float(base_score["score"]),
        "score_anytime": float(base_score["score_anytime"]),
        "score_final": float(base_score["score_final"]),
    }

    mutated_config_text = meta_model.mutate(
        mas_config=base_config_text,
        execution_logs=execution_logs,
        observations=observations,
        model_list=[
            "openai:deepseek-v4-flash-s1",
            "openai:deepseek-v4-flash-s2",
        ],
    )

    mutated_config_path = (
        args.output_dir / "mutated_config.yaml"
    )
    mutated_config_path.write_text(
        mutated_config_text,
        encoding="utf-8",
    )

    mutated_config = yaml.safe_load(
        mutated_config_text
    )

    evaluation = evaluate_config(
        mutated_config,
        instruction,
        parent_solver,
        args.output_dir / "mutated_evaluation",
        args.container,
    )

    result = evaluation["result"]

    summary = {
        "base_score": base_score["score"],
        "mutated_score": result["score"],
        "base_score_anytime": base_score["score_anytime"],
        "mutated_score_anytime": result["score_anytime"],
        "base_score_final": base_score["score_final"],
        "mutated_score_final": result["score_final"],
        "final_agent": evaluation["final_agent"],
        "levels": evaluation["levels"],
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
