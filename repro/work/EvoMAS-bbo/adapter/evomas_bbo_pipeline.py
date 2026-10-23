#!/usr/bin/env python3
"""BBO evaluator adapter using EvoMAS's original evolutionary operators."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

import yaml

ADAPTER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ADAPTER_DIR))

from native_bbo_runtime import execute as execute_mas


class DryRunMetaModel:
    """Deterministic no-API stand-in used only for pipeline wiring tests."""

    memory_evolution = False

    def select(self, task_query, task_description, pool_dir, k=2, mas_index=None):
        return [str(p) for p in sorted(Path(pool_dir).glob("*.yaml"))[:k]]

    def generate(self, mas_config_path, task_samples, task_description, model_list=None):
        return Path(mas_config_path).read_text(encoding="utf-8")

    def mutate(self, mas_config, execution_logs, observations, model_list=None):
        config = yaml.safe_load(mas_config)
        config.setdefault("meta", {})["fake_evolution_step"] = True
        return yaml.safe_dump(config, sort_keys=False)

    def crossover(self, mas_config_1, mas_config_2, logs_1, logs_2, model_list=None):
        return self.mutate(mas_config_1, logs_1, "dry-run crossover")

    def update_memory(self, **kwargs):
        return None


def metrics_from_bbo(result, runtime_summary, elapsed):
    return {
        "accuracy": float(result["score"]) * 100.0,
        "total_tokens": int(
            runtime_summary.get("metadata", {}).get("total_tokens", 0)
        ),
        "total_time": elapsed,
        "bbo_score": float(result["score"]),
        "score_anytime": float(result["score_anytime"]),
        "score_final": float(result["score_final"]),
        "mean_final_latent_objective": float(
            result["mean_final_latent_objective"]
        ),
    }


def evaluate_config(
    config_text,
    label,
    output_dir,
    instruction,
    container,
    dry_run,
):
    config_path = output_dir / f"{label}.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    if dry_run:
        config = yaml.safe_load(config_text)
        fake_step = bool(
            config.get("meta", {}).get("fake_evolution_step")
        )
        score = 0.60 if fake_step else 0.40
        return {
            "accuracy": score * 100.0,
            "total_tokens": 0,
            "total_time": 0.0,
            "bbo_score": score,
            "score_anytime": score,
            "score_final": score,
            "mean_final_latent_objective": 0.0,
        }

    if not container:
        raise ValueError("--container is required without --dry-run")

    from evomas_bbo_step0 import CandidateSelfcheckError, evaluate_solver

    run_dir = output_dir / label
    started = time.monotonic()
    runtime_summary = execute_mas(
        config_path,
        instruction,
        run_dir,
        dry_run=False,
    )
    solver = (run_dir / "candidate_solver.py").read_text(
        encoding="utf-8"
    )
    result = evaluate_solver(solver, run_dir, container)
    metrics = metrics_from_bbo(
        result,
        runtime_summary,
        time.monotonic() - started,
    )
    (run_dir / "evomas_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evomas-root", type=Path, required=True)
    parser.add_argument("--seed-pool", type=Path, required=True)
    parser.add_argument("--instruction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--container")
    parser.add_argument(
        "--meta-model",
        default="openai:deepseek-v4-flash-s2",
    )
    parser.add_argument("--num-parents", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mutation-probability",
        type=float,
        default=0.8,
    )
    parser.add_argument("--memory-evolution", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--improvement-threshold", type=float, default=0.05)
    parser.add_argument("--cost-weight", choices=("tokens", "time", "both"), default="both")
    parser.add_argument("--beta", type=float, default=1e-6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.num_parents < 1 or args.max_steps < 1:
        raise ValueError("--num-parents and --max-steps must be positive")
    if not 0.0 <= args.mutation_probability <= 1.0:
        raise ValueError("--mutation-probability must be in [0, 1]")
    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite output directory: {args.output_dir}"
        )
    if not args.seed_pool.is_dir() or not args.instruction.is_file():
        raise FileNotFoundError("seed pool or instruction file is missing")

    sys.path.insert(0, str(args.evomas_root))

    from src.meta_model.experience import consolidate_evolution_trace
    from src.meta_model.mas_index import MASIndex
    from src.meta_model.metamodel import MetaModel
    from src.meta_model.pool import ConfigurationPool, add_to_pool_if_better
    from src.meta_model.reward import compute_reward

    args.output_dir.mkdir(parents=True)
    working_pool = args.output_dir / "working_pool"
    shutil.copytree(args.seed_pool, working_pool)

    evaluation_dir = args.output_dir / "evaluations"
    evaluation_dir.mkdir()

    instruction_text = args.instruction.read_text(encoding="utf-8")
    task_query = "RSI-Exam Query-Budgeted Noisy Black-Box Optimization"
    task_description = instruction_text[:4000]
    task_samples = [{"query": task_description}]

    if args.dry_run:
        meta_model = DryRunMetaModel()
    else:
        meta_model = MetaModel(
            model_id=args.meta_model,
            memory_path=str(args.output_dir / "memory.json"),
            memory_evolution=args.memory_evolution,
            verbose=True,
        )

    mas_index = MASIndex(str(working_pool))
    parent_paths = meta_model.select(
        task_query,
        task_description,
        str(working_pool),
        k=args.num_parents,
        mas_index=mas_index,
    )
    if not parent_paths:
        raise RuntimeError("EvoMAS selection returned no parents")

    configs = []
    metrics = []

    for index, parent_path in enumerate(parent_paths):
        generated = meta_model.generate(
            parent_path,
            task_samples,
            task_description,
            model_list=["openai:deepseek-v4-flash-s2"],
        )
        parent_metrics = evaluate_config(
            generated,
            f"parent_{index:02d}",
            evaluation_dir,
            args.instruction,
            args.container,
            args.dry_run,
        )
        configs.append(generated)
        metrics.append(parent_metrics)

    rewards = [
        compute_reward(item, beta=args.beta, cost_weight=args.cost_weight)
        for item in metrics
    ]
    best_index = max(range(len(rewards)), key=rewards.__getitem__)
    operations = []

    evolution_rng = random.Random(args.seed)
    for step in range(1, args.max_steps + 1):
        old_index = best_index
        use_mutation = (
            args.num_parents == 1
            or evolution_rng.random() < args.mutation_probability
        )

        if use_mutation:
            offspring = meta_model.mutate(
                configs[best_index],
                metrics[best_index],
                f"BBO score={metrics[best_index]['bbo_score']:.6f}",
                model_list=["openai:deepseek-v4-flash-s2"],
            )
            action = "mutation"
        else:
            ranked = sorted(
                range(len(rewards)),
                key=rewards.__getitem__,
                reverse=True,
            )
            offspring = meta_model.crossover(
                configs[ranked[0]],
                configs[ranked[1]],
                metrics[ranked[0]],
                metrics[ranked[1]],
                model_list=["openai:deepseek-v4-flash-s2"],
            )
            action = "crossover"

        child_metrics = evaluate_config(
            offspring,
            f"step_{step:02d}_{action}",
            evaluation_dir,
            args.instruction,
            args.container,
            args.dry_run,
        )
        child_reward = compute_reward(
            child_metrics,
            beta=args.beta, cost_weight=args.cost_weight,
        )

        configs.append(offspring)
        metrics.append(child_metrics)
        rewards.append(child_reward)
        best_index = max(range(len(rewards)), key=rewards.__getitem__)

        operations.append({
            "step": step,
            "type": action,
            "reward": child_reward,
            "improved": best_index == len(rewards) - 1,
        })

        meta_model.update_memory(
            action_type=action,
            query=task_query,
            old_config=configs[old_index],
            new_config=offspring,
            old_accuracy=metrics[old_index]["accuracy"],
            new_accuracy=child_metrics["accuracy"],
        )

    pool = ConfigurationPool(str(working_pool))
    accepted = add_to_pool_if_better(
        pool,
        configs[best_index],
        metrics[best_index],
        metrics[:len(parent_paths)],
        task_query,
        beta=args.beta, cost_weight=args.cost_weight,
        threshold=args.improvement_threshold,
    )

    trace = consolidate_evolution_trace(
        operations,
        task_query,
        metrics[0]["accuracy"],
        metrics[best_index]["accuracy"],
        configs[best_index],
    )
    (args.output_dir / "evolution_trace.txt").write_text(
        trace + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "best_config.yaml").write_text(
        configs[best_index],
        encoding="utf-8",
    )

    summary = {
        "dry_run": args.dry_run,
        "seed_pool": str(args.seed_pool),
        "working_pool": str(working_pool),
        "num_parents": len(parent_paths),
        "max_steps": args.max_steps,
        "seed": args.seed,
        "operations": operations,
        "parent_metrics": metrics[:len(parent_paths)],
        "best_metrics": metrics[best_index],
        "best_reward": rewards[best_index],
        "best_config_sha256": hashlib.sha256(
            configs[best_index].encode()
        ).hexdigest(),
        "accepted_pool_path": accepted,
        "memory_evolution": args.memory_evolution,
        "parameters": {
            "beta": args.beta,
            "cost_weight": args.cost_weight,
            "improvement_threshold": args.improvement_threshold,
            "mutation_probability": args.mutation_probability,
        },
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
