#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutated-config", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--evomas-root", type=Path, required=True)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.evomas_root))

    from src.meta_model.experience import consolidate_evolution_trace
    from src.meta_model.pool import ConfigurationPool
    from src.meta_model.reward import should_add_to_pool

    config_text = args.mutated_config.read_text(
        encoding="utf-8",
    )
    config = yaml.safe_load(config_text)
    summary = json.loads(
        args.summary.read_text(
            encoding="utf-8",
        )
    )

    allowed_bbo_tools = set()

    for agent_id, agent in config.get("agents", {}).items():
        tools = set(agent.get("tools") or [])
        unknown = tools - allowed_bbo_tools
        if unknown:
            raise RuntimeError(
                f"unsupported BBO tools in agent {agent_id}: "
                f"{sorted(unknown)}"
            )

    base_score = float(summary["base_score"])
    mutated_score = float(summary["mutated_score"])

    parent_metrics = {
        "accuracy": base_score * 100.0,
        "total_tokens": 0,
        "total_time": 0.0,
    }

    new_metrics = {
        "accuracy": mutated_score * 100.0,
        "total_tokens": 0,
        "total_time": 0.0,
    }

    accepted = should_add_to_pool(
        new_metrics=new_metrics,
        parent_metrics=parent_metrics,
        beta=0.0,
        cost_weight="both",
        improvement_threshold=1.0,
    )

    args.pool_dir.mkdir(parents=True, exist_ok=True)
    args.memory_dir.mkdir(parents=True, exist_ok=True)

    added_path = None

    if accepted:
        pool = ConfigurationPool(str(args.pool_dir))
        added_path = pool.add_configuration(
            config_yaml=config_text,
            name="bbo_mutation_step_002",
            description=(
                "EvoMAS BBO mutation accepted by visible-score gate"
            ),
            successful_tasks=[
                {
                    "q": "bbo_noisy_continuous",
                    "score": mutated_score,
                }
            ],
            metadata={
                "base_score": base_score,
                "mutated_score": mutated_score,
                "adapter": "evomas_bbo",
            },
        )

    operations = [
        {
            "type": "mutation",
            "changes": "configuration mutation under BBO adapter",
            "accuracy_change": mutated_score - base_score,
            "improved": mutated_score > base_score,
            "base_score": base_score,
            "mutated_score": mutated_score,
        }
    ]

    trace = consolidate_evolution_trace(
        operations=operations,
        query="bbo_noisy_continuous",
        initial_accuracy=base_score,
        final_accuracy=mutated_score,
        best_config=config_text,
    )

    trace_path = args.memory_dir / "evolution_trace_step_002.txt"
    trace_path.write_text(trace + "\n", encoding="utf-8")

    result = {
        "accepted_to_pool": accepted,
        "added_path": added_path,
        "base_score": base_score,
        "mutated_score": mutated_score,
        "score_delta": mutated_score - base_score,
        "trace_path": str(trace_path),
    }

    (args.memory_dir / "archive_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
