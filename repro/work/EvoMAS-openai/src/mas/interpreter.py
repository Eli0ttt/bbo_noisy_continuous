#!/usr/bin/env python3
"""
EVOMAS MAS Interpreter

This script runs MAS experiments and saves raw model outputs to the output/ directory.
Evaluation is done separately using src.utils.evaluator for decoupled pipeline.

Pipeline:
    1. Run models -> Save raw outputs to output/{dataset}/{config}/{task_id}.txt
    2. Run src.utils.evaluator -> Evaluate saved outputs

Usage:
    # Run single experiment and save outputs
    python -m src.mas.interpreter --config mas_pools/bbeh/majority_vote.yaml --dataset bbeh_mini --save-outputs

    # Run with specific number of tasks
    python -m src.mas.interpreter --config mas_pools/bbeh/majority_vote.yaml --dataset bbeh_word_sorting --num-tasks 10 --save-outputs

    # Run batch experiments with default configs
    python -m src.mas.interpreter --batch --num-tasks 5 --save-outputs

    # Run specific task IDs
    python -m src.mas.interpreter --config mas_pools/bbeh/single_codeagent.yaml --dataset bbeh_object_counting --task-ids 0,1,2 --save-outputs

    # Parallel processing with multiple workers
    python -m src.mas.interpreter --config mas_pools/bbeh/single_codeagent.yaml --dataset bbeh_mini --num-tasks 10 --workers 4 --save-outputs

    # Custom output directory
    python -m src.mas.interpreter --config mas_pools/bbeh/majority_vote.yaml --dataset bbeh_mini --num-tasks 5 --save-outputs --output-dir my_output

    # Then evaluate saved outputs
    python -m src.dataset.bbeh_evaluator --dataset bbeh_mini
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict

from src.utils import MasRunner, MasRunResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Available BBEH datasets
BBEH_DATASETS = [
    "bbeh_word_sorting",
    "bbeh_object_counting",
    "bbeh_boardgame_qa",
    "bbeh_boolean_expressions",
    "bbeh_buggy_tables",
    "bbeh_causal_understanding",
    "bbeh_disambiguation_qa",
    "bbeh_dyck_languages",
    "bbeh_geometric_shapes",
    "bbeh_hyperbaton",
    "bbeh_linguini",
    "bbeh_movie_recommendation",
    "bbeh_multistep_arithmetic",
    "bbeh_object_properties",
    "bbeh_sarc_triples",
    "bbeh_shuffled_objects",
    "bbeh_spatial_reasoning",
    "bbeh_sportqa",
    "bbeh_temporal_sequence",
]

# Available configurations
AVAILABLE_CONFIGS = [
    "mas_pools/bbeh/single_codeagent.yaml",
    "mas_pools/bbeh/majority_vote.yaml",
]


class ExperimentRunner:
    """Main experiment runner for EVOMAS."""

    def __init__(
        self,
        verbose: bool = True,
        save_individual_outputs: bool = False,
        output_dir: str = "output",
        use_cache: bool = True,
        skip_evaluation: bool = True,
        llm_as_judge: Optional[str] = None,
        task_timeout: float = 600.0
    ):
        """
        Initialize experiment runner.

        Args:
            verbose: Whether to show detailed logs
            save_individual_outputs: Save each task output to separate file
            output_dir: Directory for task outputs
            use_cache: Enable caching of task outputs (default: True)
            skip_evaluation: Skip evaluation step (useful for patch generation) (default: True)
            llm_as_judge: Model ID for LLM-as-judge evaluation (None = use dataset evaluator)
            task_timeout: Maximum time per task in seconds (default: 600)
        """
        self.verbose = verbose
        self.save_individual_outputs = save_individual_outputs
        self.llm_as_judge = llm_as_judge
        self.output_dir = output_dir
        self.use_cache = use_cache
        self.skip_evaluation = skip_evaluation
        self.task_timeout = task_timeout

    def run_single_experiment(
        self,
        config_path: str,
        dataset_name: str,
        num_tasks: Optional[int] = None,
        task_ids: Optional[List[Union[int, str]]] = None,
        dataset_split: str = "test",
        experiment_name: Optional[str] = None,
        workers: int = 1
    ) -> Dict[str, Any]:
        """
        Run a single experiment with given configuration and dataset.

        Args:
            config_path: Path to MAS configuration file
            dataset_name: Name of the dataset
            num_tasks: Number of tasks to run
            task_ids: Specific task IDs to run
            dataset_split: Dataset split to use
            experiment_name: Custom name for experiment
            workers: Number of worker processes for parallel execution (default: 1)

        Returns:
            Dictionary with experiment results and metadata
        """
        # Generate experiment name
        if experiment_name is None:
            config_name = Path(config_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{config_name}_{dataset_name}_{timestamp}"

        logger.info("=" * 80)
        logger.info(f"EXPERIMENT: {experiment_name}")
        logger.info("=" * 80)
        logger.info(f"Configuration: {config_path}")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Split: {dataset_split}")
        if task_ids:
            logger.info(f"Task IDs: {task_ids}")
        else:
            logger.info(f"Number of tasks: {num_tasks or 'all'}")

        # Create MAS runner
        try:
            runner = MasRunner(
                config_path=config_path,
                dataset_name=dataset_name,
                dataset_split=dataset_split,
                verbose=self.verbose,
                save_individual_outputs=self.save_individual_outputs,
                output_dir=self.output_dir,
                skip_evaluation=self.skip_evaluation,  # Use configured evaluation setting
                use_cache=self.use_cache,
                llm_as_judge=self.llm_as_judge,
                task_timeout=self.task_timeout
            )
        except Exception as e:
            logger.error(f"Failed to initialize MasRunner: {e}")
            return {
                "experiment_name": experiment_name,
                "status": "failed",
                "error": str(e),
                "config_path": config_path,
                "dataset_name": dataset_name,
            }

        # Run experiment
        start_time = datetime.now()
        try:
            # Use parallel processing if workers > 1
            if workers > 1:
                logger.info(f"Using parallel processing with {workers} workers")
                if task_ids is not None:
                    results, token_costs, time_costs = runner.run_parallel(task_ids=task_ids, workers=workers)
                else:
                    results, token_costs, time_costs = runner.run_parallel(num_tasks=num_tasks, workers=workers)
            else:
                if task_ids is not None:
                    results, token_costs, time_costs = runner.run(task_ids=task_ids)
                else:
                    results, token_costs, time_costs = runner.run(num_tasks=num_tasks)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Get statistics
            stats = runner.get_summary_stats(results)

            # Add token and time cost statistics (available but not used in summary yet)
            stats['token_costs'] = token_costs
            stats['time_costs'] = time_costs

            logger.info("\n" + "=" * 80)
            logger.info("EXPERIMENT COMPLETED")
            logger.info("=" * 80)
            logger.info(f"Duration: {duration:.2f}s")
            if "accuracy" in stats:
                acc = stats['accuracy']
                logger.info(f"Accuracy: {acc:.2%}" if acc <= 1.0 else f"Accuracy: {acc:.1f}/100")

            # Output location
            config_name = _get_config_name_with_model(config_path)
            output_location = Path(self.output_dir) / dataset_name / config_name
            logger.info(f"Outputs saved to: {output_location}/")

            return {
                "experiment_name": experiment_name,
                "status": "completed",
                "config_path": config_path,
                "dataset_name": dataset_name,
                "duration_seconds": duration,
                "statistics": stats,
            }

        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            return {
                "experiment_name": experiment_name,
                "status": "failed",
                "error": str(e),
                "config_path": config_path,
                "dataset_name": dataset_name,
                "duration_seconds": duration,
            }

    def run_batch_experiments(
        self,
        configs: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
        num_tasks: int = 5,
        dataset_split: str = "test"
    ) -> List[Dict[str, Any]]:
        """
        Run batch experiments with multiple configurations and datasets.

        Args:
            configs: List of config paths (default: all available)
            datasets: List of dataset names (default: subset of BBEH)
            num_tasks: Number of tasks per experiment
            dataset_split: Dataset split to use

        Returns:
            List of experiment results
        """
        # Use defaults if not provided
        if configs is None:
            configs = AVAILABLE_CONFIGS
        if datasets is None:
            # Use a subset for quick testing
            datasets = [
                "bbeh_word_sorting",
                "bbeh_object_counting",
                "bbeh_multistep_arithmetic",
            ]

        logger.info("=" * 80)
        logger.info("BATCH EXPERIMENTS")
        logger.info("=" * 80)
        logger.info(f"Configurations: {len(configs)}")
        logger.info(f"Datasets: {len(datasets)}")
        logger.info(f"Tasks per experiment: {num_tasks}")
        logger.info(f"Total experiments: {len(configs) * len(datasets)}")

        all_results = []

        for config in configs:
            for dataset in datasets:
                experiment_name = f"batch_{Path(config).stem}_{dataset}"
                result = self.run_single_experiment(
                    config_path=config,
                    dataset_name=dataset,
                    num_tasks=num_tasks,
                    dataset_split=dataset_split,
                    experiment_name=experiment_name
                )
                all_results.append(result)

        # Print batch summary
        logger.info("\n" + "=" * 80)
        logger.info("BATCH EXPERIMENTS COMPLETED")
        logger.info("=" * 80)

        successful = sum(1 for r in all_results if r["status"] == "completed")
        failed = len(all_results) - successful

        logger.info(f"Total experiments: {len(all_results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")

        return all_results


def _get_config_name_with_model(config_path: str) -> str:
    """
    Get configuration name with model_id appended for filesystem paths.

    Args:
        config_path: Path to the MAS configuration file

    Returns:
        Config name with sanitized model_id (e.g., "single_minisweagent_gpt-4o")
    """
    config_name = Path(config_path).stem

    try:
        # Load MAS spec to get model_id
        from src.mas import load_mas_from_file
        mas_spec = load_mas_from_file(config_path)

        # Get model_id from the first agent
        if mas_spec.agents:
            first_agent = next(iter(mas_spec.agents.values()))
            model_id = first_agent.model_id

            # Sanitize model_id for filesystem use
            # Remove provider prefix (e.g., "openai:", "bedrock:", "anthropic:")
            if ":" in model_id:
                model_id = model_id.split(":", 1)[1]

            # Replace special characters with hyphens, keep only alphanumeric and hyphens
            safe_model_id = model_id.replace(".", "-").replace("/", "-").replace("_", "-")
            safe_model_id = "".join(c for c in safe_model_id if c.isalnum() or c == "-")

            # Append to config name
            config_name = f"{config_name}_{safe_model_id}"
    except Exception:
        # If we can't load the spec, just return the config name without model_id
        pass

    return config_name


def interpret_mas(
    config_path: str,
    dataset_name: str,
    num_tasks: Optional[int] = None,
    task_ids: Optional[List[Union[int, str]]] = None,
    dataset_split: str = "test",
    save_outputs: bool = True,
    output_dir: str = "output",
    verbose: bool = True,
    use_cache: bool = True,
    skip_evaluation: bool = True,
    llm_as_judge: Optional[str] = None,
    task_timeout: float = 600.0
) -> Dict[str, Any]:
    """
    Interpret and execute a MAS configuration programmatically.

    This function can be called by other scripts (e.g., meta model) to run
    a MAS configuration and get the execution results and logs.

    Args:
        config_path: Path to MAS configuration file
        dataset_name: Name of the dataset to run on
        num_tasks: Number of tasks to run (None = all)
        task_ids: Specific task IDs to run (overrides num_tasks)
        dataset_split: Dataset split to use (default: "test")
        save_outputs: Whether to save individual task outputs
        output_dir: Directory for outputs
        verbose: Whether to show detailed logs
        use_cache: Skip tasks that already have output files (default: True)
        skip_evaluation: Skip evaluation step, useful for patch generation (default: True)
        llm_as_judge: Model ID for LLM-as-judge evaluation (None = use dataset evaluator)
        task_timeout: Maximum time per task in seconds (default: 600, auto-increased for SWE-bench)

    Returns:
        Dictionary with execution results:
        {
            "status": "completed" or "failed",
            "config_path": str,
            "dataset_name": str,
            "duration_seconds": float,
            "statistics": {...},  # if completed
            "error": str,  # if failed
            "output_location": str  # where outputs were saved
        }

    Example:
        >>> result = interpret_mas(
        ...     config_path="mas_pools/bbeh/majority_vote.yaml",
        ...     dataset_name="bbeh_mini",
        ...     num_tasks=5,
        ...     save_outputs=True
        ... )
        >>> print(f"Status: {result['status']}")
        >>> print(f"Accuracy: {result['statistics']['accuracy']:.2%}")
    """
    # When LLM-as-judge is provided, automatically enable evaluation
    if llm_as_judge and skip_evaluation:
        skip_evaluation = False

    # Create experiment runner
    runner = ExperimentRunner(
        verbose=verbose,
        save_individual_outputs=save_outputs,
        output_dir=output_dir,
        use_cache=use_cache,
        skip_evaluation=skip_evaluation,
        llm_as_judge=llm_as_judge,
        task_timeout=task_timeout
    )

    # Run the experiment
    result = runner.run_single_experiment(
        config_path=config_path,
        dataset_name=dataset_name,
        num_tasks=num_tasks,
        task_ids=task_ids,
        dataset_split=dataset_split
    )

    # Add output location to result
    config_name = _get_config_name_with_model(config_path)
    result["output_location"] = str(Path(output_dir) / dataset_name / config_name)

    return result


def run_mas_cli(args: Optional[List[str]] = None) -> int:
    """
    CLI entry point for MAS interpreter.

    Args:
        args: Command-line arguments (defaults to sys.argv if None)

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = argparse.ArgumentParser(
        description="Run EVOMAS MAS interpreter with different configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single experiment
  python -m src.mas.interpreter --config mas_pools/bbeh/majority_vote.yaml --dataset bbeh_word_sorting --num-tasks 10 --save-outputs

  # Save outputs to custom directory
  python -m src.mas.interpreter --config mas_pools/bbeh/majority_vote.yaml --dataset bbeh_mini
  python -m src.mas.interpreter --config mas_pools/workbench/analytics/single_codeagent.yaml --dataset workbench_analytics --save-outputs

  # Batch experiments with default configs
  python -m src.mas.interpreter --batch --num-tasks 5 --save-outputs

  # Run on specific tasks
  python -m src.mas.interpreter --config mas_pools/bbeh/single_codeagent.yaml --dataset bbeh_object_counting --task-ids 0,1,2,3,4 --save-outputs
        """
    )

    # Experiment mode
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch experiments with multiple configs and datasets"
    )

    # Single experiment arguments
    parser.add_argument(
        "--config",
        type=str,
        help="Path to MAS configuration file"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name (e.g., bbeh_word_sorting)"
    )

    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Number of tasks to run (default: all)"
    )

    parser.add_argument(
        "--task-ids",
        type=str,
        help="Comma-separated list of task IDs to run (e.g., 0,1,2,3)"
    )

    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to use (default: test)"
    )

    # Output arguments
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help="Save individual task outputs to separate files for easy inspection"
    )

    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for task outputs (default: output)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )

    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available BBEH datasets and exit"
    )

    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available configurations and exit"
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable output caching (re-run all tasks even if output files exist)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel processing (default: 1, use 4-8 for speedup)"
    )

    parsed_args = parser.parse_args(args)

    # List datasets
    if parsed_args.list_datasets:
        print("Available BBEH Datasets:")
        for dataset in BBEH_DATASETS:
            print(f"  - {dataset}")
        return 0

    # List configs
    if parsed_args.list_configs:
        print("Available Configurations:")
        for config in AVAILABLE_CONFIGS:
            exists = "" if Path(config).exists() else ""
            print(f"  {exists} {config}")
        return 0

    # Create experiment runner
    runner = ExperimentRunner(
        verbose=not parsed_args.quiet,
        save_individual_outputs=parsed_args.save_outputs,
        output_dir=parsed_args.output_dir,
        use_cache=not parsed_args.no_cache
    )

    try:
        if parsed_args.batch:
            # Run batch experiments (uses default configs and datasets)
            results = runner.run_batch_experiments(
                num_tasks=parsed_args.num_tasks or 5,
                dataset_split=parsed_args.split
            )

            # Check if any experiments failed
            failed = sum(1 for r in results if r["status"] != "completed")
            if failed > 0:
                logger.warning(f"{failed} experiments failed")
                return 1

        else:
            # Run single experiment
            if not parsed_args.config or not parsed_args.dataset:
                parser.error("--config and --dataset are required for single experiment")

            # Validate config exists
            if not Path(parsed_args.config).exists():
                logger.error(f"Configuration file not found: {parsed_args.config}")
                return 1

            # Parse task IDs if provided
            # Support both integer and string task IDs (SWE-bench uses string IDs like "astropy__astropy-12907")
            task_ids = None
            if parsed_args.task_ids:
                raw_ids = [tid.strip() for tid in parsed_args.task_ids.split(",")]
                # Try to convert to integers if all look like integers, otherwise keep as strings
                try:
                    task_ids = [int(tid) for tid in raw_ids]
                except ValueError:
                    # Keep as strings (for SWE-bench style IDs)
                    task_ids = raw_ids

            result = runner.run_single_experiment(
                config_path=parsed_args.config,
                dataset_name=parsed_args.dataset,
                num_tasks=parsed_args.num_tasks,
                task_ids=task_ids,
                dataset_split=parsed_args.split,
                workers=parsed_args.workers
            )

            if result["status"] != "completed":
                return 1

        return 0

    except KeyboardInterrupt:
        logger.warning("\nExperiment interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_mas_cli())
