#!/usr/bin/env python3
"""
SWE-bench Dataset Evaluator with Parallel Instance Evaluation

This evaluator:
1. Scans all settings under output/{dataset_name}/
2. For each setting, evaluates all instances in parallel
3. Each worker evaluates one instance at a time
4. Creates environment on-demand for each instance
5. Deletes environment after evaluation (to save storage)
6. Generates results.txt under each setting directory

Usage:
    python -m src.dataset.swe_evaluator --dataset swe_bench_lite
    python -m src.dataset.swe_evaluator --dataset swe_bench_verified --workers 8
    nohup python src/dataset/swe_evaluator.py --dataset swe_bench_verified --workers 4 > eval_swe_bench_verified.log 2>&1 &

Features:
- Parallel instance evaluation with multiple workers
- On-demand environment creation and cleanup
- Per-setting results.txt generation
- Automatic scanning of all settings
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from functools import partial
import multiprocessing
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import local evaluator functions directly to avoid __init__.py loading
import importlib.util
spec = importlib.util.spec_from_file_location(
    "local_swe_evaluator",
    project_root / "src" / "dataset" / "local_swe_evaluator.py"
)
local_swe_evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_swe_evaluator)

evaluate_patch = local_swe_evaluator.evaluate_patch
load_instance_data = local_swe_evaluator.load_instance_data
normalize_instance_id = local_swe_evaluator.normalize_instance_id


def get_repo_from_instance_id(instance_id: str) -> str:
    """Extract repo name from instance_id (e.g., 'django__django-11433' -> 'django')."""
    # Format: repo__project-issue or repo__project_issue
    parts = instance_id.split('__')
    if len(parts) >= 1:
        return parts[0]
    return instance_id


def evaluate_repo_batch(
    batch: List[tuple],
    dataset: str,
    evals_dir_str: str
) -> List[Dict[str, Any]]:
    """
    Evaluate a batch of instances from the same repo sequentially.

    This ensures no parallel access to the same repo directory.

    Args:
        batch: List of (instance_id, patch_content) tuples
        dataset: Dataset name
        evals_dir_str: Directory path as string (for pickle compatibility)

    Returns:
        List of evaluation results
    """
    # Configure logging for this child process
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True  # Force reconfiguration in child process
    )
    child_logger = logging.getLogger(__name__)

    # Convert string back to Path for internal use
    evals_dir = Path(evals_dir_str)

    results = []
    for i, (instance_id, patch_content) in enumerate(batch, 1):
        child_logger.info(f"[Child] Processing {i}/{len(batch)}: {instance_id}")
        try:
            result = evaluate_single_instance(instance_id, patch_content, dataset, evals_dir)
            results.append(result)
            child_logger.info(f"[Child] Completed {instance_id}: {result.get('resolved', 'ERROR')}")
        except Exception as e:
            child_logger.error(f"[Child] Exception for {instance_id}: {e}")
            child_logger.error(traceback.format_exc())
            results.append({
                'instance_id': instance_id,
                'error': str(e),
                'resolved': 'NO'
            })
    return results


def evaluate_single_instance(
    instance_id: str,
    patch_content: str,
    dataset: str,
    evals_dir: Path = None
) -> Dict[str, Any]:
    """
    Evaluate a single instance with environment cleanup.

    This function:
    1. Creates environment on-demand
    2. Evaluates the patch
    3. Deletes environment after evaluation
    4. Saves individual evaluation result to evals/ directory

    Args:
        instance_id: Instance ID to evaluate
        patch_content: Patch content (diff format)
        dataset: Dataset name
        evals_dir: Directory to save individual evaluation results

    Returns:
        Evaluation result dictionary
    """
    try:
        # Evaluate with use_cache=False to delete env after eval
        result = evaluate_patch(
            instance_id=instance_id,
            patch_content=patch_content,
            dataset=dataset,
            use_cache=False  # Delete environment after evaluation
        )

        # Save individual evaluation result if evals_dir provided
        if evals_dir:
            eval_file = evals_dir / f"{instance_id}.txt"
            with open(eval_file, 'w', encoding='utf-8') as f:
                f.write(f"Instance: {instance_id}\n")
                f.write(f"Dataset: {dataset}\n")
                f.write("=" * 80 + "\n\n")

                resolved = result.get('resolved', 'NO')
                f.write(f"Resolution Status: {resolved}\n\n")

                if 'error' in result:
                    f.write(f"Error: {result['error']}\n")
                elif 'fail_to_pass' in result:
                    ftp = result['fail_to_pass']
                    ptp = result['pass_to_pass']
                    f.write(f"FAIL_TO_PASS: {ftp['passed']}/{ftp['total']} passed\n")
                    f.write(f"  - Total: {ftp['total']}\n")
                    f.write(f"  - Passed: {ftp['passed']}\n")
                    f.write(f"  - Failed: {ftp['failed']}\n\n")

                    f.write(f"PASS_TO_PASS: {ptp['passed']}/{ptp['total']} passed\n")
                    f.write(f"  - Total: {ptp['total']}\n")
                    f.write(f"  - Passed: {ptp['passed']}\n")
                    f.write(f"  - Failed: {ptp['failed']}\n\n")

                    if 'test_status_map' in result:
                        f.write("=" * 80 + "\n")
                        f.write("Test Status Details\n")
                        f.write("=" * 80 + "\n\n")
                        for test_name, status in result['test_status_map'].items():
                            f.write(f"{status}: {test_name}\n")

        return result

    except Exception as e:
        logger.error(f"Error evaluating {instance_id}: {e}")
        result = {
            'instance_id': instance_id,
            'error': str(e),
            'resolved': 'NO'
        }

        # Save error result if evals_dir provided
        if evals_dir:
            eval_file = evals_dir / f"{instance_id}.txt"
            with open(eval_file, 'w', encoding='utf-8') as f:
                f.write(f"Instance: {instance_id}\n")
                f.write(f"Dataset: {dataset}\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"ERROR: {str(e)}\n")

        return result


class SWEBenchEvaluator:
    """SWE-bench evaluator with parallel instance processing"""

    def __init__(self, dataset_name: str, max_workers: int = 4, verbose: bool = False):
        """Initialize SWE-bench evaluator."""
        self.dataset_name = dataset_name
        self.max_workers = max_workers
        self.verbose = verbose

        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # Validate dataset
        valid_datasets = ['swe_bench_lite', 'swe_bench_verified', 'swe_bench']
        if dataset_name.lower() not in valid_datasets:
            raise ValueError(
                f"Dataset '{dataset_name}' is not valid. "
                f"Valid options: {valid_datasets}"
            )

    def evaluate_correctness(self, result: str, ground_truth: Any) -> Optional[bool]:
        """
        Evaluate correctness of a single result.

        For SWE-bench, real-time evaluation is not possible as it requires
        running tests in Docker/git environments. This method returns None
        to indicate that evaluation should be done separately using the
        batch evaluation methods (evaluate_setting, evaluate_all_settings).

        Args:
            result: The generated patch content
            ground_truth: The expected patch (not used for SWE-bench)

        Returns:
            None - SWE-bench requires batch evaluation via Docker
        """
        # SWE-bench evaluation requires running tests in Docker environments
        # This cannot be done inline during MAS execution
        # Use evaluate_setting() or evaluate_all_settings() for batch evaluation
        return None

    def find_setting_directories(self) -> List[Path]:
        """Find all setting directories in the output folder."""
        output_base_dir = project_root / "output" / self.dataset_name

        if not output_base_dir.exists():
            logger.error(f"Output directory not found: {output_base_dir}")
            return []

        logger.info(f"Scanning output directory: {output_base_dir}")

        setting_dirs = []
        for setting_dir in output_base_dir.iterdir():
            if not setting_dir.is_dir():
                continue

            # Check if this directory has .txt patch files
            txt_files = list(setting_dir.glob("*.txt"))
            if txt_files:
                setting_dirs.append(setting_dir)
                logger.info(f"  Found setting: {setting_dir.name} ({len(txt_files)} patches)")

        logger.info(f"Found {len(setting_dirs)} settings")
        return setting_dirs

    def load_predictions(self, setting_dir: Path) -> Dict[str, str]:
        """
        Load predictions from .txt patch files in the setting directory.

        Args:
            setting_dir: Path to setting directory containing .txt patch files

        Returns:
            Dictionary mapping instance_id to patch content
        """
        predictions = {}

        try:
            # Find all .txt patch files
            txt_files = list(setting_dir.glob("*.txt"))

            for txt_file in txt_files:
                # Extract instance_id from filename (e.g., astropy__astropy-12907.txt)
                instance_id = txt_file.stem

                # Read patch content
                with open(txt_file, 'r', encoding='utf-8') as f:
                    patch_content = f.read()

                if patch_content.strip():
                    predictions[instance_id] = patch_content

            logger.info(f"  Loaded {len(predictions)} predictions from .txt files")
            return predictions

        except Exception as e:
            logger.error(f"  Failed to load predictions: {e}")
            return {}

    def evaluate_setting(self, setting_dir: Path) -> Dict[str, Any]:
        """
        Evaluate all instances in a setting with parallel processing.

        Args:
            setting_dir: Path to setting directory

        Returns:
            Dictionary with evaluation results
        """
        setting_name = setting_dir.name
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating setting: {setting_name}")
        logger.info(f"{'='*80}")

        # Load predictions from .txt patch files
        predictions = self.load_predictions(setting_dir)

        if not predictions:
            logger.error(f"  No valid predictions found")
            return {
                'setting_name': setting_name,
                'status': 'failed',
                'error': 'No valid predictions'
            }

        # Create evals/ directory for individual results
        evals_dir = setting_dir / "evals"
        evals_dir.mkdir(exist_ok=True)
        logger.info(f"  Individual results will be saved to: {evals_dir}")

        # Group instances by repo to avoid parallel conflicts on same repo
        repo_batches = defaultdict(list)
        for instance_id, patch_content in predictions.items():
            repo = get_repo_from_instance_id(instance_id)
            repo_batches[repo].append((instance_id, patch_content))

        # Adjust workers to match number of repos (no point having idle workers)
        effective_workers = min(self.max_workers, len(repo_batches))

        logger.info(f"  Grouped into {len(repo_batches)} repos: {list(repo_batches.keys())}")
        logger.info(f"  Starting parallel evaluation with {effective_workers} workers (one repo per worker)")
        logger.info(f"  Total instances: {sum(len(b) for b in repo_batches.values())}")

        # Evaluate repo batches in parallel (each batch runs sequentially within)
        results = []
        completed = 0
        total_instances = sum(len(b) for b in repo_batches.values())

        # Use spawn method to avoid fork issues with conda environments
        mp_context = multiprocessing.get_context('spawn')

        with ProcessPoolExecutor(max_workers=effective_workers, mp_context=mp_context) as executor:
            # Submit one batch per repo
            # Note: Convert evals_dir to string for pickle compatibility
            future_to_repo = {
                executor.submit(
                    evaluate_repo_batch,
                    batch,
                    self.dataset_name,
                    str(evals_dir)
                ): repo
                for repo, batch in repo_batches.items()
            }

            logger.info(f"  Submitted {len(future_to_repo)} repo batches to workers")

            # Collect results as repo batches complete
            # Set per-instance timeout (20 min per instance max)
            for future in as_completed(future_to_repo):
                repo = future_to_repo[future]
                batch_size = len(repo_batches[repo])
                # 20 minutes per instance, minimum 30 minutes
                batch_timeout = max(1800, batch_size * 1200)

                try:
                    batch_results = future.result(timeout=batch_timeout)
                    for result in batch_results:
                        completed += 1
                        results.append(result)

                        # Log progress
                        instance_id = result.get('instance_id', 'unknown')
                        resolved = result.get('resolved', 'NO')
                        status_icon = "" if resolved == 'FULL' else "" if resolved == 'PARTIAL' else ""
                        logger.info(f"  [{completed}/{total_instances}] {status_icon} {instance_id}: {resolved}")

                    logger.info(f"  Completed repo batch: {repo} ({len(batch_results)} instances)")

                except TimeoutError:
                    logger.error(f"  Timeout for repo {repo} (>{batch_timeout}s)")
                    # Mark all instances in this repo as timed out
                    for instance_id, _ in repo_batches[repo]:
                        completed += 1
                        results.append({
                            'instance_id': instance_id,
                            'error': f'Batch timeout after {batch_timeout}s',
                            'resolved': 'NO'
                        })

                except Exception as e:
                    logger.error(f"  Exception for repo {repo}: {e}")
                    logger.error(traceback.format_exc())
                    # Mark all instances in this repo as failed
                    for instance_id, _ in repo_batches[repo]:
                        completed += 1
                        results.append({
                            'instance_id': instance_id,
                            'error': str(e),
                            'resolved': 'NO'
                        })

        # Calculate statistics
        total = len(results)
        fully_resolved = sum(1 for r in results if r.get('resolved') == 'FULL')
        partially_resolved = sum(1 for r in results if r.get('resolved') == 'PARTIAL')
        not_resolved = sum(1 for r in results if r.get('resolved') == 'NO')
        errors = sum(1 for r in results if 'error' in r)

        resolution_rate = fully_resolved / total if total > 0 else 0.0

        summary = {
            'setting_name': setting_name,
            'status': 'completed',
            'total_instances': total,
            'fully_resolved': fully_resolved,
            'partially_resolved': partially_resolved,
            'not_resolved': not_resolved,
            'errors': errors,
            'resolution_rate': resolution_rate,
            'resolution_rate_percentage': resolution_rate * 100.0,
            'results': results
        }

        # Save results to setting directory
        self.save_setting_results(setting_dir, summary)

        logger.info(f"\n  Setting Summary:")
        logger.info(f"    Total: {total}")
        logger.info(f"    Fully Resolved: {fully_resolved} ({resolution_rate:.2%})")
        logger.info(f"    Partially Resolved: {partially_resolved}")
        logger.info(f"    Not Resolved: {not_resolved}")
        logger.info(f"    Errors: {errors}")

        return summary

    def save_setting_results(self, setting_dir: Path, summary: Dict[str, Any]):
        """
        Save results.txt for a setting.

        Args:
            setting_dir: Path to setting directory
            summary: Evaluation summary dictionary
        """
        results_file = setting_dir / "results.txt"

        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"SWE-bench Evaluation Results: {summary['setting_name']}\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Setting: {summary['setting_name']}\n")
            f.write(f"Evaluation Date: {datetime.now().isoformat()}\n\n")

            f.write("Summary:\n")
            f.write(f"  Total Instances: {summary['total_instances']}\n")
            f.write(f"  Fully Resolved: {summary['fully_resolved']} ({summary['resolution_rate']:.4f}, {summary['resolution_rate_percentage']:.1f}%)\n")
            f.write(f"  Partially Resolved: {summary['partially_resolved']}\n")
            f.write(f"  Not Resolved: {summary['not_resolved']}\n")
            f.write(f"  Errors: {summary['errors']}\n\n")

            f.write("=" * 80 + "\n")
            f.write("Per-Instance Results\n")
            f.write("=" * 80 + "\n\n")

            # Sort by resolution status
            results = summary['results']
            sorted_results = sorted(
                results,
                key=lambda x: (
                    0 if x.get('resolved') == 'FULL' else
                    1 if x.get('resolved') == 'PARTIAL' else 2,
                    x.get('instance_id', '')
                )
            )

            for result in sorted_results:
                instance_id = result.get('instance_id', 'unknown')
                resolved = result.get('resolved', 'NO')
                status_icon = "" if resolved == 'FULL' else "" if resolved == 'PARTIAL' else ""

                f.write(f"{status_icon} {instance_id}: {resolved}\n")

                if 'error' in result:
                    f.write(f"  Error: {result['error']}\n")
                elif 'fail_to_pass' in result:
                    ftp = result['fail_to_pass']
                    ptp = result['pass_to_pass']
                    f.write(f"  FAIL_TO_PASS: {ftp['passed']}/{ftp['total']} passed\n")
                    f.write(f"  PASS_TO_PASS: {ptp['passed']}/{ptp['total']} passed\n")

                f.write("\n")

        logger.info(f"  Results saved to: {results_file}")

        # Also save detailed JSON
        results_json = setting_dir / "evaluation_results.json"
        with open(results_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"  Detailed results saved to: {results_json}")

    def evaluate_all_settings(self) -> Dict[str, Any]:
        """
        Evaluate all settings in the dataset.

        Returns:
            Dictionary with overall evaluation results
        """
        logger.info(f"Starting SWE-bench evaluation for: {self.dataset_name}")
        logger.info(f"Workers: {self.max_workers}")
        logger.info("=" * 80)

        # Find all setting directories
        setting_dirs = self.find_setting_directories()

        if not setting_dirs:
            logger.error("No setting directories found")
            return {}

        # Evaluate each setting
        all_results = []

        for idx, setting_dir in enumerate(setting_dirs, 1):
            logger.info(f"\n{'#'*80}")
            logger.info(f"# Setting {idx}/{len(setting_dirs)}")
            logger.info(f"{'#'*80}")

            try:
                setting_result = self.evaluate_setting(setting_dir)
                all_results.append(setting_result)
            except Exception as e:
                logger.error(f"Failed to evaluate {setting_dir.name}: {e}")
                all_results.append({
                    'setting_name': setting_dir.name,
                    'status': 'failed',
                    'error': str(e)
                })

        # Compile overall summary
        summary = {
            'dataset_name': self.dataset_name,
            'evaluation_timestamp': datetime.now().isoformat(),
            'total_settings': len(all_results),
            'max_workers': self.max_workers,
            'settings': all_results
        }

        # Calculate overall statistics
        completed_settings = [r for r in all_results if r.get('status') == 'completed']
        if completed_settings:
            total_instances = sum(r.get('total_instances', 0) for r in completed_settings)
            total_resolved = sum(r.get('fully_resolved', 0) for r in completed_settings)
            overall_rate = total_resolved / total_instances if total_instances > 0 else 0.0

            summary['overall_statistics'] = {
                'total_settings_evaluated': len(completed_settings),
                'total_instances': total_instances,
                'total_fully_resolved': total_resolved,
                'overall_resolution_rate': overall_rate,
                'overall_resolution_rate_percentage': overall_rate * 100.0
            }
        else:
            summary['overall_statistics'] = {
                'total_settings_evaluated': 0,
                'total_instances': 0,
                'total_fully_resolved': 0,
                'overall_resolution_rate': 0.0,
                'overall_resolution_rate_percentage': 0.0
            }

        return summary

    def save_dataset_summary(self, summary: Dict[str, Any]):
        """
        Save dataset-level summary.

        Args:
            summary: Overall evaluation summary
        """
        output_base_dir = project_root / "output" / self.dataset_name

        # Save JSON
        summary_json = output_base_dir / "evaluation_summary.json"
        with open(summary_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"\nDataset summary saved to: {summary_json}")

        # Save text summary
        summary_txt = output_base_dir / "evaluation_summary.txt"
        with open(summary_txt, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"SWE-bench Dataset Evaluation Summary: {self.dataset_name}\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Evaluation Date: {summary['evaluation_timestamp']}\n")
            f.write(f"Total Settings: {summary['total_settings']}\n")
            f.write(f"Workers Used: {summary['max_workers']}\n\n")

            overall = summary.get('overall_statistics', {})
            f.write("Overall Statistics:\n")
            f.write(f"  Settings Evaluated: {overall.get('total_settings_evaluated', 0)}\n")
            f.write(f"  Total Instances: {overall.get('total_instances', 0)}\n")
            f.write(f"  Fully Resolved: {overall.get('total_fully_resolved', 0)}\n")
            rate = overall.get('overall_resolution_rate', 0.0)
            f.write(f"  Overall Resolution Rate: {rate:.4f} ({rate * 100:.2f}%)\n\n")

            f.write("=" * 80 + "\n")
            f.write("Results by Setting\n")
            f.write("=" * 80 + "\n\n")

            # Sort by resolution rate
            settings = summary.get('settings', [])
            sorted_settings = sorted(
                settings,
                key=lambda x: x.get('resolution_rate', 0.0),
                reverse=True
            )

            for setting in sorted_settings:
                name = setting.get('setting_name', 'unknown')
                status = setting.get('status', 'unknown')

                f.write(f"Setting: {name}\n")
                f.write(f"  Status: {status}\n")

                if status == 'completed':
                    total = setting.get('total_instances', 0)
                    resolved = setting.get('fully_resolved', 0)
                    rate = setting.get('resolution_rate', 0.0)
                    f.write(f"  Instances: {total}\n")
                    f.write(f"  Resolved: {resolved} ({rate:.4f}, {rate * 100:.2f}%)\n")
                elif 'error' in setting:
                    f.write(f"  Error: {setting['error']}\n")

                f.write("\n")

        logger.info(f"Dataset summary text saved to: {summary_txt}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="SWE-bench Dataset Evaluator with Parallel Instance Processing"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=['swe_bench_lite', 'swe_bench_verified', 'swe_bench'],
        help="Dataset name"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for instance evaluation (default: 4)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Initialize evaluator
    try:
        evaluator = SWEBenchEvaluator(
            dataset_name=args.dataset,
            max_workers=args.workers,
            verbose=args.verbose
        )
    except ValueError as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)

    try:
        # Run evaluation
        summary = evaluator.evaluate_all_settings()

        if not summary:
            logger.error("Evaluation failed - no results generated")
            sys.exit(1)

        # Save dataset summary
        evaluator.save_dataset_summary(summary)

        logger.info("\n" + "=" * 80)
        logger.info("Evaluation completed successfully!")
        logger.info("=" * 80)

        # Print final summary
        overall = summary.get('overall_statistics', {})
        logger.info(f"\nFinal Summary:")
        logger.info(f"  Dataset: {args.dataset}")
        logger.info(f"  Settings Evaluated: {overall.get('total_settings_evaluated', 0)}")
        logger.info(f"  Total Instances: {overall.get('total_instances', 0)}")
        logger.info(f"  Fully Resolved: {overall.get('total_fully_resolved', 0)}")
        rate = overall.get('overall_resolution_rate', 0.0)
        logger.info(f"  Overall Resolution Rate: {rate:.4f} ({rate * 100:.2f}%)")

    except KeyboardInterrupt:
        logger.warning("\nEvaluation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
