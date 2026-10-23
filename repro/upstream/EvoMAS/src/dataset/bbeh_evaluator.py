#!/usr/bin/env python3
"""
BBEH Dataset Evaluator

This module provides dataset-wide evaluation capabilities for BBEH datasets:
- BBEH (built-in correctness evaluation with fuzzy matching)
- Other datasets (simple string matching)

Usage:
    # Evaluate a single dataset
    python bbeh_evaluator.py --dataset bbeh_word_sorting
    python -m src.dataset.bbeh_evaluator --dataset bbeh_mini

    # Batch evaluate all BBEH datasets
    python bbeh_evaluator.py

Single Dataset Mode:
- Evaluates all settings under output/{dataset_name}/
- Saves dataset_analysis.json and dataset_analysis.txt in the dataset folder

Batch Mode (no --dataset argument):
- Auto-scans and evaluates all bbeh_* directories in output/
- For each dataset:
  1. Saves metrics.txt in each setting folder with per-task results
  2. Saves results.txt in the dataset folder summarizing all settings
- Runs clear_output.py once at the beginning

Note: SWE-bench datasets are not supported in this version.
"""

import sys
import json
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Evaluator supports BBEH datasets only

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class BBEHEvaluator:
    """Built-in BBEH evaluation functions"""

    @staticmethod
    def strip_latex(response: str) -> str:
        """Strip LaTeX formatting from responses."""
        if response.startswith("$") and response.endswith("$"):
            response = response[1:-1]
        if "boxed{" in response and response.endswith("}"):
            response = response[0:-1].split("boxed{")[1]
        if "text{" in response and response.endswith("}"):
            response = response[0:-1].split("text{")[1]
        if "texttt{" in response and response.endswith("}"):
            response = response[0:-1].split("texttt{")[1]
        return response

    @staticmethod
    def extract_answer(sample: str) -> str:
        """Extract the final answer from the sample."""
        answer_prefixes = [
            "The answer is:",
            "The final answer is ",
            "The final answer is: ",
            "The answer is "
        ]
        answer = sample
        for answer_prefix in answer_prefixes:
            if answer_prefix in answer:
                answer = answer.split(answer_prefix)[-1].strip()
        if answer.endswith("."):
            answer = answer[:-1]
        return BBEHEvaluator.strip_latex(answer)

    @staticmethod
    def fuzzy_match(prediction: str, reference: str) -> bool:
        """Fuzzy match function for BBEH."""
        if prediction == reference:
            return True

        # (a) vs a
        if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
            return prediction[1] == reference
        if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
            return reference[1] == prediction

        # Numbers
        try:
            if float(prediction) == float(reference):
                return True
        except ValueError:
            pass

        # Quote issues
        if prediction.replace("'", "") == reference.replace("'", ""):
            return True

        # Bracket issues
        if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
            return True

        # Question mark issues
        if prediction.endswith("?") and prediction[:-1] == reference:
            return True

        return False

    @staticmethod
    def preprocess_sample(sample: str) -> str:
        """Preprocess sample for evaluation."""
        prediction = BBEHEvaluator.extract_answer(sample.strip()).lower()
        prediction = prediction.replace(", ", ",").replace("**", "")
        prediction = prediction.split("\n")[0]
        prediction = prediction[0:-1] if prediction.endswith(".") else prediction
        return prediction

    @staticmethod
    def preprocess_reference(reference: str) -> str:
        """Preprocess reference for evaluation."""
        reference = reference.strip().lower()
        reference = reference.replace(", ", ",")
        return reference

    @staticmethod
    def evaluate_correctness(sample: str, reference: str) -> bool:
        """Evaluate correctness for BBEH datasets."""
        prediction = BBEHEvaluator.preprocess_sample(sample)
        reference = BBEHEvaluator.preprocess_reference(reference)
        return BBEHEvaluator.fuzzy_match(prediction, reference)


class DatasetEvaluator:
    """Dataset-wide evaluator for all supported datasets"""

    def __init__(self, dataset_name: str, verbose: bool = False):
        """Initialize dataset evaluator."""
        self.dataset_name = dataset_name
        self.verbose = verbose

        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # Initialize BBEH evaluator
        self.bbeh_evaluator = BBEHEvaluator()

    def run_clear_output(self):
        """Run clear_output.py to clean invalid outputs."""
        logger.info("Running clear_output.py to clean invalid outputs...")

        clear_output_path = project_root / "src" / "utils" / "clear_output.py"
        if not clear_output_path.exists():
            logger.warning("Warning: clear_output.py not found, skipping cleanup")
            return

        try:
            result = subprocess.run(
                [sys.executable, str(clear_output_path)],
                cwd=str(project_root),
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                logger.info("Clear output completed successfully")
            else:
                logger.warning(f"Clear output completed with warnings: {result.stderr}")
        except Exception as e:
            logger.error(f"Error running clear_output.py: {e}")

    def load_dataset_ground_truth(self) -> Dict[str, str]:
        """Load dataset ground truth mapping for BBEH datasets."""
        # Skip SWE-bench datasets
        if 'swe' in self.dataset_name.lower():
            raise ValueError(f"SWE-bench datasets are not supported: {self.dataset_name}")

        # Handle BBEH and other datasets
        possible_paths = [
            project_root / f"dataset/bbeh/benchmark_tasks/{self.dataset_name}/test.json",
            project_root / f"dataset/{self.dataset_name}/test.json",
            project_root / f"dataset/{self.dataset_name}/benchmark_tasks/{self.dataset_name}/test.json"
        ]

        dataset_path = None
        for path in possible_paths:
            if path.exists():
                dataset_path = path
                break

        if dataset_path is None:
            raise FileNotFoundError(f"Dataset not found: {self.dataset_name}")

        logger.info(f"Loading dataset from: {dataset_path}")

        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        task_gt_map = {}
        for task in dataset:
            task_gt_map[str(task['id'])] = task['gt']

        logger.info(f"Loaded {len(task_gt_map)} tasks from dataset")
        return task_gt_map

    def parse_setting_name(self, directory_name: str) -> Dict[str, str]:
        """Parse directory name to extract mode and model information."""
        parts = directory_name.split('_')
        setting = {}

        if directory_name.startswith('client_'):
            setting['mode'] = 'client'
            model_parts = parts[1:]
        elif directory_name.startswith('codeagent_'):
            setting['mode'] = 'codeagent'
            model_parts = parts[1:]
        elif directory_name.startswith('team_codeagent_'):
            setting['mode'] = 'team_codeagent'
            model_parts = parts[2:]
        elif directory_name.startswith('sweagent_'):
            setting['mode'] = 'sweagent'
            model_parts = parts[1:]
        else:
            setting['mode'] = 'unknown'
            model_parts = parts

        # Extract model name and config
        model_parts_clean = []
        config_parts = []

        for part in model_parts:
            if part.startswith('c') and part[1:].isdigit():
                config_parts.append(part)
            elif part.startswith('t') and part[1:].isdigit():
                config_parts.append(part)
            else:
                model_parts_clean.append(part)

        setting['model'] = '_'.join(model_parts_clean) if model_parts_clean else 'unknown'

        if config_parts:
            setting['config'] = '_'.join(config_parts)

        return setting

    def evaluate_setting_directory(self, output_dir: Path, task_gt_map: Dict[str, str], save_metrics: bool = False) -> Dict[str, Any]:
        """Evaluate a single setting directory."""
        logger.info(f"  Evaluating: {output_dir.name}")

        setting_info = self.parse_setting_name(output_dir.name)
        output_files = list(output_dir.glob("*.txt"))

        total_outputs = len(output_files)
        correct_count = 0
        evaluated_count = 0
        evaluation_details = []

        for output_file in output_files:
            try:
                task_id = output_file.stem

                # Skip if task not in ground truth
                if task_id not in task_gt_map:
                    continue

                # Read output
                with open(output_file, 'r', encoding='utf-8') as f:
                    generated_output = f.read().strip()

                ground_truth = task_gt_map[task_id]

                # Evaluate based on dataset type
                if 'bbeh' in self.dataset_name.lower():
                    is_correct = self.bbeh_evaluator.evaluate_correctness(generated_output, ground_truth)
                else:
                    # Simple string match for other datasets (not used for SWE-bench)
                    is_correct = generated_output.strip().lower() == ground_truth.strip().lower()

                if is_correct:
                    correct_count += 1

                evaluated_count += 1

                evaluation_details.append({
                    'task_id': task_id,
                    'generated': generated_output,
                    'expected': ground_truth,
                    'correct': is_correct
                })

            except Exception as e:
                logger.debug(f"    Skipping {output_file.name}: {e}")
                continue

        accuracy = correct_count / evaluated_count if evaluated_count > 0 else 0.0

        result = {
            'setting_name': output_dir.name,
            'setting_info': setting_info,
            'total_outputs': total_outputs,
            'evaluated_outputs': evaluated_count,
            'correct_outputs': correct_count,
            'incorrect_outputs': evaluated_count - correct_count,
            'accuracy': accuracy,
            'accuracy_percentage': accuracy * 100.0,
            'evaluation_details': evaluation_details
        }

        logger.info(f"    Results: {correct_count}/{evaluated_count} correct ({accuracy:.4f}, {accuracy * 100:.2f}%)")

        # Save metrics.txt if requested
        if save_metrics:
            self.save_setting_metrics(output_dir, result)

        return result

    def save_setting_metrics(self, output_dir: Path, result: Dict[str, Any]):
        """Save metrics.txt for a single setting."""
        metrics_file = output_dir / "metrics.txt"

        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write(f"BBEH Evaluation Metrics: {result['setting_name']}\n")
            f.write("="*70 + "\n\n")

            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Setting: {result['setting_name']}\n")
            info = result['setting_info']
            f.write(f"Mode: {info.get('mode', 'unknown')}\n")
            f.write(f"Model: {info.get('model', 'unknown')}\n")
            if 'config' in info:
                f.write(f"Config: {info['config']}\n")
            f.write(f"\nTotal outputs: {result['total_outputs']}\n")
            f.write(f"Evaluated: {result['evaluated_outputs']}\n\n")

            f.write("Summary:\n")
            f.write(f"  Correct: {result['correct_outputs']}/{result['evaluated_outputs']} ({result['accuracy_percentage']:.1f}%)\n")
            f.write(f"  Incorrect: {result['incorrect_outputs']}\n")
            f.write(f"  Accuracy: {result['accuracy']:.4f}\n\n")

            f.write("="*70 + "\n")
            f.write("Per-Task Results\n")
            f.write("="*70 + "\n\n")

            for detail in result['evaluation_details']:
                status = "" if detail['correct'] else ""
                f.write(f"Task {detail['task_id']}: {status}\n")
                f.write(f"  Expected: {detail['expected']}\n")
                f.write(f"  Generated: {detail['generated'][:100]}{'...' if len(detail['generated']) > 100 else ''}\n")
                f.write(f"  Correct: {detail['correct']}\n\n")

        logger.debug(f"    Metrics saved to: {metrics_file}")


    def analyze_dataset_performance(self, save_metrics: bool = False, save_results: bool = True) -> Dict[str, Any]:
        """Dataset-wide performance analysis."""
        logger.info(f"Starting dataset-wide performance evaluation for: {self.dataset_name}")
        logger.info("=" * 80)

        # Step 1: Run clear_output.py (skip in batch mode to avoid redundancy)
        if not save_metrics:
            self.run_clear_output()

        # Step 2: Load dataset ground truth
        try:
            task_gt_map = self.load_dataset_ground_truth()
        except FileNotFoundError as e:
            logger.error(f"Error: {e}")
            return {}

        # Step 3: Find output directory
        output_base_dir = project_root / f"output/{self.dataset_name}"
        if not output_base_dir.exists():
            logger.error(f"Error: Output directory not found: {output_base_dir}")
            return {}

        logger.info(f"Scanning output directory: {output_base_dir}")

        # Step 4: Process each setting directory
        all_results = []
        setting_dirs = [d for d in output_base_dir.iterdir() if d.is_dir()]

        if not setting_dirs:
            logger.error("No setting directories found")
            return {}

        logger.info(f"Found {len(setting_dirs)} setting directories")

        # Skip SWE-bench datasets
        if 'swe' in self.dataset_name.lower():
            logger.error("SWE-bench evaluation is not supported in this version")
            logger.error("Please use BBEH datasets only")
            return {}

        for setting_dir in sorted(setting_dirs):
            # Use heuristic evaluation for BBEH and other datasets
            result = self.evaluate_setting_directory(setting_dir, task_gt_map, save_metrics=save_metrics)
            all_results.append(result)

        # Step 5: Compile summary statistics
        summary = {
            'dataset_name': self.dataset_name,
            'evaluation_timestamp': datetime.now().isoformat(),
            'total_settings': len(all_results),
            'dataset_size': len(task_gt_map),
            'results_by_setting': all_results
        }

        # Calculate overall statistics
        total_evaluated = sum(r['evaluated_outputs'] for r in all_results)
        total_correct = sum(r['correct_outputs'] for r in all_results)
        overall_accuracy = total_correct / total_evaluated if total_evaluated > 0 else 0.0

        summary['overall_statistics'] = {
            'total_evaluated_outputs': total_evaluated,
            'total_correct_outputs': total_correct,
            'total_incorrect_outputs': total_evaluated - total_correct,
            'overall_accuracy': overall_accuracy,
            'overall_accuracy_percentage': overall_accuracy * 100.0
        }

        # Step 6: Save results
        if save_results:
            results_file_json = output_base_dir / "dataset_analysis.json"
            results_file_txt = output_base_dir / "dataset_analysis.txt"

            # Save JSON format
            with open(results_file_json, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            # Save human-readable text format
            with open(results_file_txt, 'w', encoding='utf-8') as f:
                f.write(f"Dataset Performance Analysis for {self.dataset_name}\n")
                f.write("=" * 80 + "\n")
                f.write(f"Evaluation Date: {summary['evaluation_timestamp']}\n")
                f.write(f"Dataset Size: {len(task_gt_map)} tasks\n")
                f.write(f"Settings Evaluated: {len(all_results)}\n\n")

                f.write("OVERALL STATISTICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Evaluated Outputs: {total_evaluated}\n")
                f.write(f"Total Correct: {total_correct}\n")
                f.write(f"Total Incorrect: {total_evaluated - total_correct}\n")
                f.write(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)\n\n")

                f.write("RESULTS BY SETTING\n")
                f.write("-" * 40 + "\n")

                # Sort by accuracy for better readability
                sorted_results = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)

                for result in sorted_results:
                    info = result['setting_info']
                    f.write(f"\nSetting: {result['setting_name']}\n")
                    f.write(f"  Mode: {info.get('mode', 'unknown')}\n")
                    f.write(f"  Model: {info.get('model', 'unknown')}\n")
                    if 'config' in info:
                        f.write(f"  Config: {info['config']}\n")
                    f.write(f"  Total Outputs: {result['total_outputs']}\n")
                    f.write(f"  Evaluated: {result['evaluated_outputs']}\n")
                    f.write(f"  Correct: {result['correct_outputs']}\n")
                    f.write(f"  Accuracy: {result['accuracy']:.4f} ({result['accuracy_percentage']:.2f}%)\n")

            logger.info(f"\nResults saved to:")
            logger.info(f"  {results_file_json}")
            logger.info(f"  {results_file_txt}")

        # Save results.txt in batch mode
        if save_metrics:
            self.save_dataset_results(output_base_dir, all_results)

        # Step 7: Display summary
        logger.info("\n" + "=" * 80)
        logger.info("DATASET ANALYSIS COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Dataset: {self.dataset_name}")
        logger.info(f"Settings evaluated: {len(all_results)}")
        logger.info(f"Total outputs evaluated: {total_evaluated}")
        logger.info(f"Overall accuracy: {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)")

        return summary

    def save_dataset_results(self, output_base_dir: Path, all_results: list):
        """Save results.txt summarizing all settings for this dataset."""
        results_file = output_base_dir / "results.txt"

        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write(f"BBEH Evaluation Summary: {self.dataset_name}\n")
            f.write("="*70 + "\n\n")

            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Total settings evaluated: {len(all_results)}\n\n")

            # Write header
            f.write(f"{'Setting':<50} {'Correct':>8} {'Rate':>6}\n")
            f.write("-"*70 + "\n")

            # Sort by accuracy for better readability
            sorted_results = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)

            # Write each setting's results
            for result in sorted_results:
                setting_name = result['setting_name']
                correct = result['correct_outputs']
                evaluated = result['evaluated_outputs']
                rate = result['accuracy_percentage']

                f.write(f"{setting_name:<50} {correct:>3}/{evaluated:<3} {rate:>5.1f}%\n")

            f.write("\n" + "="*70 + "\n")
            f.write("Detailed Metrics by Setting\n")
            f.write("="*70 + "\n\n")

            for result in sorted_results:
                info = result['setting_info']
                f.write(f"\n{result['setting_name']}\n")
                f.write("-"*70 + "\n")
                f.write(f"  Mode: {info.get('mode', 'unknown')}\n")
                f.write(f"  Model: {info.get('model', 'unknown')}\n")
                if 'config' in info:
                    f.write(f"  Config: {info['config']}\n")
                f.write(f"  Total outputs: {result['total_outputs']}\n")
                f.write(f"  Evaluated: {result['evaluated_outputs']}\n")
                f.write(f"  Correct: {result['correct_outputs']} ({result['accuracy_percentage']:.1f}%)\n")
                f.write(f"  Incorrect: {result['incorrect_outputs']}\n")
                f.write(f"  Accuracy: {result['accuracy']:.4f}\n")

        logger.info(f"  Results summary saved to: {results_file}")


def evaluate_all_bbeh_datasets(output_root: str = "output", verbose: bool = False):
    """
    Evaluate all BBEH datasets and generate summary results.

    For each dataset (e.g., bbeh_word_sorting):
    1. Evaluate each setting subfolder and save metrics.txt
    2. Create results.txt summarizing all settings for that dataset

    Args:
        output_root: Root output directory (default: "output")
        verbose: Enable verbose logging
    """
    output_root = Path(output_root)

    # Find all bbeh_* directories
    bbeh_dirs = sorted([d for d in output_root.iterdir()
                       if d.is_dir() and d.name.startswith('bbeh_')])

    if not bbeh_dirs:
        print(f"No bbeh_* directories found in {output_root}")
        return

    print("="*70)
    print(f"Evaluating All BBEH Datasets in {output_root}")
    print("="*70)
    print(f"Found {len(bbeh_dirs)} datasets\n")

    # Run clear_output once at the beginning
    logger.info("Running clear_output.py once for all datasets...")
    clear_output_path = project_root / "src" / "utils" / "clear_output.py"
    if clear_output_path.exists():
        try:
            result = subprocess.run(
                [sys.executable, str(clear_output_path)],
                cwd=str(project_root),
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info("Clear output completed successfully")
        except Exception as e:
            logger.warning(f"Clear output failed: {e}")

    # Process each dataset
    for dataset_dir in bbeh_dirs:
        dataset_name = dataset_dir.name

        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}")

        try:
            # Initialize evaluator for this dataset
            evaluator = DatasetEvaluator(dataset_name, verbose=verbose)

            # Run evaluation with metrics saving enabled
            summary = evaluator.analyze_dataset_performance(save_metrics=True, save_results=False)

            if summary and 'overall_statistics' in summary:
                stats = summary['overall_statistics']
                print(f"  Overall: {stats['total_correct_outputs']}/{stats['total_evaluated_outputs']} "
                      f"({stats['overall_accuracy_percentage']:.1f}%)")
            else:
                print(f"  No results generated")

        except Exception as e:
            print(f"  Error: {e}")
            if verbose:
                import traceback
                traceback.print_exc()

    print(f"\n{'='*70}")
    print("Evaluation Complete!")
    print(f"{'='*70}")


def main():
    """Main function with simplified argument parsing."""
    parser = argparse.ArgumentParser(description="Dataset-wide Evaluator for BBEH")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name (e.g., bbeh_word_sorting). If not provided, evaluates all bbeh_* datasets."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--output_root",
        default="output",
        help="Root directory containing bbeh_* folders (default: output)"
    )

    args = parser.parse_args()

    # If specific dataset provided, evaluate that dataset only
    if args.dataset:
        # Initialize evaluator
        evaluator = DatasetEvaluator(args.dataset, verbose=args.verbose)

        try:
            # Run dataset-wide analysis
            results = evaluator.analyze_dataset_performance()

            if not results:
                logger.error("Evaluation failed - no results generated")
                sys.exit(1)

            logger.info("Evaluation completed successfully!")

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)

    # Otherwise, evaluate all BBEH datasets
    else:
        try:
            evaluate_all_bbeh_datasets(args.output_root, verbose=args.verbose)
        except Exception as e:
            logger.error(f"Batch evaluation failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
