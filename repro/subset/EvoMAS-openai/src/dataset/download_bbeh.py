#!/usr/bin/env python3
"""
BBEH (BIG-Bench Extra Hard) Dataset Downloader

This script downloads BBEH datasets from the official Google DeepMind repository
and organizes them in the format expected by the MAS framework.

BBEH is a benchmark containing 23 challenging reasoning tasks designed to push
the boundaries of LLM evaluation.

Source: https://github.com/google-deepmind/bbeh

Usage:
    python src/dataset/download_bbeh.py --all
    python src/dataset/download_bbeh.py --task bbeh_word_sorting
    python src/dataset/download_bbeh.py --list
"""

import os
import sys
import json
import argparse
import logging
import tempfile
import shutil
import subprocess
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# BBEH GitHub repository
BBEH_REPO_URL = "https://github.com/google-deepmind/bbeh.git"

# Expected BBEH tasks (based on the official repository structure)
# Note: bbeh_mini is not in the official repo - it's a custom subset
BBEH_TASKS = [
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
    "bbeh_nycc",
    "bbeh_object_counting",
    "bbeh_object_properties",
    "bbeh_sarc_triples",
    "bbeh_shuffled_objects",
    "bbeh_spatial_reasoning",
    "bbeh_sportqa",
    "bbeh_temporal_sequence",
    "bbeh_time_arithmetic",
    "bbeh_web_of_lies",
    "bbeh_word_sorting",
    "bbeh_zebra_puzzles"
]


def clone_bbeh_repo(temp_dir: str) -> str:
    """
    Clone the BBEH repository to a temporary directory.

    Args:
        temp_dir: Temporary directory path

    Returns:
        Path to the cloned repository
    """
    logger.info(f"Cloning BBEH repository from {BBEH_REPO_URL}")

    repo_path = os.path.join(temp_dir, "bbeh")

    try:
        # Clone the repository
        subprocess.run(
            ["git", "clone", "--depth", "1", BBEH_REPO_URL, repo_path],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Successfully cloned BBEH repository to {repo_path}")
        return repo_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone repository: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.error("Git is not installed or not in PATH. Please install git first.")
        raise


def find_task_files(repo_path: str, task_name: str) -> Optional[str]:
    """
    Find the task.json file for a specific BBEH task in the cloned repository.

    Args:
        repo_path: Path to the cloned repository
        task_name: Name of the BBEH task

    Returns:
        Path to the task.json file, or None if not found
    """
    # Common possible locations for task data files
    # Note: BBEH uses task.json, not test.json
    possible_paths = [
        os.path.join(repo_path, "bbeh", "benchmark_tasks", task_name, "task.json"),
        os.path.join(repo_path, "benchmark_tasks", task_name, "task.json"),
        os.path.join(repo_path, "data", "benchmark_tasks", task_name, "task.json"),
        os.path.join(repo_path, task_name, "task.json"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            logger.debug(f"Found task file at: {path}")
            return path

    return None


def convert_to_mas_format(data: List[Dict[str, Any]], task_name: str) -> List[Dict[str, Any]]:
    """
    Convert BBEH data to MAS framework format.

    BBEH format:
    - input: question/problem statement
    - target: ground truth answer

    MAS format:
    - id: task identifier
    - query: input question
    - gt: ground truth answer
    - tag: list of tags
    - source: "BBEH"

    Args:
        data: Original BBEH data
        task_name: Name of the task

    Returns:
        Converted data in MAS format
    """
    converted_data = []

    for i, item in enumerate(data):
        try:
            # BBEH uses 'input' and 'target' fields
            query = item.get('input') or item.get('query') or item.get('question')
            gt = item.get('target') or item.get('gt') or item.get('answer')

            if not query:
                logger.warning(f"No query/input field found in task {task_name} item {i}")
                continue

            if gt is None:  # Allow empty string as ground truth
                logger.warning(f"No ground truth field found in task {task_name} item {i}")
                continue

            # Create MAS format item
            converted_item = {
                'id': item.get('id', i),
                'query': query,
                'gt': str(gt),  # Ensure it's a string
                'tag': item.get('tag', [task_name.replace('_', '-').upper()]),
                'source': item.get('source', 'BBEH')
            }

            converted_data.append(converted_item)

        except Exception as e:
            logger.warning(f"Error converting item {i} in task {task_name}: {e}")
            continue

    return converted_data


def download_task(
    repo_path: str,
    task_name: str,
    output_dir: Path,
    force_download: bool = False
) -> bool:
    """
    Download and process a single BBEH task.

    Args:
        repo_path: Path to the cloned BBEH repository
        task_name: Name of the task to download
        output_dir: Output directory for the dataset
        force_download: Whether to overwrite existing files

    Returns:
        True if successful, False otherwise
    """
    # Create task output directory
    task_dir = output_dir / "benchmark_tasks" / task_name
    task_file = task_dir / "test.json"

    # Check if already exists
    if task_file.exists() and not force_download:
        logger.info(f"Skipping {task_name} - file already exists: {task_file}")
        return True

    # Find the task file in the repository
    source_file = find_task_files(repo_path, task_name)

    if not source_file:
        logger.warning(f"Could not find data file for task: {task_name}")
        return False

    try:
        # Load the data
        with open(source_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # BBEH format uses 'examples' key containing the list of tasks
        if isinstance(data, dict):
            # Check if there's a 'examples' or 'data' key
            if 'examples' in data:
                data = data['examples']
            elif 'data' in data:
                data = data['data']
            elif 'tasks' in data:
                data = data['tasks']
            else:
                logger.warning(f"Unexpected data format for {task_name}. Expected 'examples', 'data', or 'tasks' key")
                return False

        if not isinstance(data, list):
            logger.warning(f"Data for {task_name} is not a list")
            return False

        # Convert to MAS format
        converted_data = convert_to_mas_format(data, task_name)

        if not converted_data:
            logger.warning(f"No valid data found for task: {task_name}")
            return False

        # Create output directory
        task_dir.mkdir(parents=True, exist_ok=True)

        # Save the converted data
        with open(task_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully downloaded {task_name}: {len(converted_data)} examples")
        return True

    except Exception as e:
        logger.error(f"Error processing task {task_name}: {e}")
        return False


def download_all_tasks(force_download: bool = False) -> None:
    """
    Download all BBEH tasks.

    Args:
        force_download: Whether to re-download existing files
    """
    # Set up output directory
    repo_root = Path(__file__).parent.parent.parent
    dataset_dir = repo_root / "dataset" / "bbeh"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {dataset_dir}")

    # Create temporary directory for cloning
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Clone the repository
            repo_path = clone_bbeh_repo(temp_dir)

            # Download each task
            success_count = 0
            fail_count = 0

            for task_name in BBEH_TASKS:
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing task: {task_name}")
                logger.info(f"{'='*60}")

                if download_task(repo_path, task_name, dataset_dir, force_download):
                    success_count += 1
                else:
                    fail_count += 1

            # Summary
            logger.info(f"\n{'='*60}")
            logger.info(f"Download Summary:")
            logger.info(f"  Successful: {success_count}/{len(BBEH_TASKS)}")
            logger.info(f"  Failed: {fail_count}/{len(BBEH_TASKS)}")
            logger.info(f"{'='*60}")

        except Exception as e:
            logger.error(f"Failed to download BBEH datasets: {e}")
            raise


def download_single_task(task_name: str, force_download: bool = False) -> None:
    """
    Download a single BBEH task.

    Args:
        task_name: Name of the task to download
        force_download: Whether to re-download if exists
    """
    if task_name not in BBEH_TASKS:
        logger.error(f"Unknown task: {task_name}")
        logger.info(f"Available tasks: {', '.join(BBEH_TASKS)}")
        return

    # Set up output directory
    repo_root = Path(__file__).parent.parent.parent
    dataset_dir = repo_root / "dataset" / "bbeh"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading task: {task_name}")
    logger.info(f"Output directory: {dataset_dir}")

    # Create temporary directory for cloning
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Clone the repository
            repo_path = clone_bbeh_repo(temp_dir)

            # Download the task
            if download_task(repo_path, task_name, dataset_dir, force_download):
                logger.info(f"Successfully downloaded {task_name}")
            else:
                logger.error(f"Failed to download {task_name}")

        except Exception as e:
            logger.error(f"Failed to download task {task_name}: {e}")
            raise


def list_available_tasks() -> None:
    """List all available BBEH tasks."""
    logger.info("Available BBEH tasks:")
    for i, task in enumerate(BBEH_TASKS, 1):
        print(f"  {i:2d}. {task}")
    print(f"\nTotal: {len(BBEH_TASKS)} tasks")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Download and organize BBEH datasets from Google DeepMind"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all BBEH tasks"
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Download a specific task (e.g., bbeh_word_sorting)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available BBEH tasks and exit"
    )

    args = parser.parse_args()

    if args.list:
        list_available_tasks()
        return

    if args.all:
        logger.info("Downloading all BBEH tasks...")
        download_all_tasks(force_download=args.force)
    elif args.task:
        download_single_task(args.task, force_download=args.force)
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python src/dataset/download_bbeh.py --all")
        print("  python src/dataset/download_bbeh.py --task bbeh_word_sorting")
        print("  python src/dataset/download_bbeh.py --list")


if __name__ == "__main__":
    main()
