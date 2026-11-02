#!/usr/bin/env python3
"""
SWE-bench Dataset Downloader and Processor

This script downloads SWE-bench datasets from Hugging Face and converts them
to the format expected by the MAS framework.

Supports:
- SWE-bench (main dataset)
- SWE-bench Lite (smaller subset)
- SWE-bench Verified (validated subset)

Usage:
    python src/dataset/swe_bench.py --dataset swe-bench --split test
    python src/dataset/swe_bench.py --dataset swe-bench-lite --split dev
    python src/dataset/swe_bench.py --all  # Download all variants
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_dataset_config(dataset_name: str) -> Dict[str, str]:
    """
    Get Hugging Face dataset configuration for different SWE-bench variants.

    Args:
        dataset_name: Name of the dataset (swe-bench, swe-bench-lite, swe-bench-verified)

    Returns:
        Dictionary with dataset configuration
    """
    configs = {
        "swe-bench": {
            "hf_name": "princeton-nlp/SWE-bench",
            "local_name": "swe_bench"
        },
        "swe-bench-lite": {
            "hf_name": "princeton-nlp/SWE-bench_Lite",
            "local_name": "swe_bench_lite"
        },
        "swe-bench-verified": {
            "hf_name": "princeton-nlp/SWE-bench_Verified",
            "local_name": "swe_bench_verified"
        }
    }

    if dataset_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")

    return configs[dataset_name]


def convert_swe_bench_to_mas_format(swe_bench_item: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    Convert a SWE-bench dataset item to MAS framework format.

    SWE-bench fields (based on HuggingFace schema):
    - repo: repository name
    - instance_id: unique identifier
    - base_commit: base commit hash
    - patch: code changes needed
    - test_patch: test-related changes
    - problem_statement: description of the issue
    - hints_text: optional hints
    - created_at: timestamp
    - version: dataset version
    - FAIL_TO_PASS: failure to pass information
    - PASS_TO_PASS: pass to pass information
    - environment_setup_commit: environment setup commit

    MAS format expected fields:
    - id: task identifier
    - query: input/question for the model
    - gt: ground truth/expected output
    - tag: list of tags
    - source: source dataset name

    Args:
        swe_bench_item: Original SWE-bench item
        index: Index for creating ID if instance_id is not usable

    Returns:
        Dictionary in MAS format
    """

    # Use instance_id exactly as provided - required for official SWE-bench evaluation
    # No fallback to ensure we only use official instance IDs that the harness recognizes
    if 'instance_id' not in swe_bench_item:
        raise KeyError(f"Missing required 'instance_id' in SWE-bench item at index {index}")

    task_id = swe_bench_item['instance_id']

    # Create comprehensive query combining problem statement and context
    repo = swe_bench_item.get('repo', 'unknown')
    problem = swe_bench_item.get('problem_statement', '')
    hints = swe_bench_item.get('hints_text', '')

    # Build query with context
    query_parts = [
        f"Repository: {repo}",
        f"Problem Statement:\n{problem}"
    ]

    if hints and hints.strip():
        query_parts.append(f"Hints:\n{hints}")

    query_parts.append("\nPlease provide a patch to resolve this issue.")

    query = "\n\n".join(query_parts)

    # Use the patch as ground truth
    ground_truth = swe_bench_item.get('patch', '')

    # Create tags based on repository and other metadata
    tags = ['swe-bench']
    if repo:
        tags.append(f"repo:{repo}")

    # Add version if available
    version = swe_bench_item.get('version', '')
    if version:
        tags.append(f"version:{version}")

    return {
        'id': task_id,
        'query': query,
        'gt': ground_truth,
        'tag': tags,
        'source': 'SWE-BENCH',
        # Store original fields for reference
        'metadata': {
            'repo': repo,
            'instance_id': swe_bench_item.get('instance_id'),
            'base_commit': swe_bench_item.get('base_commit'),
            'test_patch': swe_bench_item.get('test_patch'),
            'created_at': swe_bench_item.get('created_at'),
            'environment_setup_commit': swe_bench_item.get('environment_setup_commit'),
            'FAIL_TO_PASS': swe_bench_item.get('FAIL_TO_PASS'),
            'PASS_TO_PASS': swe_bench_item.get('PASS_TO_PASS')
        }
    }


def download_and_convert_dataset(
    dataset_name: str,
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
    force_download: bool = False
) -> None:
    """
    Download SWE-bench dataset from Hugging Face and convert to MAS format.

    Note: SWE-bench official splits are 'dev' and 'test' (not 'train').

    Args:
        dataset_name: Name of the dataset variant
        split: Dataset split to download (None for official dev/test splits)
        max_samples: Maximum samples per split (None for all)
        force_download: Whether to re-download existing files
    """

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not found. Install with: pip install datasets")
        return

    # Get dataset configuration
    config = get_dataset_config(dataset_name)
    hf_name = config["hf_name"]
    local_name = config["local_name"]

    # Set up local directory
    repo_root = Path(__file__).parent.parent.parent
    dataset_dir = repo_root / "dataset" / local_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {dataset_name} from {hf_name}")
    logger.info(f"Saving to: {dataset_dir}")

    # Define available splits per dataset variant
    dataset_splits = {
        "swe-bench": ["dev", "train", "test"],          # Main dataset has all splits
        "swe-bench-lite": ["dev", "test"],             # Lite has dev and test
        "swe-bench-verified": ["test"]                 # Verified has only test
    }

    # Get splits for this specific dataset
    available_splits_for_dataset = dataset_splits.get(dataset_name, ["dev", "test"])

    if split:
        # Download specific split
        if split not in available_splits_for_dataset:
            logger.warning(f"Requested split '{split}' may not be available for {dataset_name}. Expected splits: {available_splits_for_dataset}")
        splits_to_download = [split]
    else:
        # Download all available splits for this dataset
        splits_to_download = available_splits_for_dataset
        logger.info(f"Downloading all available splits for {dataset_name}: {available_splits_for_dataset}")

    try:
        # Load dataset from Hugging Face
        logger.info(f"Loading dataset: {hf_name}")
        full_dataset = load_dataset(hf_name)

        # Check available splits
        available_splits = list(full_dataset.keys())
        logger.info(f"Available splits in dataset: {available_splits}")

        splits_to_process = {}
        for split_name in splits_to_download:
            if split_name in available_splits:
                splits_to_process[split_name] = full_dataset[split_name]
                logger.info(f"Added split '{split_name}' for processing ({len(full_dataset[split_name])} samples)")
            else:
                logger.warning(f"Split '{split_name}' not found in dataset. Available: {available_splits}")

    except Exception as e:
        logger.error(f"Failed to load dataset {hf_name}: {e}")
        logger.info("This might be due to authentication issues or dataset access restrictions")
        logger.info("Make sure you have proper Hugging Face credentials if the dataset requires authentication")
        return

    # Process each split
    for split_name, split_data in splits_to_process.items():
        output_file = dataset_dir / f"{split_name}.json"

        # Skip if file exists and not forcing download
        if output_file.exists() and not force_download:
            logger.info(f"Skipping {split_name} - file already exists: {output_file}")
            continue

        logger.info(f"Processing split '{split_name}' with {len(split_data)} samples")

        # Convert to MAS format
        converted_data = []
        sample_count = 0

        for i, item in enumerate(split_data):
            if max_samples and sample_count >= max_samples:
                logger.info(f"Reached max_samples limit ({max_samples}) for split {split_name}")
                break

            try:
                converted_item = convert_swe_bench_to_mas_format(item, i)
                converted_data.append(converted_item)
                sample_count += 1

                if (sample_count % 100 == 0):
                    logger.info(f"Converted {sample_count} samples for split {split_name}")

            except Exception as e:
                logger.warning(f"Failed to convert sample {i} in split {split_name}: {e}")
                continue

        # Save converted data
        logger.info(f"Saving {len(converted_data)} converted samples to {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully saved split '{split_name}' to {output_file}")

    logger.info(f"Dataset {dataset_name} download and conversion completed!")


def list_available_datasets() -> List[str]:
    """List all available SWE-bench dataset variants."""
    configs = {
        "swe-bench": "princeton-nlp/SWE-bench",
        "swe-bench-lite": "princeton-nlp/SWE-bench-Lite",
        "swe-bench-verified": "princeton-nlp/SWE-bench-Verified"
    }
    return list(configs.keys())


def download_all_datasets(max_samples: Optional[int] = None, force_download: bool = False) -> None:
    """Download all available SWE-bench dataset variants."""
    datasets = list_available_datasets()

    logger.info(f"Downloading all SWE-bench variants: {datasets}")

    for dataset in datasets:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing dataset: {dataset}")
            logger.info(f"{'='*60}")

            download_and_convert_dataset(
                dataset_name=dataset,
                split=None,  # Download all splits
                max_samples=max_samples,
                force_download=force_download
            )
        except Exception as e:
            logger.error(f"Failed to download dataset {dataset}: {e}")
            continue

    logger.info("\nAll dataset downloads completed!")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Download and convert SWE-bench datasets")

    parser.add_argument(
        "--dataset",
        choices=list_available_datasets(),
        help="SWE-bench dataset variant to download"
    )
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        help="Specific split to download (default: all splits)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples per split (default: all samples)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all SWE-bench dataset variants"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit"
    )

    args = parser.parse_args()

    if args.list:
        datasets = list_available_datasets()
        print("Available SWE-bench datasets:")
        for dataset in datasets:
            config = get_dataset_config(dataset)
            print(f"  {dataset} -> {config['hf_name']}")
        return

    if args.all:
        download_all_datasets(
            max_samples=args.max_samples,
            force_download=args.force
        )
    elif args.dataset:
        download_and_convert_dataset(
            dataset_name=args.dataset,
            split=args.split,
            max_samples=args.max_samples,
            force_download=args.force
        )
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python src/dataset/swe_bench.py --dataset swe-bench-lite --split test")
        print("  python src/dataset/swe_bench.py --all")
        print("  python src/dataset/swe_bench.py --list")


if __name__ == "__main__":
    main()
