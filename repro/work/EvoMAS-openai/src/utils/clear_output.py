#!/usr/bin/env python3
"""
Clear Output - Clean and fix invalid answer files from all output directories

Usage:
    python -m src.utils.clear_output
    python src/utils/clear_output.py

This script scans all output directories and cleans invalid answer files:
- For SWE-bench: Extracts clean patches from generation logs and markdown
- For BBEH: Extracts final answers
- Removes files that cannot be cleaned
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent  # src/utils/clear_output.py -> project root
sys.path.insert(0, str(project_root))

from src.utils.verifier import AnswerVerifier
from src.utils.output_cleaner import clean_output_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_output_directories(base_path: Path) -> list[Path]:
    """Find all directories to scan for .txt files."""
    output_dirs = []

    if not base_path.exists():
        return output_dirs

    # Check if base_path itself has .txt files
    txt_files = list(base_path.glob("*.txt"))
    if txt_files:
        output_dirs.append(base_path)

    # Look for subdirectories that might contain output files
    for item in base_path.iterdir():
        if item.is_dir():
            # Check if directory contains .txt files
            txt_files = list(item.glob("*.txt"))
            if txt_files:
                output_dirs.append(item)

    return output_dirs


def main():
    """Main function to clean and verify outputs."""
    print("Clear Output - Cleaning and verifying answer files...")
    print("=" * 80)

    # Find project root and output directory
    project_root = Path(__file__).parent.parent.parent
    output_base = project_root / "output"

    if not output_base.exists():
        print(f"Output directory not found: {output_base}")
        return

    print(f"Scanning output directory: {output_base}")

    # Initialize verifier
    verifier = AnswerVerifier()

    total_cleaned = 0
    total_removed = 0
    total_scanned = 0
    total_already_valid = 0

    # Process each dataset directory
    for dataset_dir in output_base.iterdir():
        if not dataset_dir.is_dir():
            continue

        print(f"\nProcessing dataset: {dataset_dir.name}")

        # Find output directories in this dataset
        output_dirs = find_output_directories(dataset_dir)

        if not output_dirs:
            print(f"   ℹNo output directories found")
            continue

        # Process each output directory
        for output_dir in output_dirs:
            print(f"   Scanning: {output_dir.name}")

            # STEP 1: Clean all files first (extract patches, remove markdown, etc.)
            print(f"      Cleaning outputs...")
            clean_stats = clean_output_directory(output_dir, dataset_type='auto')
            total_cleaned += clean_stats['cleaned']
            total_scanned += clean_stats['total']

            if clean_stats['cleaned'] > 0:
                print(f"      Cleaned {clean_stats['cleaned']}/{clean_stats['total']} files")
            else:
                print(f"      ℹNo files needed cleaning")

            # STEP 2: After cleaning, verify remaining files
            print(f"      Verifying outputs...")
            results = verifier.scan_directory(output_dir)
            invalid_files = results['invalid']
            valid_files = results['valid']

            total_already_valid += len(valid_files)

            if invalid_files:
                # Remove files that couldn't be cleaned
                removed_count = verifier.remove_invalid_files(invalid_files, dry_run=False)
                total_removed += removed_count
                print(f"      Removed {removed_count} invalid files that couldn't be cleaned")
            else:
                print(f"      All {len(valid_files)} files are now valid")

    # Summary
    print("\n" + "=" * 80)
    print("CLEANUP SUMMARY")
    print("=" * 80)
    print(f"Total files scanned: {total_scanned}")
    print(f"Files cleaned: {total_cleaned}")
    print(f"Files already valid: {total_already_valid}")
    print(f"Invalid files removed: {total_removed}")
    print(f"Final valid files: {total_scanned - total_removed}")

    if total_cleaned > 0:
        print(f"Cleanup completed! Cleaned {total_cleaned} files, removed {total_removed} invalid files.")
    elif total_removed > 0:
        print(f"Cleanup completed! Removed {total_removed} invalid files.")
    else:
        print("No cleaning needed. All outputs are already valid!")


if __name__ == "__main__":
    main()
