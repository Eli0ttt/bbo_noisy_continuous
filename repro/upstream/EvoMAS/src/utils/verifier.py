#!/usr/bin/env python3
"""
Answer Verifier - Detects and removes invalid answer files

This module provides functionality to identify and clean up invalid answer files
from the output directories. It handles various types of invalid answers including
empty files, error messages, and non-informative responses.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Set

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnswerVerifier:
    """Verifies and cleans invalid answer files from output directories."""

    def __init__(self):
        """Initialize the verifier with invalid patterns."""
        # Patterns that indicate invalid answers
        self.invalid_patterns = [
            # Empty or unknown responses
            r'^yes,\s*yes,\s*unknown\s*$',

            # No final response pattern
            r'^No final response\s*$',
            r'No final response',

            # Unable to provide responses
            r'Unable to provide.*',
            r'Cannot provide.*',

            # Error messages - General patterns first
            r'^Error:',  # Any line starting with "Error:"
            r'error:.*Bad Request',
            r'simulated due to error',
            r'Remote model response.*error',
            r'Exception.*occurred',
            r'Error.*processing',
            r'execution failed',  # Catches "Agent aggregator execution failed"
            r'ExpiredTokenException',
            r'TimeoutException',
            r'ThrottlingException',
            r'ServiceUnavailable',
            r'InternalServerError',
            r'rate limit',
            r'RateLimitError',

            # Refusal messages
            r'I cannot.*',
            r'I don\'t.*',
            r'I am unable.*',
            r'I\'m sorry.*',

            # Only truly meaningless filler patterns (empty or just punctuation)
            r'^\s*[,.\-_]*\s*$',  # Only punctuation and whitespace
        ]

        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                                 for pattern in self.invalid_patterns[:-1]]  # Exclude the last pattern
        self.short_response_pattern = re.compile(self.invalid_patterns[-1], re.IGNORECASE)

    def is_swe_bench_dataset(self, file_path: Path) -> bool:
        """
        Determine if this file belongs to a SWE-bench dataset.

        Args:
            file_path: Path to the file being validated

        Returns:
            True if it's a SWE-bench dataset, False for BBEH or other datasets
        """
        # Check the directory path for SWE-bench indicators
        path_str = str(file_path).lower()
        return 'swe' in path_str and ('bench' in path_str or 'swe_bench' in path_str)

    def is_valid_swe_patch(self, content: str) -> bool:
        """
        Check if content is a valid SWE-bench patch.

        Args:
            content: File content to validate

        Returns:
            True if content starts with "diff", False otherwise
        """
        content = content.strip()
        return content.startswith('diff')

    def is_empty_file(self, file_path: Path) -> bool:
        """Check if file is empty or contains only whitespace."""
        try:
            if file_path.stat().st_size == 0:
                return True

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
                return len(content) == 0
        except Exception as e:
            logger.error(f"Error checking if file is empty {file_path}: {e}")
            return False

    def is_short_meaningless(self, content: str) -> bool:
        """Check if content is too short to be meaningful AND looks like filler text."""
        clean_content = re.sub(r'\s+', '', content)

        # Check for repeated punctuation patterns (meaningless)
        if re.match(r'^[,.\-_]{2,}$', clean_content):
            return True

        # For very short content (1-2 chars), only flag if it's clearly not an answer
        if len(clean_content) <= 2:
            # Allow letters and numbers (A, B, C, D, 0, 1, 20, 42, No, etc.)
            # Allow letter/number with single punctuation (A), 1.)
            if re.match(r'^[A-Za-z0-9]+[).]?$', clean_content):
                return False
            # Flag random punctuation or whitespace-only
            return True

        # For 3+ character content, be very conservative
        # Only flag if it's clearly meaningless patterns
        if len(clean_content) >= 3:
            # Flag if it's all the same repeated character (but not alphanumeric)
            if len(set(clean_content)) == 1 and not clean_content[0].isalnum():
                return True

        return False

    def contains_invalid_pattern(self, content: str) -> bool:
        """Check if content matches any invalid patterns."""
        # Check against compiled patterns
        for pattern in self.compiled_patterns:
            if pattern.search(content):
                return True

        # Check for short meaningless content
        if self.is_short_meaningless(content):
            return True

        return False

    def is_invalid_answer(self, file_path: Path) -> tuple[bool, str]:
        """
        Check if a file contains an invalid answer.

        Returns:
            tuple: (is_invalid, reason)
        """
        try:
            # Check if file is empty
            if self.is_empty_file(file_path):
                return True, "Empty file"

            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()

            # Apply different validation logic based on dataset type
            if self.is_swe_bench_dataset(file_path):
                # SWE-bench validation: only check if it starts with "diff"
                if self.is_valid_swe_patch(content):
                    return False, "Valid SWE-bench patch"
                else:
                    return True, "Invalid SWE-bench patch (does not start with 'diff')"
            else:
                # BBEH dataset validation: use original complex judge logic
                if self.contains_invalid_pattern(content):
                    return True, "Contains invalid pattern"
                return False, "Valid BBEH answer"

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return True, f"Error reading file: {e}"

    def scan_directory(self, directory: Path) -> Dict[str, List[Path]]:
        """
        Scan directory for invalid answer files.

        Returns:
            Dict with 'invalid' and 'valid' file lists
        """
        results = {'invalid': [], 'valid': []}

        if not directory.exists():
            logger.error(f"Directory does not exist: {directory}")
            return results

        # Find all .txt files
        txt_files = list(directory.glob("*.txt"))

        logger.info(f"Scanning {len(txt_files)} files in {directory}")

        for file_path in txt_files:
            is_invalid, reason = self.is_invalid_answer(file_path)

            if is_invalid:
                results['invalid'].append(file_path)
                logger.debug(f"Invalid: {file_path} - {reason}")
            else:
                results['valid'].append(file_path)

        return results

    def scan_all_directories(self, base_path: Path) -> Dict[str, Dict[str, List[Path]]]:
        """
        Scan all subdirectories in base path for invalid files.

        Returns:
            Dict mapping directory names to their scan results
        """
        all_results = {}

        if not base_path.exists():
            logger.error(f"Base directory does not exist: {base_path}")
            return all_results

        # Find all subdirectories
        subdirs = [d for d in base_path.iterdir() if d.is_dir()]

        for subdir in subdirs:
            logger.info(f"Scanning directory: {subdir}")
            results = self.scan_directory(subdir)
            all_results[subdir.name] = results

            invalid_count = len(results['invalid'])
            valid_count = len(results['valid'])
            logger.info(f"  Found {invalid_count} invalid, {valid_count} valid files")

        return all_results

    def remove_invalid_files(self, file_paths: List[Path], dry_run: bool = True) -> int:
        """
        Remove invalid files.

        Args:
            file_paths: List of file paths to remove
            dry_run: If True, only log what would be removed

        Returns:
            Number of files processed
        """
        removed_count = 0

        for file_path in file_paths:
            try:
                if dry_run:
                    logger.info(f"Would remove: {file_path}")
                else:
                    file_path.unlink()
                    logger.info(f"Removed: {file_path}")
                removed_count += 1
            except Exception as e:
                logger.error(f"Error removing {file_path}: {e}")

        return removed_count

    def generate_report(self, all_results: Dict[str, Dict[str, List[Path]]]) -> str:
        """Generate a summary report of the verification results."""
        report_lines = ["Answer Verification Report", "=" * 30, ""]

        total_invalid = 0
        total_valid = 0

        for dir_name, results in all_results.items():
            invalid_count = len(results['invalid'])
            valid_count = len(results['valid'])
            total_invalid += invalid_count
            total_valid += valid_count

            report_lines.extend([
                f"Directory: {dir_name}",
                f"  Invalid files: {invalid_count}",
                f"  Valid files: {valid_count}",
                f"  Total files: {invalid_count + valid_count}",
                ""
            ])

        report_lines.extend([
            "Overall Summary:",
            f"  Total invalid files: {total_invalid}",
            f"  Total valid files: {total_valid}",
            f"  Total files: {total_invalid + total_valid}",
            f"  Invalid percentage: {100 * total_invalid / (total_invalid + total_valid):.1f}%" if (total_invalid + total_valid) > 0 else "  Invalid percentage: 0.0%"
        ])

        return "\n".join(report_lines)


def main():
    """Main function to run the verifier."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify and clean invalid answer files")
    parser.add_argument("path", help="Path to scan (directory or base directory)")
    parser.add_argument("--remove", action="store_true", help="Actually remove invalid files (default is dry run)")
    parser.add_argument("--recursive", action="store_true", help="Scan all subdirectories")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    verifier = AnswerVerifier()
    base_path = Path(args.path)

    if args.recursive:
        # Scan all subdirectories
        all_results = verifier.scan_all_directories(base_path)

        # Generate and print report
        report = verifier.generate_report(all_results)
        print(report)

        # Collect all invalid files
        all_invalid_files = []
        for results in all_results.values():
            all_invalid_files.extend(results['invalid'])

        if all_invalid_files:
            print(f"\nProcessing {len(all_invalid_files)} invalid files...")
            removed_count = verifier.remove_invalid_files(all_invalid_files, dry_run=not args.remove)

            if args.remove:
                print(f"Removed {removed_count} invalid files.")
            else:
                print(f"Dry run: would remove {removed_count} files. Use --remove to actually delete them.")
    else:
        # Scan single directory
        results = verifier.scan_directory(base_path)

        invalid_count = len(results['invalid'])
        valid_count = len(results['valid'])

        print(f"Scan results for {base_path}:")
        print(f"  Invalid files: {invalid_count}")
        print(f"  Valid files: {valid_count}")

        if results['invalid']:
            removed_count = verifier.remove_invalid_files(results['invalid'], dry_run=not args.remove)

            if args.remove:
                print(f"Removed {removed_count} invalid files.")
            else:
                print(f"Dry run: would remove {removed_count} files. Use --remove to actually delete them.")


if __name__ == "__main__":
    main()
