#!/usr/bin/env python3
"""
Output Cleaner - Shared utilities for cleaning LLM outputs

This module provides functions to clean LLM outputs for different dataset types:
- BBEH: Extract final answers
- SWE-bench: Extract clean patches starting with "diff"
"""

import re
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def clean_swe_bench_output(content: str) -> str:
    """
    Clean up SWE-bench diff content by extracting only the valid source code patch.

    This function:
    1. Searches for diff blocks that modify actual source files (not .traj, .sweagent_output, etc.)
    2. Extracts only relevant source code diffs
    3. Removes markdown code fences
    4. Removes trailing generation logs or explanations

    Args:
        content: Raw content that may contain generation logs, markdown, and patch

    Returns:
        Cleaned patch content that starts with "diff"
    """
    content = content.strip()

    # Files/patterns to exclude from the patch (not actual source code changes)
    exclude_patterns = [
        '.sweagent_output/',
        '.traj',
        '_helpers',  # submodules like astropy_helpers
        'Subproject commit',
    ]

    # Helper function to check if a diff block should be excluded
    def should_exclude_diff(diff_header: str) -> bool:
        for pattern in exclude_patterns:
            if pattern in diff_header:
                return True
        return False

    # Step 1: Try to extract from markdown code fence first
    diff_match = re.search(r'```diff\s*\n(.*?)\n```', content, re.DOTALL)
    if diff_match:
        content = diff_match.group(1).strip()
    else:
        code_match = re.search(r'```\s*\n(.*?)\n```', content, re.DOTALL)
        if code_match:
            potential_diff = code_match.group(1).strip()
            if potential_diff.startswith('diff '):
                content = potential_diff

    # Step 2: Split content into individual diff blocks
    lines = content.split('\n')
    diff_blocks = []
    current_block = []
    current_header = ""

    for line in lines:
        if line.startswith('diff --git') or line.startswith('diff '):
            # Save previous block if it exists and is valid
            if current_block and current_header and not should_exclude_diff(current_header):
                diff_blocks.append('\n'.join(current_block))
            # Start new block
            current_block = [line]
            current_header = line
        elif current_block:
            # Continue current block
            current_block.append(line)

    # Don't forget the last block
    if current_block and current_header and not should_exclude_diff(current_header):
        diff_blocks.append('\n'.join(current_block))

    # Step 3: If no valid diff blocks found, return empty or original
    if not diff_blocks:
        logger.debug(f"No valid source diff found in content (first 100 chars): {content[:100]}")
        # Try to return original content if it looks like a valid diff
        if content.strip().startswith('diff '):
            return content.strip()
        return ""

    # Step 4: Join all valid diff blocks
    clean_diff = '\n'.join(diff_blocks).strip()

    # Step 5: Remove any trailing markdown fences or non-diff content
    # NOTE: Don't strip lines when checking - context lines start with space ' '
    clean_lines = clean_diff.split('\n')
    last_valid_idx = len(clean_lines) - 1

    for i in range(len(clean_lines) - 1, -1, -1):
        line = clean_lines[i]
        stripped = line.strip()

        # Skip markdown fences
        if stripped == '```':
            last_valid_idx = i - 1
            continue

        # Empty lines are valid in diffs (context)
        if not stripped:
            continue

        # Check if line is a valid diff line (don't strip - context lines start with space)
        is_valid_diff_line = (
            line.startswith(('diff ', '---', '+++', '@@', '+', '-', ' ', 'index ', 'new file', 'deleted file', 'Binary files', '\\'))
            or stripped.startswith(('diff ', '---', '+++', '@@', '+', '-', 'index ', 'new file', 'deleted file', 'Binary files', '\\'))
        )

        if not is_valid_diff_line:
            last_valid_idx = i - 1
            continue

        # Found a valid diff line, stop searching
        last_valid_idx = i
        break

    result = '\n'.join(clean_lines[:last_valid_idx + 1]).strip()
    # Ensure patch ends with newline (required for git apply)
    if result and not result.endswith('\n'):
        result += '\n'
    return result


def clean_bbeh_output(content: str) -> str:
    """
    Extract the final answer from BBEH task response.

    Args:
        content: Full LLM response

    Returns:
        Extracted final answer
    """
    content = content.strip()

    # Pattern 1: Content after "Final answer:" (our prompt format)
    final_answer_match = re.search(r'Final answer:\s*(.+?)(?:\n|$)', content, re.IGNORECASE | re.MULTILINE)
    if final_answer_match:
        return final_answer_match.group(1).strip()

    # Pattern 2: \boxed{answer}
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', content)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Pattern 3: **Answer:** variations
    answer_patterns = [
        r'\*\*Answer:\*\*\s*(.+?)(?:\n|$)',
        r'Answer:\s*(.+?)(?:\n|$)',
        r'The answer is:?\s*(.+?)(?:\n|$)',
        r'Result:\s*(.+?)(?:\n|$)'
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()

    # Pattern 4: Look for "unknown" specifically for BBEH tasks
    if 'unknown' in content.lower():
        return 'unknown'

    # If response is already short and clean, return as-is
    if len(content) < 50 and '\n' not in content:
        return content

    # Pattern 5: Look for the last meaningful line
    lines = [line.strip() for line in content.strip().split('\n') if line.strip()]
    if lines:
        last_line = lines[-1]
        # If last line is short and doesn't look like explanation text
        if (len(last_line.split()) <= 10 and
            not last_line.lower().startswith(('therefore', 'thus', 'so', 'in conclusion', 'to summarize'))):
            return last_line

    # Fallback: return first line if it's reasonable, otherwise first 100 chars
    if lines and len(lines[0]) < 100:
        return lines[0]

    return content[:100] + "..." if len(content) > 100 else content


def is_valid_workbench_output(content: str) -> bool:
    """
    Check if WorkBench output is valid (has FUNCTION_CALLS: with actual function calls, no errors).

    Validation checks:
    1. Not empty
    2. No error indicators
    3. Has FUNCTION_CALLS: section
    4. Has at least one actual function call

    Args:
        content: Full LLM response

    Returns:
        True if valid (has FUNCTION_CALLS with at least one call, no errors), False otherwise
    """
    content = content.strip()

    # Check 1: Must not be empty
    if not content:
        logger.debug("Invalid WorkBench output: empty content")
        return False

    content_lower = content.lower()

    # Check 2: Must not contain error indicators (case-insensitive)
    error_indicators = [
        'error:',
        'exception:',
        'traceback',
        'failed to',
        'could not',
        'validation error',
        'validationexception',
        'cannot',
        'unable to'
    ]
    for indicator in error_indicators:
        if indicator in content_lower:
            logger.debug(f"Invalid WorkBench output: contains error indicator '{indicator}'")
            return False

    # Check 3: Must have FUNCTION_CALLS: section
    if 'FUNCTION_CALLS:' not in content:
        logger.debug("Invalid WorkBench output: missing 'FUNCTION_CALLS:' section")
        return False

    # Extract content after FUNCTION_CALLS:
    function_section = content.split('FUNCTION_CALLS:')[-1].strip()

    # Function section must not be empty
    if not function_section:
        logger.debug("Invalid WorkBench output: empty FUNCTION_CALLS section")
        return False

    # Check 4: Must have actual function calls
    # Look for patterns like domain.function.func(...) or domain_function(...)
    function_patterns = [
        r'\w+\.\w+\.func\([^)]*\)',  # official: domain.function.func(...)
        r'\w+_\w+(?:_\w+)*\([^)]*\)',  # evomas: domain_function(...)
    ]

    for pattern in function_patterns:
        if re.search(pattern, function_section):
            return True

    logger.debug(f"Invalid WorkBench output: no function calls found after 'FUNCTION_CALLS:' (content: {function_section[:100]})")
    return False


def clean_workbench_output(content: str) -> str:
    """
    Validate and clean WorkBench output.

    For WorkBench, we don't modify the content, we just validate it.
    Returns empty string if invalid, original content if valid.

    Args:
        content: Full LLM response

    Returns:
        Original content if valid, empty string if invalid
    """
    if is_valid_workbench_output(content):
        return content.strip()
    else:
        return ""


def clean_output_file(file_path: Path, dataset_type: str = 'auto') -> bool:
    """
    Clean an output file in-place based on dataset type.

    Args:
        file_path: Path to the output file
        dataset_type: 'swe_bench', 'bbeh', or 'auto' to detect from path

    Returns:
        True if file was cleaned, False if no cleaning needed or failed
    """
    try:
        # Read original content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Auto-detect dataset type if needed
        if dataset_type == 'auto':
            path_str = str(file_path).lower()
            if 'swe' in path_str and ('bench' in path_str or 'swe_bench' in path_str):
                dataset_type = 'swe_bench'
            elif 'workbench' in path_str:
                dataset_type = 'workbench'
            else:
                dataset_type = 'bbeh'

        # Clean based on dataset type
        if dataset_type == 'swe_bench':
            cleaned_content = clean_swe_bench_output(content)
        elif dataset_type == 'workbench':
            cleaned_content = clean_workbench_output(content)
        else:
            cleaned_content = clean_bbeh_output(content)

        # If cleaning produced empty content (invalid file), delete it
        # Check this BEFORE checking if content changed (to catch empty original files)
        if not cleaned_content:
            logger.warning(f"Invalid output detected, removing: {file_path}")
            file_path.unlink()  # Delete the file
            return True  # Counted as "cleaned" (removed)

        # Check if cleaning changed anything
        if cleaned_content == content.strip():
            return False

        # Write cleaned content back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)

        logger.info(f"Cleaned: {file_path}")
        return True

    except Exception as e:
        logger.error(f"Error cleaning {file_path}: {e}")
        return False


def clean_output_directory(directory: Path, dataset_type: str = 'auto') -> dict:
    """
    Clean all .txt files in a directory.

    Args:
        directory: Path to directory containing output files
        dataset_type: 'swe_bench', 'bbeh', 'workbench', or 'auto'

    Returns:
        Dictionary with statistics: {cleaned, skipped, failed, total}
    """
    stats = {'cleaned': 0, 'skipped': 0, 'failed': 0, 'total': 0}

    if not directory.exists():
        logger.error(f"Directory does not exist: {directory}")
        return stats

    # Skip special files that should not be cleaned
    skip_files = {'metrics.txt', 'results.txt', 'summary.txt'}

    txt_files = [f for f in directory.glob("*.txt") if f.name not in skip_files]
    stats['total'] = len(txt_files)

    logger.info(f"Cleaning {len(txt_files)} files in {directory}")

    for file_path in txt_files:
        try:
            if clean_output_file(file_path, dataset_type):
                stats['cleaned'] += 1
            else:
                stats['skipped'] += 1
        except Exception as e:
            logger.error(f"Failed to clean {file_path}: {e}")
            stats['failed'] += 1

    return stats
