#!/usr/bin/env python3
"""
Local SWE-bench Evaluator with On-Demand Environment Creation

This evaluator:
1. Takes instance_id and patch as input
2. Creates conda environments on-demand
3. Applies patches and runs tests locally
4. Aligns with auto-code-rover and SWE-bench official metrics
5. Supports caching environments or deleting after use

Usage:
    # Test with ground truth patch
    python local_swe_evaluator.py \
        --dataset swe_bench_lite \
        --instance_id astropy__astropy-12907 \
        --use_gt

    # Evaluate custom patch
    python local_swe_evaluator.py \
        --dataset swe_bench_lite \
        --instance_id django__django-11333 \
        --patch_file my_patch.diff \
        --use_cache  # Keep environment after evaluation

    # Evaluate without caching (delete env after)
    python local_swe_evaluator.py \
        --dataset swe_bench_verified \
        --instance_id astropy__astropy-12907 \
        --patch_file patch.diff
"""

import argparse
import json
import re
import subprocess
import sys
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from utils directory
from src.utils.lazy_env_manager import ensure_env_ready, cleanup_env
from src.utils.env_utils import cd, repo_reset_and_clean_checkout, run_string_cmd_in_conda


class TestStatus(Enum):
    """Test status enum matching SWE-bench/auto-code-rover"""
    FAILED = "FAILED"
    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


class ResolvedStatus(Enum):
    """Resolution status matching auto-code-rover"""
    FULL = "FULL"           # All FAIL_TO_PASS tests pass, no new failures
    PARTIAL = "PARTIAL"     # Some FAIL_TO_PASS tests pass
    NO = "NO"               # No FAIL_TO_PASS tests pass


def normalize_instance_id(instance_id: str) -> Tuple[str, str]:
    """
    Normalize instance ID to handle format differences.

    Args:
        instance_id: Can be "astropy__astropy-12907" or "astropy__astropy_12907"

    Returns:
        Tuple of (underscore_format, dash_format)
    """
    # Convert to underscore format
    underscore_format = instance_id.replace('-', '_')

    # Convert to dash format (for tasks_map lookup)
    # Only replace the last underscore-digits with dash-digits
    parts = instance_id.split('__')
    if len(parts) == 2:
        repo_part = parts[0]
        issue_part = parts[1].replace('_', '-')
        dash_format = f"{repo_part}__{issue_part}"
    else:
        dash_format = instance_id

    return underscore_format, dash_format


def load_instance_data(instance_id: str, dataset: str) -> Optional[Dict]:
    """
    Load instance data from test.json.

    Args:
        instance_id: Instance ID (handles both formats)
        dataset: Dataset name

    Returns:
        Instance data dict or None if not found
    """
    test_json_path = PROJECT_ROOT / "dataset" / dataset / "test.json"

    if not test_json_path.exists():
        print(f"test.json not found at {test_json_path}")
        return None

    with open(test_json_path) as f:
        data = json.load(f)

    # Try to find instance with either format
    _, dash_format = normalize_instance_id(instance_id)

    for item in data:
        item_id = item["metadata"]["instance_id"]
        if item_id == dash_format or item_id == instance_id:
            return item

    print(f"Instance {instance_id} not found in test.json")
    return None


def load_configs(dataset: str) -> Tuple[Dict, Dict]:
    """Load setup_map and tasks_map."""
    config_dir = PROJECT_ROOT / "dataset" / dataset

    with open(config_dir / "setup_map.json") as f:
        setup_map = json.load(f)

    with open(config_dir / "tasks_map.json") as f:
        tasks_map = json.load(f)

    return setup_map, tasks_map


@contextmanager
def apply_patch(patch_content: str, repo_path: str):
    """
    Apply patch temporarily, rollback after use.
    Matches auto-code-rover's implementation.
    """
    try:
        with NamedTemporaryFile(buffering=0, suffix=".diff") as f:
            f.write(patch_content.encode())
            apply_cmd = ["git", "apply", f.name]
            result = subprocess.run(
                apply_cmd,
                cwd=repo_path,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print(f"Patch application warning: {result.stderr[:200]}")

        yield result.returncode == 0

    finally:
        # Rollback changes
        with cd(repo_path):
            subprocess.run(["git", "reset", "--hard"], capture_output=True)
            subprocess.run(["git", "clean", "-fd"], capture_output=True)


def parse_log_pytest(log: str) -> Dict[str, str]:
    """
    Parser for test logs generated with PyTest framework.
    From auto-code-rover/SWE-bench.
    """
    test_status_map = {}
    for line in log.split("\n"):
        if any([line.startswith(x.value) for x in TestStatus]):
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map


def parse_log_django(log: str) -> Dict[str, str]:
    """Parser for Django test logs."""
    test_status_map = {}
    lines = log.split("\n")

    # Track current test name for multi-line output
    current_test = None

    for line in lines:
        line = line.strip()

        # Check if this line is a test name (format: test_name (module.Class))
        if ' (' in line and ')' in line and not line.startswith(' '):
            # Possible test name line
            potential_test = line
            if potential_test.endswith(')'):
                current_test = potential_test
            elif ' ... ' in potential_test:
                # Test name and status on same line
                current_test = None

        # Check for status on current or previous line's test
        if line.endswith(" ... ok"):
            test = line.split(" ... ok")[0]
            # If we have current_test, the test name was on the previous line
            if current_test:
                test = current_test
                current_test = None
            if test:
                test_status_map[test] = TestStatus.PASSED.value
        elif " ... skipped" in line:
            test = line.split(" ... skipped")[0]
            if current_test:
                test = current_test
                current_test = None
            if test:
                test_status_map[test] = TestStatus.SKIPPED.value
        elif line.endswith(" ... FAIL"):
            test = line.split(" ... FAIL")[0]
            if current_test:
                test = current_test
                current_test = None
            if test:
                test_status_map[test] = TestStatus.FAILED.value
        elif line.startswith("FAIL:"):
            test = line.split()[1].strip()
            test_status_map[test] = TestStatus.FAILED.value
            current_test = None
        elif line.endswith(" ... ERROR"):
            test = line.split(" ... ERROR")[0]
            if current_test:
                test = current_test
                current_test = None
            if test:
                test_status_map[test] = TestStatus.ERROR.value
        elif line.startswith("ERROR:"):
            test = line.split()[1].strip()
            test_status_map[test] = TestStatus.ERROR.value
            current_test = None

    return test_status_map


# Map repo to parser (from auto-code-rover/SWE-bench)
REPO_PARSER_MAP = {
    "astropy": parse_log_pytest,
    "django": parse_log_django,
    "flask": parse_log_pytest,
    "matplotlib": parse_log_pytest,
    "pylint": parse_log_pytest,
    "pytest": parse_log_pytest,
    "requests": parse_log_pytest,
    "scikit-learn": parse_log_pytest,
    "seaborn": parse_log_pytest,
    "sphinx": parse_log_pytest,
    "sympy": parse_log_pytest,
    "xarray": parse_log_pytest,
}


def get_parser_for_repo(repo_name: str):
    """Get appropriate log parser for repository."""
    for key in REPO_PARSER_MAP:
        if key in repo_name.lower():
            return REPO_PARSER_MAP[key]
    return parse_log_pytest  # Default


def convert_test_name_for_django(test_name: str) -> str:
    """
    Convert test name from SWE-bench format to Django format.

    Example:
        "test_override_file_upload_permissions (test_utils.tests.OverrideSettingsTests)"
        -> "test_utils.tests.OverrideSettingsTests.test_override_file_upload_permissions"
    """
    # Extract test method and class
    if ' (' in test_name and ')' in test_name:
        method = test_name.split(' (')[0]
        class_path = test_name.split(' (')[1].replace(')', '')
        return f"{class_path}.{method}"
    return test_name


def run_tests(env_name: str, repo_path: str, test_cmd: str = "pytest",
              test_list: list = None, repo_name: str = "") -> Tuple[str, int]:
    """
    Run tests in conda environment.

    Args:
        env_name: Conda environment name
        repo_path: Path to repository
        test_cmd: Base test command
        test_list: List of test names to run
        repo_name: Repository name (to determine Django vs others)

    Returns:
        Tuple of (output, returncode)
    """
    # Handle Django specially
    if 'django' in repo_name.lower() and test_list:
        # Convert test names to Django format, filtering out invalid ones
        django_tests = []
        for t in test_list:
            converted = convert_test_name_for_django(t)
            # Skip test names that are not valid test identifiers
            # (e.g., "assertRaisesMessage shouldn't interpret RE special chars.")
            if not converted.replace('.', '').replace('_', '').isalnum():
                continue
            django_tests.append(converted)

        # Django: cd tests && python runtests.py test1 test2...
        if 'cd tests' in test_cmd and django_tests:
            # Write test names to a temp file to avoid shell escaping issues
            test_file = Path(repo_path) / "tests" / ".swe_test_list.txt"
            with open(test_file, 'w') as f:
                for test in django_tests:
                    f.write(f"{test}\n")

            # Modify command to read from file
            # For now, just pass tests directly but with proper escaping
            test_args = ' '.join(f"'{t}'" for t in django_tests)
            full_cmd = f"{test_cmd} {test_args}"
        else:
            full_cmd = test_cmd
    elif '-' not in test_cmd:
        # Not Django, no flags in test_cmd
        full_cmd = f"{test_cmd} -xvs"
    else:
        # Command already has flags
        full_cmd = test_cmd

    print(f"Running tests: {full_cmd[:200]}...")

    # Run in conda environment
    result = run_string_cmd_in_conda(
        full_cmd,
        env_name,
        capture_output=True,
        text=True,
        cwd=repo_path,
        timeout=600  # 10 minutes timeout
    )

    return result.stdout + result.stderr, result.returncode


def evaluate_patch(
    instance_id: str,
    patch_content: str,
    dataset: str = "swe_bench_lite",
    use_cache: bool = False
) -> Dict:
    """
    Evaluate a patch on an instance.

    Args:
        instance_id: Instance ID
        patch_content: Patch content (diff format)
        dataset: Dataset name
        use_cache: If True, keep environment; if False, delete after eval

    Returns:
        Evaluation result dict
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {instance_id}")
    print(f"Dataset: {dataset}")
    print(f"Use cache: {use_cache}")
    print(f"{'='*60}")

    # Step 1: Load instance data
    instance_data = load_instance_data(instance_id, dataset)
    if not instance_data:
        return {"error": "instance_not_found", "instance_id": instance_id}

    # Step 2: Load configs
    setup_map, tasks_map = load_configs(dataset)

    # Handle ID format differences
    _, dash_format = normalize_instance_id(instance_id)

    if dash_format not in setup_map:
        print(f"Instance {dash_format} not found in setup_map")
        return {"error": "setup_not_found", "instance_id": instance_id}

    setup = setup_map[dash_format]
    task = tasks_map[dash_format]

    # Step 3: Ensure environment ready
    print(f"\n[1/5] Ensuring environment ready...")
    env_name = ensure_env_ready(dash_format, dataset=dataset)

    if not env_name:
        return {"error": "env_setup_failed", "instance_id": instance_id}

    print(f"Environment ready: {env_name}")

    try:
        # Step 4: Reset repository
        print(f"\n[2/5] Resetting repository to commit {task['base_commit'][:8]}...")
        repo_path = Path(setup['repo_path'])

        with cd(repo_path):
            repo_reset_and_clean_checkout(task['base_commit'])

        print(f"Repository reset")

        # Step 5: Apply patches and run tests
        print(f"\n[3/5] Applying patches and running tests...")

        # Get test lists
        fail_to_pass = task.get('FAIL_TO_PASS', [])
        pass_to_pass = task.get('PASS_TO_PASS', [])
        test_patch = task.get('test_patch', '')

        print(f"FAIL_TO_PASS tests: {len(fail_to_pass)}")
        print(f"PASS_TO_PASS tests: {len(pass_to_pass)}")

        # Apply test_patch first (filters which tests to run)
        with apply_patch(test_patch, str(repo_path)) as test_patch_applied:
            if not test_patch_applied:
                print(f"Test patch may not have applied cleanly")

            # Then apply solution patch
            with apply_patch(patch_content, str(repo_path)) as patch_applied:
                if not patch_applied:
                    print(f"Solution patch may not have applied cleanly")

                # Run tests with specific test names
                repo_name = task['repo'].split('/')[-1]
                test_output, returncode = run_tests(
                    env_name,
                    str(repo_path),
                    setup.get('test_cmd', 'pytest'),
                    test_list=fail_to_pass + pass_to_pass,
                    repo_name=repo_name
                )

        # Step 6: Parse results
        print(f"\n[4/5] Parsing test results...")

        # Save test output for debugging
        with open(f"/tmp/test_output_{instance_id.replace('/', '_')}.log", 'w') as f:
            f.write(test_output)
        print(f"Test output saved to /tmp/test_output_{instance_id.replace('/', '_')}.log")

        repo_name = task['repo'].split('/')[-1]
        parser = get_parser_for_repo(repo_name)
        test_status_map = parser(test_output)

        # Calculate metrics (matching auto-code-rover)
        fail_to_pass_passed = 0
        fail_to_pass_failed = 0
        pass_to_pass_passed = 0
        pass_to_pass_failed = 0

        for test in fail_to_pass:
            status = test_status_map.get(test, TestStatus.ERROR.value)
            if status == TestStatus.PASSED.value:
                fail_to_pass_passed += 1
            else:
                fail_to_pass_failed += 1

        for test in pass_to_pass:
            status = test_status_map.get(test, TestStatus.PASSED.value)
            if status == TestStatus.PASSED.value:
                pass_to_pass_passed += 1
            else:
                pass_to_pass_failed += 1

        # Determine resolution status
        if fail_to_pass_passed == len(fail_to_pass) and pass_to_pass_failed == 0:
            resolved_status = ResolvedStatus.FULL.value
        elif fail_to_pass_passed > 0:
            resolved_status = ResolvedStatus.PARTIAL.value
        else:
            resolved_status = ResolvedStatus.NO.value

        # Build result
        result = {
            "instance_id": instance_id,
            "resolved": resolved_status,
            "fail_to_pass": {
                "total": len(fail_to_pass),
                "passed": fail_to_pass_passed,
                "failed": fail_to_pass_failed,
            },
            "pass_to_pass": {
                "total": len(pass_to_pass),
                "passed": pass_to_pass_passed,
                "failed": pass_to_pass_failed,
            },
            "test_status_map": test_status_map,
        }

        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Resolution: {resolved_status}")
        print(f"FAIL_TO_PASS: {fail_to_pass_passed}/{len(fail_to_pass)} passed")
        print(f"PASS_TO_PASS: {pass_to_pass_passed}/{len(pass_to_pass)} passed")

        return result

    finally:
        # Step 7: Cleanup if not using cache
        if not use_cache:
            print(f"\n[5/5] Cleaning up environment...")
            cleanup_env(env_name)
            print(f"Environment removed")
        else:
            print(f"\n[5/5] Environment cached for future use")


def main():
    parser = argparse.ArgumentParser(
        description="Local SWE-bench Evaluator with On-Demand Environments"
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["swe_bench_lite", "swe_bench_verified"],
        help="Dataset name"
    )

    parser.add_argument(
        "--instance_id",
        required=True,
        help="Instance ID (e.g., django__django-11333)"
    )

    parser.add_argument(
        "--patch_file",
        type=str,
        help="Path to patch file (diff format)"
    )

    parser.add_argument(
        "--use_gt",
        action="store_true",
        help="Use ground truth patch from test.json for testing"
    )

    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Keep environment after evaluation (default: delete)"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)"
    )

    args = parser.parse_args()

    # Load patch
    if args.use_gt:
        print("Using ground truth patch from test.json...")
        instance_data = load_instance_data(args.instance_id, args.dataset)

        if not instance_data:
            print("Failed to load instance data")
            return 1

        patch_content = instance_data.get("gt", "")
        if not patch_content:
            print("No ground truth patch found in instance data")
            return 1

        print(f"Loaded ground truth patch ({len(patch_content)} bytes)")

    elif args.patch_file:
        patch_file = Path(args.patch_file)
        if not patch_file.exists():
            print(f"Patch file not found: {patch_file}")
            return 1

        with open(patch_file) as f:
            patch_content = f.read()

        print(f"Loaded patch from {patch_file} ({len(patch_content)} bytes)")

    else:
        print("Must provide either --patch_file or --use_gt")
        return 1

    # Evaluate
    result = evaluate_patch(
        args.instance_id,
        patch_content,
        args.dataset,
        args.use_cache
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\nResults saved to {output_path}")

    # Print summary
    if "error" in result:
        print(f"\nEvaluation failed: {result['error']}")
        return 1

    print(f"\nEvaluation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
