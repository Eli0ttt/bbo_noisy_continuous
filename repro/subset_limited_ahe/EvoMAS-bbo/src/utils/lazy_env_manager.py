#!/usr/bin/env python3
"""
Lazy Environment Manager - Create conda environments on-demand during evaluation

This is the recommended approach: create environments only when needed,
rather than pre-building all 300-500 environments.

Usage:
    from lazy_env_manager import ensure_env_ready

    # In your evaluation code:
    instance_id = "django__django-11333"
    env_name = ensure_env_ready(instance_id, dataset="swe_bench_lite")

    # Now use the environment
    result = run_string_cmd_in_conda(test_cmd, env_name, ...)
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import threading

# Global lock for thread-safe environment creation
_env_lock = threading.Lock()
_created_envs = set()  # Track which envs we've already created

_EVOMAS_DIR = Path(os.environ.get("EVOMAS_DIR", str(Path(".").resolve())))


def load_configs(dataset_name: str) -> Tuple[dict, dict]:
    """Load setup_map and tasks_map for a dataset."""
    dataset_dir = _EVOMAS_DIR / "dataset" / dataset_name

    with open(dataset_dir / "setup_map.json") as f:
        setup_map = json.load(f)

    with open(dataset_dir / "tasks_map.json") as f:
        tasks_map = json.load(f)

    return setup_map, tasks_map


def env_exists(env_name: str) -> bool:
    """Check if a conda environment exists."""
    result = subprocess.run(
        ["conda", "env", "list"],
        capture_output=True,
        text=True
    )
    return env_name in result.stdout


def create_conda_env(env_name: str, python_version: str = "3.9") -> bool:
    """Create a conda environment."""
    print(f"Creating environment: {env_name}")

    try:
        result = subprocess.run(
            ["conda", "create", "-n", env_name, f"python={python_version}", "-y"],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            print(f"Warning: Environment creation had issues: {result.stderr[:200]}")
            return False

        return True

    except Exception as e:
        print(f"Error creating environment: {e}")
        return False


def install_base_packages(env_name: str) -> bool:
    """Install base testing and build packages."""
    # Test packages
    test_packages = ["pytest", "pytest-cov", "coverage", "xmlrunner", "decorator", "hypothesis"]

    # Build dependencies needed by many repos (astropy, scipy, etc.)
    # Pin numpy<2.0 for compatibility with older packages that use numpy.core
    build_packages = [
        "numpy<2.0",
        "cython",
        "setuptools",
        "setuptools_scm",
        "wheel",
        "extension-helpers",
        "pyerfa",  # Required by astropy
        "pyyaml",
        "packaging",
    ]

    try:
        # Install build packages first (needed for pip install -e .)
        subprocess.run(
            ["conda", "run", "-n", env_name, "pip", "install"] + build_packages,
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Then install test packages
        subprocess.run(
            ["conda", "run", "-n", env_name, "pip", "install"] + test_packages,
            capture_output=True,
            text=True,
            timeout=300,
            check=True
        )
        return True
    except Exception as e:
        print(f"Warning: Package installation had issues: {e}")
        return True  # Continue anyway


def setup_repository(env_name: str, repo_path: str, commit: str,
                     pre_install: list, install_cmd: str) -> bool:
    """Setup repository in the environment."""
    repo_path = Path(repo_path)

    if not repo_path.exists():
        print(f"Warning: Repository not found at {repo_path}")
        return False

    # Checkout commit
    try:
        subprocess.run(
            ["git", "checkout", commit],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=60,
            check=True
        )
    except Exception as e:
        print(f"Warning: Git checkout failed: {e}")
        return False

    # Pre-install commands
    for cmd in pre_install:
        if not cmd:
            continue

        try:
            full_cmd = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {env_name} && cd {repo_path} && {cmd}"
            subprocess.run(
                ["bash", "-c", full_cmd],
                capture_output=True,
                text=True,
                timeout=600
            )
        except Exception as e:
            print(f"Warning: Pre-install command failed: {e}")

    # Main install
    if install_cmd:
        try:
            full_cmd = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {env_name} && cd {repo_path} && {install_cmd}"
            result = subprocess.run(
                ["bash", "-c", full_cmd],
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode != 0:
                print(f"Warning: Install command failed: {result.stderr[:200]}")
                return False
        except Exception as e:
            print(f"Warning: Install failed: {e}")
            return False

    return True


def ensure_env_ready(instance_id: str, dataset: str = "swe_bench_lite",
                     force_recreate: bool = False) -> Optional[str]:
    """
    Ensure environment is ready for an instance.
    Creates environment on-demand if it doesn't exist.

    Args:
        instance_id: Instance ID (e.g., "django__django-11333")
        dataset: Dataset name ("swe_bench_lite" or "swe_bench_verified")
        force_recreate: If True, recreate even if exists

    Returns:
        Environment name if successful, None if failed

    Example:
        >>> env_name = ensure_env_ready("django__django-11333")
        >>> if env_name:
        ...     result = run_string_cmd_in_conda(test_cmd, env_name, ...)
    """
    # Load configs
    setup_map, tasks_map = load_configs(dataset)

    if instance_id not in setup_map:
        print(f"Error: Instance {instance_id} not found in setup_map")
        return None

    setup = setup_map[instance_id]
    task = tasks_map[instance_id]

    env_name = setup['env_name']

    # Thread-safe check and creation
    with _env_lock:
        # Check if we've already created this environment in this session
        if env_name in _created_envs and not force_recreate:
            if env_exists(env_name):
                return env_name

        # Check if environment exists
        if env_exists(env_name) and not force_recreate:
            print(f"Environment {env_name} already exists")
            _created_envs.add(env_name)
            return env_name

        # Create environment on-demand
        print(f"Setting up environment on-demand for {instance_id}...")

        # Step 1: Create conda environment
        python_version = setup.get('python', '3.9')
        if not create_conda_env(env_name, python_version):
            print(f"Failed to create environment for {instance_id}")
            return None

        # Step 2: Install base packages
        install_base_packages(env_name)

        # Step 3: Setup repository
        setup_repository(
            env_name,
            setup['repo_path'],
            task['base_commit'],
            setup.get('pre_install', []),
            setup.get('install', '')
        )

        # Mark as created
        _created_envs.add(env_name)
        print(f"✓ Environment {env_name} ready")

        return env_name


def cleanup_env(env_name: str) -> bool:
    """
    Remove a conda environment to free disk space.

    Args:
        env_name: Environment name to remove

    Returns:
        True if successful
    """
    try:
        subprocess.run(
            ["conda", "env", "remove", "-n", env_name, "-y"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True
        )
        _created_envs.discard(env_name)
        return True
    except Exception as e:
        print(f"Failed to remove environment {env_name}: {e}")
        return False


def get_env_status(dataset: str = "swe_bench_lite") -> dict:
    """
    Get status of environments for a dataset.

    Returns:
        Dict with 'total', 'created', 'missing' counts and lists
    """
    setup_map, _ = load_configs(dataset)

    # Get all conda environments
    result = subprocess.run(
        ["conda", "env", "list"],
        capture_output=True,
        text=True
    )
    conda_envs = set()
    for line in result.stdout.split('\n'):
        parts = line.strip().split()
        if parts and not parts[0].startswith('#'):
            conda_envs.add(parts[0])

    created = []
    missing = []

    for instance_id, setup in setup_map.items():
        env_name = setup['env_name']
        if env_name in conda_envs:
            created.append(instance_id)
        else:
            missing.append(instance_id)

    return {
        'total': len(setup_map),
        'created': len(created),
        'missing': len(missing),
        'created_list': created,
        'missing_list': missing
    }


# Example usage
if __name__ == "__main__":
    # Test lazy environment creation
    instance_id = "django__django-11333"

    print(f"Testing lazy environment creation for {instance_id}...")
    env_name = ensure_env_ready(instance_id, dataset="swe_bench_lite")

    if env_name:
        print(f"✓ Environment ready: {env_name}")

        # Check environment status
        status = get_env_status("swe_bench_lite")
        print(f"\nEnvironment status:")
        print(f"  Total instances: {status['total']}")
        print(f"  Environments created: {status['created']}")
        print(f"  Environments missing: {status['missing']}")
    else:
        print("✗ Failed to setup environment")
