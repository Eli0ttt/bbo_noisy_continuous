#!/usr/bin/env python3
"""
Build conda environments for SWE-bench instances.

This script creates per-instance conda environments similar to how
auto-code-rover/ExpeRepair expect them to be set up.

Usage:
    # Build environments for specific instances
    python build_envs.py --dataset swe_bench_lite --instances django__django-11333 astropy__astropy-12907

    # Build environments for all instances (WARNING: Takes a long time!)
    python build_envs.py --dataset swe_bench_lite --all

    # Build from a file
    python build_envs.py --dataset swe_bench_lite --instances-file instances.txt
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Paths — resolve relative to project root, overridable via env var
EVOMAS_DIR = Path(os.environ.get("EVOMAS_DIR", str(Path(".").resolve())))
DATASET_DIR = EVOMAS_DIR / "dataset"


def load_configs(dataset_name: str):
    """Load setup_map.json and tasks_map.json."""
    dataset_dir = DATASET_DIR / dataset_name

    setup_map_path = dataset_dir / "setup_map.json"
    tasks_map_path = dataset_dir / "tasks_map.json"

    if not setup_map_path.exists():
        print(f"setup_map.json not found at {setup_map_path}")
        print(f"Please run generate_configs_from_local.py first")
        return None, None

    if not tasks_map_path.exists():
        print(f"tasks_map.json not found at {tasks_map_path}")
        return None, None

    with open(setup_map_path) as f:
        setup_map = json.load(f)

    with open(tasks_map_path) as f:
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
    if env_exists(env_name):
        print(f"  Environment '{env_name}' already exists, skipping...")
        return True

    print(f"  Creating conda environment '{env_name}' with Python {python_version}...")

    try:
        result = subprocess.run(
            ["conda", "create", "-n", env_name, f"python={python_version}", "-y"],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            print(f"  Failed to create environment: {result.stderr}")
            return False

        print(f"  Environment created")
        return True

    except subprocess.TimeoutExpired:
        print(f"  Environment creation timed out")
        return False
    except Exception as e:
        print(f"  Error creating environment: {e}")
        return False


def install_base_packages(env_name: str) -> bool:
    """Install base testing packages in the environment."""
    packages = [
        "pytest",
        "pytest-cov",
        "coverage",
        "xmlrunner",
        "decorator",
    ]

    print(f"  Installing base packages: {', '.join(packages)}...")

    try:
        result = subprocess.run(
            ["conda", "run", "-n", env_name, "pip", "install"] + packages,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            print(f"  Warning: Some packages may have failed to install")
            print(f"  {result.stderr[:200]}")
            return True  # Continue anyway

        print(f"  Base packages installed")
        return True

    except subprocess.TimeoutExpired:
        print(f"  Package installation timed out")
        return True  # Continue anyway
    except Exception as e:
        print(f"  Error installing packages: {e}")
        return True  # Continue anyway


def setup_repo(instance_id: str, setup: dict, task: dict) -> bool:
    """Checkout repository to correct commit and install dependencies."""
    repo_path = Path(setup['repo_path'])
    env_name = setup['env_name']
    commit = task['base_commit']

    if not repo_path.exists():
        print(f"  Repository not found at {repo_path}")
        return False

    print(f"  Checking out commit {commit[:8]}...")

    # Checkout commit
    try:
        result = subprocess.run(
            ["git", "checkout", commit],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"  Git checkout failed: {result.stderr[:200]}")
            return False

    except Exception as e:
        print(f"  Error during checkout: {e}")
        return False

    # Run pre-install commands
    for cmd in setup.get('pre_install', []):
        if not cmd:
            continue

        print(f"  Running pre-install: {cmd[:60]}...")
        try:
            # Use bash -c with conda activation
            full_cmd = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {env_name} && cd {repo_path} && {cmd}"

            result = subprocess.run(
                ["bash", "-c", full_cmd],
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode != 0:
                print(f"  Pre-install command failed: {result.stderr[:200]}")
                # Continue anyway

        except subprocess.TimeoutExpired:
            print(f"  Pre-install command timed out")
        except Exception as e:
            print(f"  Error during pre-install: {e}")

    # Run main install command
    install_cmd = setup.get('install', '')
    if install_cmd:
        print(f"  Running install: {install_cmd[:60]}...")
        try:
            full_cmd = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {env_name} && cd {repo_path} && {install_cmd}"

            result = subprocess.run(
                ["bash", "-c", full_cmd],
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode != 0:
                print(f"  Install command failed: {result.stderr[:200]}")
                return False

            print(f"  Repository installed")

        except subprocess.TimeoutExpired:
            print(f"  Install command timed out")
            return False
        except Exception as e:
            print(f"  Error during install: {e}")
            return False

    return True


def build_environment(instance_id: str, setup_map: dict, tasks_map: dict, skip_repo_setup: bool = False) -> bool:
    """Build environment for a single instance."""
    if instance_id not in setup_map:
        print(f"Instance {instance_id} not found in setup_map")
        return False

    setup = setup_map[instance_id]
    task = tasks_map[instance_id]

    env_name = setup['env_name']
    python_version = setup.get('python', '3.9')

    print(f"\n{'='*60}")
    print(f"Building environment for: {instance_id}")
    print(f"{'='*60}")
    print(f"  Environment: {env_name}")
    print(f"  Python: {python_version}")
    print(f"  Repository: {setup['repo_path']}")

    # Step 1: Create conda environment
    if not create_conda_env(env_name, python_version):
        return False

    # Step 2: Install base packages
    if not install_base_packages(env_name):
        print(f"  Continuing despite package installation issues...")

    # Step 3: Setup repository
    if not skip_repo_setup:
        if not setup_repo(instance_id, setup, task):
            print(f"  Repository setup had issues, but environment is created")
            return True  # Environment exists, just repo setup failed

    print(f"  Environment built successfully")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Build conda environments for SWE-bench instances"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["swe_bench_lite", "swe_bench_verified"],
        help="Dataset name"
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        help="Specific instance IDs to build environments for"
    )
    parser.add_argument(
        "--instances-file",
        type=str,
        help="File containing instance IDs (one per line)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build environments for ALL instances (WARNING: Takes a long time!)"
    )
    parser.add_argument(
        "--skip-repo-setup",
        action="store_true",
        help="Only create conda environments, skip repository setup"
    )

    args = parser.parse_args()

    # Load configs
    print(f"Loading configs for {args.dataset}...")
    setup_map, tasks_map = load_configs(args.dataset)

    if setup_map is None or tasks_map is None:
        return 1

    print(f"Loaded {len(setup_map)} instances")

    # Determine which instances to build
    instances_to_build = []

    if args.all:
        instances_to_build = list(setup_map.keys())
        print(f"\nWARNING: Building environments for ALL {len(instances_to_build)} instances!")
        print(f"This will take several hours and use ~{len(instances_to_build) * 0.5:.1f} GB disk space")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return 0

    elif args.instances_file:
        instances_file = Path(args.instances_file)
        if not instances_file.exists():
            print(f"File not found: {instances_file}")
            return 1

        with open(instances_file) as f:
            instances_to_build = [line.strip() for line in f if line.strip()]

        print(f"Read {len(instances_to_build)} instances from {instances_file}")

    elif args.instances:
        instances_to_build = args.instances

    else:
        print("Error: Must specify --instances, --instances-file, or --all")
        return 1

    # Validate instances
    invalid_instances = [i for i in instances_to_build if i not in setup_map]
    if invalid_instances:
        print(f"\nWarning: {len(invalid_instances)} instances not found in setup_map:")
        for inst in invalid_instances[:5]:
            print(f"  - {inst}")
        if len(invalid_instances) > 5:
            print(f"  ... and {len(invalid_instances) - 5} more")

    instances_to_build = [i for i in instances_to_build if i in setup_map]

    if not instances_to_build:
        print("No valid instances to build")
        return 1

    print(f"\nBuilding environments for {len(instances_to_build)} instances")

    # Build environments
    success_count = 0
    failed_instances = []

    for idx, instance_id in enumerate(instances_to_build, 1):
        print(f"\n[{idx}/{len(instances_to_build)}]")

        success = build_environment(
            instance_id,
            setup_map,
            tasks_map,
            skip_repo_setup=args.skip_repo_setup
        )

        if success:
            success_count += 1
        else:
            failed_instances.append(instance_id)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {len(instances_to_build)}")
    print(f"Success: {success_count}")
    print(f"Failed: {len(failed_instances)}")

    if failed_instances:
        print(f"\nFailed instances:")
        for inst in failed_instances[:10]:
            print(f"  - {inst}")
        if len(failed_instances) > 10:
            print(f"  ... and {len(failed_instances) - 10} more")

    return 0 if len(failed_instances) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
