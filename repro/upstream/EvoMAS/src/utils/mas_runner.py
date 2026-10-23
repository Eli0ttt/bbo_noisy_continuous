#!/usr/bin/env python3
"""
MAS Runner - A reusable runtime sandbox for running MAS on different datasets.

This module provides a flexible class-based approach to create and run multi-agent systems
with different configurations and datasets, replacing the need for individual run_* scripts.
"""

import logging
import fcntl
import os
import signal
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import time

logger = logging.getLogger(__name__)

REPOS_DIR = os.environ.get(
    "EVOMAS_REPOS_DIR",
    str(Path("dataset/repos").resolve())
)


class RepoLock:
    """
    File-based lock for repository access to prevent concurrent modifications.
    Uses flock for cross-process synchronization.
    """

    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        # Create lock file in /tmp to avoid polluting the repo
        safe_name = repo_path.replace('/', '_').replace(' ', '_')
        self.lock_file = f"/tmp/evomas_repo_lock_{safe_name}.lock"
        self._lock_fd = None

    def acquire(self, timeout: float = 300.0) -> bool:
        """
        Acquire the lock with timeout.

        Args:
            timeout: Maximum time to wait for lock (seconds)

        Returns:
            True if lock acquired, False if timeout
        """
        import time as time_module
        start_time = time_module.time()

        # Create lock file if it doesn't exist
        self._lock_fd = open(self.lock_file, 'w')

        while True:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                logger.debug(f"Acquired lock for repo: {self.repo_path}")
                return True
            except IOError:
                if time_module.time() - start_time >= timeout:
                    logger.warning(f"Timeout waiting for repo lock: {self.repo_path}")
                    self._lock_fd.close()
                    self._lock_fd = None
                    return False
                time_module.sleep(0.5)

    def release(self):
        """Release the lock."""
        if self._lock_fd:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                self._lock_fd.close()
                logger.debug(f"Released lock for repo: {self.repo_path}")
            except Exception as e:
                logger.warning(f"Error releasing lock: {e}")
            finally:
                self._lock_fd = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


class WorkbenchDomainLock:
    """Per-domain fcntl lock for WorkBench tasks.

    WorkBench tool modules (src/tools/workbench_tools_*.py) keep domain data
    in module-level globals (EMAILS, CALENDAR_EVENTS, CRM_DATA, ...). Because
    Python caches modules, every thread in the process sees the same state.
    Two concurrent tasks on the same domain would race on those globals and
    on the backing dataset/workbench/<domain>/data.csv, producing silent
    corruption.

    This lock serializes per-domain task execution. Tasks on *different*
    domains still run concurrently. Multi-domain tasks acquire all domain
    locks (in sorted order to prevent deadlock).

    Uses fcntl file locks so the protection is both thread-safe and
    process-safe, matching the RepoLock pattern used for SWE-bench.
    """

    _ALL_DOMAINS = (
        "analytics",
        "calendar",
        "customer_relationship_manager",
        "email",
        "project_management",
    )

    @classmethod
    def domains_for_dataset(cls, dataset_name: str):
        dsn = (dataset_name or "").lower()
        if "workbench" not in dsn:
            return []
        if "multi_domain" in dsn:
            return list(cls._ALL_DOMAINS)
        for d in cls._ALL_DOMAINS:
            if d in dsn:
                return [d]
        return []

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.domains = WorkbenchDomainLock.domains_for_dataset(dataset_name)
        self._held: List = []  # list of (domain, fd)

    def acquire(self, timeout: float = 300.0) -> bool:
        """Acquire all domain locks this dataset needs, in sorted order."""
        import time as time_module
        if not self.domains:
            return True  # nothing to lock — not a workbench dataset
        start = time_module.time()
        for domain in sorted(self.domains):
            lock_file = f"/tmp/evomas_workbench_lock_{domain}.lock"
            fd = open(lock_file, 'w')
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._held.append((domain, fd))
                    break
                except IOError:
                    if time_module.time() - start >= timeout:
                        logger.warning(f"Timeout waiting for workbench lock: {domain}")
                        fd.close()
                        self.release()
                        return False
                    time_module.sleep(0.5)
        return True

    def release(self):
        while self._held:
            domain, fd = self._held.pop()
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
                fd.close()
                logger.debug(f"Released workbench lock for domain: {domain}")
            except Exception as e:
                logger.warning(f"Error releasing workbench lock for {domain}: {e}")

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


def cleanup_temp_files(max_age_hours: float = 24.0, dry_run: bool = False) -> Dict[str, int]:
    """
    Clean up stale EvoMAS temporary files and directories.

    Args:
        max_age_hours: Maximum age in hours before files are considered stale
        dry_run: If True, only report what would be deleted without actually deleting

    Returns:
        Dict with 'files_deleted', 'dirs_deleted', 'bytes_freed' counts
    """
    import shutil
    import glob

    stats = {'files_deleted': 0, 'dirs_deleted': 0, 'bytes_freed': 0}
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    # Patterns for EvoMAS temp files
    patterns = [
        '/tmp/evomas_*',
        '/tmp/miniswe_task_*',
        '/tmp/sweagent_task_*',
        '/tmp/test_output_*.log',
    ]

    for pattern in patterns:
        for path_str in glob.glob(pattern):
            path = Path(path_str)
            try:
                # Check file age
                mtime = path.stat().st_mtime
                age = current_time - mtime

                if age > max_age_seconds:
                    if path.is_dir():
                        size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                        if dry_run:
                            logger.info(f"Would delete directory: {path} ({size} bytes, {age/3600:.1f}h old)")
                        else:
                            shutil.rmtree(path)
                            logger.info(f"Deleted directory: {path}")
                        stats['dirs_deleted'] += 1
                        stats['bytes_freed'] += size
                    else:
                        size = path.stat().st_size
                        if dry_run:
                            logger.info(f"Would delete file: {path} ({size} bytes, {age/3600:.1f}h old)")
                        else:
                            path.unlink()
                            logger.info(f"Deleted file: {path}")
                        stats['files_deleted'] += 1
                        stats['bytes_freed'] += size

            except Exception as e:
                logger.warning(f"Error processing {path}: {e}")

    # Also clean up stale repo lock files
    for lock_file in glob.glob('/tmp/evomas_repo_lock_*.lock'):
        lock_path = Path(lock_file)
        try:
            mtime = lock_path.stat().st_mtime
            age = current_time - mtime
            if age > max_age_seconds:
                if dry_run:
                    logger.info(f"Would delete stale lock: {lock_path}")
                else:
                    lock_path.unlink()
                    logger.info(f"Deleted stale lock: {lock_path}")
                stats['files_deleted'] += 1
        except Exception as e:
            logger.warning(f"Error processing lock file {lock_path}: {e}")

    return stats


# Global variables for worker processes
WORKER_RUNNER = None
WORKER_CONFIG_PATH = None
WORKER_DATASET_NAME = None
WORKER_DATASET_SPLIT = None
WORKER_VERBOSE = None
WORKER_SAVE_OUTPUTS = None
WORKER_OUTPUT_DIR = None
WORKER_USE_CACHE = None


@dataclass
class MasRunResult:
    """Result from running MAS on a task."""
    task_id: str
    result: Optional[str] = None
    ground_truth: Optional[Any] = None
    correct: Optional[bool] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


def init_mas_worker(config_path: str, dataset_name: str, dataset_split: str,
                    verbose: bool, save_outputs: bool, output_dir: str, use_cache: bool):
    """Initialize worker process with MasRunner instance."""
    global WORKER_RUNNER, WORKER_CONFIG_PATH, WORKER_DATASET_NAME, WORKER_DATASET_SPLIT
    global WORKER_VERBOSE, WORKER_SAVE_OUTPUTS, WORKER_OUTPUT_DIR, WORKER_USE_CACHE

    WORKER_CONFIG_PATH = config_path
    WORKER_DATASET_NAME = dataset_name
    WORKER_DATASET_SPLIT = dataset_split
    WORKER_VERBOSE = verbose
    WORKER_SAVE_OUTPUTS = save_outputs
    WORKER_OUTPUT_DIR = output_dir
    WORKER_USE_CACHE = use_cache

    # Create MasRunner once per worker
    WORKER_RUNNER = MasRunner(
        config_path=config_path,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        verbose=False,  # Reduce verbosity in workers
        save_individual_outputs=save_outputs,
        output_dir=output_dir,
        skip_evaluation=False,
        use_cache=use_cache
    )

    worker_logger = logging.getLogger(f"mas_worker_init")
    worker_logger.info(f"MAS worker initialized with {config_path}")


def process_single_task_parallel(args):
    """Process a single task in a worker process."""
    task, worker_id = args
    global WORKER_RUNNER

    worker_logger = logging.getLogger(f"mas_worker_{worker_id}")

    try:
        # Run single task using the initialized runner
        result = WORKER_RUNNER.run_single_task(task.id)
        worker_logger.info(f"Worker {worker_id}: Completed task {task.id}")
        return result
    except Exception as e:
        worker_logger.error(f"Worker {worker_id}: Error processing task {task.id}: {e}")
        return MasRunResult(
            task_id=task.id,
            result=None,
            ground_truth=task.gt if hasattr(task, 'gt') else None,
            correct=False,
            error=str(e)
        )


class MasRunner:
    """
    A reusable runtime sandbox for running MAS on different datasets.

    This class encapsulates the entire pipeline:
    1. Load MAS configuration
    2. Load dataset
    3. Initialize runtime
    4. Run tasks
    5. Evaluate results

    Example:
        # Basic usage
        runner = MasRunner(
            config_path="mas_pools/bbeh/majority_vote.yaml",
            dataset_name="bbeh_word_sorting"
        )
        results = runner.run(num_tasks=3)

        # Run specific task
        result = runner.run_single_task(task_id=0)

        # Custom configuration
        runner = MasRunner(
            config_path="path/to/config.yaml",
            dataset_name="custom_dataset",
            dataset_split="validation",
            log_level=logging.DEBUG
        )
        results = runner.run()
    """

    def __init__(
        self,
        config_path: Union[str, Path],
        dataset_name: str,
        dataset_split: str = "test",
        log_level: int = logging.INFO,
        verbose: bool = True,
        save_individual_outputs: bool = True,  # Default to True now
        output_dir: Optional[Union[str, Path]] = None,
        skip_evaluation: bool = True,  # Default to skip eval during run
        use_cache: bool = True,  # Enable caching by default
        llm_as_judge: Optional[str] = None,  # LLM-as-judge model ID
        task_timeout: float = 600.0  # Task timeout in seconds (default: 10 minutes)
    ):
        """
        Initialize MAS runner.

        Args:
            config_path: Path to MAS configuration file
            dataset_name: Name of the dataset to load
            dataset_split: Dataset split to use (default: "test")
            log_level: Logging level (default: INFO)
            verbose: Whether to show detailed logs (default: True)
            save_individual_outputs: Save each task output to separate file (default: True)
            output_dir: Base directory for individual outputs (default: "output")
            skip_evaluation: Skip evaluation during run (use batch evaluator later) (default: True)
            use_cache: Skip tasks that already have output files (default: True)
            task_timeout: Maximum time for a single task in seconds (default: 300 = 5 min)
        """
        self.config_path = Path(config_path)
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.verbose = verbose
        self.save_individual_outputs = save_individual_outputs
        # Resolve output_dir to absolute path to handle working directory changes
        self.output_dir = (Path(output_dir) if output_dir else Path("output")).resolve()
        self.skip_evaluation = skip_evaluation
        self.use_cache = use_cache
        self.llm_as_judge = llm_as_judge
        # Auto-increase timeout for SWE-bench datasets (agents need more time due to code exploration)
        if task_timeout == 600.0 and ('swe' in dataset_name.lower() or dataset_name in ['swebench', 'swebench_lite', 'swebench_verified']):
            self.task_timeout = 1800.0  # 30 minutes for SWE-bench tasks
        else:
            self.task_timeout = task_timeout

        # Configure logging
        if not verbose:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(log_level)

        # Will be initialized lazily
        self._mas_spec = None
        self._runtime = None
        self._dataset = None
        self._evaluator = None

        # Store all metadata for consolidated results.json
        self._all_metadata = []

    def _load_mas_config(self):
        """Load MAS configuration from file."""
        if self._mas_spec is not None:
            return

        if self.verbose:
            logger.info("=" * 80)
            logger.info("EVOMAS - MAS Runner")
            logger.info("=" * 80)
            logger.info("\nLoading MAS configuration...")

        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            from src.mas import load_mas_from_file

            self._mas_spec = load_mas_from_file(str(self.config_path))

            if self.verbose:
                logger.info(f"Loaded MAS: {self._mas_spec.name}")
                logger.info(f"   Backend: {self._mas_spec.backend}")
                logger.info(f"   Agents: {list(self._mas_spec.agents.keys())}")
                logger.info(f"   Execution order: {self._mas_spec.get_execution_order()}")

        except Exception as e:
            logger.error(f"Failed to load MAS configuration: {e}")
            raise

    def _load_dataset(self):
        """Load dataset."""
        if self._dataset is not None:
            return

        if self.verbose:
            logger.info("\nLoading dataset...")

        try:
            from src.dataset import load_dataset

            self._dataset = load_dataset(self.dataset_name, split=self.dataset_split)

            if self.verbose:
                logger.info(f"Loaded dataset: {self.dataset_name}")
                logger.info(f"   Split: {self.dataset_split}")
                logger.info(f"   Total tasks: {len(self._dataset)}")

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def _initialize_runtime(self):
        """Initialize MAS runtime."""
        if self._runtime is not None:
            return

        if self.verbose:
            logger.info("\nInitializing MAS runtime...")

        try:
            from src.mas import MasRuntime

            self._runtime = MasRuntime(self._mas_spec)

            if self.verbose:
                logger.info("Runtime initialized")

        except Exception as e:
            logger.error(f"Failed to initialize runtime: {e}")
            raise

    def _get_evaluator(self):
        """Get appropriate evaluator for dataset.

        If llm_as_judge is configured, uses LLM-as-Judge evaluator as the
        reward signal (used during evolution). When llm_as_judge is None,
        falls back to dataset-specific ground-truth evaluators (used for
        final evaluation).
        """
        if self._evaluator is not None:
            return self._evaluator

        # Use LLM-as-Judge when configured (evolution reward signal)
        if self.llm_as_judge:
            from src.dataset.llm_as_judge import LLMAsJudgeEvaluator
            self._evaluator = LLMAsJudgeEvaluator(
                model_id=self.llm_as_judge,
                dataset_name=self.dataset_name
            )
            if self.verbose:
                logger.info(f"Using LLM-as-Judge evaluator ({self.llm_as_judge}) for {self.dataset_name}")
            return self._evaluator

        # Dataset-specific ground-truth evaluators (final evaluation)
        if 'bbeh' in self.dataset_name.lower() or 'aime' in self.dataset_name.lower():
            from src.dataset import BBEHEvaluator
            self._evaluator = BBEHEvaluator()
            if self.verbose:
                logger.info(f"Using BBEHEvaluator for {self.dataset_name}")

        elif 'swe' in self.dataset_name.lower():
            from src.dataset import SWEBenchEvaluator
            self._evaluator = SWEBenchEvaluator(dataset_name=self.dataset_name, verbose=self.verbose)
            if self.verbose:
                logger.info(f"Using SWEBenchEvaluator for {self.dataset_name}")

        elif 'workbench' in self.dataset_name.lower():
            from src.dataset import WorkBenchEvaluator
            self._evaluator = WorkBenchEvaluator()
            if self.verbose:
                logger.info(f"Using WorkBenchEvaluator for {self.dataset_name}")

        return self._evaluator

    def _get_config_name_with_model(self) -> str:
        """
        Get configuration name with model_id appended for filesystem paths.

        Returns:
            Config name with sanitized model_id (e.g., "single_minisweagent_gpt-4o")
        """
        config_name = self.config_path.stem

        # Get model_id from the first agent in the MAS spec
        if self._mas_spec and self._mas_spec.agents:
            # Get the first agent's model_id
            first_agent = next(iter(self._mas_spec.agents.values()))
            model_id = first_agent.model_id

            # Sanitize model_id for filesystem use
            # Remove provider prefix (e.g., "openai:", "bedrock:", "anthropic:")
            if ":" in model_id:
                model_id = model_id.split(":", 1)[1]

            # Replace special characters with hyphens, keep only alphanumeric and hyphens
            safe_model_id = model_id.replace(".", "-").replace("/", "-").replace("_", "-")
            safe_model_id = "".join(c for c in safe_model_id if c.isalnum() or c == "-")

            # Append to config name
            config_name = f"{config_name}_{safe_model_id}"

        return config_name

    def _checkout_base_commit(self, repo_path: str, base_commit: str) -> bool:
        """
        Checkout the base commit for a SWE-bench task.

        This ensures the repository is at the correct state (before the fix)
        so the agent can apply the patch.

        Args:
            repo_path: Path to the repository
            base_commit: The base commit hash to checkout

        Returns:
            True if checkout successful, False otherwise
        """
        import subprocess

        if not repo_path or not Path(repo_path).exists():
            logger.warning(f"Repository path does not exist: {repo_path}")
            return False

        if not base_commit:
            logger.warning("No base_commit provided")
            return False

        try:
            if self.verbose:
                logger.info(f"Checking out base commit: {base_commit[:8]}...")

            # Remove stale git lock files (from crashed processes)
            git_dir = Path(repo_path) / ".git"
            lock_files = [
                git_dir / "index.lock",
                git_dir / "HEAD.lock",
                git_dir / "config.lock",
            ]
            for lock_file in lock_files:
                if lock_file.exists():
                    lock_file.unlink()
                    logger.debug(f"Removed stale lock file: {lock_file}")

            # CRITICAL: First explicitly remove .sweagent_output directory if it exists
            # This directory can persist between tasks and contaminate submissions
            sweagent_output = Path(repo_path) / ".sweagent_output"
            if sweagent_output.exists():
                import shutil
                shutil.rmtree(sweagent_output, ignore_errors=True)
                logger.debug(f"Removed .sweagent_output directory before checkout")

            # Reset ALL changes (staged and unstaged)
            subprocess.run(
                ["git", "reset", "--hard", "HEAD"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True
            )

            # Clean ALL untracked files including ignored ones
            subprocess.run(
                ["git", "clean", "-fdx"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True
            )

            # Fetch the commit if not available locally
            result = subprocess.run(
                ["git", "cat-file", "-t", base_commit],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logger.info(f"Commit {base_commit[:8]} not found locally, fetching...")
                subprocess.run(
                    ["git", "fetch", "origin", base_commit],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=120
                )

            # Checkout the base commit
            subprocess.run(
                ["git", "checkout", base_commit],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True
            )

            # CRITICAL: Verify that HEAD is now at the expected commit
            verify_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            current_head = verify_result.stdout.strip()

            # Compare the first 40 chars (full SHA) or match prefix
            if not current_head.startswith(base_commit[:8]):
                logger.error(f"Checkout verification failed! Expected {base_commit[:8]}, got {current_head[:8]}")
                return False

            if self.verbose:
                logger.info(f"Verified checkout at commit: {current_head[:8]}")

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to checkout base commit {base_commit[:8]}: {e}")
            if e.stderr:
                logger.error(f"   Error: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Error during checkout: {e}")
            return False

    def _recover_repository(self, repo_path: str) -> bool:
        """
        Recover repository to clean state after task execution.

        This prevents contamination between instances by:
        1. Removing stale git lock files
        2. Resetting all tracked files (git reset --hard)
        3. Removing all untracked files (git clean -fd)

        Args:
            repo_path: Path to the repository to recover

        Returns:
            True if recovery successful, False otherwise
        """
        import subprocess

        if not repo_path or not Path(repo_path).exists():
            logger.warning(f"Repository path does not exist: {repo_path}")
            return False

        # Check if it's a git repository
        git_dir = Path(repo_path) / ".git"
        if not git_dir.exists():
            logger.warning(f"Not a git repository: {repo_path}")
            return False

        try:
            if self.verbose:
                logger.info(f"Recovering repository: {repo_path}")

            # Remove stale git lock files (from crashed processes)
            lock_files = [
                git_dir / "index.lock",
                git_dir / "HEAD.lock",
                git_dir / "config.lock",
            ]
            for lock_file in lock_files:
                if lock_file.exists():
                    lock_file.unlink()
                    logger.debug(f"Removed stale lock file: {lock_file}")

            # Reset all tracked files to HEAD
            subprocess.run(
                ["git", "reset", "--hard", "HEAD"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True
            )

            # Remove all untracked files and directories
            subprocess.run(
                ["git", "clean", "-fd"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True
            )

            if self.verbose:
                logger.info(f"Repository recovered successfully")

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to recover repository: {e}")
            if e.stderr:
                logger.error(f"   Error: {e.stderr}")
            return False

    def _cleanup_sweagent_artifacts(self) -> None:
        """
        Clean up SWE-agent artifacts from home directory and /tmp.

        SWE-agent creates test files like reproduce_issue.py in the home
        directory which can persist between runs and cause issues.
        This method removes these artifacts after each task.
        """
        import glob
        import shutil

        # Patterns for SWE-agent created files in home directories
        home_dirs = [Path.home(), Path("/root")]
        artifact_patterns = [
            "reproduce_issue.py",
            "reproduce_bug.py",
            "test_fix.py",
            "test_*.py",
            "debug_*.py",
            "check_*.py",
            "verify_*.py",
            "run_test*.py",
            "*.patch",
            # Additional patterns for files created by agents
            "*_test.py",
            "explore_*.py",
            "final_*.py",
            "*_demo.py",
            "*_verification.py",
            "*_comprehensive*.py",
            "pr_*.py",
            "edge_case*.py",
        ]

        for home_dir in home_dirs:
            if not home_dir.exists():
                continue

            for pattern in artifact_patterns:
                for filepath in home_dir.glob(pattern):
                    try:
                        if filepath.is_file():
                            filepath.unlink()
                            logger.debug(f"Cleaned up SWE-agent artifact: {filepath}")
                    except Exception as e:
                        logger.warning(f"Failed to remove {filepath}: {e}")

        # Clean up /tmp files created by SWE-agent
        tmp_patterns = [
            "/tmp/sweagent_*",
            "/tmp/swe_agent_*",
            "/tmp/reproduce_*",
            "/tmp/test_fix_*",
        ]

        for pattern in tmp_patterns:
            for filepath in glob.glob(pattern):
                try:
                    path = Path(filepath)
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        path.unlink()
                    logger.debug(f"Cleaned up temp artifact: {filepath}")
                except Exception as e:
                    logger.warning(f"Failed to remove {filepath}: {e}")

    def _evaluate_result(self, result: str, ground_truth: Any, query: str = None) -> Optional[bool]:
        """
        Evaluate result against ground truth.

        Args:
            result: MAS output
            ground_truth: Expected answer
            query: Original task query (passed to LLM-as-judge for context)

        Returns:
            True if correct, False if incorrect, None if no evaluator available
        """
        evaluator = self._get_evaluator()
        if evaluator is None:
            return None

        # Pass query to LLM-as-judge if the evaluator supports it
        if hasattr(evaluator, 'evaluate_correctness'):
            import inspect
            sig = inspect.signature(evaluator.evaluate_correctness)
            if 'query' in sig.parameters:
                return evaluator.evaluate_correctness(result, ground_truth, query=query)
        return evaluator.evaluate_correctness(result, ground_truth)

    def _save_task_output(self, task, result: MasRunResult):
        """
        Save individual task output to file.

        Args:
            task: Task object with query and ground truth
            result: MasRunResult with the output
        """
        if not self.save_individual_outputs:
            return

        # Create directory structure: output_dir/{dataset_name}/{config_name}/
        config_name = self._get_config_name_with_model()
        task_output_dir = self.output_dir / self.dataset_name / config_name
        task_output_dir.mkdir(parents=True, exist_ok=True)

        # Save raw output to {task_id}.txt
        output_file = task_output_dir / f"{result.task_id}.txt"

        # Determine content to write
        # Prioritize actual result over error (agent may have succeeded even if evaluation failed)
        if result.result:
            # Clean the result to extract final answer
            content_to_save = self._clean_output(result.result)
        elif result.error:
            # Only save error if there's no result
            content_to_save = f"ERROR: {result.error}"
        else:
            # Empty result
            content_to_save = ""

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content_to_save)

        # Store metadata in memory for consolidated results.json
        meta_data = {
            "task_id": result.task_id,
            "dataset": self.dataset_name,
            "configuration": config_name,
            "query": task.query,
            "ground_truth": result.ground_truth,
            "error": result.error,
            "raw_output": result.result[:500] if result.result else None  # Store first 500 chars of raw output
        }

        # Only include evaluation if it was done
        if not self.skip_evaluation and result.correct is not None:
            meta_data["correct"] = result.correct

        self._all_metadata.append(meta_data)

        if self.verbose:
            logger.info(f"Saved output to: {output_file}")

    def _save_consolidated_results(self):
        """
        Save all metadata to a consolidated results.json file.
        """
        if not self.save_individual_outputs or not self._all_metadata:
            return

        # Create directory structure: output_dir/{dataset_name}/{config_name}/
        config_name = self._get_config_name_with_model()
        task_output_dir = self.output_dir / self.dataset_name / config_name
        task_output_dir.mkdir(parents=True, exist_ok=True)

        # Save consolidated results.json
        import json
        results_file = task_output_dir / "results.json"

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self._all_metadata, f, indent=2, ensure_ascii=False)

        if self.verbose:
            logger.info(f"Saved consolidated metadata to: {results_file} ({len(self._all_metadata)} tasks)")

    def _is_task_cached(self, task_id: Union[int, str]) -> bool:
        """
        Check if a task output file already exists and is valid.

        Args:
            task_id: Task ID to check

        Returns:
            True if cached output exists and is valid, False otherwise
        """
        if not self.use_cache or not self.save_individual_outputs:
            return False

        # Check if output file exists
        config_name = self._get_config_name_with_model()
        output_file = self.output_dir / self.dataset_name / config_name / f"{task_id}.txt"

        if not output_file.exists():
            return False

        # Check if file is empty or contains only whitespace
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    return False

            # File exists and has content
            return True

        except Exception as e:
            logger.debug(f"Error checking cache for task {task_id}: {e}")
            return False

    def _clean_output(self, output: str) -> str:
        """
        Clean output to extract the final answer.
        Removes extra explanations and keeps only the answer.

        Args:
            output: Raw output from model

        Returns:
            Cleaned output
        """
        if not output:
            return ""

        # For BBEH datasets, try to extract just the answer
        if 'bbeh' in self.dataset_name.lower():
            from src.utils.output_cleaner import clean_bbeh_output
            return clean_bbeh_output(output)

        # For SWE-bench datasets
        if 'swe' in self.dataset_name.lower():
            from src.utils.output_cleaner import clean_swe_bench_output
            return clean_swe_bench_output(output)

        # Default: return as-is (strip whitespace)
        return output.strip()

    def _run_with_timeout(self, task_query: str, timeout: float) -> Tuple[str, Dict[str, Any]]:
        """
        Run MAS with a timeout.

        Args:
            task_query: The task query to run
            timeout: Timeout in seconds

        Returns:
            Tuple of (result, metadata)

        Raises:
            TimeoutError: If the task exceeds the timeout (but includes partial metadata)
        """
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._runtime.run, task_query)
            try:
                result, metadata = future.result(timeout=timeout)
                return result, metadata
            except FuturesTimeoutError:
                logger.error(f"Task timed out after {timeout:.0f} seconds")
                # Try to get partial metadata from runtime context even on timeout
                partial_metadata = self._get_partial_metadata_on_timeout()
                # Raise a custom exception that includes the partial metadata
                error = TimeoutError(f"Task execution exceeded timeout of {timeout:.0f} seconds")
                error.partial_metadata = partial_metadata
                raise error

    def _get_partial_metadata_on_timeout(self) -> Dict[str, Any]:
        """
        Try to extract partial metadata from runtime when timeout occurs.
        This captures token usage even if the task didn't complete.
        """
        partial_metadata = {}
        try:
            # Try to get metadata from runtime's context if available
            if hasattr(self._runtime, 'context') and self._runtime.context:
                context = self._runtime.context
                # Aggregate metadata from trace entries
                total_input_tokens = 0
                total_output_tokens = 0
                total_api_calls = 0
                total_instance_cost = 0.0

                if hasattr(context, 'trace'):
                    for trace_entry in context.trace:
                        if 'metadata' in trace_entry and trace_entry['metadata']:
                            metadata = trace_entry['metadata']
                            # Handle standard token names
                            total_input_tokens += metadata.get('input_tokens', 0)
                            total_output_tokens += metadata.get('output_tokens', 0)
                            # Handle OpenAI-style names
                            total_input_tokens += metadata.get('prompt_tokens', 0)
                            total_output_tokens += metadata.get('completion_tokens', 0)
                            # Handle SWE-agent model_stats
                            if 'model_stats' in metadata and metadata['model_stats']:
                                model_stats = metadata['model_stats']
                                total_input_tokens += model_stats.get('tokens_sent', 0)
                                total_output_tokens += model_stats.get('tokens_received', 0)
                                total_api_calls += model_stats.get('api_calls', 0)
                                total_instance_cost += model_stats.get('instance_cost', 0.0)

                partial_metadata = {
                    'input_tokens': total_input_tokens,
                    'output_tokens': total_output_tokens,
                    'api_calls': total_api_calls,
                    'instance_cost': total_instance_cost,
                    'timeout': True
                }
                logger.info(f"Captured partial metadata on timeout: {total_input_tokens} input, {total_output_tokens} output tokens")

            # Also try to get stats directly from runner if available
            if hasattr(self._runtime, 'runner') and self._runtime.runner:
                runner = self._runtime.runner
                if hasattr(runner, '_last_agent') and runner._last_agent:
                    agent = runner._last_agent
                    if hasattr(agent, 'model') and hasattr(agent.model, 'stats'):
                        stats = agent.model.stats
                        partial_metadata['model_stats'] = {
                            'tokens_sent': getattr(stats, 'tokens_sent', 0),
                            'tokens_received': getattr(stats, 'tokens_received', 0),
                            'api_calls': getattr(stats, 'api_calls', 0),
                            'instance_cost': getattr(stats, 'instance_cost', 0.0),
                        }
                        logger.info(f"Got SWE-agent stats on timeout: {stats.tokens_sent} sent, {stats.tokens_received} received")

        except Exception as e:
            logger.warning(f"Failed to get partial metadata on timeout: {e}")

        return partial_metadata

    def run_single_task(self, task_id: Union[int, str]) -> MasRunResult:
        """
        Run MAS on a single task.

        Args:
            task_id: Task ID to run

        Returns:
            MasRunResult containing the result and evaluation
        """
        # Ensure everything is initialized
        self._load_mas_config()
        self._load_dataset()

        # Get task
        task = self._dataset.get_by_id(task_id)
        if task is None:
            raise ValueError(f"Task ID {task_id} not found in dataset")

        # Check if task is already cached
        if self._is_task_cached(task.id):
            if self.verbose:
                logger.info(f"Task {task.id} already cached, skipping execution")

            # Return a placeholder result indicating it was cached
            return MasRunResult(
                task_id=task.id,
                result="[Cached]",
                ground_truth=task.gt,
                correct=None,
                metadata={"cached": True}
            )

        # Initialize runtime only if we need to run the task
        self._initialize_runtime()

        if self.verbose:
            logger.info("\n" + "-" * 80)
            logger.info(f"Running Task ID: {task.id}")
            logger.info("-" * 80)
            logger.info(f"Query: {task.query[:100]}...")
            logger.info(f"Ground Truth: {task.gt}")

        # Track repository path for recovery in finally block
        repo_path = None
        repo_lock = None
        wb_lock = None

        try:
            # For WorkBench, acquire per-domain locks so concurrent tasks on the
            # same domain serialize on the shared in-memory/CSV state.
            if 'workbench' in self.dataset_name.lower():
                wb_lock = WorkbenchDomainLock(self.dataset_name)
                if wb_lock.domains and not wb_lock.acquire(timeout=600):
                    raise RuntimeError(
                        f"Failed to acquire WorkBench domain lock for {self.dataset_name}"
                    )
                if wb_lock.domains:
                    logger.info(f"Acquired WorkBench domain lock: {wb_lock.domains}")

            # Enhance task query for SWE-bench instances
            task_query = task.query
            if (self.dataset_name.startswith('swe') or
                self.dataset_name in ['swebench', 'swebench_lite', 'swebench_verified']):
                # Get repository information
                if hasattr(task, 'metadata') and task.metadata:
                    repo_name = task.metadata.get('repo', '')
                    if repo_name:
                        # Extract repo name (e.g., "django/django" -> "django")
                        repo_simple = repo_name.split('/')[-1]
                        repo_path = str(Path(REPOS_DIR) / repo_simple)

                        # Check for source subdirectory (e.g., django-src for django)
                        # Some repos have submodule structure with {name}-src containing actual code
                        src_subdir = Path(repo_path) / f"{repo_simple}-src"
                        if src_subdir.exists() and (src_subdir / ".git").exists():
                            repo_path = str(src_subdir)
                            logger.info(f"Using source subdirectory: {repo_path}")

                if repo_path and Path(repo_path).exists():
                    # Acquire lock to prevent concurrent access to the same repo
                    repo_lock = RepoLock(repo_path)
                    if not repo_lock.acquire(timeout=600):  # 10 min timeout
                        raise RuntimeError(f"Failed to acquire lock for repo: {repo_path}")
                    logger.info(f"Acquired lock for {repo_path}")

                    # Checkout base_commit before running agent (critical for SWE-bench)
                    base_commit = task.metadata.get('base_commit') if hasattr(task, 'metadata') and task.metadata else None
                    if base_commit:
                        if not self._checkout_base_commit(repo_path, base_commit):
                            raise RuntimeError(f"Failed to checkout base commit {base_commit[:8]} for {task.id}")

                    # Check backend type - minisweagent/sweagent have their own templates with proper formatting
                    backend = self._mas_spec.backend if self._mas_spec else None

                    if backend in ("minisweagent", "sweagent"):
                        # For minisweagent/sweagent: pass minimal info, let agent's config templates handle formatting
                        # The instance_template in config uses {{task}} for the problem statement
                        task_query = f"""Repository: {repo_path}
Instance ID: {task.id}

Problem Statement:
{task.query}"""
                    else:
                        # For other backends (smolagents, etc.): include explicit instructions
                        task_query = f"""Repository: {repo_path}
Instance ID: {task.id}

{task.query}

CRITICAL INSTRUCTIONS:
1. Read and analyze the relevant files in the repository at {repo_path}
2. Identify the exact code changes needed to fix the issue
3. Generate a git diff patch in unified diff format

Your response MUST be ONLY the git diff patch, starting with "diff --git" and following this exact format:

diff --git a/path/to/file.py b/path/to/file.py
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
 context line
-removed line
+added line
 context line

Do NOT include any explanations, markdown formatting, or other text. ONLY output the raw diff patch."""
                else:
                    logger.warning(f"Repository path not found for SWE-bench instance {task.id}")

            # Run MAS with timeout - returns tuple of (result, metadata)
            if self.task_timeout and self.task_timeout > 0:
                logger.info(f"Running task with {self.task_timeout:.0f}s timeout...")
                result, metadata = self._run_with_timeout(task_query, self.task_timeout)
            else:
                result, metadata = self._runtime.run(task_query)

            if self.verbose:
                logger.info(f"\nResult: {result[:200]}...")

            # Evaluate only if not skipping evaluation
            is_correct = None
            if not self.skip_evaluation:
                is_correct = self._evaluate_result(result, task.gt, query=getattr(task, 'query', None))
                if self.verbose and is_correct is not None:
                    status = "CORRECT" if is_correct else "INCORRECT"
                    logger.info(f"Evaluation: {status}")

            mas_result = MasRunResult(
                task_id=task.id,
                result=result,
                ground_truth=task.gt,
                correct=is_correct,
                metadata=metadata
            )

            # Save individual output if enabled
            self._save_task_output(task, mas_result)

            # Save consolidated results.json
            self._save_consolidated_results()

            return mas_result

        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()

            # Check if exception has partial_metadata (from timeout)
            error_metadata = None
            if hasattr(e, 'partial_metadata') and e.partial_metadata:
                error_metadata = e.partial_metadata
                logger.info(f"Using partial metadata from timeout: {error_metadata}")
            elif 'metadata' in locals():
                error_metadata = metadata

            # Preserve result and metadata if agent ran successfully
            mas_result = MasRunResult(
                task_id=task.id,
                result=result if 'result' in locals() else None,
                error=str(e),
                ground_truth=task.gt,
                metadata=error_metadata
            )

            # Save individual output even for errors
            self._save_task_output(task, mas_result)

            # Save consolidated results.json
            self._save_consolidated_results()

            return mas_result

        finally:
            # Recover repository to clean state (for SWE-bench instances)
            if repo_path and Path(repo_path).exists():
                self._recover_repository(repo_path)

            # Clean up SWE-agent artifacts from home directory and /tmp
            self._cleanup_sweagent_artifacts()

            # Release repository lock
            if repo_lock:
                repo_lock.release()
                logger.info(f"Released lock for {repo_path}")

            # Release WorkBench domain lock(s)
            if wb_lock and wb_lock.domains:
                wb_lock.release()
                logger.info(f"Released WorkBench domain lock: {wb_lock.domains}")

    def run(
        self,
        num_tasks: Optional[int] = None,
        task_ids: Optional[List[Union[int, str]]] = None,
        start_idx: int = 0
    ) -> Tuple[List[MasRunResult], Dict[str, int], Dict[str, float]]:
        """
        Run MAS on multiple tasks.

        Args:
            num_tasks: Number of tasks to run from start_idx (default: all tasks)
            task_ids: Specific task IDs to run (overrides num_tasks)
            start_idx: Starting index for tasks (default: 0)

        Returns:
            Tuple of (results, token_costs, time_costs) where:
            - results: List of MasRunResult objects
            - token_costs: Dict with 'input_tokens', 'output_tokens', 'total_tokens'
            - time_costs: Dict with 'total_time', 'avg_time'
        """
        # Ensure everything is initialized
        self._load_mas_config()
        self._load_dataset()
        self._initialize_runtime()

        # Initialize cost tracking
        total_input_tokens = 0
        total_output_tokens = 0
        total_time = 0.0
        successful_runs = 0

        # Select tasks
        if task_ids is not None:
            tasks = []
            for task_id in task_ids:
                task = self._dataset.get_by_id(task_id)
                if task is not None:
                    tasks.append(task)
                else:
                    logger.warning(f"Task ID {task_id} not found, skipping")
        else:
            all_tasks = list(self._dataset)
            if num_tasks is None:
                tasks = all_tasks[start_idx:]
            else:
                tasks = all_tasks[start_idx:start_idx + num_tasks]

        # Check which tasks are already cached
        results = []
        tasks_to_run = []
        cached_count = 0
        for task in tasks:
            if self._is_task_cached(task.id):
                cached_count += 1
                # Add cached result
                results.append(MasRunResult(
                    task_id=task.id,
                    result="[Cached]",
                    ground_truth=task.gt,
                    correct=None,
                    metadata={"cached": True}
                ))
            else:
                tasks_to_run.append(task)

        if self.verbose:
            if cached_count > 0:
                logger.info(f"Found {cached_count} cached tasks, will run {len(tasks_to_run)} tasks")
            else:
                logger.info(f"\nRunning {len(tasks_to_run)} tasks...")

        # Run non-cached tasks
        for i, task in enumerate(tasks_to_run, 1):
            if self.verbose:
                logger.info("\n" + "-" * 80)
                logger.info(f"Task {i}/{len(tasks_to_run)} - ID: {task.id}")
                logger.info("-" * 80)
                logger.info(f"Query: {task.query[:100]}...")
                logger.info(f"Ground Truth: {task.gt}")

            # Track repository path for recovery
            repo_path = None
            repo_lock = None
            wb_lock = None

            try:
                # For WorkBench, acquire per-domain locks BEFORE resetting state
                # or running the task so concurrent workers on the same domain
                # cannot race on the shared module globals / data.csv.
                if 'workbench' in self.dataset_name.lower():
                    wb_lock = WorkbenchDomainLock(self.dataset_name)
                    if wb_lock.domains and not wb_lock.acquire(timeout=600):
                        raise RuntimeError(
                            f"Failed to acquire WorkBench domain lock for {self.dataset_name}"
                        )
                    if wb_lock.domains:
                        logger.info(f"Acquired WorkBench domain lock: {wb_lock.domains}")

                # Reset WorkBench tool state before each task to prevent data pollution
                self._reset_workbench_state()

                # Track execution time
                start_time = time.time()

                # Enhance task query for SWE-bench instances
                task_query = task.query
                if (self.dataset_name.startswith('swe') or
                    self.dataset_name in ['swebench', 'swebench_lite', 'swebench_verified']):
                    # Get repository information
                    if hasattr(task, 'metadata') and task.metadata:
                        repo_name = task.metadata.get('repo', '')
                        if repo_name:
                            # Extract repo name (e.g., "django/django" -> "django")
                            repo_simple = repo_name.split('/')[-1]
                            repo_path = str(Path(REPOS_DIR) / repo_simple)

                            # Check for source subdirectory (e.g., django-src for django)
                            # Some repos have submodule structure with {name}-src containing actual code
                            src_subdir = Path(repo_path) / f"{repo_simple}-src"
                            if src_subdir.exists() and (src_subdir / ".git").exists():
                                repo_path = str(src_subdir)
                                logger.info(f"Using source subdirectory: {repo_path}")

                            # Acquire lock to prevent concurrent access
                            repo_lock = RepoLock(repo_path)
                            if not repo_lock.acquire(timeout=600):
                                raise RuntimeError(f"Failed to acquire lock for repo: {repo_path}")
                            logger.info(f"Acquired lock for {repo_path}")

                            # Checkout base_commit before running agent (critical for SWE-bench)
                            base_commit = task.metadata.get('base_commit') if task.metadata else None
                            if base_commit:
                                if not self._checkout_base_commit(repo_path, base_commit):
                                    raise RuntimeError(f"Failed to checkout base commit {base_commit[:8]} for {task.id}")

                    if repo_path and Path(repo_path).exists():
                        task_query = f"""Repository: {repo_path}
Instance ID: {task.id}

{task.query}

CRITICAL INSTRUCTIONS:
1. Read and analyze the relevant files in the repository at {repo_path}
2. Identify the exact code changes needed to fix the issue
3. Generate a git diff patch in unified diff format

Your response MUST be ONLY the git diff patch, starting with "diff --git" and following this exact format:

diff --git a/path/to/file.py b/path/to/file.py
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
 context line
-removed line
+added line
 context line

Do NOT include any explanations, markdown formatting, or other text. ONLY output the raw diff patch."""
                    else:
                        if repo_path:
                            logger.warning(f"Repository path not found for SWE-bench instance {task.id}: {repo_path}")

                # Run MAS with timeout - returns (result, metadata)
                if self.task_timeout and self.task_timeout > 0:
                    result, metadata = self._run_with_timeout(task_query, self.task_timeout)
                else:
                    result, metadata = self._runtime.run(task_query)

                # Calculate execution time
                execution_time = time.time() - start_time
                total_time += execution_time

                # Track token usage
                if metadata:
                    total_input_tokens += metadata.get('input_tokens', 0)
                    total_output_tokens += metadata.get('output_tokens', 0)
                    successful_runs += 1

                if self.verbose:
                    logger.info(f"\nResult: {result[:200]}...")

                # Evaluate only if not skipping evaluation
                is_correct = None
                if not self.skip_evaluation:
                    is_correct = self._evaluate_result(result, task.gt, query=getattr(task, 'query', None))
                    if self.verbose and is_correct is not None:
                        status = "CORRECT" if is_correct else "INCORRECT"
                        logger.info(f"Evaluation: {status}")

                mas_result = MasRunResult(
                    task_id=task.id,
                    result=result,
                    ground_truth=task.gt,
                    correct=is_correct,
                    metadata=metadata
                )

                # Save individual output if enabled
                self._save_task_output(task, mas_result)

                results.append(mas_result)

            except Exception as e:
                logger.error(f"Task {task.id} failed: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()

                # Preserve result and metadata if agent ran successfully before error
                mas_result = MasRunResult(
                    task_id=task.id,
                    result=result if 'result' in locals() else None,
                    error=str(e),
                    ground_truth=task.gt,
                    metadata=metadata if 'metadata' in locals() else None
                )

                # Save individual output even for errors
                self._save_task_output(task, mas_result)

                results.append(mas_result)

            finally:
                # Recover repository to clean state (for SWE-bench instances)
                if repo_path and Path(repo_path).exists():
                    self._recover_repository(repo_path)

                # Clean up SWE-agent artifacts from home directory and /tmp
                self._cleanup_sweagent_artifacts()

                # Release repository lock
                if repo_lock:
                    repo_lock.release()
                    logger.info(f"Released lock for {repo_path}")

                # Release WorkBench domain lock(s)
                if wb_lock and wb_lock.domains:
                    wb_lock.release()
                    logger.info(f"Released WorkBench domain lock: {wb_lock.domains}")

        # Print summary
        self._print_summary(results)

        # Save consolidated results.json
        self._save_consolidated_results()

        # Prepare cost dictionaries
        token_costs = {
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'total_tokens': total_input_tokens + total_output_tokens
        }

        time_costs = {
            'total_time': total_time,
            'avg_time': total_time / successful_runs if successful_runs > 0 else 0.0
        }

        return results, token_costs, time_costs

    def run_parallel(
        self,
        num_tasks: Optional[int] = None,
        task_ids: Optional[List[Union[int, str]]] = None,
        start_idx: int = 0,
        workers: int = 4
    ) -> Tuple[List[MasRunResult], Dict[str, int], Dict[str, float]]:
        """
        Run MAS on multiple tasks using parallel processing.

        Args:
            num_tasks: Number of tasks to run from start_idx (default: all tasks)
            task_ids: Specific task IDs to run (overrides num_tasks)
            start_idx: Starting index for tasks (default: 0)
            workers: Number of worker processes (default: 4)

        Returns:
            Tuple of (results, token_costs, time_costs) where:
            - results: List of MasRunResult objects
            - token_costs: Dict with 'input_tokens', 'output_tokens', 'total_tokens'
            - time_costs: Dict with 'total_time', 'avg_time'
        """
        # Ensure everything is initialized
        self._load_mas_config()
        self._load_dataset()
        self._initialize_runtime()

        # Initialize cost tracking
        total_input_tokens = 0
        total_output_tokens = 0
        total_time = 0.0
        successful_runs = 0

        # Limit workers to CPU count
        workers = min(workers, cpu_count())

        # Select tasks
        if task_ids is not None:
            tasks = []
            for task_id in task_ids:
                task = self._dataset.get_by_id(task_id)
                if task is not None:
                    tasks.append(task)
                else:
                    logger.warning(f"Task ID {task_id} not found, skipping")
        else:
            all_tasks = list(self._dataset)
            if num_tasks is None:
                tasks = all_tasks[start_idx:]
            else:
                tasks = all_tasks[start_idx:start_idx + num_tasks]

        # Check which tasks are already cached
        results = []
        tasks_to_run = []
        cached_count = 0
        for task in tasks:
            if self._is_task_cached(task.id):
                cached_count += 1
                # Add cached result
                results.append(MasRunResult(
                    task_id=task.id,
                    result="[Cached]",
                    ground_truth=task.gt,
                    correct=None,
                    metadata={"cached": True}
                ))
            else:
                tasks_to_run.append(task)

        if self.verbose:
            if cached_count > 0:
                logger.info(f"Found {cached_count} cached tasks, will run {len(tasks_to_run)} tasks with {workers} workers")
            else:
                logger.info(f"\nRunning {len(tasks_to_run)} tasks with {workers} workers...")

        # Run non-cached tasks in parallel
        if tasks_to_run:
            start_time = time.time()

            # Group tasks by repo to avoid concurrent access to same repo
            # Tasks for the same repo will be processed sequentially by the same worker
            repo_task_groups = {}
            for task in tasks_to_run:
                repo_name = 'default'
                if hasattr(task, 'metadata') and task.metadata:
                    repo_full = task.metadata.get('repo', '')
                    if repo_full:
                        repo_name = repo_full.split('/')[-1]
                if repo_name not in repo_task_groups:
                    repo_task_groups[repo_name] = []
                repo_task_groups[repo_name].append(task)

            # Assign repos to workers in round-robin fashion
            # Each worker gets a list of repos, and processes all tasks for those repos
            worker_repo_assignments = [[] for _ in range(workers)]
            for i, repo_name in enumerate(repo_task_groups.keys()):
                worker_id = i % workers
                worker_repo_assignments[worker_id].append(repo_name)

            logger.info(f"Task distribution by repo:")
            for i, repos in enumerate(worker_repo_assignments):
                task_count = sum(len(repo_task_groups[r]) for r in repos)
                logger.info(f"   Worker {i}: {len(repos)} repos, {task_count} tasks")

            # Flatten the task list ordered by worker assignment
            # Tasks for the same repo stay together and get processed by same worker
            ordered_tasks = []
            for worker_id, repos in enumerate(worker_repo_assignments):
                for repo_name in repos:
                    for task in repo_task_groups[repo_name]:
                        ordered_tasks.append((task, worker_id))

            with Pool(
                processes=workers,
                initializer=init_mas_worker,
                initargs=(
                    self.config_path,
                    self.dataset_name,
                    self.dataset_split,
                    False,  # verbose=False in workers
                    self.save_individual_outputs,
                    self.output_dir,
                    self.use_cache
                )
            ) as pool:
                # Process tasks in parallel (ordered to minimize repo conflicts)
                parallel_results = pool.map(process_single_task_parallel, ordered_tasks)

                # Add parallel results to results list
                results.extend(parallel_results)

                # Aggregate token and time costs from parallel results
                for result in parallel_results:
                    if result.metadata and not result.metadata.get('cached'):
                        total_input_tokens += result.metadata.get('input_tokens', 0)
                        total_output_tokens += result.metadata.get('output_tokens', 0)
                        successful_runs += 1

            processing_time = time.time() - start_time
            total_time = processing_time
            logger.info(f"Parallel processing completed in {processing_time:.2f}s")
            logger.info(f"Throughput: {len(tasks_to_run) / processing_time:.2f} tasks/second")

        # Print summary
        self._print_summary(results)

        # Save consolidated results.json
        self._save_consolidated_results()

        # Prepare cost dictionaries
        token_costs = {
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'total_tokens': total_input_tokens + total_output_tokens
        }

        time_costs = {
            'total_time': total_time,
            'avg_time': total_time / successful_runs if successful_runs > 0 else 0.0
        }

        return results, token_costs, time_costs

    def _print_summary(self, results: List[MasRunResult]):
        """Print summary of results."""
        if not self.verbose:
            return

        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)

        total_tasks = len(results)
        cached_tasks = sum(1 for r in results if r.metadata and r.metadata.get("cached"))
        failed_tasks = sum(1 for r in results if r.error is not None and not (r.metadata and r.metadata.get("cached")))

        logger.info(f"Total tasks: {total_tasks}")
        if cached_tasks > 0:
            logger.info(f"Cached (skipped): {cached_tasks}")
            logger.info(f"Actually run: {total_tasks - cached_tasks}")
        logger.info(f"Failed: {failed_tasks}")

        # Check if we have correctness evaluations
        evaluated_results = [r for r in results if r.correct is not None]
        if evaluated_results:
            correct_count = sum(1 for r in evaluated_results if r.correct)
            accuracy = correct_count / len(evaluated_results) if evaluated_results else 0

            logger.info(f"Correct: {correct_count}")
            logger.info(f"Incorrect: {len(evaluated_results) - correct_count}")
            logger.info(f"Accuracy: {accuracy:.2%}")

        logger.info("=" * 80)

    def get_summary_stats(self, results: List[MasRunResult]) -> Dict[str, Any]:
        """
        Get summary statistics from results.

        Args:
            results: List of MasRunResult objects

        Returns:
            Dictionary with summary statistics
        """
        total_tasks = len(results)
        cached_tasks = sum(1 for r in results if r.metadata and r.metadata.get("cached"))
        failed_tasks = sum(1 for r in results if r.error is not None and not (r.metadata and r.metadata.get("cached")))
        evaluated_results = [r for r in results if r.correct is not None]

        stats = {
            'total_tasks': total_tasks,
            'cached_tasks': cached_tasks,
            'actually_run': total_tasks - cached_tasks,
            'failed_tasks': failed_tasks,
            'successful_tasks': total_tasks - failed_tasks - cached_tasks,
        }

        if evaluated_results:
            # Support both bool scores (legacy) and float scores (multi-aspect judge)
            scores = [r.correct for r in evaluated_results]
            if all(isinstance(s, bool) for s in scores):
                correct_count = sum(1 for s in scores if s)
                accuracy = correct_count / len(evaluated_results)
            else:
                # Float scores from multi-aspect judge (0-100): average them
                correct_count = sum(1 for s in scores if (s is True) or (isinstance(s, (int, float)) and s >= 50))
                accuracy = sum(float(s) for s in scores) / len(evaluated_results)
            stats.update({
                'evaluated_tasks': len(evaluated_results),
                'correct': correct_count,
                'incorrect': len(evaluated_results) - correct_count,
                'accuracy': accuracy
            })

        return stats

    def _reset_workbench_state(self):
        """
        Reset WorkBench tool state to prevent data pollution between tasks.

        This ensures each task operates on fresh data by resetting all
        WorkBench domain tools to their original state.
        """
        # Only reset if this is a WorkBench dataset
        if 'workbench' not in self.dataset_name.lower():
            return

        try:
            # Import and reset each WorkBench domain
            domains_to_reset = []

            # Determine which domains to reset based on dataset name
            if 'email' in self.dataset_name:
                from src.tools import workbench_tools_email
                domains_to_reset.append(workbench_tools_email)
            elif 'calendar' in self.dataset_name:
                from src.tools import workbench_tools_calendar
                domains_to_reset.append(workbench_tools_calendar)
            elif 'analytics' in self.dataset_name:
                from src.tools import workbench_tools_all
                # Analytics reset is in workbench_tools_all
                if hasattr(workbench_tools_all, 'reset_analytics_state'):
                    workbench_tools_all.reset_analytics_state()
            elif 'project_management' in self.dataset_name:
                from src.tools import workbench_tools_all
                if hasattr(workbench_tools_all, 'reset_project_management_state'):
                    workbench_tools_all.reset_project_management_state()
            elif 'customer_relationship_manager' in self.dataset_name:
                from src.tools import workbench_tools_all
                if hasattr(workbench_tools_all, 'reset_crm_state'):
                    workbench_tools_all.reset_crm_state()
            elif 'multi_domain' in self.dataset_name:
                # Reset all domains for multi-domain tasks
                from src.tools import workbench_tools_email, workbench_tools_calendar, workbench_tools_all
                domains_to_reset.extend([workbench_tools_email, workbench_tools_calendar])
                if hasattr(workbench_tools_all, 'reset_analytics_state'):
                    workbench_tools_all.reset_analytics_state()
                if hasattr(workbench_tools_all, 'reset_project_management_state'):
                    workbench_tools_all.reset_project_management_state()
                if hasattr(workbench_tools_all, 'reset_crm_state'):
                    workbench_tools_all.reset_crm_state()
                if hasattr(workbench_tools_all, 'reset_company_directory_state'):
                    workbench_tools_all.reset_company_directory_state()

            # Call reset_state() on each domain module
            for domain_module in domains_to_reset:
                if hasattr(domain_module, 'reset_state'):
                    domain_module.reset_state()
                    logger.debug(f"Reset state for {domain_module.__name__}")

        except Exception as e:
            logger.warning(f"Failed to reset WorkBench state: {e}")
            # Don't fail the task if reset fails - just log warning
