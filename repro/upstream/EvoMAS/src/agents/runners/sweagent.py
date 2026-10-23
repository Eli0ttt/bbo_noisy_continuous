"""
Runner for SWE-agent framework.

This runner adapts the official SWE-agent to work with EvoMAS infrastructure,
providing a direct integration without modifying SWE-agent's core logic,
prompts, or tools.

The runner delegates to SWE-agent's native components:
- Agent: sweagent.agent.agents.DefaultAgent
- Model: sweagent.agent.models (via litellm)
- Environment: sweagent.environment.swe_env.SWEEnv

Supports two modes:
- use_docker=True: Uses Docker deployment (requires Docker)
- use_docker=False: Uses local deployment (no Docker required)
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from ..spec import AgentSpec, AgentResult
from .base import BaseAgentRunner

logger = logging.getLogger(__name__)

# Try to import SWE-agent
try:
    from sweagent.agent.agents import DefaultAgent, TemplateConfig
    from sweagent.agent.models import GenericAPIModelConfig, get_model
    from sweagent.agent.problem_statement import TextProblemStatement
    from sweagent.environment.swe_env import SWEEnv
    from sweagent.environment.repo import PreExistingRepoConfig
    from sweagent.tools.tools import ToolConfig, ToolHandler
    from sweagent.agent.history_processors import DefaultHistoryProcessor
    from sweagent import CONFIG_DIR
    SWEAGENT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"SWE-agent not available: {e}")
    SWEAGENT_AVAILABLE = False
    DefaultAgent = None
    SWEEnv = None

# Configuration paths — resolve relative to project root, overridable via env vars
EVOMAS_CONFIG_DIR = Path(os.environ.get(
    "EVOMAS_SWEAGENT_CONFIG_DIR",
    str(Path("config/sweagent_configs").resolve())
))
SWEAGENT_CONFIG_DIR = Path(os.environ.get(
    "SWEAGENT_CONFIG_DIR",
    str(Path("config/sweagent_configs").resolve())
))


def load_sweagent_config(config_name: str = "default") -> Dict[str, Any]:
    """
    Load SWE-agent configuration from file.

    Args:
        config_name: Configuration name or path

    Returns:
        Configuration dictionary
    """
    import yaml

    # Check EvoMAS config dir first
    config_path = EVOMAS_CONFIG_DIR / f"{config_name}.yaml"
    if not config_path.exists():
        config_path = SWEAGENT_CONFIG_DIR / f"{config_name}.yaml"
    if not config_path.exists():
        # Use default from SWE-agent
        config_path = SWEAGENT_CONFIG_DIR / "default.yaml"

    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)

    return {}


class SWEAgentRunner(BaseAgentRunner):
    """
    Runner for SWE-agent framework.

    This runner provides a direct integration with SWE-agent, using its native
    components. It supports:
    - Local execution (no Docker required, uses conda environment)
    - Docker execution (production mode)

    The runner adapts EvoMAS model IDs to litellm format used by SWE-agent.
    """

    def __init__(
        self,
        working_dir: Optional[Path] = None,
        config: str = "default",
        use_docker: bool = False,  # Default to no Docker for EvoMAS
        **kwargs
    ):
        """
        Initialize SWE-agent runner.

        Args:
            working_dir: Working directory for code execution
            config: Configuration name or path
            use_docker: Whether to use Docker for execution (default False)
        """
        if not SWEAGENT_AVAILABLE:
            logger.warning("SWE-agent not available, runner will fail")

        self.working_dir = working_dir or Path.cwd()
        self.config_name = config
        self.use_docker = use_docker

        # Load configuration
        self.full_config = load_sweagent_config(config)

        logger.info(f"SWEAgentRunner initialized with config: {config}, use_docker: {use_docker}")

    def _convert_model_id(self, model_id: str) -> str:
        """
        Convert EvoMAS model ID to litellm format.

        EvoMAS format: "provider:model_name"
        litellm format varies by provider:
        - OpenAI: "model_name" (auto-detected)
        - Anthropic: "anthropic/model_name"
        - Bedrock: "bedrock/model_name"

        Args:
            model_id: EvoMAS model ID

        Returns:
            litellm-compatible model ID
        """
        if ':' not in model_id:
            return model_id

        provider, model_name = model_id.split(':', 1)
        provider = provider.lower()

        if provider == 'openai':
            # litellm auto-detects OpenAI models
            return model_name
        elif provider == 'bedrock':
            # Bedrock uses "bedrock/model_name" format
            return f"bedrock/{model_name}"
        elif provider == 'anthropic':
            # Anthropic uses "anthropic/model_name" format
            return f"anthropic/{model_name}"
        elif provider == 'azure':
            # litellm uses "azure/deployment_name" format. For Claude on Azure,
            # set litellm's Azure env vars from our AZURE_* env vars.
            is_claude = "claude" in model_name.lower()
            if is_claude:
                os.environ["AZURE_API_BASE"] = os.environ.get("AZURE_Anthropic_ENDPOINT", "")
                os.environ["AZURE_API_KEY"] = os.environ.get("AZURE_Anthropic_API_KEY", "")
            else:
                os.environ["AZURE_API_BASE"] = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
                os.environ["AZURE_API_KEY"] = os.environ.get("AZURE_OPENAI_API_KEY", "")
            os.environ["AZURE_API_VERSION"] = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            return f"azure/{model_name}"
        else:
            # Generic format for other providers
            return f"{provider}/{model_name}"

    def _extract_repo_path(self, task: str) -> Optional[Path]:
        """Extract repository path from task query."""
        # First try to extract absolute path after "Repository:"
        match = re.search(r'Repository:\s*(/[^\s\n]+)', task)
        if match:
            return Path(match.group(1))

        # Try to find "directory /path/to/repo" pattern (common in SWE-bench tasks)
        match = re.search(r'directory\s+(/[^\s\n\.]+)', task)
        if match:
            path = Path(match.group(1))
            if path.exists():
                return path

        # Try to find any absolute path ending with a repo-like name
        match = re.search(r'(/home/[^\s\n]+/repos/[^\s\n\.]+)', task)
        if match:
            path = Path(match.group(1))
            if path.exists():
                return path

        return None

    def _extract_repo_name(self, task: str) -> Optional[str]:
        """Extract repository name from task query."""
        # Try "Repository: owner/repo" format
        match = re.search(r'Repository:\s*([^/\s]+/[^\s\n]+)', task)
        if match:
            repo_str = match.group(1)
            # If it's a path, extract just the repo name
            if '/' in repo_str and not repo_str.startswith('/'):
                return repo_str.split('/')[-1]
            return repo_str
        return None

    def _extract_problem_statement(self, task: str) -> str:
        """
        Extract problem statement from EvoMAS task format.

        For proper alignment with SWE-agent:
        - If task contains "Problem Statement:", extract just that part
        - Otherwise, pass the task as-is
        """
        # Check for EvoMAS wrapped format
        if "Problem Statement:" in task:
            # Extract everything after "Problem Statement:"
            match = re.search(r'Problem Statement:\s*\n(.*)', task, re.DOTALL)
            if match:
                return match.group(1).strip()

        return task

    def _reset_repo(self, repo_path: Path):
        """Reset repository to clean state."""
        try:
            subprocess.run(["git", "reset", "--hard"], cwd=str(repo_path),
                          capture_output=True, timeout=30)
            subprocess.run(["git", "clean", "-fd"], cwd=str(repo_path),
                          capture_output=True, timeout=30)

            # Explicitly remove .sweagent_output directory if it exists
            # (belt and suspenders - git clean should handle this, but be sure)
            sweagent_output_dir = repo_path / ".sweagent_output"
            if sweagent_output_dir.exists():
                import shutil
                shutil.rmtree(sweagent_output_dir, ignore_errors=True)
                logger.debug(f"Cleaned up .sweagent_output directory in {repo_path}")
        except Exception as e:
            logger.warning(f"Failed to reset repo {repo_path}: {e}")

    def _get_git_diff(self, repo_path: Path) -> str:
        """Get git diff of changes in repository, excluding SWE-agent output files.

        This tries multiple strategies to capture changes:
        1. Unstaged changes (git diff)
        2. Staged changes (git diff --staged)
        3. Combined unstaged + staged (git diff HEAD)
        """
        exclude_paths = [
            ":(exclude).sweagent_output",
            ":(exclude)*.traj",
            ":(exclude)*_helpers",  # Exclude submodules like astropy_helpers
        ]

        try:
            # Strategy 1: Try git diff HEAD (shows all uncommitted changes vs last commit)
            # This captures both staged and unstaged changes
            result = subprocess.run(
                ["git", "diff", "HEAD", "--", "."] + exclude_paths,
                cwd=str(repo_path),
                capture_output=True, text=True, timeout=30
            )
            if result.stdout.strip():
                logger.debug(f"Got diff from 'git diff HEAD' ({len(result.stdout)} bytes)")
                return result.stdout

            # Strategy 2: Try unstaged changes only
            result = subprocess.run(
                ["git", "diff", "--", "."] + exclude_paths,
                cwd=str(repo_path),
                capture_output=True, text=True, timeout=30
            )
            if result.stdout.strip():
                logger.debug(f"Got diff from 'git diff' ({len(result.stdout)} bytes)")
                return result.stdout

            # Strategy 3: Try staged changes only
            result = subprocess.run(
                ["git", "diff", "--staged", "--", "."] + exclude_paths,
                cwd=str(repo_path),
                capture_output=True, text=True, timeout=30
            )
            if result.stdout.strip():
                logger.debug(f"Got diff from 'git diff --staged' ({len(result.stdout)} bytes)")
                return result.stdout

            logger.debug(f"No git diff found in {repo_path}")
            return ""

        except Exception as e:
            logger.warning(f"Failed to get git diff from {repo_path}: {e}")
            return ""

    def _filter_sweagent_artifacts(self, diff_content: str) -> str:
        """Filter out SWE-agent artifacts from diff content.

        Removes diff blocks for:
        - .sweagent_output/ directory
        - *.traj files
        - *_helpers submodules

        Args:
            diff_content: Raw git diff content

        Returns:
            Filtered diff with only source code changes
        """
        if not diff_content:
            return ""

        # Patterns to exclude from the patch
        exclude_patterns = [
            '.sweagent_output/',
            '.traj',
            '_helpers',  # submodules like astropy_helpers
            'Subproject commit',
        ]

        # Split into individual diff blocks
        lines = diff_content.split('\n')
        filtered_blocks = []
        current_block = []
        current_header = ""
        should_exclude = False

        for line in lines:
            if line.startswith('diff --git') or line.startswith('diff '):
                # Save previous block if it wasn't excluded
                if current_block and not should_exclude:
                    filtered_blocks.append('\n'.join(current_block))

                # Start new block
                current_block = [line]
                current_header = line

                # Check if this block should be excluded
                should_exclude = any(pattern in line for pattern in exclude_patterns)
            elif current_block:
                # Continue current block
                current_block.append(line)

        # Don't forget the last block
        if current_block and not should_exclude:
            filtered_blocks.append('\n'.join(current_block))

        result = '\n'.join(filtered_blocks).strip()
        if result and not result.endswith('\n'):
            result += '\n'
        return result

    def create_agent(self, spec: AgentSpec, working_dir: Path) -> Any:
        """
        Create a SWE-agent instance from specification.

        Args:
            spec: Agent specification
            working_dir: Working directory for the agent

        Returns:
            SWE-agent instance
        """
        if not SWEAGENT_AVAILABLE:
            raise ImportError("SWE-agent is not available")

        # Ensure API keys are loaded from EvoMAS .env
        try:
            from dotenv import load_dotenv
            evomas_root = Path(__file__).parent.parent.parent.parent
            env_file = evomas_root / '.env'
            if env_file.exists():
                load_dotenv(env_file)
                logger.debug(f"Loaded environment from {env_file}")
        except ImportError:
            logger.warning("dotenv not available, relying on existing environment")

        # Convert model ID to litellm format
        litellm_model_id = self._convert_model_id(spec.model_id)
        logger.info(f"Using litellm model: {litellm_model_id}")

        # Load tools configuration from SWE-agent default config
        default_config_path = CONFIG_DIR / "default.yaml"
        if default_config_path.exists():
            import yaml
            with open(default_config_path) as f:
                default_config = yaml.safe_load(f)

            agent_config = default_config.get('agent', {})
            tools_config = agent_config.get('tools', {})
            templates_config = agent_config.get('templates', {})
        else:
            tools_config = {}
            templates_config = {}

        # Create tool handler
        # For local mode, reduce execution timeout to speed up submit failures
        # (EvoMAS extracts patch via git diff fallback anyway)
        if not self.use_docker:
            tools_config = dict(tools_config) if tools_config else {}
            tools_config['execution_timeout'] = 120  # 2 minutes for build commands
            tools_config['max_consecutive_execution_timeouts'] = 5  # Allow more retries
        tool_config = ToolConfig(**tools_config) if tools_config else ToolConfig()
        tools = ToolHandler.from_config(tool_config)

        # Create templates config
        templates = TemplateConfig(**templates_config) if templates_config else TemplateConfig()

        # Create model config
        # Note: Use drop_params=True to avoid Bedrock errors with unsupported params
        model_config = GenericAPIModelConfig(
            name=litellm_model_id,
            temperature=spec.temperature if hasattr(spec, 'temperature') else 0.0,
            top_p=None,  # Don't set top_p for Bedrock compatibility
            per_instance_cost_limit=20.0,  # $20 budget for complex tasks
            completion_kwargs={
                "drop_params": True,  # Drop unsupported params like top_p for Bedrock
            },
        )

        # Get model using SWE-agent's native model loading (requires tools config)
        model = get_model(model_config, tool_config)

        # Create the agent
        agent = DefaultAgent(
            model=model,
            tools=tools,
            templates=templates,
            history_processors=[DefaultHistoryProcessor()],
            max_requeries=3,
            name=spec.id,
        )

        logger.info(f"Created SWE-agent for {spec.id} with model {litellm_model_id}")
        return agent

    def _cleanup_tools_dir(self):
        """Clean up /root/tools directory before running to avoid file exists errors."""
        tools_dir = Path("/root/tools")
        try:
            if tools_dir.exists() and not self.use_docker:
                for item in tools_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                logger.debug("Cleaned up /root/tools directory")
        except (PermissionError, OSError) as e:
            logger.debug(f"Skipping /root/tools cleanup: {e}")

    def _cleanup_swe_agent_state(self):
        """
        Clean up SWE-agent state files before running a new instance.

        CRITICAL: This prevents stale data from previous instances from
        contaminating new instances. In particular:
        - /root/model.patch: Contains the git diff from previous submission
        - /root/state.json: Contains working directory state

        If not cleaned, a failed submission command may read the OLD model.patch
        from a previous instance, causing wrong patch output.
        """
        if self.use_docker:
            return  # Docker creates fresh containers

        state_files = [
            Path("/root/model.patch"),
            Path("/root/state.json"),
        ]

        for state_file in state_files:
            try:
                if state_file.exists():
                    state_file.unlink()
                    logger.debug(f"Cleaned up SWE-agent state file: {state_file}")
            except (PermissionError, OSError) as e:
                logger.debug(f"Skipping state file cleanup {state_file}: {e}")

    def _create_environment(self, repo_path: Optional[Path]) -> SWEEnv:
        """
        Create SWE-agent environment.

        Args:
            repo_path: Path to the repository (if any)

        Returns:
            SWEEnv instance
        """
        # Clean tools directory and state files before creating environment (for local mode)
        self._cleanup_tools_dir()
        self._cleanup_swe_agent_state()

        from swerex.deployment.config import get_deployment

        if self.use_docker:
            from swerex.deployment.config import DockerDeploymentConfig
            deployment_config = DockerDeploymentConfig(
                image="python:3.11",
                python_standalone_dir="/root"
            )
        else:
            from swerex.deployment.local import LocalDeploymentConfig
            deployment_config = LocalDeploymentConfig()

        deployment = get_deployment(deployment_config)

        # For pre-existing repos, use PreExistingRepoConfig
        # The repo is already at the path, we just need to tell SWE-agent about it
        repo_config = None
        if repo_path and repo_path.exists():
            repo_name = repo_path.name
            # Use PreExistingRepoConfig - assumes repo is accessible at /{repo_name}
            # For local mode, we'll set up the path via post_startup_commands
            if not self.use_docker:
                # For local mode, don't use repo config - use post_startup_commands
                # Note: autosubmission may fail but EvoMAS fallback (_get_git_diff) handles this
                repo_config = None
            else:
                repo_config = PreExistingRepoConfig(repo_name=repo_name)

        # For local mode, navigate to repo via post_startup_commands
        # Also reset git state to ensure consistency with parent process checkout
        # CRITICAL: Write state.json with correct working_dir because the state command
        # (_state_anthropic) uses os.getcwd() which in local mode returns the parent
        # process's directory, not the shell session's directory after cd.
        post_startup_commands = []
        if repo_path and repo_path.exists() and not self.use_docker:
            post_startup_commands = [
                f"cd {str(repo_path)}",
                "git reset --hard HEAD",  # Ensure working directory matches HEAD
                "git clean -fd",  # Remove untracked files
                "export ROOT=$(pwd -P)",
                # Write state.json with current shell working directory (after cd)
                # Using $(pwd -P) to get the actual shell path, not os.getcwd()
                'echo \'{"working_dir": "\'$(pwd -P)\'"}\' > /root/state.json',
            ]

        env = SWEEnv(
            deployment=deployment,
            repo=repo_config,
            post_startup_commands=post_startup_commands,
        )

        return env

    def run(self, spec: AgentSpec, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Run SWE-agent on a task.

        Args:
            spec: Agent specification
            task: Task/query for the agent
            context: Optional context from other agents

        Returns:
            AgentResult with execution outcome
        """
        if not SWEAGENT_AVAILABLE:
            return AgentResult(
                agent_id=spec.id,
                content="",
                success=False,
                metadata={},
                error="SWE-agent is not available"
            )

        repo_path = self._extract_repo_path(task)
        use_repo_dir = repo_path is not None and repo_path.exists()

        if use_repo_dir:
            logger.info(f"Running SWE-agent in repository: {repo_path}")
            work_dir = repo_path
            self._reset_repo(repo_path)
        else:
            work_dir = Path(tempfile.mkdtemp(prefix="sweagent_task_"))
            logger.info(f"Running SWE-agent in isolated dir: {work_dir}")

        # Create output directory
        output_dir = work_dir / ".sweagent_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Create agent
            agent = self.create_agent(spec, work_dir)

            # Store reference to agent for timeout stats access
            self._last_agent = agent

            # CRITICAL FIX: In local mode, the state command's os.getcwd() returns the
            # parent process's working directory, not the shell session's. We need to
            # explicitly set the working_dir via mock_state to ensure the agent works
            # in the correct repository directory.
            if use_repo_dir and not self.use_docker:
                logger.info(f"Setting mock_state working_dir to: {repo_path}")
                agent.tools.mock_state = {"working_dir": str(repo_path)}

            # Create environment
            env = self._create_environment(repo_path if use_repo_dir else None)

            # Start environment
            env.start()

            # Enhance task with context from previous agents
            # Note: Exclude 'task' field (redundant with main task) and empty fields
            enhanced_task = task
            if context:
                # Filter out redundant/empty context fields
                excluded_keys = {'task'}  # 'task' is already the main input
                relevant_context = {}
                for key, value in context.items():
                    if key in excluded_keys:
                        continue
                    # Skip empty containers
                    if isinstance(value, (dict, list)) and not value:
                        continue
                    if value is None or value == "":
                        continue
                    relevant_context[key] = value

                # Only append context if there's relevant content
                if relevant_context:
                    context_str = "\n\nContext from previous agents:\n"
                    for key, value in relevant_context.items():
                        if isinstance(value, dict):
                            import json
                            context_str += f"- {key}: {json.dumps(value, indent=2)}\n"
                        else:
                            context_str += f"- {key}: {value}\n"
                    enhanced_task = task + context_str

            # Extract problem statement
            problem_statement_text = self._extract_problem_statement(enhanced_task)

            # Extract instance ID from task for unique trajectory file naming
            # Look for "Instance ID: xxx" pattern in the task query
            instance_id = "evomas_task"  # default
            instance_match = re.search(r'Instance ID:\s*([^\s\n]+)', task)
            if instance_match:
                instance_id = instance_match.group(1)
                # Sanitize for filesystem use
                instance_id = instance_id.replace('/', '_').replace('\\', '_')

            # Create problem statement object with unique ID
            problem_statement = TextProblemStatement(
                text=problem_statement_text,
                id=instance_id
            )

            # Run agent (setup is called internally by run())
            logger.info("Running SWE-agent...")
            result = agent.run(
                env=env,
                problem_statement=problem_statement,
                output_dir=output_dir
            )

            # Extract the patch/output
            output = ""
            if hasattr(result, 'info') and result.info:
                raw_submission = result.info.get('submission', '')
                logger.info(f"Raw submission length: {len(raw_submission) if raw_submission else 0}")
                logger.info(f"Raw submission first 200 chars: {raw_submission[:200] if raw_submission else 'None'}")

                output = raw_submission
                if not output:
                    output = result.info.get('exit_status', '')

                # Filter out .sweagent_output files from submission
                # (SWE-agent uses 'git add -A' which includes trajectory files)
                if output:
                    output = self._filter_sweagent_artifacts(output)
                    logger.info(f"Filtered output length: {len(output) if output else 0}")
                    logger.info(f"Filtered output first 200 chars: {output[:200] if output else 'None'}")

            # Also try to get git diff as fallback
            patch_content = ""
            if use_repo_dir:
                logger.info(f"Attempting git diff in: {repo_path}")
                # Check if repo exists and has .git
                if not (repo_path / ".git").exists():
                    logger.warning(f"No .git directory found in {repo_path}")
                # Show git status for debugging
                try:
                    status_result = subprocess.run(
                        ["git", "status", "--short"],
                        cwd=str(repo_path),
                        capture_output=True, text=True, timeout=10
                    )
                    logger.info(f"Git status in {repo_path}: {status_result.stdout[:500] if status_result.stdout else 'empty'}")
                except Exception as e:
                    logger.warning(f"Failed to get git status: {e}")

                patch_content = self._get_git_diff(repo_path)
                logger.info(f"Git diff fallback length: {len(patch_content) if patch_content else 0}")
                if not patch_content:
                    # Try staged changes (excluding SWE-agent output files)
                    try:
                        staged_result = subprocess.run(
                            ["git", "diff", "--cached", "--", ".", ":(exclude).sweagent_output", ":(exclude)*.traj"],
                            cwd=str(repo_path),
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        patch_content = staged_result.stdout
                    except Exception:
                        pass

            # Priority: git diff > submission
            # We prefer our own filtered git diff because SWE-agent's submission
            # can include stale data from previous tasks (trajectory files, etc.)
            final_output = patch_content if patch_content else output
            logger.info(f"Final output source: {'git diff' if patch_content else 'submission'} ({len(final_output) if final_output else 0} bytes)")

            # Determine success
            success = bool(final_output and final_output.strip())

            # Get model stats if available
            model_stats = {}
            if hasattr(agent, 'model') and hasattr(agent.model, 'stats'):
                stats = agent.model.stats
                model_stats = {
                    'instance_cost': getattr(stats, 'instance_cost', 0),
                    'tokens_sent': getattr(stats, 'tokens_sent', 0),
                    'tokens_received': getattr(stats, 'tokens_received', 0),
                    'api_calls': getattr(stats, 'api_calls', 0),
                }

            agent_result = AgentResult(
                agent_id=spec.id,
                content=final_output,
                metadata={
                    'model_stats': model_stats,
                    'had_patch': bool(patch_content),
                    'used_repo_dir': use_repo_dir,
                    'config': self.config_name,
                    'use_docker': self.use_docker,
                },
                error=None if success else "No output generated"
            )

            logger.info(f"SWE-agent {spec.id} completed")
            return agent_result

        except Exception as e:
            logger.error(f"Error running SWE-agent {spec.id}: {e}", exc_info=True)
            return AgentResult(
                agent_id=spec.id,
                content="",
                success=False,
                metadata={},
                error=str(e)
            )
        finally:
            # Cleanup
            if use_repo_dir:
                self._reset_repo(repo_path)
            else:
                try:
                    shutil.rmtree(work_dir, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"Failed to clean up {work_dir}: {e}")

            # Close environment
            try:
                if 'env' in locals():
                    env.close()
            except Exception as e:
                logger.warning(f"Failed to close environment: {e}")
