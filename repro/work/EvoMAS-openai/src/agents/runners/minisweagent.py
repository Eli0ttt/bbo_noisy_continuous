"""
Runner for mini-swe-agent framework.

This runner supports multiple configuration modes:
1. config_default: Full SWE-bench configuration (recommended for best performance)
2. config_simple: Minimal configuration (backward compatible)
3. Custom config: Load from specified path

Configuration is loaded from:
- config/agent_configs/ (relative to project root, or set EVOMAS_AGENT_CONFIG_DIR)
- Or original mini-swe-agent config directory (set MINISWEAGENT_CONFIG_DIR)
"""

import logging
import os
import shutil
import subprocess
import tempfile
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

from ..spec import AgentSpec, AgentResult
from .base import BaseAgentRunner

logger = logging.getLogger(__name__)

# Configuration paths — resolve relative to project root, overridable via env vars
EVOMAS_CONFIG_DIR = Path(os.environ.get(
    "EVOMAS_AGENT_CONFIG_DIR",
    str(Path("config/agent_configs").resolve())
))
MINISWEAGENT_CONFIG_DIR = Path(os.environ.get(
    "MINISWEAGENT_CONFIG_DIR",
    str(Path("config/minisweagent").resolve())
))

# Available configurations
CONFIG_ALIASES = {
    "default": EVOMAS_CONFIG_DIR / "config_default.yaml",
    "simple": EVOMAS_CONFIG_DIR / "config_simple.yaml",
    "swebench": MINISWEAGENT_CONFIG_DIR / "extra" / "swebench.yaml",
    "mini_default": MINISWEAGENT_CONFIG_DIR / "default.yaml",
}

# Try to import mini-swe-agent
try:
    from minisweagent.agents.default import DefaultAgent, AgentConfig
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent import Model
    MINISWEAGENT_AVAILABLE = True
except ImportError:
    logger.warning("mini-swe-agent not available")
    MINISWEAGENT_AVAILABLE = False
    DefaultAgent = None
    AgentConfig = None
    LocalEnvironment = None


def get_config_path(config_spec: str) -> Path:
    """
    Resolve configuration specification to actual path.

    Args:
        config_spec: Config name ("default", "simple", "swebench") or path

    Returns:
        Path to configuration file
    """
    # Check if it's an alias
    if config_spec.lower() in CONFIG_ALIASES:
        return CONFIG_ALIASES[config_spec.lower()]

    # Check if it's a direct path
    config_path = Path(config_spec)
    if config_path.exists():
        return config_path

    # Add .yaml extension if missing
    if not config_path.suffix:
        config_path = config_path.with_suffix(".yaml")

    # Check in EvoMAS config dir
    evomas_path = EVOMAS_CONFIG_DIR / config_path.name
    if evomas_path.exists():
        return evomas_path

    # Check in mini-swe-agent config dir
    mini_path = MINISWEAGENT_CONFIG_DIR / config_path.name
    if mini_path.exists():
        return mini_path

    mini_extra_path = MINISWEAGENT_CONFIG_DIR / "extra" / config_path.name
    if mini_extra_path.exists():
        return mini_extra_path

    raise FileNotFoundError(
        f"Configuration not found: {config_spec}. "
        f"Available: {list(CONFIG_ALIASES.keys())} or provide a valid path"
    )


def load_config(config_spec: str = "default") -> Dict[str, Any]:
    """
    Load agent configuration from file.

    Args:
        config_spec: Config name or path

    Returns:
        Configuration dictionary
    """
    config_path = get_config_path(config_spec)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")
    return config


class EvoMASModelAdapter:
    """
    Adapter to make EvoMAS models compatible with mini-swe-agent's Model protocol.
    """

    def __init__(self, model_wrapper, config: Optional[Dict[str, Any]] = None):
        self.model_wrapper = model_wrapper
        self.config = config or {}
        self.cost = 0.0
        self.n_calls = 0

        # Get the underlying smolagents model if available
        if hasattr(model_wrapper, 'smolagents_model'):
            self.smolagents_model = model_wrapper.smolagents_model
        else:
            self.smolagents_model = None

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        """Query the model following mini-swe-agent's protocol."""
        self.n_calls += 1

        try:
            if self.smolagents_model:
                from smolagents import ChatMessage, MessageRole

                chat_messages = []
                for msg in messages:
                    role_str = msg['role'].upper()
                    if hasattr(MessageRole, role_str):
                        role = getattr(MessageRole, role_str)
                    else:
                        role = MessageRole.USER if msg['role'] == 'user' else MessageRole.ASSISTANT

                    chat_messages.append(ChatMessage(role=role, content=msg['content']))

                response = self.smolagents_model.generate(chat_messages, **kwargs)

                if hasattr(response, 'token_usage') and response.token_usage is not None:
                    token_usage = response.token_usage
                    total_tokens = token_usage.input_tokens + token_usage.output_tokens
                    self.cost += total_tokens * 0.00001

                return {'content': response.content}
            else:
                prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                return {'content': f"[Model response to: {prompt[:100]}...]"}

        except Exception as e:
            logger.error(f"Error in model query: {e}")
            return {'content': f"Error: {str(e)}"}

    def get_template_vars(self) -> dict[str, Any]:
        return {
            'model_name': getattr(self.model_wrapper, 'model_name', 'unknown'),
            'model_id': getattr(self.model_wrapper, 'model_id', 'unknown'),
        }


class MinisweagentRunner(BaseAgentRunner):
    """
    Runner for mini-swe-agent framework with configurable settings.

    Supports:
    - config_default: Full SWE-bench configuration (250 steps, detailed prompts)
    - config_simple: Minimal configuration (20 steps, basic prompts)
    - Custom configs from path
    """

    def __init__(
        self,
        working_dir: Optional[Path] = None,
        config: str = "default",
        **kwargs
    ):
        """
        Initialize mini-swe-agent runner.

        Args:
            working_dir: Working directory for code execution
            config: Configuration name ("default", "simple", "swebench") or path
        """
        if not MINISWEAGENT_AVAILABLE:
            logger.warning("mini-swe-agent not available, runner will fail")

        self.working_dir = working_dir or Path.cwd()
        self.config_name = config

        # Load configuration
        try:
            self.full_config = load_config(config)
            self.agent_config = self.full_config.get('agent', {})
            self.env_config = self.full_config.get('environment', {})
        except FileNotFoundError as e:
            logger.warning(f"Config not found, using defaults: {e}")
            self.full_config = {}
            self.agent_config = {}
            self.env_config = {}

        logger.info(f"MinisweagentRunner initialized with config: {config}")
        logger.info(f"  step_limit: {self.agent_config.get('step_limit', 'default')}")
        logger.info(f"  cost_limit: {self.agent_config.get('cost_limit', 'default')}")

    def create_agent(self, spec: AgentSpec, working_dir: Path) -> Any:
        """Create a mini-swe-agent with configured settings."""
        if not MINISWEAGENT_AVAILABLE:
            raise ImportError("mini-swe-agent is not available")

        # Ensure API keys are loaded from EvoMAS .env
        try:
            from dotenv import load_dotenv
            import os
            evomas_root = Path(__file__).parent.parent.parent.parent
            env_file = evomas_root / '.env'
            if env_file.exists():
                load_dotenv(env_file)
                logger.debug(f"Loaded environment from {env_file}")
        except ImportError:
            logger.warning("dotenv not available, relying on existing environment")

        # Use mini-swe-agent's native model loading (uses litellm directly)
        # This ensures we use the exact same model interface as the original
        from minisweagent.models import get_model as miniswe_get_model

        # Convert EvoMAS model_id format to litellm format
        # e.g., "openai:gpt-4.1" -> "gpt-4.1" (litellm auto-detects OpenAI)
        # e.g., "anthropic:claude-3-5-sonnet" -> "anthropic/claude-3-5-sonnet"
        model_id = spec.model_id
        if ':' in model_id:
            provider, model_name = model_id.split(':', 1)
            if provider.lower() == 'openai':
                # litellm auto-detects OpenAI models
                model_id = model_name
            else:
                # Other providers use "provider/model" format
                model_id = f"{provider}/{model_name}"

        logger.info(f"Loading model with litellm: {model_id}")
        model = miniswe_get_model(model_id)

        # Create local environment
        local_env_config = {
            'cwd': str(working_dir),
            'timeout': self.env_config.get('timeout', 60),
            'env': self.env_config.get('env', {})
        }
        env = LocalEnvironment(**local_env_config)

        # Prepare agent config - replace /testbed with actual path
        agent_config = self.agent_config.copy()
        for key in ['system_template', 'instance_template', 'action_observation_template',
                    'format_error_template', 'timeout_template']:
            if key in agent_config and agent_config[key]:
                agent_config[key] = agent_config[key].replace('/testbed', str(working_dir))

        # Create agent
        agent = DefaultAgent(
            model=model,
            env=env,
            **agent_config
        )

        logger.info(f"Created mini-swe-agent for {spec.id} with model {spec.model_id}")
        logger.info(f"  Config: {self.config_name}, step_limit={agent_config.get('step_limit')}")
        return agent

    def _extract_repo_path(self, task: str) -> Optional[Path]:
        """Extract repository path from SWE-bench task query."""
        import re
        match = re.search(r'Repository:\s*(/[^\s\n]+)', task)
        if match:
            return Path(match.group(1))
        return None

    def _reset_repo(self, repo_path: Path):
        """Reset repository to clean state."""
        try:
            subprocess.run(["git", "reset", "--hard"], cwd=str(repo_path),
                          capture_output=True, timeout=30)
            subprocess.run(["git", "clean", "-fd"], cwd=str(repo_path),
                          capture_output=True, timeout=30)
        except Exception as e:
            logger.warning(f"Failed to reset repo {repo_path}: {e}")

    def _get_git_diff(self, repo_path: Path) -> str:
        """Get git diff of changes in repository."""
        try:
            result = subprocess.run(["git", "diff"], cwd=str(repo_path),
                                   capture_output=True, text=True, timeout=30)
            return result.stdout
        except Exception as e:
            logger.warning(f"Failed to get git diff from {repo_path}: {e}")
            return ""

    def _extract_problem_statement(self, task: str) -> str:
        """
        Extract problem statement from EvoMAS task format.

        For proper alignment with mini-swe-agent:
        - If task contains "Problem Statement:", extract just that part
        - Otherwise, pass the task as-is (for raw SWE-bench format)

        The agent's instance_template will handle formatting via {{task}}.
        """
        import re

        # Check if this is EvoMAS wrapped format (legacy with CRITICAL INSTRUCTIONS)
        if "Problem Statement:" in task and "CRITICAL INSTRUCTIONS:" in task:
            # Extract just the problem statement, ignore the CRITICAL INSTRUCTIONS
            # as the agent's template already has proper instructions
            match = re.search(r'Problem Statement:\s*\n(.*?)(?=\n\nCRITICAL INSTRUCTIONS:)',
                             task, re.DOTALL)
            if match:
                return match.group(1).strip()

        # Check for new simpler format (Problem Statement at end)
        # Use greedy .* to capture everything including blank lines in the problem
        if "Problem Statement:" in task:
            match = re.search(r'Problem Statement:\s*\n(.*)', task, re.DOTALL)
            if match:
                return match.group(1).strip()

        # Return as-is for raw SWE-bench format (just problem_statement)
        return task

    def run(self, spec: AgentSpec, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Run mini-swe-agent with configured settings."""
        repo_path = self._extract_repo_path(task)
        use_repo_dir = repo_path is not None and repo_path.exists()

        if use_repo_dir:
            logger.info(f"Running mini-swe-agent in repository: {repo_path}")
            work_dir = repo_path
            self._reset_repo(repo_path)
        else:
            work_dir = Path(tempfile.mkdtemp(prefix="miniswe_task_"))
            logger.info(f"Running mini-swe-agent in isolated dir: {work_dir}")

        try:
            agent = self.create_agent(spec, work_dir)

            # Enhance task with context
            enhanced_task = task
            if context:
                context_str = "\n\nContext from previous agents:\n"
                for key, value in context.items():
                    context_str += f"- {key}: {value}\n"
                enhanced_task = task + context_str

            problem_statement = self._extract_problem_statement(enhanced_task)

            logger.info(f"Running agent with step_limit={self.agent_config.get('step_limit')}")
            exit_status, exit_message = agent.run(problem_statement)

            # The exit_message from Submitted exception contains the git diff
            # (from: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached)
            # This is the primary output for SWE-bench patches
            output = exit_message

            # For local execution (non-Docker), also capture git diff as fallback
            # In case the agent didn't use the submission command properly
            patch_content = ""
            if use_repo_dir:
                # Try to get uncommitted changes
                patch_content = self._get_git_diff(repo_path)
                if not patch_content:
                    # Also try staged changes
                    try:
                        import subprocess
                        result = subprocess.run(
                            ["git", "diff", "--cached"],
                            cwd=str(repo_path),
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        patch_content = result.stdout
                    except Exception:
                        pass
                if patch_content:
                    logger.info(f"Captured git diff fallback ({len(patch_content)} bytes)")

            # Priority: submission output > git diff fallback
            # If exit_message looks like a valid diff, use it
            if exit_message and exit_message.strip().startswith('diff --git'):
                final_output = exit_message
                logger.info("Using submission output (valid diff format)")
            elif patch_content:
                final_output = patch_content
                logger.info("Using git diff fallback")
            else:
                final_output = output
                logger.info("Using raw output (no diff captured)")
            success = exit_status == "Submitted"

            result = AgentResult(
                agent_id=spec.id,
                content=final_output,
                metadata={
                    'exit_status': exit_status,
                    'exit_message': exit_message,
                    'model_cost': agent.model.cost,
                    'model_calls': agent.model.n_calls,
                    'total_messages': len(agent.messages),
                    'had_patch_file': bool(patch_content),
                    'used_repo_dir': use_repo_dir,
                    'config': self.config_name,
                    'step_limit': self.agent_config.get('step_limit'),
                },
                error=None if success else exit_message
            )

            logger.info(f"Agent {spec.id} completed: {exit_status}")
            logger.info(f"  Model calls: {agent.model.n_calls}, Cost: ${agent.model.cost:.4f}")
            return result

        except Exception as e:
            logger.error(f"Error running mini-swe-agent {spec.id}: {e}", exc_info=True)
            return AgentResult(
                agent_id=spec.id,
                content="",
                success=False,
                metadata={},
                error=str(e)
            )
        finally:
            if use_repo_dir:
                self._reset_repo(repo_path)
            else:
                try:
                    shutil.rmtree(work_dir, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"Failed to clean up {work_dir}: {e}")
