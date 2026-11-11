"""
Runner for smolagents framework.
"""

import logging
import re
import subprocess
from typing import Any, Dict, Optional
from pathlib import Path

from ..spec import AgentSpec, AgentResult
from .base import BaseAgentRunner

logger = logging.getLogger(__name__)

# Path to WorkBench custom prompts
# __file__ is in src/agents/runners/smolagents.py
# Need to go up 3 levels to src/, then into prompts/templates/workbench
WORKBENCH_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "templates" / "workbench"

try:
    from smolagents import CodeAgent, ToolCallingAgent
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    logger.warning("smolagents not available")
    SMOLAGENTS_AVAILABLE = False
    CodeAgent = None
    ToolCallingAgent = None


class SmolagentsRunner(BaseAgentRunner):
    """Runner for smolagents framework (CodeAgent, ToolCallingAgent)."""

    def __init__(self):
        """Initialize smolagents runner."""
        if not SMOLAGENTS_AVAILABLE:
            logger.warning("smolagents not available, runner will fail")

    def _extract_repo_path(self, task: str) -> Optional[Path]:
        """Extract repository path from SWE-bench task query."""
        match = re.search(r'Repository:\s*(/[^\s\n]+)', task)
        if match:
            return Path(match.group(1))
        return None

    def _is_swebench_task(self, task: str) -> bool:
        """Check if this is a SWE-bench task."""
        return "Repository:" in task and "diff --git" in task

    def _reset_repo(self, repo_path: Path):
        """Reset repository to clean state."""
        try:
            subprocess.run(
                ["git", "reset", "--hard"],
                cwd=str(repo_path),
                capture_output=True,
                timeout=30
            )
            subprocess.run(
                ["git", "clean", "-fd"],
                cwd=str(repo_path),
                capture_output=True,
                timeout=30
            )
        except Exception as e:
            logger.warning(f"Failed to reset repo {repo_path}: {e}")

    def create_agent(self, spec: AgentSpec) -> Any:
        """Create a smolagents agent from specification.

        Args:
            spec: Agent specification

        Returns:
            smolagents agent instance (CodeAgent or ToolCallingAgent)
        """
        if not SMOLAGENTS_AVAILABLE:
            raise ImportError("smolagents is not available")

        # Import model loader
        from src.models import get_model

        # Prepare model parameters
        model_params = {
            "max_tokens": spec.max_tokens,
            "temperature": spec.temperature
        }

        # Add device parameter if specified (for local models)
        if spec.device is not None:
            model_params["device"] = spec.device
            logger.info(f"Using device: {spec.device} for model {spec.model_id}")

        # Load the model
        model = get_model(spec.model_id, **model_params)

        # Determine agent class
        agent_class = CodeAgent
        if spec.agent_type == "ToolCallingAgent":
            agent_class = ToolCallingAgent
        elif spec.agent_type == "CodeAgent":
            agent_class = CodeAgent
        else:
            logger.warning(f"Unknown agent type {spec.agent_type}, defaulting to CodeAgent")
            agent_class = CodeAgent

        # Create agent
        agent = agent_class(
            tools=spec.tools if spec.tools else [],
            model=model.smolagents_model,
            additional_authorized_imports=["src.models", "src.tools"]
        )

        logger.info(f"Created {spec.agent_type} for agent {spec.id}")
        return agent

    def run(self, spec: AgentSpec, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Run a smolagents agent.

        Args:
            spec: Agent specification
            task: Task/query for the agent
            context: Optional context from other agents

        Returns:
            AgentResult with execution outcome
        """
        # Check if this is a SWE-bench task
        repo_path = self._extract_repo_path(task)
        is_swebench = repo_path is not None and repo_path.exists()

        if is_swebench:
            logger.info(f"Detected SWE-bench task with repository: {repo_path}")
            # Reset repo to clean state before running
            self._reset_repo(repo_path)

        try:
            # Import model loader
            from src.models import get_model

            # Prepare model parameters
            model_params = {
                "max_tokens": spec.max_tokens,
                "temperature": spec.temperature
            }

            # Add device parameter if specified (for local models)
            if spec.device is not None:
                model_params["device"] = spec.device
                logger.info(f"Using device: {spec.device} for model {spec.model_id}")

            # Load the model
            model = get_model(spec.model_id, **model_params)

            # Create the agent (CodeAgent or ToolCallingAgent)
            agent_class = CodeAgent
            if spec.agent_type == "ToolCallingAgent":
                agent_class = ToolCallingAgent
            elif spec.agent_type == "CodeAgent":
                agent_class = CodeAgent
            else:
                logger.warning(f"Unknown agent type {spec.agent_type}, defaulting to CodeAgent")
                agent_class = CodeAgent

            # Determine which tools to use
            tools_to_use = []

            # For SWE-bench tasks, add file reading tools
            if is_swebench:
                try:
                    from smolagents import tool

                    @tool
                    def read_file(file_path: str) -> str:
                        """
                        Read the contents of a file from the repository.

                        Args:
                            file_path: Path to the file relative to repository root, or absolute path

                        Returns:
                            The contents of the file as a string
                        """
                        try:
                            # Handle both relative and absolute paths
                            if file_path.startswith('/'):
                                full_path = Path(file_path)
                            else:
                                full_path = repo_path / file_path

                            if not full_path.exists():
                                return f"Error: File not found: {file_path}"

                            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                                return f.read()
                        except Exception as e:
                            return f"Error reading file: {str(e)}"

                    @tool
                    def list_files(directory: str = ".") -> str:
                        """
                        List files in a directory of the repository.

                        Args:
                            directory: Directory path relative to repository root (default: root)

                        Returns:
                            List of files and directories
                        """
                        try:
                            if directory.startswith('/'):
                                dir_path = Path(directory)
                            else:
                                dir_path = repo_path / directory

                            if not dir_path.exists():
                                return f"Error: Directory not found: {directory}"

                            items = []
                            for item in sorted(dir_path.iterdir()):
                                if item.name.startswith('.'):
                                    continue
                                prefix = "[DIR] " if item.is_dir() else "[FILE]"
                                items.append(f"{prefix} {item.name}")

                            return "\n".join(items[:100])  # Limit to 100 items
                        except Exception as e:
                            return f"Error listing directory: {str(e)}"

                    @tool
                    def search_in_files(pattern: str, file_glob: str = "*.py") -> str:
                        """
                        Search for a pattern in files matching the glob pattern.

                        Args:
                            pattern: Text pattern to search for
                            file_glob: Glob pattern for files to search (default: *.py)

                        Returns:
                            Matching lines with file paths and line numbers
                        """
                        try:
                            import fnmatch
                            results = []
                            for file_path in repo_path.rglob(file_glob):
                                if '.git' in str(file_path):
                                    continue
                                try:
                                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                                        for i, line in enumerate(f, 1):
                                            if pattern.lower() in line.lower():
                                                rel_path = file_path.relative_to(repo_path)
                                                results.append(f"{rel_path}:{i}: {line.strip()}")
                                                if len(results) >= 50:
                                                    return "\n".join(results) + "\n... (truncated)"
                                except:
                                    pass
                            return "\n".join(results) if results else f"No matches found for '{pattern}'"
                        except Exception as e:
                            return f"Error searching: {str(e)}"

                    tools_to_use.extend([read_file, list_files, search_in_files])
                    logger.info(f"Added SWE-bench file tools for repository: {repo_path}")
                except Exception as e:
                    logger.warning(f"Failed to create SWE-bench tools: {e}")

            if spec.tools:
                for tool_spec in spec.tools:
                    # Check if it's a WorkBench tool
                    if tool_spec == "workbench_email":
                        from src.tools.workbench_tools_email import get_all_email_tools
                        tools_to_use.extend(get_all_email_tools())
                        logger.info(f"Loaded {len(get_all_email_tools())} WorkBench email tools for agent {spec.id}")
                    elif tool_spec == "workbench_calendar":
                        from src.tools.workbench_tools_calendar import get_all_calendar_tools
                        tools_to_use.extend(get_all_calendar_tools())
                        logger.info(f"Loaded {len(get_all_calendar_tools())} WorkBench calendar tools for agent {spec.id}")
                    elif tool_spec == "workbench_analytics":
                        from src.tools.workbench_tools_all import get_all_analytics_tools
                        tools_to_use.extend(get_all_analytics_tools())
                        logger.info(f"Loaded {len(get_all_analytics_tools())} WorkBench analytics tools for agent {spec.id}")
                    elif tool_spec == "workbench_project_management":
                        from src.tools.workbench_tools_all import get_all_project_management_tools
                        tools_to_use.extend(get_all_project_management_tools())
                        logger.info(f"Loaded {len(get_all_project_management_tools())} WorkBench project_management tools for agent {spec.id}")
                    elif tool_spec == "workbench_customer_relationship_manager":
                        from src.tools.workbench_tools_all import get_all_crm_tools
                        tools_to_use.extend(get_all_crm_tools())
                        logger.info(f"Loaded {len(get_all_crm_tools())} WorkBench CRM tools for agent {spec.id}")
                    elif tool_spec == "workbench_multi_domain":
                        # Multi-domain uses all tools
                        from src.tools.workbench_tools_email import get_all_email_tools
                        from src.tools.workbench_tools_calendar import get_all_calendar_tools
                        from src.tools.workbench_tools_all import (
                            get_all_analytics_tools, get_all_project_management_tools,
                            get_all_crm_tools, get_all_company_directory_tools
                        )
                        tools_to_use.extend(get_all_email_tools())
                        tools_to_use.extend(get_all_calendar_tools())
                        tools_to_use.extend(get_all_analytics_tools())
                        tools_to_use.extend(get_all_project_management_tools())
                        tools_to_use.extend(get_all_crm_tools())
                        tools_to_use.extend(get_all_company_directory_tools())
                        logger.info(f"Loaded all {len(tools_to_use)} WorkBench tools for multi-domain agent {spec.id}")
                    else:
                        # Regular tool - would be loaded normally
                        pass

            # Create agent with model
            agent = agent_class(
                tools=tools_to_use,
                model=model.smolagents_model,
                additional_authorized_imports=["src.models", "src.tools", "src.data_generation", "json"]
            )

            # Build full prompt with context
            full_prompt = task

            # Check if this is a WorkBench task (needs special instructions)
            is_workbench = any(t.startswith("workbench_") for t in (spec.tools or []))
            if is_workbench:
                workbench_instructions = self._get_workbench_instructions()
                if workbench_instructions:
                    full_prompt = workbench_instructions + "\n\n" + full_prompt

            if context and context.get('reports'):
                full_prompt += "\n\nContext from other agents:\n"
                for agent_id, report in context['reports'].items():
                    full_prompt += f"\n--- {agent_id} ---\n{report}\n"

            # Use agent.run() method (NOT direct model call)
            logger.info(f"Running {spec.agent_type} {spec.id} with model {spec.model_id}")

            # CodeAgent.run() returns the final answer string
            content = agent.run(full_prompt)

            # Convert to string if needed
            if not isinstance(content, str):
                content = str(content)

            # Check if content is empty or None
            if not content or content.strip() == "":
                logger.warning(f"Agent {spec.id} returned empty content - treating as failure")
                return AgentResult(
                    agent_id=spec.id,
                    content="",
                    success=False,
                    error="Agent returned empty response - execution may have failed",
                    metadata={'agent_type': spec.agent_type}
                )

            logger.info(f"Agent {spec.id} completed successfully with {len(content)} chars")

            # Extract token usage from agent logs if available
            token_usage = self._extract_token_usage_from_agent(agent, spec.model_id)

            # Build metadata with token usage
            metadata = {
                'agent_type': spec.agent_type,
                'model_id': spec.model_id
            }

            # Add token usage if available
            if token_usage:
                metadata.update(token_usage)
                logger.info(f"Agent {spec.id} token usage: {token_usage.get('input_tokens', 0)} input, {token_usage.get('output_tokens', 0)} output")

            return AgentResult(
                agent_id=spec.id,
                content=content,
                success=True,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Agent {spec.id} execution failed: {e}")
            import traceback
            traceback.print_exc()
            return AgentResult(
                agent_id=spec.id,
                content="",
                success=False,
                error=str(e),
                metadata={'agent_type': spec.agent_type}
            )
        finally:
            # For SWE-bench: reset repo to clean state after running
            if is_swebench:
                self._reset_repo(repo_path)

    def _get_workbench_instructions(self) -> Optional[str]:
        """
        Load WorkBench evaluation instructions to prepend to task.

        Returns:
            Instructions string or None if file not found
        """
        try:
            instructions_file = WORKBENCH_PROMPT_PATH / "system_prompt.txt"

            if not instructions_file.exists():
                logger.warning(f"WorkBench instructions file not found: {instructions_file}")
                return None

            with open(instructions_file, 'r') as f:
                instructions = f.read()

            logger.info("Loaded WorkBench evaluation instructions")
            return instructions

        except Exception as e:
            logger.warning(f"Failed to load WorkBench instructions: {e}")
            return None

    def _extract_token_usage_from_agent(self, agent, model_id: str) -> Optional[Dict[str, int]]:
        """
        Extract token usage information from the model instance.

        Args:
            agent: The CodeAgent instance
            model_id: Model identifier to determine provider

        Returns:
            Dictionary with 'input_tokens', 'output_tokens', 'total_tokens' or None
        """
        try:
            # Try to get the model from the agent
            model = agent.model if hasattr(agent, 'model') else None
            if not model:
                logger.debug(f"No model found on agent for {model_id}")
                return None

            logger.debug(f"Model type: {type(model).__name__}")
            logger.debug(f"Model has parent: {hasattr(model, 'parent')}")
            logger.debug(f"Model has get_cumulative_token_usage: {hasattr(model, 'get_cumulative_token_usage')}")

            # The model object is actually a MockSmolagentsModel wrapper
            # We need to get the parent model instance which has the actual token tracking
            if hasattr(model, 'parent'):
                # This is our custom OpenAI/Anthropic model wrapper
                parent_model = model.parent
                logger.debug(f"Parent model type: {type(parent_model).__name__}")
                if hasattr(parent_model, 'get_cumulative_token_usage'):
                    usage = parent_model.get_cumulative_token_usage()
                    logger.debug(f"Got token usage from parent: {usage}")
                    return usage

            # For Bedrock models or other types, try to get token usage from model attributes
            if hasattr(model, 'get_cumulative_token_usage'):
                usage = model.get_cumulative_token_usage()
                logger.debug(f"Got token usage from model: {usage}")
                return usage

            logger.debug(f"No token usage method found for model {model_id}")
            return None

        except Exception as e:
            logger.warning(f"Failed to extract token usage from model: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
