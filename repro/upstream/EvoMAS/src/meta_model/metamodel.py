"""
Meta Model for evolving Multi-Agent Systems.

Based on the formal definition:
- MAS Configuration: C = (V, E, {m_i}, {Γ_i}, {p_i})
- MAS Gradient: g = f(τ, R(M))
- Meta-Model: πθ(Y_edit | M, g)
"""

import yaml
import random
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from src.models.model import get_model
from src.prompts.registry import PromptRegistry
from src.prompts.render import render_prompt
from src.mas.interpreter import interpret_mas
from src.meta_model.experience import ActionExperience, MemoryStore

logger = logging.getLogger(__name__)


def extract_yaml_from_response(response: str) -> str:
    """
    Extract YAML configuration from LLM response.

    Looks for YAML between ```yaml and ``` markers or after "Updated Configuration:" marker.

    Args:
        response: LLM response text

    Returns:
        Extracted YAML string
    """
    import re

    # Try to find YAML code block
    yaml_pattern = r"```yaml\s*\n(.*?)\n```"
    match = re.search(yaml_pattern, response, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()

    # Try to find after "Updated Configuration:" or "Offspring Configuration:"
    config_patterns = [
        r"Updated Configuration:\s*```yaml\s*\n(.*?)\n```",
        r"Offspring Configuration:\s*```yaml\s*\n(.*?)\n```",
        r"Updated Configuration:\s*\n(.*?)(?:\n\n|$)",
        r"Offspring Configuration:\s*\n(.*?)(?:\n\n|$)"
    ]

    for pattern in config_patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # If no markers found, try to parse entire response as YAML
    logger.warning("No YAML markers found, attempting to parse entire response")
    return response.strip()


def validate_mas_config(config_str: str) -> bool:
    """
    Validate that a string is valid MAS YAML configuration.

    Args:
        config_str: YAML string to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        config_dict = yaml.safe_load(config_str)

        # Check required fields
        if not isinstance(config_dict, dict):
            return False

        required_fields = ['name', 'backend', 'agents']
        for field in required_fields:
            if field not in config_dict:
                logger.error(f"Missing required field: {field}")
                return False

        # Check agents structure
        if not isinstance(config_dict['agents'], dict) or len(config_dict['agents']) == 0:
            logger.error("Invalid agents structure")
            return False

        return True

    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        return False


class MetaModel:
    """Meta model for evolutionary optimization of MAS configurations."""

    def __init__(
        self,
        model_id: str = "bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        verbose: bool = True,
        memory_path: Optional[str] = None,
        memory_evolution: bool = True,
    ):
        """
        Initialize meta model.

        Args:
            model_id: Model to use for meta-reasoning
            temperature: Sampling temperature
            max_tokens: Maximum tokens for generation
            verbose: Whether to show detailed logs
            memory_path: Path to load/save memory (optional). If the file
                exists it is loaded; if it does not exist, memory starts empty
                and (when memory_evolution is True) the file is created at
                first save.
            memory_evolution: If True (default), update_memory() records
                new experiences and persists them when memory_path is set.
                If False, memory is read-only — existing experiences still
                inform selection/generation, but no new entries are added or
                saved. Use False for evaluation-only runs where the existing
                memory should not be mutated by this run.
        """
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose

        # Load meta model
        self.model = get_model(
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Load prompt templates
        self.prompt_registry = PromptRegistry()

        # Initialize memory
        if memory_path and Path(memory_path).exists():
            self.memory = MemoryStore.load(memory_path)
            logger.info(f"Loaded {len(self.memory)} experiences from memory ({memory_path})")
        else:
            self.memory = MemoryStore()
            logger.info(f"Initialized empty memory (path={memory_path})")

        self.memory_path = memory_path
        self.memory_evolution = memory_evolution
        if not memory_evolution:
            logger.info("Memory evolution disabled — memory is read-only for this run")

        logger.info(f"Initialized MetaModel with {model_id}")

    def select(
        self,
        task_query: str,
        task_description: str,
        pool_dir: str,
        k: int = 2,
        mas_index=None
    ) -> List[str]:
        """
        Select action: Choose k parent configurations from pool using LLM reasoning.

        Implements: {C_1, ..., C_k} ~ π^S(· | q, D(C̄))

        Args:
            task_query: Task query string
            task_description: Description of task characteristics
            pool_dir: Directory containing MAS configuration pool
            k: Number of configurations to select
            mas_index: Optional MASIndex with rich performance metadata

        Returns:
            List of k selected configuration file paths
        """
        logger.info(f"SELECT: Choosing {k} configurations using LLM")

        # Load pool metadata
        from src.meta_model.selection import load_pool_metadata
        pool_metadata = load_pool_metadata(pool_dir)

        if not pool_metadata:
            raise ValueError(f"No configurations found in pool: {pool_dir}")

        if len(pool_metadata) < k:
            logger.warning(f"Pool has {len(pool_metadata)} configs, but requested {k}")
            k = len(pool_metadata)

        # Use MAS index for rich metadata if available, otherwise fallback to YAML metadata
        if mas_index is not None:
            pool_metadata_str = mas_index.get_selection_context(max_configs=20)
        else:
            pool_metadata_str = ""
            for name, meta in pool_metadata.items():
                pool_metadata_str += f"\n### {name}\n"
                pool_metadata_str += f"- Description: {meta.get('description', 'N/A')}\n"
                pool_metadata_str += f"- Number of agents: {meta.get('num_agents', 'N/A')}\n"
                successful_tasks = meta.get('successful_tasks', [])
                if successful_tasks:
                    pool_metadata_str += f"- Successful tasks: {len(successful_tasks)}\n"
                    for task in successful_tasks[:2]:
                        pool_metadata_str += f"  - {task.get('notes', task.get('q', ''))[:100]}\n"

        # Load prompt template
        template = self.prompt_registry.get("meta_select")

        # Prepare prompt variables
        prompt_vars = {
            "task_query": task_query[:500],  # Truncate long queries
            "task_description": task_description,
            "pool_metadata": pool_metadata_str,
            "k": str(k)
        }

        # Add memory context if available
        if len(self.memory) > 0:
            memory_context = self.get_memory_context(max_experiences=3)
            prompt_vars["memory_context"] = f"\n\n## Previous Experiences\n\n{memory_context}"
        else:
            prompt_vars["memory_context"] = ""

        # Render prompt
        prompt = render_prompt(template, prompt_vars)

        logger.info("Calling meta model for selection...")

        # Call meta model
        try:
            import asyncio
            import json
            import re

            if asyncio.iscoroutinefunction(self.model.__call__):
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                response = loop.run_until_complete(self.model(prompt))
            else:
                response = self.model(prompt)

            if not isinstance(response, str):
                response = str(response)

            logger.info(f"Received response ({len(response)} chars)")

            # Extract JSON from response
            json_pattern = r"```json\s*\n(.*?)\n```"
            match = re.search(json_pattern, response, re.DOTALL)

            selected_names = []
            if match:
                try:
                    selection_data = json.loads(match.group(1))
                    selected_names = [item["name"] for item in selection_data.get("selected", [])]
                except json.JSONDecodeError:
                    logger.warning("Failed to parse selection JSON")

            # Validate selected names exist in pool
            valid_selections = []
            for name in selected_names[:k]:
                if name in pool_metadata:
                    valid_selections.append(pool_metadata[name]["path"])
                    logger.info(f"  Selected: {name}")

            # Fallback: if LLM selection failed, use programmatic selection
            if len(valid_selections) < k:
                logger.warning(f"LLM selected {len(valid_selections)}/{k}, falling back to programmatic selection")
                from src.meta_model.selection import selection_operator
                fallback = selection_operator(task_query, pool_dir, k - len(valid_selections))
                for path in fallback:
                    if path not in valid_selections:
                        valid_selections.append(path)
                        if len(valid_selections) >= k:
                            break

            logger.info(f"Selected {len(valid_selections)} configurations")
            return valid_selections[:k]

        except Exception as e:
            logger.error(f"Selection failed: {e}")
            logger.warning("Falling back to programmatic selection")
            from src.meta_model.selection import selection_operator
            return selection_operator(task_query, pool_dir, k)

    def generate(
        self,
        mas_config_path: Optional[str],
        task_samples: List[Dict[str, Any]],
        task_description: str,
        model_list: Optional[List[str]] = None
    ) -> str:
        """
        Generate action: Adapt selected MAS configuration to current tasks.

        Args:
            mas_config_path: Path to selected MAS configuration (None for empty pool)
            task_samples: Sample tasks for analysis
            task_description: Description of task characteristics
            model_list: List of available models for MAS agents

        Returns:
            Generated MAS configuration as YAML string
        """
        # Handle empty pool case - use default template
        if mas_config_path is None or mas_config_path == "None":
            logger.info("GENERATE: Creating new MAS from scratch (empty pool)")
            mas_config = """# No parent configuration - generate from scratch
# Use the MAS Configuration Structure above as reference
name: new_mas
description: Auto-generated MAS for the target tasks
backend: smolagents

agents:
  worker:
    id: worker
    role: worker
    agent_type: CodeAgent
    model_id: bedrock:us.anthropic.claude-3-5-sonnet-20241022-v2:0
    prompt: worker
    tools: []
    max_tokens: 4096
    temperature: 0.7
    device: null

topology:
  reports_to:
    worker: []

execution:
  parallel_workers: false
  timeout: 600
  max_retries: 0
"""
        else:
            logger.info(f"GENERATE: Adapting {mas_config_path}")
            # Load existing config
            with open(mas_config_path, 'r') as f:
                mas_config = f.read()

        # Load prompt template
        template = self.prompt_registry.get("meta_generate")

        # Format task samples
        task_samples_str = "\n".join([
            f"Task {i}: {task['query'][:100]}..."
            for i, task in enumerate(task_samples[:3])
        ])

        # Format model list
        if model_list:
            model_list_str = "\n".join([f"  - {model}" for model in model_list])
            model_constraint = f"\n\n**Available Models for MAS Agents:**\nYou can ONLY use the following models for agent model_id fields:\n{model_list_str}\n"
        else:
            model_constraint = ""

        # Prepare prompt variables
        prompt_vars = {
            "mas_config": mas_config,
            "task_samples": task_samples_str,
            "task_description": task_description,
            "model_constraint": model_constraint
        }

        # Add memory context if available
        if len(self.memory) > 0:
            memory_context = self.get_memory_context(max_experiences=3)
            prompt_vars["memory_context"] = f"\n\n## Previous Experiences\n\n{memory_context}"
        else:
            prompt_vars["memory_context"] = ""

        # Render prompt
        prompt = render_prompt(template, prompt_vars)

        logger.info("Calling meta model for generation...")

        # Call meta model
        try:
            import asyncio
            if asyncio.iscoroutinefunction(self.model.__call__):
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                response = loop.run_until_complete(self.model(prompt))
            else:
                response = self.model(prompt)

            if not isinstance(response, str):
                response = str(response)

            logger.info(f"Received response ({len(response)} chars)")

            # Extract YAML from response
            generated_config = extract_yaml_from_response(response)

            # Validate generated config
            if not validate_mas_config(generated_config):
                logger.warning("Generated config validation failed, falling back to original")
                return mas_config

            logger.info("Successfully generated new MAS configuration")
            return generated_config

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            logger.warning("Falling back to original configuration")
            return mas_config

    def mutate(
        self,
        mas_config: str,
        execution_logs: Dict[str, Any],
        observations: str,
        model_list: Optional[List[str]] = None
    ) -> str:
        """
        Mutate action: Modify MAS components based on execution observations.

        Args:
            mas_config: Current MAS configuration (YAML string)
            execution_logs: Execution logs including accuracy, errors
            observations: Human-readable observations from logs
            model_list: List of available models for MAS agents

        Returns:
            Mutated MAS configuration as YAML string
        """
        logger.info("MUTATE: Modifying MAS based on observations")

        # Load prompt template
        template = self.prompt_registry.get("meta_mutate")

        # Format model list
        if model_list:
            model_list_str = "\n".join([f"  - {model}" for model in model_list])
            model_constraint = f"\n\n**Available Models for MAS Agents:**\nYou can ONLY use the following models for agent model_id fields:\n{model_list_str}\n"
        else:
            model_constraint = ""

        # Prepare prompt variables
        prompt_vars = {
            "mas_config": mas_config,
            "execution_logs": str(execution_logs),
            "accuracy": execution_logs.get('accuracy', 'N/A'),
            "errors": execution_logs.get('errors', 'None'),
            "observations": observations,
            "model_constraint": model_constraint
        }

        # Add memory context if available
        if len(self.memory) > 0:
            memory_context = self.get_memory_context(max_experiences=3)
            prompt_vars["memory_context"] = f"\n\n## Previous Experiences\n\n{memory_context}"
        else:
            prompt_vars["memory_context"] = ""

        # Render prompt
        prompt = render_prompt(template, prompt_vars)

        logger.info("Calling meta model for mutation...")

        # Call meta model
        try:
            import asyncio
            if asyncio.iscoroutinefunction(self.model.__call__):
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                response = loop.run_until_complete(self.model(prompt))
            else:
                response = self.model(prompt)

            if not isinstance(response, str):
                response = str(response)

            logger.info(f"Received response ({len(response)} chars)")

            # Extract YAML from response
            mutated_config = extract_yaml_from_response(response)

            # Validate mutated config
            if not validate_mas_config(mutated_config):
                logger.warning("Mutated config validation failed, falling back to original")
                return mas_config

            logger.info("Successfully mutated MAS configuration")
            return mutated_config

        except Exception as e:
            logger.error(f"Mutation failed: {e}")
            logger.warning("Falling back to original configuration")
            return mas_config

    def crossover(
        self,
        mas_config_1: str,
        mas_config_2: str,
        logs_1: Dict[str, Any],
        logs_2: Dict[str, Any],
        model_list: Optional[List[str]] = None
    ) -> str:
        """
        Crossover action: Combine two MAS configurations.

        Args:
            mas_config_1: First parent MAS configuration
            mas_config_2: Second parent MAS configuration
            logs_1: Execution logs for parent 1
            logs_2: Execution logs for parent 2
            model_list: List of available models for MAS agents

        Returns:
            Offspring MAS configuration as YAML string
        """
        logger.info("CROSSOVER: Combining two MAS configurations")

        # Load prompt template
        template = self.prompt_registry.get("meta_crossover")

        # Analyze logs for strengths/weaknesses
        acc_1 = logs_1.get('accuracy', 0.0)
        acc_2 = logs_2.get('accuracy', 0.0)

        strengths_1 = f"Better performance" if acc_1 > acc_2 else "Alternative approach"
        weaknesses_1 = f"Lower accuracy ({acc_1:.2%})" if acc_1 < acc_2 else f"Current accuracy ({acc_1:.2%})"

        strengths_2 = f"Better performance" if acc_2 > acc_1 else "Alternative approach"
        weaknesses_2 = f"Lower accuracy ({acc_2:.2%})" if acc_2 < acc_1 else f"Current accuracy ({acc_2:.2%})"

        # Format model list
        if model_list:
            model_list_str = "\n".join([f"  - {model}" for model in model_list])
            model_constraint = f"\n\n**Available Models for MAS Agents:**\nYou can ONLY use the following models for agent model_id fields:\n{model_list_str}\n"
        else:
            model_constraint = ""

        # Prepare prompt variables
        prompt_vars = {
            "mas_config_1": mas_config_1,
            "mas_config_2": mas_config_2,
            "accuracy_1": f"{acc_1:.2%}",
            "accuracy_2": f"{acc_2:.2%}",
            "strengths_1": strengths_1,
            "weaknesses_1": weaknesses_1,
            "strengths_2": strengths_2,
            "weaknesses_2": weaknesses_2,
            "execution_logs": str({"parent_1": logs_1, "parent_2": logs_2}),
            "model_constraint": model_constraint
        }

        # Add memory context if available
        if len(self.memory) > 0:
            memory_context = self.get_memory_context(max_experiences=3)
            prompt_vars["memory_context"] = f"\n\n## Previous Experiences\n\n{memory_context}"
        else:
            prompt_vars["memory_context"] = ""

        # Render prompt
        prompt = render_prompt(template, prompt_vars)

        logger.info("Calling meta model for crossover...")

        # Call meta model
        try:
            import asyncio
            if asyncio.iscoroutinefunction(self.model.__call__):
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                response = loop.run_until_complete(self.model(prompt))
            else:
                response = self.model(prompt)

            if not isinstance(response, str):
                response = str(response)

            logger.info(f"Received response ({len(response)} chars)")

            # Extract YAML from response
            offspring_config = extract_yaml_from_response(response)

            # Validate offspring config
            if not validate_mas_config(offspring_config):
                logger.warning("Offspring config validation failed, returning better parent")
                return mas_config_1 if acc_1 >= acc_2 else mas_config_2

            logger.info("Successfully generated offspring MAS configuration")
            return offspring_config

        except Exception as e:
            logger.error(f"Crossover failed: {e}")
            logger.warning("Returning better parent configuration")
            return mas_config_1 if acc_1 >= acc_2 else mas_config_2

    def update_memory(
        self,
        action_type: str,
        query: str,
        old_config: str,
        new_config: str,
        old_accuracy: float,
        new_accuracy: float
    ) -> Optional[ActionExperience]:
        """
        Update memory action: Analyze action outcome and extract experience.

        Skipped (returns None) when self.memory_evolution is False.

        Args:
            action_type: Type of action (generate, mutate, crossover)
            query: The task/query being solved
            old_config: Previous MAS configuration (YAML string)
            new_config: New MAS configuration (YAML string)
            old_accuracy: Accuracy before action
            new_accuracy: Accuracy after action

        Returns:
            ActionExperience with analysis, or None if memory evolution is disabled.
        """
        if not getattr(self, "memory_evolution", True):
            logger.info("UPDATE_MEMORY: skipped (memory_evolution=False)")
            return None

        logger.info(f"UPDATE_MEMORY: Analyzing {action_type} action outcome")

        # Determine success
        success = new_accuracy > old_accuracy
        result = "Improved" if success else "Declined" if new_accuracy < old_accuracy else "→ No change"

        # Compare configurations to identify changes
        config_changes = self._compare_configs(old_config, new_config)

        # Load prompt template
        template = self.prompt_registry.get("meta_update_memory")

        # Render prompt
        prompt = render_prompt(template, {
            "action_type": action_type,
            "query": query[:500],  # Truncate long queries
            "old_config": old_config,
            "new_config": new_config,
            "old_accuracy": f"{old_accuracy:.2%}",
            "new_accuracy": f"{new_accuracy:.2%}",
            "result": result,
            "config_changes": config_changes
        })

        logger.info("Calling meta model for memory update...")

        # Call meta model
        try:
            import asyncio
            if asyncio.iscoroutinefunction(self.model.__call__):
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                response = loop.run_until_complete(self.model(prompt))
            else:
                response = self.model(prompt)

            if not isinstance(response, str):
                response = str(response)

            logger.info(f"Received analysis ({len(response)} chars)")

            # Create experience
            experience = ActionExperience(
                query=query[:500],
                action=action_type,
                config_changes=config_changes,
                old_accuracy=old_accuracy,
                new_accuracy=new_accuracy,
                success=success,
                analysis=response
            )

            # Add to memory
            self.memory.add(experience)
            logger.info(f"Added experience to memory (total: {len(self.memory)})")

            # Save memory if path provided
            if self.memory_path:
                self.memory.save(self.memory_path)
                logger.info(f"Saved memory to {self.memory_path}")

            return experience

        except Exception as e:
            logger.error(f"Memory update failed: {e}")
            # Still create a basic experience even if analysis fails
            experience = ActionExperience(
                query=query[:500],
                action=action_type,
                config_changes=config_changes,
                old_accuracy=old_accuracy,
                new_accuracy=new_accuracy,
                success=success,
                analysis=f"Analysis failed: {e}"
            )
            self.memory.add(experience)
            return experience

    def _compare_configs(self, old_config: str, new_config: str) -> str:
        """
        Compare two YAML configurations and describe the changes.

        Args:
            old_config: Old configuration YAML
            new_config: New configuration YAML

        Returns:
            Human-readable description of changes
        """
        try:
            old_dict = yaml.safe_load(old_config)
            new_dict = yaml.safe_load(new_config)

            changes = []

            # Check name change
            if old_dict.get('name') != new_dict.get('name'):
                changes.append(f"Name: {old_dict.get('name')} → {new_dict.get('name')}")

            # Check backend change
            if old_dict.get('backend') != new_dict.get('backend'):
                changes.append(f"Backend: {old_dict.get('backend')} → {new_dict.get('backend')}")

            # Check agent changes
            old_agents = old_dict.get('agents', {})
            new_agents = new_dict.get('agents', {})

            # Added agents
            added = set(new_agents.keys()) - set(old_agents.keys())
            if added:
                changes.append(f"Added agents: {', '.join(added)}")

            # Removed agents
            removed = set(old_agents.keys()) - set(new_agents.keys())
            if removed:
                changes.append(f"Removed agents: {', '.join(removed)}")

            # Modified agents
            for agent_name in set(old_agents.keys()) & set(new_agents.keys()):
                old_agent = old_agents[agent_name]
                new_agent = new_agents[agent_name]

                # Check model changes
                if old_agent.get('model_id') != new_agent.get('model_id'):
                    changes.append(
                        f"{agent_name} model: {old_agent.get('model_id')} → {new_agent.get('model_id')}"
                    )

                # Check prompt changes (just note if changed)
                if old_agent.get('prompt') != new_agent.get('prompt'):
                    changes.append(f"{agent_name} prompt modified")

                # Check reports_to changes
                if old_agent.get('reports_to') != new_agent.get('reports_to'):
                    changes.append(
                        f"{agent_name} reports_to: {old_agent.get('reports_to')} → {new_agent.get('reports_to')}"
                    )

            if not changes:
                return "No significant structural changes detected"

            return "\n".join(f"- {change}" for change in changes)

        except Exception as e:
            return f"Unable to compare configs: {e}"

    def get_memory_context(self, max_experiences: int = 5) -> str:
        """
        Get memory context for inclusion in prompts.

        Args:
            max_experiences: Maximum number of recent experiences to include

        Returns:
            Formatted memory context string
        """
        return self.memory.to_context_string(max_experiences)

    def evaluate_mas(
        self,
        mas_config_path: str,
        dataset_name: str,
        num_eval_tasks: int = 10,
        seed: int = 42,
        output_dir: str = "output"
    ) -> Dict[str, Any]:
        """
        Evaluate a MAS configuration using the interpreter.

        Args:
            mas_config_path: Path to MAS configuration file
            dataset_name: Dataset to evaluate on
            num_eval_tasks: Number of tasks to evaluate (default: 10)
            seed: Random seed for task selection (default: 42)
            output_dir: Output directory

        Returns:
            Evaluation results from interpreter
        """
        logger.info(f"EVALUATE: Running {mas_config_path} on {num_eval_tasks} tasks (seed={seed})")

        # Get random task IDs
        random.seed(seed)
        task_ids = list(range(num_eval_tasks))  # For now, just use first N tasks

        # Call interpreter
        result = interpret_mas(
            config_path=mas_config_path,
            dataset_name=dataset_name,
            task_ids=task_ids,
            save_outputs=True,
            output_dir=output_dir,
            verbose=self.verbose
        )

        return result


def run_evolution(
    dataset_name: str,
    pool_dir: str = "mas_pools/bbeh",
    num_eval_tasks: int = 10,
    max_steps: int = 3,
    seed: int = 42,
    output_dir: str = "output"
) -> Dict[str, Any]:
    """
    Run evolutionary MAS optimization loop.

    .. deprecated::
        This function is deprecated. Use ``main.run_evolution_pipeline()`` instead,
        which implements the full EvoMAS algorithm (pool update, memory consolidation,
        reward function, mutate-or-crossover selection).

    Args:
        dataset_name: Dataset to optimize for
        pool_dir: Directory containing MAS pool
        num_eval_tasks: Number of tasks for evaluation
        max_steps: Maximum evolutionary steps
        seed: Random seed
        output_dir: Output directory

    Returns:
        Results dictionary with final MAS and performance
    """
    import warnings
    warnings.warn(
        "metamodel.run_evolution() is deprecated. Use main.run_evolution_pipeline() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    from main import run_evolution_pipeline
    return run_evolution_pipeline(
        dataset_name=dataset_name,
        pool_dir=pool_dir,
        num_eval_tasks=num_eval_tasks,
        max_steps=max_steps,
        seed=seed,
        output_dir=output_dir
    )
