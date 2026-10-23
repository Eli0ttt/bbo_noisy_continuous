"""
Runtime for executing MAS.
"""

import logging
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .spec import MasSpec
from src.topology.context import Context
from src.agents.spec import AgentResult, get_backend_for_agent_type
from src.agents.runners.base import BaseAgentRunner
# Lazy imports for runners to avoid loading unused packages
# from src.agents.runners.smolagents import SmolagentsRunner
# from src.agents.runners.minisweagent import MinisweagentRunner
# from src.agents.runners.sweagent import SWEAgentRunner

logger = logging.getLogger(__name__)


def _get_runner_class(backend: str):
    """Lazily import and return the runner class for a backend."""
    if backend == "smolagents":
        from src.agents.runners.smolagents import SmolagentsRunner
        return SmolagentsRunner
    elif backend == "minisweagent":
        from src.agents.runners.minisweagent import MinisweagentRunner
        return MinisweagentRunner
    elif backend == "sweagent":
        from src.agents.runners.sweagent import SWEAgentRunner
        return SWEAgentRunner
    elif backend == "langchain":
        raise NotImplementedError("Langchain backend is not yet implemented")
    else:
        raise ValueError(f"Unsupported backend: {backend}")


class MasRuntime:
    """Runtime for executing multi-agent systems."""

    def __init__(self, mas_spec: MasSpec):
        """Initialize MAS runtime.

        Args:
            mas_spec: MAS specification
        """
        self.mas_spec = mas_spec
        self._runners = {}  # Cache runners by backend type
        self.runner = self._get_runner()  # Default runner for MAS-level backend

    def _get_runner(self, backend: str = None) -> BaseAgentRunner:
        """Get the appropriate agent runner based on backend.

        Args:
            backend: Backend name (uses MAS-level backend if None)

        Returns:
            Agent runner instance
        """
        backend = backend or self.mas_spec.backend

        # Get agent config from execution settings
        agent_config = self.mas_spec.execution.agent_config

        # Create cache key that includes config
        cache_key = f"{backend}:{agent_config}"

        # Return cached runner if available
        if cache_key in self._runners:
            return self._runners[cache_key]

        # Create new runner with config (lazy import)
        RunnerClass = _get_runner_class(backend)
        if backend == "smolagents":
            runner = RunnerClass()
        else:
            runner = RunnerClass(config=agent_config)
            logger.info(f"Created {RunnerClass.__name__} with config: {agent_config}")

        # Cache and return
        self._runners[cache_key] = runner
        return runner

    def _get_runner_for_agent(self, agent_spec) -> BaseAgentRunner:
        """Get the appropriate runner for a specific agent.

        Priority order for backend selection:
        1. Explicit agent-level backend (if specified)
        2. Auto-inferred from agent_type (e.g., CodeAgent -> smolagents)
        3. MAS-level default backend

        Args:
            agent_spec: Agent specification

        Returns:
            Agent runner instance
        """
        # Priority 1: Explicit agent-level backend
        backend = getattr(agent_spec, 'backend', None)

        # Priority 2: Auto-infer from agent_type
        if not backend:
            agent_type = getattr(agent_spec, 'agent_type', 'CodeAgent')
            backend = get_backend_for_agent_type(agent_type)

        # Priority 3: Fall back to MAS-level backend
        if not backend:
            backend = self.mas_spec.backend

        return self._get_runner(backend)

    def run(self, task: str) -> tuple:
        """Execute the MAS on a given task.

        Args:
            task: The task/query to process

        Returns:
            Tuple of (final_result: str, metadata: dict) containing:
            - final_result: The final output from the MAS
            - metadata: Dictionary with 'input_tokens', 'output_tokens', etc.
        """
        logger.info("=" * 60)
        logger.info(f"Starting MAS: {self.mas_spec.name}")
        logger.info("=" * 60)

        # Initialize context
        context = Context(task=task)

        # Get execution order
        execution_order = self.mas_spec.get_execution_order()
        logger.info(f"Execution order: {execution_order}")

        # DAG Compiler: topology-driven execution for ALL configurations
        logger.info("Executing via DAG compiler")
        self._execute_dag(execution_order, task, context)

        # Get final result
        final_agent_id = execution_order[-1]
        final_result = context.reports.get(final_agent_id, "No result")

        # Aggregate metadata from all executed agents
        metadata = self._aggregate_metadata_from_context(context)

        logger.info("=" * 60)
        logger.info(f"MAS Execution Completed: {self.mas_spec.name}")
        logger.info("=" * 60)

        return final_result, metadata

    def _execute_dag(self, execution_order: list, task: str, context: Context):
        """DAG compiler: execute agents by topology levels with parallel support.

        This is the universal execution engine. It handles ANY combination
        of agent roles by reading the topology DAG. Agents are grouped into
        dependency levels and executed level-by-level, with parallel execution
        within each level when enabled.

        Args:
            execution_order: Flat execution order (used as fallback reference)
            task: Task to execute
            context: Execution context
        """
        levels = self.mas_spec.get_execution_levels()
        logger.info(f"DAG levels: {levels}")

        force_sequential = "Repository:" in task and "/repos/" in task
        if force_sequential:
            logger.info("SWE-bench repo task: forcing sequential execution")

        for level_idx, level in enumerate(levels):
            pending = [aid for aid in level if aid not in context.reports]
            if not pending:
                continue

            if (len(pending) > 1
                    and self.mas_spec.execution.parallel_workers
                    and not force_sequential):
                logger.info(f"Level {level_idx}: {pending} in parallel")
                self._execute_agents_parallel(pending, task, context)
            else:
                logger.info(f"Level {level_idx}: {pending} sequentially")
                for agent_id in pending:
                    self._execute_agent(agent_id, task, context)

    def _execute_agents_parallel(self, agent_ids: list, task: str, context: Context):
        """Execute agents in parallel using ThreadPoolExecutor.

        Generic parallel executor that works for any agent role.

        Args:
            agent_ids: List of agent IDs to execute in parallel
            task: Task to execute
            context: Execution context
        """
        with ThreadPoolExecutor(max_workers=len(agent_ids)) as executor:
            futures = {}
            for agent_id in agent_ids:
                agent_spec = self.mas_spec.agents[agent_id]
                runner = self._get_runner_for_agent(agent_spec)
                future = executor.submit(runner.run, agent_spec, task, context.model_dump())
                futures[future] = agent_id

            for future in as_completed(futures):
                agent_id = futures[future]
                try:
                    result = future.result()
                    if not result.success:
                        raise RuntimeError(f"Agent {agent_id} failed: {result.error}")
                    context.add_report(agent_id, result.content, metadata=result.metadata)
                    logger.info(f"Agent {agent_id} completed successfully")
                except Exception as e:
                    logger.error(f"Agent {agent_id} failed: {e}")
                    context.add_report(agent_id, f"Error: {str(e)}")

    def _execute_workers_parallel(self, worker_ids: list, task: str, context: Context):
        """Execute workers in parallel.

        Args:
            worker_ids: List of worker agent IDs
            task: Task to execute
            context: Execution context
        """
        with ThreadPoolExecutor(max_workers=len(worker_ids)) as executor:
            futures = {}
            for agent_id in worker_ids:
                agent_spec = self.mas_spec.agents[agent_id]
                runner = self._get_runner_for_agent(agent_spec)
                future = executor.submit(
                    runner.run,
                    agent_spec,
                    task,
                    context.model_dump()
                )
                futures[future] = agent_id

            for future in as_completed(futures):
                agent_id = futures[future]
                try:
                    result: AgentResult = future.result()

                    # Check if agent execution failed
                    if not result.success:
                        error_msg = result.error or "Agent execution failed with unknown error"
                        logger.error(f"Worker {agent_id} failed: {error_msg}")
                        raise RuntimeError(f"Agent {agent_id} execution failed: {error_msg}")

                    context.add_report(
                        agent_id,
                        result.content,
                        metadata=result.metadata
                    )
                    logger.info(f"Worker {agent_id} completed successfully")
                except Exception as e:
                    logger.error(f"Worker {agent_id} failed: {e}")
                    context.add_report(agent_id, f"Error: {str(e)}")

    def _execute_agent(self, agent_id: str, task: str, context: Context):
        """Execute a single agent.

        Args:
            agent_id: Agent ID
            task: Task to execute
            context: Execution context
        """
        logger.info(f"Executing agent: {agent_id}")

        agent_spec = self.mas_spec.agents[agent_id]
        runner = self._get_runner_for_agent(agent_spec)

        try:
            result: AgentResult = runner.run(
                agent_spec,
                task,
                context.model_dump()
            )

            # Check if agent execution failed
            if not result.success:
                error_msg = result.error or "Agent execution failed with unknown error"
                logger.error(f"Agent {agent_id} failed: {error_msg}")
                raise RuntimeError(f"Agent {agent_id} execution failed: {error_msg}")

            context.add_report(
                agent_id,
                result.content,
                metadata=result.metadata
            )

            logger.info(f"Agent {agent_id} completed successfully")

        except Exception as e:
            logger.error(f"Agent {agent_id} execution failed: {e}")
            context.add_report(agent_id, f"Error: {str(e)}")

    def _aggregate_metadata_from_context(self, context: Context) -> Dict[str, Any]:
        """Aggregate metadata (token usage) from all agents in the context.

        Args:
            context: Execution context with agent traces

        Returns:
            Dictionary with aggregated token usage statistics
        """
        total_input_tokens = 0
        total_output_tokens = 0
        total_thinking_tokens = 0
        total_api_calls = 0
        total_instance_cost = 0.0

        # Iterate through all trace entries to collect token usage
        for trace_entry in context.trace:
            if 'metadata' in trace_entry and trace_entry['metadata']:
                metadata = trace_entry['metadata']

                # Accumulate token counts
                if 'input_tokens' in metadata:
                    total_input_tokens += metadata.get('input_tokens', 0)
                if 'output_tokens' in metadata:
                    total_output_tokens += metadata.get('output_tokens', 0)
                if 'thinking_tokens' in metadata:
                    total_thinking_tokens += metadata.get('thinking_tokens', 0)

                # Handle OpenAI-style token names
                if 'prompt_tokens' in metadata:
                    total_input_tokens += metadata.get('prompt_tokens', 0)
                if 'completion_tokens' in metadata:
                    total_output_tokens += metadata.get('completion_tokens', 0)
                if 'reasoning_tokens' in metadata:
                    total_thinking_tokens += metadata.get('reasoning_tokens', 0)

                # Handle SWE-agent style model_stats
                if 'model_stats' in metadata and metadata['model_stats']:
                    model_stats = metadata['model_stats']
                    total_input_tokens += model_stats.get('tokens_sent', 0)
                    total_output_tokens += model_stats.get('tokens_received', 0)
                    total_api_calls += model_stats.get('api_calls', 0)
                    total_instance_cost += model_stats.get('instance_cost', 0.0)

        # Build aggregated metadata
        aggregated = {
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'total_tokens': total_input_tokens + total_output_tokens + total_thinking_tokens,
            'api_calls': total_api_calls,
            'instance_cost': total_instance_cost,
        }

        # Add thinking tokens if present
        if total_thinking_tokens > 0:
            aggregated['thinking_tokens'] = total_thinking_tokens

        return aggregated

    def _execute_debate(self, debater_ids: list, task: str, context: Context):
        """Execute multi-round debate between debaters.

        Args:
            debater_ids: List of debater agent IDs
            task: Task to debate
            context: Execution context
        """
        from copy import deepcopy

        num_rounds = self.mas_spec.execution.debate_rounds

        # Store debater responses across rounds
        debater_responses = {agent_id: None for agent_id in debater_ids}

        for round_num in range(1, num_rounds + 1):
            logger.info(f"=" * 60)
            logger.info(f"DEBATE ROUND {round_num}/{num_rounds}")
            logger.info(f"=" * 60)

            if round_num == 1:
                # Round 1: Independent initial responses
                logger.info("Debaters providing initial independent responses...")
                for agent_id in debater_ids:
                    agent_spec = self.mas_spec.agents[agent_id]

                    # Use debater_initial prompt for round 1
                    agent_spec_copy = deepcopy(agent_spec)
                    if agent_spec_copy.prompt == "debater_round":
                        agent_spec_copy.prompt = "debater_initial"

                    try:
                        result: AgentResult = self.runner.run(
                            agent_spec_copy,
                            task,
                            context.model_dump()
                        )

                        if not result.success:
                            logger.error(f"Debater {agent_id} failed: {result.error}")
                            debater_responses[agent_id] = f"Error: {result.error}"
                        else:
                            debater_responses[agent_id] = result.content
                            logger.info(f"Debater {agent_id} completed round {round_num}")
                    except Exception as e:
                        logger.error(f"Debater {agent_id} failed: {e}")
                        debater_responses[agent_id] = f"Error: {str(e)}"
            else:
                # Rounds 2-N: Refine based on others' responses
                logger.info("Debaters refining responses based on debate...")

                for agent_id in debater_ids:
                    agent_spec = self.mas_spec.agents[agent_id]

                    # Prepare context with other debaters' responses
                    debate_context = deepcopy(context)

                    # Add this debater's previous response
                    debate_context.shared["your_previous_response"] = debater_responses[agent_id]

                    # Add other debaters' responses
                    other_responses = []
                    for other_id in debater_ids:
                        if other_id != agent_id:
                            other_responses.append(
                                f"**{other_id}**: {debater_responses[other_id]}"
                            )
                    debate_context.shared["other_responses"] = "\n\n".join(other_responses)

                    # Use debater_round prompt for rounds 2+
                    agent_spec_copy = deepcopy(agent_spec)
                    if agent_spec_copy.prompt == "debater_initial":
                        agent_spec_copy.prompt = "debater_round"

                    try:
                        result: AgentResult = self.runner.run(
                            agent_spec_copy,
                            task,
                            debate_context.model_dump()
                        )

                        if not result.success:
                            logger.error(f"Debater {agent_id} failed: {result.error}")
                            debater_responses[agent_id] = f"Error: {result.error}"
                        else:
                            debater_responses[agent_id] = result.content
                            logger.info(f"Debater {agent_id} completed round {round_num}")
                    except Exception as e:
                        logger.error(f"Debater {agent_id} failed: {e}")
                        debater_responses[agent_id] = f"Error: {str(e)}"

        # Add final debater responses to context
        for agent_id, response in debater_responses.items():
            context.add_report(agent_id, response)

        logger.info(f"Debate completed after {num_rounds} rounds")

    def _execute_smoa(self, task: str, context: Context):
        """Execute Sparse Mixture-of-Agents (SMoA) with layers, judge, and moderator.

        Args:
            task: Task to process
            context: Execution context
        """
        from copy import deepcopy
        import re

        num_layers = self.mas_spec.execution.smoa_layers
        top_k = self.mas_spec.execution.smoa_top_k
        judge_id = self.mas_spec.get_judge_agent()
        moderator_id = self.mas_spec.get_moderator_agent()

        # Store selected responses across layers
        selected_responses = []

        for layer_num in range(1, num_layers + 1):
            logger.info(f"=" * 60)
            logger.info(f"SMoA LAYER {layer_num}/{num_layers}")
            logger.info(f"=" * 60)

            # Get processors for this layer
            layer_processors = self.mas_spec.get_processor_agents(layer=layer_num)

            if not layer_processors:
                logger.warning(f"No processors found for layer {layer_num}")
                continue

            logger.info(f"Executing {len(layer_processors)} processors in layer {layer_num}...")

            # Execute all processors in parallel
            processor_responses = {}
            with ThreadPoolExecutor(max_workers=len(layer_processors)) as executor:
                futures = {}
                for agent_id in layer_processors:
                    agent_spec = self.mas_spec.agents[agent_id]

                    # Prepare context with role description and previous responses
                    processor_context = deepcopy(context)
                    role_desc = agent_spec.metadata.get("role_description", "") if agent_spec.metadata else ""
                    processor_context.shared["role_description"] = role_desc

                    # Add previous layer's selected responses
                    if selected_responses:
                        prev_text = "\n\n".join([
                            f"**Response {i+1}**: {resp}"
                            for i, resp in enumerate(selected_responses)
                        ])
                        processor_context.shared["previous_responses"] = prev_text
                    else:
                        processor_context.shared["previous_responses"] = "None (this is the first layer)"

                    future = executor.submit(
                        self.runner.run,
                        agent_spec,
                        task,
                        processor_context.model_dump()
                    )
                    futures[future] = agent_id

                for future in as_completed(futures):
                    agent_id = futures[future]
                    try:
                        result: AgentResult = future.result()
                        if result.success:
                            processor_responses[agent_id] = result.content
                            logger.info(f"Processor {agent_id} completed")
                        else:
                            logger.error(f"Processor {agent_id} failed: {result.error}")
                            processor_responses[agent_id] = f"Error: {result.error}"
                    except Exception as e:
                        logger.error(f"Processor {agent_id} failed: {e}")
                        processor_responses[agent_id] = f"Error: {str(e)}"

            # Call Judge to select top-k responses
            logger.info(f"Calling judge to select top-{top_k} responses...")
            judge_context = deepcopy(context)
            judge_context.shared["top_k"] = str(top_k)

            # Format processor responses for judge
            responses_text = "\n\n".join([
                f"**Response {i+1}** (from {agent_id}):\n{response}"
                for i, (agent_id, response) in enumerate(processor_responses.items())
            ])
            judge_context.shared["processor_responses"] = responses_text

            judge_spec = self.mas_spec.agents[judge_id]
            judge_result = self.runner.run(judge_spec, task, judge_context.model_dump())

            # Parse judge's selection
            if judge_result.success:
                judge_output = judge_result.content.strip()
                logger.info(f"Judge output: {judge_output}")

                # Extract numbers from judge output
                selected_indices = []
                try:
                    # Try to parse comma-separated numbers
                    numbers = re.findall(r'\d+', judge_output)
                    selected_indices = [int(n) - 1 for n in numbers[:top_k]]  # Convert to 0-indexed
                except Exception as e:
                    logger.error(f"Failed to parse judge output: {e}")
                    # Fallback: select first top_k
                    selected_indices = list(range(min(top_k, len(processor_responses))))

                # Get selected responses
                processor_list = list(processor_responses.items())
                selected_responses = []
                for idx in selected_indices:
                    if 0 <= idx < len(processor_list):
                        agent_id, response = processor_list[idx]
                        selected_responses.append(response)
                        logger.info(f"Selected response from {agent_id}")

            else:
                logger.error(f"Judge failed: {judge_result.error}")
                # Fallback: select first top_k
                selected_responses = list(processor_responses.values())[:top_k]

            # Call Moderator to check for early stopping
            if layer_num < num_layers and moderator_id:
                logger.info("Calling moderator to check for early stopping...")
                moderator_context = deepcopy(context)
                moderator_context.shared["current_round"] = str(layer_num)
                moderator_context.shared["max_rounds"] = str(num_layers)
                moderator_context.shared["processor_responses"] = "\n\n".join([
                    f"**Selected Response {i+1}**:\n{resp}"
                    for i, resp in enumerate(selected_responses)
                ])

                moderator_spec = self.mas_spec.agents[moderator_id]
                moderator_result = self.runner.run(moderator_spec, task, moderator_context.model_dump())

                if moderator_result.success:
                    decision = moderator_result.content.strip().upper()
                    logger.info(f"Moderator decision: {decision}")

                    if "STOP" in decision:
                        logger.info("Moderator decided to stop early - consensus reached")
                        break
                else:
                    logger.error(f"Moderator failed: {moderator_result.error}")

        # Store selected responses in context for aggregator
        context.shared["final_responses"] = "\n\n".join([
            f"**Response {i+1}**:\n{resp}"
            for i, resp in enumerate(selected_responses)
        ])

        logger.info(f"SMoA completed after {layer_num} layers")

    def _execute_peer_review(self, reviewer_ids: list, task: str, context: Context):
        """Execute peer review process with Create, Review, Revise stages.

        Args:
            reviewer_ids: List of reviewer agent IDs
            task: Task to solve
            context: Execution context
        """
        from copy import deepcopy

        # Store solutions and reviews
        initial_solutions = {}
        reviews = {}  # reviews[from_id][to_id] = review content
        final_solutions = {}

        # ====================
        # Stage 1: Create
        # ====================
        logger.info("=" * 60)
        logger.info("STAGE 1: CREATE (Independent Solutions)")
        logger.info("=" * 60)

        # Execute all reviewers in parallel to create initial solutions
        with ThreadPoolExecutor(max_workers=len(reviewer_ids)) as executor:
            futures = {}
            for agent_id in reviewer_ids:
                agent_spec = self.mas_spec.agents[agent_id]

                # Use reviewer_create prompt
                agent_spec_copy = deepcopy(agent_spec)
                agent_spec_copy.prompt = "reviewer_create"

                future = executor.submit(
                    self.runner.run,
                    agent_spec_copy,
                    task,
                    context.model_dump()
                )
                futures[future] = agent_id

            for future in as_completed(futures):
                agent_id = futures[future]
                try:
                    result: AgentResult = future.result()
                    if result.success:
                        initial_solutions[agent_id] = result.content
                        logger.info(f"Reviewer {agent_id} created initial solution")
                    else:
                        logger.error(f"Reviewer {agent_id} failed: {result.error}")
                        initial_solutions[agent_id] = f"Error: {result.error}"
                except Exception as e:
                    logger.error(f"Reviewer {agent_id} failed: {e}")
                    initial_solutions[agent_id] = f"Error: {str(e)}"

        # ====================
        # Stage 2: Review
        # ====================
        logger.info("=" * 60)
        logger.info("STAGE 2: REVIEW (Peer Reviews)")
        logger.info("=" * 60)

        # Each reviewer reviews all other reviewers' solutions
        for reviewer_id in reviewer_ids:
            reviews[reviewer_id] = {}

            logger.info(f"Reviewer {reviewer_id} reviewing peers...")

            for peer_id in reviewer_ids:
                if peer_id == reviewer_id:
                    continue  # Don't review yourself

                # Prepare context with reviewer's own solution and peer's solution
                review_context = deepcopy(context)
                review_context.shared["your_solution"] = initial_solutions.get(reviewer_id, "")
                review_context.shared["peer_solution"] = initial_solutions.get(peer_id, "")

                agent_spec = self.mas_spec.agents[reviewer_id]
                agent_spec_copy = deepcopy(agent_spec)
                agent_spec_copy.prompt = "reviewer_review"

                try:
                    result: AgentResult = self.runner.run(
                        agent_spec_copy,
                        task,
                        review_context.model_dump()
                    )

                    if result.success:
                        reviews[reviewer_id][peer_id] = result.content
                        logger.info(f"  → Reviewed {peer_id}'s solution")
                    else:
                        logger.error(f"  → Review of {peer_id} failed: {result.error}")
                        reviews[reviewer_id][peer_id] = f"Error: {result.error}"
                except Exception as e:
                    logger.error(f"  → Review of {peer_id} failed: {e}")
                    reviews[reviewer_id][peer_id] = f"Error: {str(e)}"

        # ====================
        # Stage 3: Revise
        # ====================
        logger.info("=" * 60)
        logger.info("STAGE 3: REVISE (Incorporate Feedback)")
        logger.info("=" * 60)

        # Each reviewer revises their solution based on received reviews
        with ThreadPoolExecutor(max_workers=len(reviewer_ids)) as executor:
            futures = {}
            for agent_id in reviewer_ids:
                # Collect all reviews received by this agent
                received_reviews_list = []
                for reviewer_id in reviewer_ids:
                    if reviewer_id != agent_id and agent_id in reviews.get(reviewer_id, {}):
                        review_content = reviews[reviewer_id][agent_id]
                        received_reviews_list.append(
                            f"**Review from {reviewer_id}**:\n{review_content}"
                        )

                # Prepare context for revision
                revise_context = deepcopy(context)
                revise_context.shared["your_solution"] = initial_solutions.get(agent_id, "")
                revise_context.shared["received_reviews"] = "\n\n".join(received_reviews_list) if received_reviews_list else "No reviews received"

                agent_spec = self.mas_spec.agents[agent_id]
                agent_spec_copy = deepcopy(agent_spec)
                agent_spec_copy.prompt = "reviewer_revise"

                future = executor.submit(
                    self.runner.run,
                    agent_spec_copy,
                    task,
                    revise_context.model_dump()
                )
                futures[future] = agent_id

            for future in as_completed(futures):
                agent_id = futures[future]
                try:
                    result: AgentResult = future.result()
                    if result.success:
                        final_solutions[agent_id] = result.content
                        context.add_report(agent_id, result.content)
                        logger.info(f"Reviewer {agent_id} revised solution")
                    else:
                        logger.error(f"Reviewer {agent_id} revision failed: {result.error}")
                        # Fallback to initial solution
                        final_solutions[agent_id] = initial_solutions.get(agent_id, "")
                        context.add_report(agent_id, final_solutions[agent_id])
                except Exception as e:
                    logger.error(f"Reviewer {agent_id} revision failed: {e}")
                    # Fallback to initial solution
                    final_solutions[agent_id] = initial_solutions.get(agent_id, "")
                    context.add_report(agent_id, final_solutions[agent_id])

        logger.info(f"Peer review completed (3 stages)")

    def _execute_croto(self, team_worker_ids: list, task: str, context: Context):
        """Execute Cross-Team Orchestration (CROTO) with multiple teams and hierarchical aggregation.

        Args:
            team_worker_ids: List of team worker agent IDs
            task: Task to solve
            context: Execution context
        """
        from copy import deepcopy
        import math

        num_phases = self.mas_spec.execution.croto_phases
        partition_size = self.mas_spec.execution.croto_partition_size
        aggregator_id = self.mas_spec.get_aggregator_agents()[0] if self.mas_spec.get_aggregator_agents() else None

        # Store team solutions across phases
        team_solutions = {agent_id: None for agent_id in team_worker_ids}
        aggregated_solution = None

        for phase_num in range(1, num_phases + 1):
            logger.info("=" * 60)
            logger.info(f"CROTO PHASE {phase_num}/{num_phases}")
            logger.info("=" * 60)

            # ========================================
            # Step 1: All teams work independently
            # ========================================
            logger.info(f"Teams working independently on phase {phase_num}...")

            with ThreadPoolExecutor(max_workers=len(team_worker_ids)) as executor:
                futures = {}
                for agent_id in team_worker_ids:
                    agent_spec = self.mas_spec.agents[agent_id]
                    team_id = agent_spec.metadata.get("team_id", "unknown") if agent_spec.metadata else "unknown"

                    # Prepare context
                    team_context = deepcopy(context)
                    team_context.shared["team_id"] = str(team_id)
                    team_context.shared["phase"] = f"Phase {phase_num}"

                    # Add aggregated solution if available
                    if aggregated_solution:
                        team_context.shared["previous_solutions"] = f"Aggregated solution from previous phase:\n{aggregated_solution}"
                    else:
                        team_context.shared["previous_solutions"] = "None (this is the first phase)"

                    future = executor.submit(
                        self.runner.run,
                        agent_spec,
                        task,
                        team_context.model_dump()
                    )
                    futures[future] = agent_id

                for future in as_completed(futures):
                    agent_id = futures[future]
                    try:
                        result: AgentResult = future.result()
                        if result.success:
                            team_solutions[agent_id] = result.content
                            logger.info(f"Team {agent_id} completed phase {phase_num}")
                        else:
                            logger.error(f"Team {agent_id} failed: {result.error}")
                            team_solutions[agent_id] = f"Error: {result.error}"
                    except Exception as e:
                        logger.error(f"Team {agent_id} failed: {e}")
                        team_solutions[agent_id] = f"Error: {str(e)}"

            # ========================================
            # Step 2: Hierarchical Aggregation
            # ========================================
            logger.info(f"Performing hierarchical aggregation...")

            # Collect all team solutions
            solutions = list(team_solutions.values())

            # Hierarchical aggregation in rounds
            round_num = 0
            while len(solutions) > 1:
                round_num += 1
                logger.info(f"  Aggregation round {round_num}: {len(solutions)} solutions")

                # Partition into groups
                num_groups = math.ceil(len(solutions) / partition_size)
                grouped_solutions = []
                for i in range(num_groups):
                    start_idx = i * partition_size
                    end_idx = min((i + 1) * partition_size, len(solutions))
                    group = solutions[start_idx:end_idx]
                    grouped_solutions.append(group)

                # Aggregate each group
                next_round_solutions = []
                for group_idx, group in enumerate(grouped_solutions):
                    if len(group) == 1:
                        # Single solution, no need to aggregate
                        next_round_solutions.append(group[0])
                        continue

                    # Prepare aggregation context
                    agg_context = deepcopy(context)
                    agg_context.shared["phase"] = f"Phase {phase_num}"

                    group_text = "\n\n".join([
                        f"**Solution {i+1}**:\n{sol}"
                        for i, sol in enumerate(group)
                    ])
                    agg_context.shared["team_solutions"] = group_text

                    # Run aggregator
                    if aggregator_id:
                        aggregator_spec = self.mas_spec.agents[aggregator_id]
                        try:
                            result: AgentResult = self.runner.run(
                                aggregator_spec,
                                task,
                                agg_context.model_dump()
                            )

                            if result.success:
                                aggregated = result.content
                                next_round_solutions.append(aggregated)
                                logger.info(f"    Group {group_idx + 1} aggregated")
                            else:
                                logger.error(f"    Group {group_idx + 1} aggregation failed: {result.error}")
                                # Fallback: use first solution
                                next_round_solutions.append(group[0])
                        except Exception as e:
                            logger.error(f"    Group {group_idx + 1} aggregation failed: {e}")
                            next_round_solutions.append(group[0])
                    else:
                        # No aggregator, just take first
                        next_round_solutions.append(group[0])

                solutions = next_round_solutions

            # Final aggregated solution
            aggregated_solution = solutions[0] if solutions else "No solution generated"
            logger.info(f"Phase {phase_num} aggregation complete")

        # Store final aggregated solution in context
        for agent_id in team_worker_ids:
            context.add_report(agent_id, aggregated_solution)

        logger.info(f"CROTO completed after {num_phases} phases")

    def _execute_topology_based(self, execution_order: list, task: str, context: Context):
        """Execute MAS based on topology (general fallback for any configuration).

        This method can handle ANY MAS configuration generated by meta model,
        as long as it has valid agents and topology.

        Args:
            execution_order: Order of agent execution from topology
            task: Task to execute
            context: Execution context
        """
        logger.info("Executing agents in topology order...")

        for agent_id in execution_order:
            # Skip if already executed
            if agent_id in context.reports:
                logger.info(f"Agent {agent_id} already executed, skipping")
                continue

            logger.info(f"Executing agent: {agent_id}")
            self._execute_agent(agent_id, task, context)

        logger.info(f"Topology-based execution completed")
