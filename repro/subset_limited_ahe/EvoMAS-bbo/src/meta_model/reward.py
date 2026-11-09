"""
Reward Function for EvoMAS.

Implements: R(C, q) = Metrics(C, q) - β · Cost(C)

Computes reward as a trade-off between task performance and execution cost.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def compute_reward(
    metrics: Dict[str, Any],
    beta: float = 1e-6,
    cost_weight: str = "both"
) -> float:
    """
    Compute reward for a configuration on a task.

    Implements: R(C, q) = Metrics(C, q) - β · Cost(C)

    Args:
        metrics: Dictionary containing:
            - accuracy: Task performance (0-100 from multi-aspect judge, or 0-1 legacy)
            - total_tokens: Token usage (optional)
            - total_time: Execution time in seconds (optional)
        beta: Cost trade-off parameter (≥ 0)
        cost_weight: Which cost metric to use ("tokens", "time", or "both")

    Returns:
        Reward value (higher is better)
    """
    # Extract performance metric
    accuracy = metrics.get('accuracy', 0.0)

    # Extract cost metrics (handle both flat and nested structures)
    total_tokens = metrics.get('total_tokens', 0)
    total_time = metrics.get('total_time', 0.0)
    if 'token_costs' in metrics:
        total_tokens = metrics['token_costs'].get('total_tokens', total_tokens)
    if 'time_costs' in metrics:
        total_time = metrics['time_costs'].get('total_time', total_time)

    # Compute cost based on weight
    if cost_weight == "tokens":
        cost = total_tokens
    elif cost_weight == "time":
        cost = total_time
    elif cost_weight == "both":
        # Normalize and combine (1 token ≈ 0.001 seconds empirically)
        cost = total_tokens + 1000 * total_time
    else:
        cost = 0.0

    # Compute reward
    reward = accuracy - beta * cost

    logger.debug(f"Reward: accuracy={accuracy:.4f}, cost={cost:.2f}, "
                f"beta={beta}, reward={reward:.4f}")

    return reward


def compare_configurations(
    metrics_1: Dict[str, Any],
    metrics_2: Dict[str, Any],
    beta: float = 1e-6,
    cost_weight: str = "both"
) -> int:
    """
    Compare two configurations based on their rewards.

    Args:
        metrics_1: Metrics for configuration 1
        metrics_2: Metrics for configuration 2
        beta: Cost trade-off parameter
        cost_weight: Which cost metric to use

    Returns:
        1 if config 1 is better, -1 if config 2 is better, 0 if equal
    """
    reward_1 = compute_reward(metrics_1, beta, cost_weight)
    reward_2 = compute_reward(metrics_2, beta, cost_weight)

    if reward_1 > reward_2:
        return 1
    elif reward_1 < reward_2:
        return -1
    else:
        return 0


def should_add_to_pool(
    new_metrics: Dict[str, Any],
    parent_metrics: Dict[str, Any],
    beta: float = 1e-6,
    cost_weight: str = "both",
    improvement_threshold: float = 0.01
) -> bool:
    """
    Determine if a new configuration should be added to the pool.

    A configuration is added if:
    R(C_new, q) > R(C_parent, q) + threshold

    Args:
        new_metrics: Metrics for new configuration
        parent_metrics: Metrics for parent configuration
        beta: Cost trade-off parameter
        cost_weight: Which cost metric to use
        improvement_threshold: Minimum improvement required to add

    Returns:
        True if configuration should be added to pool
    """
    reward_new = compute_reward(new_metrics, beta, cost_weight)
    reward_parent = compute_reward(parent_metrics, beta, cost_weight)

    improvement = reward_new - reward_parent

    logger.info(f"Reward comparison:")
    logger.info(f"  New:    {reward_new:.4f}")
    logger.info(f"  Parent: {reward_parent:.4f}")
    logger.info(f"  Improvement: {improvement:+.4f}")

    return improvement > improvement_threshold


def compute_reward_with_details(
    metrics: Dict[str, Any],
    beta: float = 1e-6,
    cost_weight: str = "both"
) -> Dict[str, Any]:
    """
    Compute reward with detailed breakdown.

    Args:
        metrics: Dictionary containing performance and cost metrics
        beta: Cost trade-off parameter
        cost_weight: Which cost metric to use

    Returns:
        Dictionary with:
            - reward: Final reward value
            - accuracy: Task performance component
            - cost: Cost component (before beta)
            - cost_penalty: Actual penalty (beta * cost)
            - breakdown: String description
    """
    accuracy = metrics.get('accuracy', 0.0)
    total_tokens = metrics.get('total_tokens', 0)
    total_time = metrics.get('total_time', 0.0)
    if 'token_costs' in metrics:
        total_tokens = metrics['token_costs'].get('total_tokens', total_tokens)
    if 'time_costs' in metrics:
        total_time = metrics['time_costs'].get('total_time', total_time)

    # Compute cost
    if cost_weight == "tokens":
        cost = total_tokens
        cost_desc = f"{total_tokens:,} tokens"
    elif cost_weight == "time":
        cost = total_time
        cost_desc = f"{total_time:.2f}s"
    elif cost_weight == "both":
        cost = total_tokens + 1000 * total_time
        cost_desc = f"{total_tokens:,} tokens + {total_time:.2f}s"
    else:
        cost = 0.0
        cost_desc = "none"

    cost_penalty = beta * cost
    reward = accuracy - cost_penalty

    breakdown = (
        f"Reward = {reward:.4f}\n"
        f"  Accuracy: {accuracy:.4f}\n"
        f"  Cost: {cost_desc} (penalty: -{cost_penalty:.4f})"
    )

    return {
        "reward": reward,
        "accuracy": accuracy,
        "cost": cost,
        "cost_penalty": cost_penalty,
        "breakdown": breakdown
    }


def select_best_configuration(
    configs_with_metrics: list,
    beta: float = 1e-6,
    cost_weight: str = "both"
) -> int:
    """
    Select the best configuration from a list based on reward.

    Args:
        configs_with_metrics: List of (config, metrics) tuples
        beta: Cost trade-off parameter
        cost_weight: Which cost metric to use

    Returns:
        Index of best configuration
    """
    if not configs_with_metrics:
        raise ValueError("Empty configuration list")

    best_idx = 0
    best_reward = float('-inf')

    for i, (_, metrics) in enumerate(configs_with_metrics):
        reward = compute_reward(metrics, beta, cost_weight)
        if reward > best_reward:
            best_reward = reward
            best_idx = i

    logger.info(f"Selected configuration {best_idx} with reward {best_reward:.4f}")

    return best_idx
