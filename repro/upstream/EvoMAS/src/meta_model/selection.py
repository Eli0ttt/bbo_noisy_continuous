"""
Configuration Selection Operator π^S for EvoMAS.

Implements: {C_1, ..., C_k} ~ π^S(· | q, D(C̄))

Selects k parent configurations from the pool based on task similarity
and historical performance metadata.
"""

import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def load_pool_metadata(pool_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load metadata for all configurations in the pool.

    Args:
        pool_dir: Directory containing MAS configuration pool

    Returns:
        Dictionary mapping config_name -> metadata
    """
    pool_path = Path(pool_dir)
    metadata = {}

    for config_file in pool_path.glob("*.yaml"):
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            metadata[config_file.stem] = {
                "name": config.get("name", config_file.stem),
                "description": config.get("description", ""),
                "successful_tasks": config.get("successful_tasks", []),
                "num_agents": len(config.get("agents", {})),
                "path": str(config_file)
            }
        except Exception as e:
            logger.warning(f"Failed to load metadata from {config_file}: {e}")

    return metadata


def compute_task_similarity(query: str, config_metadata: Dict[str, Any]) -> float:
    """
    Compute similarity between task query and configuration's successful tasks.

    Uses simple keyword overlap for now. Can be enhanced with embeddings.

    Args:
        query: Task query string
        config_metadata: Metadata for a configuration

    Returns:
        Similarity score in [0, 1]
    """
    # Extract keywords from query (simple tokenization)
    query_tokens = set(query.lower().split())

    # Get successful task notes from metadata
    successful_tasks = config_metadata.get("successful_tasks", [])

    if not successful_tasks:
        return 0.0

    # Compute overlap with successful tasks
    max_similarity = 0.0
    for task in successful_tasks:
        task_text = task.get("notes", "") + " " + task.get("q", "")
        task_tokens = set(task_text.lower().split())

        # Jaccard similarity
        if len(query_tokens) > 0:
            overlap = len(query_tokens & task_tokens)
            union = len(query_tokens | task_tokens)
            similarity = overlap / union if union > 0 else 0.0
            max_similarity = max(max_similarity, similarity)

    return max_similarity


def selection_operator(
    query: str,
    pool_dir: str,
    k: int = 2,
) -> List[str]:
    """
    Selection operator π^S: Select k parent configurations from pool.

    Implements: {C_1, ..., C_k} ~ π^S(· | q, D(C̄))

    Args:
        query: Task query
        pool_dir: Directory containing configuration pool
        k: Number of configurations to select

    Returns:
        List of k selected configuration file paths
    """
    logger.info(f"Selection: Choosing {k} configurations for task")

    # Load pool metadata
    pool_metadata = load_pool_metadata(pool_dir)

    if not pool_metadata:
        raise ValueError(f"No configurations found in pool: {pool_dir}")

    if len(pool_metadata) < k:
        logger.warning(f"Pool has {len(pool_metadata)} configs, but requested {k}")
        k = len(pool_metadata)

    # Compute task similarity scores for each configuration
    scored_configs = []
    for config_name, metadata in pool_metadata.items():
        similarity = compute_task_similarity(query, metadata)
        scored_configs.append((config_name, metadata["path"], similarity))

    # Sort by similarity (descending)
    scored_configs.sort(key=lambda x: x[2], reverse=True)

    # Select top-k configurations
    selected = scored_configs[:k]

    logger.info(f"Selected configurations:")
    for name, path, score in selected:
        logger.info(f"  - {name} (similarity: {score:.3f})")

    return [path for _, path, _ in selected]


def select_with_diversity(
    query: str,
    pool_dir: str,
    k: int = 2,
    diversity_weight: float = 0.3,
) -> List[str]:
    """
    Selection with diversity: Balance between task relevance and diversity.

    This variant balances:
    - Task similarity (relevance to current query)
    - Diversity (structural differences between selected configs)

    Args:
        query: Task query
        pool_dir: Directory containing configuration pool
        k: Number of configurations to select
        diversity_weight: Weight for diversity vs similarity (0-1)

    Returns:
        List of k selected configuration file paths
    """
    logger.info(f"Selection with diversity: Choosing {k} configurations")

    # Load pool metadata
    pool_metadata = load_pool_metadata(pool_dir)

    if not pool_metadata:
        raise ValueError(f"No configurations found in pool: {pool_dir}")

    if len(pool_metadata) < k:
        k = len(pool_metadata)

    # Compute task similarity scores
    scored_configs = []
    for config_name, metadata in pool_metadata.items():
        similarity = compute_task_similarity(query, metadata)
        scored_configs.append({
            "name": config_name,
            "path": metadata["path"],
            "similarity": similarity,
            "num_agents": metadata["num_agents"]
        })

    # Start with most similar configuration
    scored_configs.sort(key=lambda x: x["similarity"], reverse=True)
    selected = [scored_configs[0]]
    remaining = scored_configs[1:]

    # Select remaining configurations to maximize diversity
    while len(selected) < k and remaining:
        best_score = -float('inf')
        best_idx = 0

        for i, candidate in enumerate(remaining):
            # Compute diversity from already selected configs
            diversity = 0.0
            for sel in selected:
                # Simple diversity: difference in number of agents
                diversity += abs(candidate["num_agents"] - sel["num_agents"])

            diversity /= len(selected)  # Normalize

            # Combined score: similarity + diversity
            score = (1 - diversity_weight) * candidate["similarity"] + \
                    diversity_weight * (diversity / 10.0)  # Scale diversity

            if score > best_score:
                best_score = score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    logger.info(f"Selected configurations (with diversity):")
    for config in selected:
        logger.info(f"  - {config['name']} (similarity: {config['similarity']:.3f}, agents: {config['num_agents']})")

    return [config["path"] for config in selected]
