"""
MAS Configuration Pool Management for EvoMAS.

Implements pool update: C̄_{t+1} = C̄_t ∪ {C_q}

Manages the set of candidate MAS configurations that serve as parents
for evolutionary operators.
"""

import yaml
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfigurationPool:
    """
    Manages a pool of MAS configurations.

    The pool stores successful configurations discovered through evolution.
    Configurations are added if they achieve higher reward than their parents.
    """

    def __init__(self, pool_dir: str):
        """
        Initialize configuration pool.

        Args:
            pool_dir: Directory containing the pool of configurations
        """
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized pool at: {self.pool_dir}")

    def add_configuration(
        self,
        config_yaml: str,
        name: str,
        description: str,
        successful_tasks: list = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Add a new configuration to the pool.

        Implements: C̄_{t+1} = C̄_t ∪ {C_q}

        Args:
            config_yaml: Configuration as YAML string
            name: Configuration name
            description: Human-readable description
            successful_tasks: List of tasks where config performed well
            metadata: Additional metadata (accuracy, reward, etc.)

        Returns:
            Path to saved configuration file
        """
        # Parse YAML to dict
        config_dict = yaml.safe_load(config_yaml)

        # Add/update metadata fields
        config_dict['name'] = name
        config_dict['description'] = description

        if successful_tasks:
            config_dict['successful_tasks'] = successful_tasks

        if metadata:
            # Store metadata in config for future reference
            if 'meta' not in config_dict:
                config_dict['meta'] = {}
            config_dict['meta'].update(metadata)
            config_dict['meta']['added_at'] = datetime.now().isoformat()

        # Generate filename (sanitize name)
        safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
        filename = f"{safe_name}.yaml"
        filepath = self.pool_dir / filename

        # Check if file already exists
        if filepath.exists():
            logger.warning(f"Configuration {filename} already exists, overwriting")

        # Save to pool
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Added configuration to pool: {filename}")

        return str(filepath)

    def update_configuration(
        self,
        config_path: str,
        successful_tasks: list = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Update an existing configuration in the pool.

        Args:
            config_path: Path to configuration file
            successful_tasks: New successful tasks to add
            metadata: Additional metadata to update
        """
        config_path = Path(config_path)

        if not config_path.exists():
            logger.error(f"Configuration not found: {config_path}")
            return

        # Load existing config
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Update successful tasks
        if successful_tasks:
            existing_tasks = config_dict.get('successful_tasks', [])
            # Merge tasks (avoid duplicates)
            task_queries = {task.get('q') for task in existing_tasks if isinstance(task, dict)}
            for task in successful_tasks:
                if isinstance(task, dict) and task.get('q') not in task_queries:
                    existing_tasks.append(task)
            config_dict['successful_tasks'] = existing_tasks

        # Update metadata
        if metadata:
            if 'meta' not in config_dict:
                config_dict['meta'] = {}
            config_dict['meta'].update(metadata)
            config_dict['meta']['updated_at'] = datetime.now().isoformat()

        # Save updated config
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Updated configuration: {config_path.name}")

    def get_configurations(self) -> list:
        """
        Get all configurations in the pool.

        Returns:
            List of configuration file paths
        """
        configs = list(self.pool_dir.glob("*.yaml"))
        return [str(c) for c in configs]

    def get_configuration_count(self) -> int:
        """Get number of configurations in pool."""
        return len(list(self.pool_dir.glob("*.yaml")))

    def remove_configuration(self, config_path: str):
        """
        Remove a configuration from the pool.

        Args:
            config_path: Path to configuration file
        """
        config_path = Path(config_path)

        if not config_path.exists():
            logger.warning(f"Configuration not found: {config_path}")
            return

        config_path.unlink()
        logger.info(f"Removed configuration: {config_path.name}")

    def backup_pool(self, backup_dir: str):
        """
        Create a backup of the entire pool.

        Args:
            backup_dir: Directory to save backup
        """
        backup_path = Path(backup_dir) / f"pool_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copytree(self.pool_dir, backup_path)
        logger.info(f"Created pool backup: {backup_path}")

    def __len__(self) -> int:
        """Return number of configurations in pool."""
        return self.get_configuration_count()


def should_update_pool(
    new_reward: float,
    parent_rewards: list,
    threshold: float = 0.01
) -> bool:
    """
    Determine if a new configuration should be added to the pool.

    Configuration is added if: R(C_new) > max(R(C_parents)) + threshold

    Args:
        new_reward: Reward of new configuration
        parent_rewards: Rewards of parent configurations
        threshold: Minimum improvement threshold

    Returns:
        True if configuration should be added to pool
    """
    if not parent_rewards:
        # If no parents, always add
        return True

    max_parent_reward = max(parent_rewards)
    improvement = new_reward - max_parent_reward

    logger.info(f"Pool update decision:")
    logger.info(f"  New reward: {new_reward:.4f}")
    logger.info(f"  Max parent reward: {max_parent_reward:.4f}")
    logger.info(f"  Improvement: {improvement:+.4f} (threshold: {threshold})")

    return improvement > threshold


def add_to_pool_if_better(
    pool: ConfigurationPool,
    config_yaml: str,
    new_metrics: Dict[str, Any],
    parent_metrics: list,
    task_query: str,
    beta: float = 1e-6,
    cost_weight: str = "both",
    threshold: float = 0.01
) -> Optional[str]:
    """
    Add configuration to pool if it achieves higher reward than parents.

    Args:
        pool: ConfigurationPool instance
        config_yaml: Configuration as YAML string
        new_metrics: Metrics for new configuration
        parent_metrics: List of metrics for parent configurations
        task_query: Task query that was solved
        beta: Cost trade-off parameter for reward
        cost_weight: Which cost metric to use ("tokens", "time", or "both")
        threshold: Minimum improvement threshold

    Returns:
        Path to added configuration, or None if not added
    """
    from .reward import compute_reward

    # Compute rewards
    new_reward = compute_reward(new_metrics, beta=beta, cost_weight=cost_weight)
    parent_rewards = [compute_reward(m, beta=beta, cost_weight=cost_weight) for m in parent_metrics]

    # Check if should add
    if not should_update_pool(new_reward, parent_rewards, threshold):
        logger.info("Configuration not added to pool (insufficient improvement)")
        return None

    # Generate name and description
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"evolved_{timestamp}"
    description = f"Evolved configuration with reward {new_reward:.4f}"

    # Add successful task info
    successful_tasks = [{
        "q": task_query[:200],  # Truncate long queries
        "notes": f"Reward: {new_reward:.4f}, Accuracy: {new_metrics.get('accuracy', 0):.2%}"
    }]

    # Add metadata
    metadata = {
        "reward": new_reward,
        "accuracy": new_metrics.get('accuracy', 0),
        "total_tokens": new_metrics.get('total_tokens', 0),
        "evolution_source": "evolved"
    }

    # Add to pool
    config_path = pool.add_configuration(
        config_yaml=config_yaml,
        name=name,
        description=description,
        successful_tasks=successful_tasks,
        metadata=metadata
    )

    logger.info(f"Configuration added to pool: {config_path}")

    return config_path
