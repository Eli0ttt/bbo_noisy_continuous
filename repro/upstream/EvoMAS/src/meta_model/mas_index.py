"""
MAS Configuration Index for EvoMAS.

Maintains a centralized JSON index of all MAS configurations in the pool,
tracking structure summaries, performance history, solved tasks, and costs.
This index is used by the meta-model for informed selection.
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MASIndex:
    """
    Centralized index of MAS configurations and their performance history.

    Stored as a JSON file alongside the pool directory.
    Each entry tracks:
    - Structure summary (agents, topology, models)
    - Solved tasks with accuracy
    - Average cost/time
    - Win/loss record
    """

    def __init__(self, pool_dir: str):
        self.pool_dir = Path(pool_dir)
        self.index_path = self.pool_dir / "mas_index.json"
        self.index: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        """Load index from disk, or build from pool configs if not exists."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                self.index = json.load(f)
            logger.info(f"Loaded MAS index with {len(self.index)} entries")
        else:
            self._build_from_pool()

    def _build_from_pool(self):
        """Build index from existing pool YAML configs."""
        for config_file in self.pool_dir.glob("*.yaml"):
            try:
                self._index_config(config_file)
            except Exception as e:
                logger.warning(f"Failed to index {config_file.name}: {e}")
        self.save()
        logger.info(f"Built MAS index with {len(self.index)} entries from pool")

    def _index_config(self, config_path: Path):
        """Extract and store metadata from a config file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        name = config_path.stem
        agents = config.get("agents", {})
        topology = config.get("topology", {})
        reports_to = topology.get("reports_to", {})

        # Build structure summary
        agent_roles = []
        agent_models = set()
        for agent_id, agent_info in agents.items():
            role = agent_info.get("role", "unknown")
            model = agent_info.get("model_id", "unknown")
            agent_roles.append(f"{agent_id}({role})")
            agent_models.add(model)

        # Build topology summary
        edges = []
        for src, dsts in reports_to.items():
            if dsts:
                for dst in dsts:
                    edges.append(f"{src}→{dst}")

        self.index[name] = {
            "path": str(config_path),
            "name": config.get("name", name),
            "description": config.get("description", ""),
            "backend": config.get("backend", "unknown"),
            # Structure
            "num_agents": len(agents),
            "agent_roles": agent_roles,
            "agent_models": list(agent_models),
            "topology_edges": edges,
            "structure_summary": self._make_structure_summary(agents, edges),
            # Performance (accumulated over queries)
            "solved_tasks": [],  # [{task_id, query_snippet, accuracy}]
            "total_queries": 0,
            "total_wins": 0,  # Times this config was the best for a query
            "avg_accuracy": 0.0,
            "avg_tokens": 0,
            "avg_time_seconds": 0.0,
            # Metadata
            "source": config.get("meta", {}).get("evolution_source", "seed"),
            "created_at": config.get("meta", {}).get("added_at", "unknown"),
        }

    def _make_structure_summary(self, agents: dict, edges: list) -> str:
        """Create a human-readable one-line structure summary."""
        n = len(agents)
        roles = [a.get("role", "?") for a in agents.values()]
        role_counts = {}
        for r in roles:
            role_counts[r] = role_counts.get(r, 0) + 1

        role_str = ", ".join(f"{count}×{role}" if count > 1 else role
                            for role, count in role_counts.items())

        if not edges:
            topo = "independent"
        elif len(edges) == n - 1:
            topo = "chain/tree"
        elif len(edges) >= n * (n - 1) // 2:
            topo = "fully-connected"
        else:
            topo = f"{len(edges)}-edge graph"

        return f"{n} agents [{role_str}], {topo}"

    def save(self):
        """Persist index to disk."""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)

    def record_query_result(
        self,
        config_name: str,
        task_id: Any,
        query_snippet: str,
        accuracy: float,
        tokens: int = 0,
        time_seconds: float = 0.0,
        is_winner: bool = False
    ):
        """Record the result of running a config on a query."""
        if config_name not in self.index:
            logger.warning(f"Config {config_name} not in index, skipping record")
            return

        entry = self.index[config_name]
        entry["total_queries"] += 1
        if is_winner:
            entry["total_wins"] += 1

        # Update running averages
        n = entry["total_queries"]
        entry["avg_accuracy"] = ((n - 1) * entry["avg_accuracy"] + accuracy) / n
        entry["avg_tokens"] = int(((n - 1) * entry["avg_tokens"] + tokens) / n)
        entry["avg_time_seconds"] = ((n - 1) * entry["avg_time_seconds"] + time_seconds) / n

        # Record solved task (keep last 20 to avoid bloat)
        if accuracy > 0:
            entry["solved_tasks"].append({
                "task_id": str(task_id),
                "query": query_snippet[:150],
                "accuracy": accuracy,
                "timestamp": datetime.now().isoformat()
            })
            entry["solved_tasks"] = entry["solved_tasks"][-20:]

        self.save()

    def add_config(self, config_path: str):
        """Add a newly evolved config to the index."""
        self._index_config(Path(config_path))
        self.save()

    def get_selection_context(self, max_configs: int = 20) -> str:
        """
        Format the index as context for the meta-model's selection prompt.

        Returns a structured text describing each MAS in the pool with
        its structure, performance history, and solved tasks.
        """
        lines = []
        # Sort by win rate then accuracy
        sorted_entries = sorted(
            self.index.items(),
            key=lambda x: (x[1].get("total_wins", 0), x[1].get("avg_accuracy", 0)),
            reverse=True
        )

        for name, entry in sorted_entries[:max_configs]:
            lines.append(f"### {name}")
            lines.append(f"- Structure: {entry.get('structure_summary', 'N/A')}")
            lines.append(f"- Description: {entry.get('description', 'N/A')}")

            total_q = entry.get("total_queries", 0)
            if total_q > 0:
                wins = entry.get("total_wins", 0)
                lines.append(f"- Performance: {wins}/{total_q} wins, "
                             f"avg accuracy {entry['avg_accuracy']:.1%}, "
                             f"avg {entry['avg_tokens']} tokens, "
                             f"avg {entry['avg_time_seconds']:.1f}s")

                solved = entry.get("solved_tasks", [])
                if solved:
                    recent = solved[-3:]  # Last 3 solved tasks
                    task_strs = [f"{t['query'][:60]}... (acc:{t['accuracy']:.0%})" for t in recent]
                    lines.append(f"- Recent solved tasks: {'; '.join(task_strs)}")
            else:
                lines.append(f"- Performance: Not yet evaluated")

            lines.append("")

        return "\n".join(lines)

    def __len__(self):
        return len(self.index)

    def __contains__(self, name: str):
        return name in self.index
