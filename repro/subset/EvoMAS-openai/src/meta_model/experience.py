"""
Action Experience dataclass for storing meta-model action experiences.

Implements evolution trace consolidation: h_q ~ π^U(· | O_q)

Stores both individual action experiences and consolidated evolution traces
for entire queries.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import json


@dataclass
class ActionExperience:
    """
    Stores experience from meta-model actions (generate, mutate, crossover).

    This captures:
    - What action was taken
    - What configuration changes were made
    - Whether the action improved performance (success/failure)
    - Analysis of why it succeeded or failed
    """
    query: str  # The task/problem being solved
    action: str  # generate, mutate, or crossover
    config_changes: str  # Description of configuration modifications
    old_accuracy: float  # Accuracy before action
    new_accuracy: float  # Accuracy after action
    success: bool  # Whether action improved performance
    analysis: str  # Why the action succeeded or failed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActionExperience':
        """Create ActionExperience from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'ActionExperience':
        """Create ActionExperience from JSON string."""
        return cls.from_dict(json.loads(json_str))


class MemoryStore:
    """
    Simple memory store for action experiences.

    Keeps track of past actions and their outcomes to guide future decisions.
    """

    def __init__(self):
        self.experiences: List[ActionExperience] = []

    def add(self, experience: ActionExperience):
        """Add a new experience to memory."""
        self.experiences.append(experience)

    def get_all(self) -> List[ActionExperience]:
        """Get all stored experiences."""
        return self.experiences

    def get_by_action(self, action: str) -> List[ActionExperience]:
        """Get experiences filtered by action type."""
        return [exp for exp in self.experiences if exp.action == action]

    def get_successful(self) -> List[ActionExperience]:
        """Get only successful experiences."""
        return [exp for exp in self.experiences if exp.success]

    def get_failed(self) -> List[ActionExperience]:
        """Get only failed experiences."""
        return [exp for exp in self.experiences if not exp.success]

    def to_context_string(self, max_experiences: int = 5) -> str:
        """
        Convert memory to a context string for prompts.

        Args:
            max_experiences: Maximum number of recent experiences to include

        Returns:
            Formatted string of experiences for prompt context
        """
        if not self.experiences:
            return "No previous experiences available."

        recent = self.experiences[-max_experiences:]

        context = "Previous action experiences:\n\n"
        for i, exp in enumerate(recent, 1):
            result = "✓ Success" if exp.success else "✗ Failed"
            context += f"Experience {i} ({result}):\n"
            context += f"  Action: {exp.action}\n"
            context += f"  Query: {exp.query[:100]}...\n"
            context += f"  Changes: {exp.config_changes}\n"
            context += f"  Performance: {exp.old_accuracy:.2%} → {exp.new_accuracy:.2%}\n"
            context += f"  Analysis: {exp.analysis}\n\n"

        return context

    def save(self, filepath: str):
        """Save memory to JSON file."""
        with open(filepath, 'w') as f:
            json.dump([exp.to_dict() for exp in self.experiences], f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'MemoryStore':
        """Load memory from JSON file."""
        memory = cls()
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                memory.experiences = [ActionExperience.from_dict(exp) for exp in data]
        except FileNotFoundError:
            pass  # Return empty memory if file doesn't exist
        return memory

    def clear(self):
        """Clear all experiences from memory."""
        self.experiences = []

    def __len__(self) -> int:
        """Return number of stored experiences."""
        return len(self.experiences)


@dataclass
class EvolutionTrace:
    """
    Stores the evolution trace for a single query: O_q = (π^M_1, π^C_1, ..., π^M_n, π^C_n)

    This captures the sequence of mutation and crossover operations
    applied during evolution, along with their outcomes.
    """
    query: str  # Task query
    operations: List[Dict[str, Any]]  # List of operations (mutation/crossover)
    initial_accuracy: float  # Starting accuracy
    final_accuracy: float  # Final accuracy
    best_config: str  # Best configuration found (YAML)
    improvement: float  # Final - initial accuracy

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvolutionTrace':
        """Create from dictionary."""
        return cls(**data)


def consolidate_evolution_trace(
    operations: List[Dict[str, Any]],
    query: str,
    initial_accuracy: float,
    final_accuracy: float,
    best_config: str
) -> str:
    """
    Consolidate evolution trace into summary: h_q ~ π^U(· | O_q)

    This extracts patterns and insights from the sequence of evolutionary
    operations to guide future evolution.

    Args:
        operations: List of operations performed (mutation/crossover with results)
        query: Task query
        initial_accuracy: Starting accuracy
        final_accuracy: Final accuracy
        best_config: Best configuration found

    Returns:
        Consolidated summary string
    """
    trace = EvolutionTrace(
        query=query,
        operations=operations,
        initial_accuracy=initial_accuracy,
        final_accuracy=final_accuracy,
        best_config=best_config,
        improvement=final_accuracy - initial_accuracy
    )

    # Generate summary
    summary_parts = []
    summary_parts.append(f"Evolution Summary for Query: {query[:100]}...")
    summary_parts.append(f"")
    summary_parts.append(f"Performance: {initial_accuracy:.2%} → {final_accuracy:.2%} "
                        f"(Δ {trace.improvement:+.2%})")
    summary_parts.append(f"")

    # Analyze operations
    num_mutations = sum(1 for op in operations if op.get('type') == 'mutation')
    num_crossovers = sum(1 for op in operations if op.get('type') == 'crossover')

    summary_parts.append(f"Operations: {num_mutations} mutations, {num_crossovers} crossovers")
    summary_parts.append(f"")

    # Identify successful operations
    successful_ops = [op for op in operations if op.get('improved', False)]
    if successful_ops:
        summary_parts.append(f"Successful Operations ({len(successful_ops)}):")
        for op in successful_ops:
            op_type = op.get('type', 'unknown')
            acc_change = op.get('accuracy_change', 0)
            changes = op.get('changes', 'N/A')
            summary_parts.append(f"  - {op_type}: {acc_change:+.2%} ({changes})")
        summary_parts.append(f"")

    # Identify patterns
    if trace.improvement > 0:
        summary_parts.append("Key Insights:")
        if successful_ops:
            # Find most impactful operation
            best_op = max(successful_ops, key=lambda x: x.get('accuracy_change', 0))
            summary_parts.append(f"  - Most effective: {best_op.get('type')} "
                                f"({best_op.get('accuracy_change', 0):+.2%})")

        summary_parts.append(f"  - Final configuration improved over initial")
    else:
        summary_parts.append("Note: No improvement achieved in this evolution")

    return "\n".join(summary_parts)
