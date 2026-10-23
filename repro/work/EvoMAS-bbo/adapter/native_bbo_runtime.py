"""BBO task adapter around EvoMAS's original MasRuntime and SmolagentsRunner.

Generation runs on the research host. Nothing from this module is submitted as
BBO solver code; a separately checked candidate must be a self-contained file.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml

ROLE_TASKS = {
    "optimizer_designer": "Design a noise-aware optimizer. Explain query allocation, anytime quality and an implementable plan. Do not write files or call the evaluator.",
    "code_implementer": 'Produce a complete, self-contained solver.py in one fenced Python block. Implement Optimizer(dim, lower, upper, budget, seed, rng), ask(n), tell(X, y). Use only stdlib and NumPy. Keep algorithm state shapes explicit: values used in Python if/max/min must be scalars; for per-dimension bounds use NumPy elementwise operations. Do not pass NumPy vectors to Python built-in max/min or boolean conditions.',
    'reviewer_create': 'As an EvoMAS peer-review creator, propose one complete BBO optimizer. Return solver.py source using only stdlib and NumPy. Keep scalar state scalar; use NumPy elementwise operations for vector bounds. Do not access the objective oracle.',
    'reviewer_review': 'As an EvoMAS peer reviewer, inspect prior optimizer proposals for the BBO ask/tell contract, noisy observations, query budget, bounds, finite outputs, and NumPy scalar/vector errors. Return actionable findings; do not call tools or the evaluator.',
    'reviewer_revise': 'As an EvoMAS reviser, use prior proposals and reviews to produce complete self-contained solver.py source. Implement Optimizer(dim, lower, upper, budget, seed, rng), ask(n), tell(X, y, metadata=None). Use only stdlib and NumPy; avoid Python max/min or truth tests on NumPy vectors. Do not access the objective oracle.',
    'worker': 'As an EvoMAS worker, independently produce complete BBO solver.py source. Implement Optimizer(dim, lower, upper, budget, seed, rng), ask(n), tell(X, y, metadata=None). Use only stdlib and NumPy; ensure finite in-bounds queries within budget. Avoid Python max/min or truth tests on NumPy vectors.',
    'debater_initial': 'As an EvoMAS initial debater, propose a complete noise-aware BBO optimizer. Implement the required Optimizer constructor, ask(n), and tell(X, y, metadata=None) using only stdlib and NumPy. Avoid Python max/min or truth tests on NumPy vectors.',
    'debater_round': 'As an EvoMAS debate participant, analyze prior optimizer proposals and give specific improvements while preserving the BBO ask/tell contract. Avoid Python max/min or truth tests on NumPy vectors; do not access the objective oracle.',
    "aggregator": "Use the preceding agents' reports. Return exactly one complete, self-contained solver.py in one fenced Python block; repair any invalid interface. Before returning, check that ask(n) returns a finite in-bounds array of shape (n, dim), tell(X, y, metadata=None) accepts the evaluator's 3-argument call, and no NumPy vector is used as a Python scalar condition or passed to built-in max/min. Use NumPy elementwise operations for vector steps and scalarize any global step size. Do not execute it or call the evaluator.",
}
ALLOWED_MODELS = {"openai:deepseek-v4-flash-s2"}
ALLOWED_TOOLS = {"python_interpreter", "python_code_interpreter"}


def agent_instruction(agent) -> str:
    """Keep evolved prompt variants subordinate to BBO's required role contract."""
    role_defaults = {
        "worker": ROLE_TASKS["worker"],
        "creator": ROLE_TASKS["reviewer_create"],
        "reviewer": ROLE_TASKS["reviewer_review"],
        "reviser": ROLE_TASKS["reviewer_revise"],
        "debater": ROLE_TASKS.get(agent.prompt, ROLE_TASKS["debater_round"]),
        "aggregator": ROLE_TASKS["aggregator"],
    }
    canonical = ROLE_TASKS.get(agent.prompt)
    if canonical is None:
        canonical = role_defaults.get(agent.role)
    if canonical is None:
        raise ValueError(f"No BBO instruction mapped for EvoMAS role {agent.role!r}")
    variant = "" if agent.prompt in ROLE_TASKS else (
        "\n\nEvolved EvoMAS prompt variant (retain compatible strategy guidance only; "
        "the BBO task and required role output below take precedence over conflicts):\n"
        + agent.prompt
    )
    return canonical + variant

def extract_candidate(content: str) -> str:
    def source_with_optimizer(source: str) -> bool:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return False
        return any(
            isinstance(node, ast.ClassDef)
            and node.name == "Optimizer"
            and {"__init__", "ask", "tell"}.issubset(
                {method.name for method in node.body if isinstance(method, ast.FunctionDef)}
            )
            for node in tree.body
        )

    def unwrap(source: str) -> str | None:
        source = source.strip() + "\n"
        if source_with_optimizer(source):
            return source
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        embedded = [
            node.value.value.strip() + "\n"
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.startswith("solver_code")
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and source_with_optimizer(node.value.value)
        ]
        if len(embedded) == 1:
            return embedded[0]
        return None

    blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", content, re.I | re.S)
    # Some OpenAI-compatible models use XML-style code tags rather than
    # Markdown fences. Treat those as containers, never as Python source.
    blocks.extend(re.findall(r"<code>\s*(.*?)\s*</code>", content, re.I | re.S))
    for block in sorted(blocks, key=len, reverse=True):
        candidate = unwrap(block)
        if candidate is not None:
            return candidate
    # Smolagents may return executable source directly as its final answer.
    # Preserve support for the un-fenced form instead of treating it as a run failure.
    raw = content.strip()
    raw = re.sub(r"^Final answer:\s*", "", raw, flags=re.I).strip()
    candidate = unwrap(raw)
    if candidate is not None:
        return candidate
    raise ValueError("No top-level Optimizer candidate or unambiguous solver_code string in final report")

def validate(spec) -> list[list[str]]:
    agents = spec.agents
    if spec.backend != "smolagents":
        raise ValueError("BBO adapter requires the native smolagents backend")
    if not agents:
        raise ValueError("Empty MAS configuration")
    graph = spec.topology.reports_to
    if any(source not in agents or any(target not in agents for target in targets)
           for source, targets in graph.items()):
        raise ValueError("Topology names an unknown agent")
    levels = spec.get_execution_levels()
    if sorted(a for level in levels for a in level) != sorted(agents):
        raise ValueError("Incomplete or cyclic topology")
    for aid, agent in agents.items():
        if agent.backend not in (None, "smolagents") or agent.agent_type != "CodeAgent":
            raise ValueError(f"Unsupported native runner for {aid}")
        if agent.id != aid:
            raise ValueError(f"Agent ID mismatch: {aid} / {agent.id}")
        if agent.model_id not in ALLOWED_MODELS:
            raise ValueError(f"Unapproved model for {aid}: {agent.model_id}")
        if not isinstance(agent.prompt, str) or not agent.prompt.strip():
            raise ValueError(f"Empty or invalid EvoMAS prompt for {aid}")
        if set(agent.tools) - ALLOWED_TOOLS:
            raise ValueError(f"Unsupported tools for {aid}: {agent.tools}")

    terminal = levels[-1]
    if len(terminal) != 1:
        raise ValueError("BBO adapter requires one terminal candidate-producing agent")
    terminal_role = agents[terminal[0]].role
    if terminal_role != "aggregator" and not (
        len(agents) == 1 and terminal_role == "worker"
    ):
        raise ValueError("BBO candidate requires an aggregator or one upstream worker")
    return levels

def execute(config_path: Path, instruction_path: Path, output_dir: Path, dry_run: bool):
    root = Path(__file__).resolve().parents[2] / "EvoMAS-openai"
    sys.path.insert(0, str(root))
    from src.agents.spec import AgentResult
    from src.mas.spec import MasSpec
    import src.mas.runtime as runtime_module

    spec = MasSpec(**yaml.safe_load(config_path.read_text(encoding="utf-8")))
    levels = validate(spec)
    terminal_agent = levels[-1][0]
    output_dir.mkdir(parents=True, exist_ok=True)
    instruction = instruction_path.read_text(encoding="utf-8")
    task = ("RSI-Exam BBO optimizer generation. The goal is to MINIMIZE noisy y. "
            "No objective oracle is available to agents. Candidate code must obey:\n\n"
            + instruction)
    failures = []
    calls = []

    if dry_run:
        class UnderlyingRunner:
            def run(self, agent, text, context=None):
                calls.append({"agent": agent.id, "model": agent.model_id,
                              "role": agent.prompt,
                              "reports_seen": sorted((context or {}).get("reports", {})),
                              "task_has_instruction": "strict budget of 120" in text,
                              "task_has_role": agent_instruction(agent) in text})
                return AgentResult(agent_id=agent.id, content=(
                    "```python\nclass Optimizer:\n"
                    "    def __init__(self, dim, lower, upper, budget, seed, rng): pass\n"
                    "    def ask(self, n): pass\n"
                    "    def tell(self, X, y): pass\n```"
                    if agent.id == terminal_agent else f"Mock report by {agent.id}"
                ), success=True)
    else:
        from src.agents.runners.smolagents import SmolagentsRunner
        UnderlyingRunner = SmolagentsRunner

    class RoleRunner:
        def __init__(self):
            self.delegate = UnderlyingRunner()

        def run(self, agent, text, context=None):
            result = self.delegate.run(
                agent, agent_instruction(agent) + "\n\n" + text, context
            )
            (output_dir / f"agent_{agent.id}.json").write_text(
                json.dumps({
                    "agent_id": agent.id,
                    "success": result.success,
                    "error": result.error,
                    "content": result.content,
                    "metadata": result.metadata,
                }, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if not result.success:
                failures.append({"agent": agent.id, "error": result.error})
            return result

    # Preserve EvoMAS's runtime DAG, scheduling, context and runner mechanics.
    # The adapter supplies BBO-specific instructions at the runner boundary.
    original_factory = runtime_module._get_runner_class
    runtime_module._get_runner_class = lambda backend: (
        RoleRunner if backend == "smolagents" else original_factory(backend)
    )
    try:
        final, metadata = runtime_module.MasRuntime(spec).run(task)
    finally:
        runtime_module._get_runner_class = original_factory
    # Write the raw final output before parsing so malformed/novel formats remain diagnosable.
    (output_dir / "aggregator_response.txt").write_text(final, encoding="utf-8")
    if failures or final.startswith("Error:"):
        raise RuntimeError(f"Agent failed; errors={failures}, final={final[:200]}")
    candidate = extract_candidate(final)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / ("mock_candidate.py" if dry_run else "candidate_solver.py")).write_text(candidate)
    summary = {"dry_run": dry_run, "levels": levels,
               "models": {aid: a.model_id for aid, a in spec.agents.items()},
               "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
               "metadata": metadata, "calls": calls}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--instruction", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    execute(args.config, args.instruction, args.output_dir, args.dry_run)
