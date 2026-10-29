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
    'reviewer_review': 'As an EvoMAS peer reviewer, inspect prior optimizer proposals for the BBO ask/tell contract, noisy observations, query budget, bounds, finite outputs, and NumPy scalar/vector errors. Return actionable findings; do not call the evaluator.',
    'reviewer_revise': 'As an EvoMAS reviser, use prior proposals and reviews to produce complete self-contained solver.py source. Implement Optimizer(dim, lower, upper, budget, seed, rng), ask(n), tell(X, y, metadata=None). Use only stdlib and NumPy; avoid Python max/min or truth tests on NumPy vectors. Do not access the objective oracle.',
    'worker': 'As an EvoMAS worker, independently produce complete BBO solver.py source. Implement Optimizer(dim, lower, upper, budget, seed, rng), ask(n), tell(X, y, metadata=None). Use only stdlib and NumPy; ensure finite in-bounds queries within budget. Avoid Python max/min or truth tests on NumPy vectors.',
    'debater_initial': 'As an EvoMAS initial debater, propose a complete noise-aware BBO optimizer. Implement the required Optimizer constructor, ask(n), and tell(X, y, metadata=None) using only stdlib and NumPy. Avoid Python max/min or truth tests on NumPy vectors.',
    'debater_round': 'As an EvoMAS debate participant, analyze prior optimizer proposals and give specific improvements while preserving the BBO ask/tell contract. Avoid Python max/min or truth tests on NumPy vectors; do not access the objective oracle.',
    "aggregator": "Use the preceding agents' reports. Return exactly one complete, self-contained solver.py in one fenced Python block; repair any invalid interface. Before returning, inspect ask(1), ask(31), and ask(120) for finite in-bounds arrays of shape (n, dim); when filling one coordinate points[:, d], use lower[d] and upper[d], never full bound vectors; ensure tell(X, y, metadata=None) accepts the evaluator's 3-argument call, and no NumPy vector is used as a Python scalar condition or passed to built-in max/min. Use NumPy elementwise operations for vector steps and scalarize any global step size. Do not execute it or call the evaluator.",
    "team_worker": "Work with your team on a concrete BBO optimizer design and implementation. Return a complete self-contained solver.py proposal using only the standard library and NumPy; do not call the objective or evaluator.",
    "processor": "Synthesize upstream BBO solver proposals into a concrete improvement. Include the algorithm, ask/tell interface checks, and, when useful, complete solver.py code. Do not call the objective or evaluator.",
    "judge": "Compare upstream BBO solver proposals for validity, noise awareness, bounds, query use, and likely anytime performance. Report actionable findings without accessing the objective or evaluator.",
}
SUPPORTED_MODELS = (
    "openai:deepseek-v4-flash-s1",
    "openai:deepseek-v4-flash-s2",
    "openai:GLM-5.3-W4A8",
    "openai:GLM-5.3-Flash",
)
ALLOWED_MODELS = {"openai:deepseek-v4-flash-s2"}


def canonical_model_id(name: str) -> str:
    model = name.strip()
    if not model.startswith("openai:"):
        model = "openai:" + model
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported BBO model {name!r}; choose from {SUPPORTED_MODELS}")
    return model


def parse_model_pool(value: str) -> list[str]:
    models = list(dict.fromkeys(canonical_model_id(part)
                                for part in value.replace(",", " ").split()))
    if not models:
        raise ValueError("BBO model pool must not be empty")
    return models


def configure_allowed_models(models):
    ALLOWED_MODELS.clear()
    ALLOWED_MODELS.update(models)


def find_project_root(start: Path | None = None) -> Path:
    """Find the task checkout from any supported adapter nesting depth."""
    start = (start or Path(__file__)).resolve()
    for parent in (start, *start.parents):
        if (parent / "environment/methods/main/solver.py").is_file():
            return parent
    raise FileNotFoundError(
        f"Could not find environment/methods/main/solver.py above {start}"
    )


class CandidateExtractionError(ValueError):
    """The MAS ran, but produced no usable BBO source; this is a candidate failure."""


def agent_instruction(agent) -> str:
    """Keep evolved prompt variants subordinate to BBO's required role contract."""
    role_defaults = {
        "worker": ROLE_TASKS["worker"],
        "creator": ROLE_TASKS["reviewer_create"],
        "reviewer": ROLE_TASKS["reviewer_review"],
        "reviser": ROLE_TASKS["reviewer_revise"],
        "debater": ROLE_TASKS.get(agent.prompt, ROLE_TASKS["debater_round"]),
        "aggregator": ROLE_TASKS["aggregator"],
        "team_worker": ROLE_TASKS["team_worker"],
        "processor": ROLE_TASKS["processor"],
        "judge": ROLE_TASKS["judge"],
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
    assigned = set(agent.tools)
    guidance = ["\nThe public task has 120 noisy queries in 10D and scores both "
                "anytime and final best-so-far. Compare concrete, bounded "
                "noise-aware local or covariance adaptation to global random "
                "search; avoid spending most queries on initial space filling "
                "or trusting one noisy minimum. A normalized initial sigma near "
                "0.2 of each coordinate's range and a small batch are public "
                "hypotheses to test, not prescribed parameters. Repeatedly "
                "measuring an identical incumbent consumes a query but can "
                "improve a noisy estimate for later decisions; compare repeat "
                "rates empirically instead of eliminating repeats blindly. "
                "Estimate work per ask/tell "
                "against 400 sealed runs and the aggregate runtime budget. "
                "Consider bounded, vectorized surrogates and quadratic local "
                "models if measured public runtime permits. Never optimize "
                "synthetic uniqueness instead of official visible quality. "
                "Treat these as design hypotheses, not guaranteed scores. "
                "Assigned callable BBO tools for THIS agent: " +
                (", ".join(sorted(assigned)) if assigned else "none") +
                ". Do not invoke tools outside this list; evolved prose does not grant them."]
    if "bbo_read_context" in assigned:
        guidance.append("bbo_read_context(item='instruction.md') reads the public "
                        "task; item='candidate.py' reads only YOUR file. It starts "
                        "as the official uniform-random solver, then your writes "
                        "replace it. Other agents' files cannot be read.")
    if "bbo_read_public_data" in assigned:
        guidance.append("bbo_read_public_data(item='summary') gives the official "
                        "public visible dataset schema; item='instance_0' reads "
                        "one public instance. Use only these named selectors. "
                        "The submitted solver receives no instance parameters.")
    if "bbo_read_prior_feedback" in assigned:
        guidance.append("bbo_read_prior_feedback(item='summary') reads "
                        "the preceding completed version's public feedback "
                        "within this run, if available.")
    if "bbo_read_prior_trace" in assigned:
        guidance.append("bbo_read_prior_trace(item='checkpoints') reads "
                        "sampled PUBLIC visible best-so-far checkpoints at "
                        "queries 20, 60 and 120, if available. The submitted "
                        "solver never receives these latent diagnostics.")
    if "bbo_write_candidate" in assigned:
        guidance.append("bbo_write_candidate(source=...) takes complete Python "
                        "source and writes only YOUR candidate.py.")
    if "bbo_preflight_candidate" in assigned:
        guidance.append("bbo_preflight_candidate(item='candidate.py') requires "
                        "YOUR candidate.py and performs static checks only.")
    if "bbo_probe_candidate" in assigned:
        guidance.append("bbo_probe_candidate(item='candidate.py') runs YOUR "
                        "written candidate through synthetic ask/tell cases in "
                        "the offline visible container. It gives contract and "
                        "duplicate-query diagnostics, no objective or score. "
                        "Write your candidate before invoking this probe.")
    guidance.append("Other agents' reports arrive in the EvoMAS context, never "
                    "through their private files. Do not use open(), exec(), "
                    "importlib, os, sys, or traceback in CodeAgent's restricted "
                    "interpreter to inspect or execute the solver. In interpreter "
                    "steps, call only the assigned BBO tools with their documented "
                    "arguments; put optimizer Python source in a string argument "
                    "or the final response, not as interpreter statements. "
                    "Use bbo_probe_candidate for runtime contract checks; "
                    "the restricted CodeAgent interpreter cannot execute the solver. "
                    "Each agent's candidate.py is private; a downstream agent "
                    "must use the preceding report, not a prior agent's file. "
                    "A final complete "
                    "fenced Python source is required even if a file was written.")
    return canonical + variant + "\n" + " ".join(guidance)

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
    raise CandidateExtractionError("No top-level Optimizer candidate or unambiguous solver_code string in final report")

def validate(spec, tool_mode="none") -> list[list[str]]:
    if tool_mode not in {"none", "bbo"}:
        raise ValueError("unknown BBO tool mode")
    from src.tools.bbo_generation import CATALOG
    permitted_tools = set() if tool_mode == "none" else set(CATALOG)
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
        if set(agent.tools) - permitted_tools:
            raise ValueError(f"Unsupported tools for {aid}: {agent.tools}")

    terminal = [aid for aid in agents if not graph.get(aid)]
    if len(terminal) != 1 or levels[-1] != terminal:
        raise ValueError("BBO adapter requires one terminal candidate-producing agent")
    terminal_role = agents[terminal[0]].role
    if terminal_role != "aggregator" and not (
        len(agents) == 1 and terminal_role == "worker"
    ):
        raise ValueError("BBO candidate requires an aggregator or one upstream worker")
    return levels

def execute(config_path: Path, instruction_path: Path, output_dir: Path, dry_run: bool,
            tool_mode: str = "none", feedback: dict | None = None,
            container: str | None = None):
    root = Path(__file__).resolve().parents[2] / "EvoMAS-openai"
    sys.path.insert(0, str(root))
    from src.agents.spec import AgentResult
    from src.mas.spec import MasSpec
    import src.mas.runtime as runtime_module

    spec = MasSpec(**yaml.safe_load(config_path.read_text(encoding="utf-8")))
    levels = validate(spec, tool_mode)
    terminal_agent = levels[-1][0]
    output_dir.mkdir(parents=True, exist_ok=True)
    instruction = instruction_path.read_text(encoding="utf-8")
    if tool_mode == "bbo":
        work = output_dir / "tool_workspace"
        work.mkdir(exist_ok=True)
        (work / "instruction.md").write_text(instruction, encoding="utf-8")
        if feedback:
            (work / "prior_feedback.json").write_text(
                json.dumps({key: feedback.get(key) for key in
                            ("source_label", "summary", "public_trace")},
                           ensure_ascii=False), encoding="utf-8")
    task = ("RSI-Exam BBO optimizer generation. The goal is to MINIMIZE noisy y. "
            "No objective oracle is available to agents. Count actual evaluations "
            "exactly once across ask/tell: increment the counter in tell or ask, "
            "never both. In ask(1), each branch must keep every allocation nonnegative; "
            "always cap n_explore <= n_actual before n_exploit = n_actual - n_explore. "
            "Use local preflight on the final candidate when the assigned tools permit it. "
            "Candidate code must obey:\n\n" + instruction)
    starter_path = find_project_root() / "environment/methods/main/solver.py"
    starter = starter_path.read_text(encoding="utf-8")
    if tool_mode == "bbo":
        for aid in spec.agents:
            agent_scratch = output_dir / "tool_workspace" / aid
            agent_scratch.mkdir(parents=True, exist_ok=True)
            (agent_scratch / "candidate.py").write_text(starter, encoding="utf-8")
    task += ("\n\n## Official shipped starting solver (uniform random search)\n"
             "Start this run from this task baseline; prior runs' generated solver "
             "source is not inherited.\n```python\n" + starter + "\n```\n")
    if feedback:
        if hashlib.sha256(feedback["solver_source"].encode()).hexdigest() != feedback["source_sha256"]:
            raise ValueError("research feedback source hash mismatch")
        (output_dir / "research_feedback.json").write_text(
            json.dumps(feedback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        task += "\n\n## Prior research feedback (public visible data only)\n" + feedback["summary"]
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
            if not dry_run and tool_mode == "bbo":
                self.delegate.bbo_tool_root = output_dir
                self.delegate.bbo_eval_container = container

        def run(self, agent, text, context=None):
            guidance = ("Preserve the incumbent's valid interface and working search "
                        "components. Make one specific, defensible change aimed at the "
                        "visible anytime/final weakness, and produce the full solver.py. "
                        if feedback and feedback.get("source_contract_valid") else
                        "Repair the incumbent's diagnosed contract failure and produce "
                        "the full solver.py. ")
            prior = ("\n\n## Prior incumbent solver.py source\n```python\n"
                     + feedback["solver_source"] + "\n```\n"
                     + guidance + "Do not merely describe a patch.\n"
                     if feedback and agent.id == terminal_agent else "")
            result = self.delegate.run(
                agent, agent_instruction(agent) + "\n\n" + text + prior, context
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
    try:
        candidate = extract_candidate(final)
        candidate_source = "final_report"
    except CandidateExtractionError:
        # A CodeAgent may use the YAML-loaded writer and finish with a path-only
        # message. Accept only its own audited file, never another agent's file.
        if tool_mode != "bbo":
            raise
        owned = output_dir / "tool_workspace" / terminal_agent / "candidate.py"
        audit_file = output_dir / "tool_audit.jsonl"
        if owned.is_symlink() or not owned.is_file() or not audit_file.is_file():
            raise
        writes = [json.loads(row) for row in audit_file.read_text().splitlines()
                  if row.strip()]
        if not any(row.get("agent") == terminal_agent and
                   row.get("tool") == "bbo_write_candidate" and
                   row.get("event") == "called" and row.get("success") is True
                   for row in writes):
            raise
        candidate = extract_candidate(owned.read_text(encoding="utf-8"))
        candidate_source = "terminal_agent_tool_file"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / ("mock_candidate.py" if dry_run else "candidate_solver.py")).write_text(candidate)
    summary = {"dry_run": dry_run, "tool_mode": tool_mode, "levels": levels,
               "research_feedback_source": feedback["source_label"] if feedback else None,
               "research_feedback_sha256": feedback["source_sha256"] if feedback else None,
               "configured_tools": {aid: list(a.tools) for aid, a in spec.agents.items()},
               "models": {aid: a.model_id for aid, a in spec.agents.items()},
               "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
               "candidate_source": candidate_source,
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
