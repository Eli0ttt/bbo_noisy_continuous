"""BBO research-time tools. No object here is imported by a submitted solver."""
from __future__ import annotations

import ast
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from smolagents import Tool


CATALOG = {
    "bbo_read_context": "Read the public BBO instruction.md or this agent's candidate.py only.",
    "bbo_write_candidate": "Write this agent's candidate.py in its isolated research scratch folder.",
    "bbo_preflight_candidate": "Check candidate.py syntax, Optimizer signatures, imports, and obvious duplicate query counting without executing it.",
}
CATALOG_TEXT = (
    "BBO generation-stage tools (available now; YAML may assign any subset to agents):\n"
    + "\n".join(f"- {name}: {description}" for name, description in CATALOG.items())
    + "\nUse the YAML tool lists as the starting point; change them only when useful. "
      "Assign bbo_write_candidate and bbo_preflight_candidate to the same agent if it needs to preflight a file. "
      "These tools never call the objective, official selfcheck, or hidden evaluator. "
      "The submitted solver.py must use only Python standard library and NumPy."
)
_AUDIT_LOCK = threading.Lock()


def _audit(root: Path, agent_id: str, tool_name: str, event: str, success=True):
    entry = {
        "time": datetime.now(timezone.utc).isoformat(),
        "agent": agent_id, "tool": tool_name,
        "event": event, "success": bool(success),
    }
    with _AUDIT_LOCK:
        with (root / "tool_audit.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")


def inspect_candidate(source: str) -> str:
    if not isinstance(source, str) or not source.strip() or len(source) > 100_000:
        return "FAIL: candidate must be nonempty Python source under 100000 characters"
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return f"FAIL: SyntaxError line {exc.lineno}: {exc.msg}"
    optimizer = next((n for n in tree.body if isinstance(n, ast.ClassDef)
                      and n.name == "Optimizer"), None)
    if optimizer is None:
        return "FAIL: top-level class Optimizer is missing"
    methods = {n.name: n for n in optimizer.body if isinstance(n, ast.FunctionDef)}
    for method, required in (("__init__", ["self", "dim", "lower", "upper", "budget", "seed", "rng"]),
                             ("ask", ["self", "n"]), ("tell", ["self", "X", "y"])):
        if method not in methods:
            return f"FAIL: Optimizer.{method} is missing"
        names = [arg.arg for arg in methods[method].args.args]
        if names[:len(required)] != required:
            return f"FAIL: Optimizer.{method} must begin with {required}"
    import sys
    allowed = set(sys.stdlib_module_names) | {"numpy"}
    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        if any(name.split(".")[0] not in allowed for name in modules):
            return "FAIL: imports must be from the standard library or NumPy"
    def incremented_counters(method):
        return {
            node.target.attr for node in ast.walk(method)
            if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add)
            and isinstance(node.target, ast.Attribute)
            and isinstance(node.target.value, ast.Name)
            and node.target.value.id == "self"
            and node.target.attr in {"queries_used", "query_count", "n_evals"}
        }
    overlap = incremented_counters(methods["ask"]) & incremented_counters(methods["tell"])
    if overlap:
        return ("FAIL: query counter incremented in both ask and tell: "
                + ", ".join(sorted(overlap)) + "; count actual evaluated points once")
    # A minimum exploration allocation greater than one can exceed ask(1).
    for node in ast.walk(methods["ask"]):
        if not isinstance(node, ast.Assign) or not any(
                isinstance(target, ast.Name) and target.id == "n_explore"
                for target in node.targets):
            continue
        value = node.value
        if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                and value.func.id == "max" and any(
                    isinstance(arg, ast.Constant) and type(arg.value) is int and arg.value > 1
                    for arg in value.args)):
            return ("FAIL: n_explore has a minimum above one; ask(1) can make "
                    "n_exploit = n_actual - n_explore negative. Cap allocation by n_actual")
    return "PASS: static syntax, interface, and import preflight (runtime behavior not tested)"


class _BBOBase(Tool):
    output_type = "string"

    def __init__(self, root: Path, agent_id: str):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", agent_id):
            raise ValueError("unsafe agent id for tool workspace")
        self.root = Path(root).resolve()
        self.agent_id = agent_id
        self.scratch = self.root / "tool_workspace" / agent_id
        self.scratch.mkdir(parents=True, exist_ok=True)
        super().__init__()
        _audit(self.root, agent_id, self.name, "loaded")

    def _record(self, success=True):
        _audit(self.root, self.agent_id, self.name, "called", success)


class BBOReadContext(_BBOBase):
    name = "bbo_read_context"
    description = CATALOG[name]
    inputs = {"item": {"type": "string", "description": "Exactly instruction.md or candidate.py"}}

    def forward(self, item: str) -> str:
        if item == "instruction.md":
            path = self.root / "tool_workspace" / "instruction.md"
        elif item == "candidate.py":
            path = self.scratch / "candidate.py"
        else:
            self._record(False)
            return "DENIED: only instruction.md and candidate.py are readable"
        if path.is_symlink() or not path.is_file():
            self._record(False)
            return "MISSING: requested file does not exist"
        self._record()
        return path.read_text(encoding="utf-8")[:100_000]


class BBOWriteCandidate(_BBOBase):
    name = "bbo_write_candidate"
    description = CATALOG[name]
    inputs = {"source": {"type": "string", "description": "Complete Python source for candidate.py"}}

    def forward(self, source: str) -> str:
        if not isinstance(source, str) or not source.strip() or len(source) > 100_000:
            self._record(False)
            return "DENIED: source must be nonempty and at most 100000 characters"
        path = self.scratch / "candidate.py"
        if path.is_symlink():
            self._record(False)
            return "DENIED: candidate.py cannot be a symbolic link"
        import os
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(source)
        self._record()
        return "WROTE: candidate.py in this agent's research scratch folder"


class BBOPreflightCandidate(_BBOBase):
    name = "bbo_preflight_candidate"
    description = CATALOG[name]
    inputs = {"item": {"type": "string", "description": "Exactly candidate.py"}}

    def forward(self, item: str) -> str:
        path = self.scratch / "candidate.py"
        if item != "candidate.py" or path.is_symlink() or not path.is_file():
            self._record(False)
            return "DENIED: only an existing candidate.py can be preflighted"
        result = inspect_candidate(path.read_text(encoding="utf-8"))
        self._record(result.startswith("PASS"))
        return result


_FACTORIES = {
    "bbo_read_context": BBOReadContext,
    "bbo_write_candidate": BBOWriteCandidate,
    "bbo_preflight_candidate": BBOPreflightCandidate,
}


def make_tool(name: str, root: Path, agent_id: str) -> Tool:
    if name not in _FACTORIES:
        raise ValueError(f"unknown BBO generation tool: {name}")
    return _FACTORIES[name](root, agent_id)
