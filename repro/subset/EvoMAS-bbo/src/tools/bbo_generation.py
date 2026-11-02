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
    "bbo_read_context": "Read the public BBO instruction.md or this agent's candidate.py only; candidate.py begins with the official starter.",
    "bbo_read_public_data": "Read a bounded summary or one named instance from the official public visible.json; never reads sealed data.",
    "bbo_read_prior_feedback": "Read the last completed version's public visible score, error and structured research feedback within THIS run.",
    "bbo_read_prior_trace": "Read bounded 20/60/120-query diagnostic trajectories sampled on PUBLIC visible cases from the preceding version in THIS run.",
    "bbo_read_history": "List or read any complete or failed solver.py candidate from THIS research run; no earlier-run solver or hidden data.",
    "bbo_write_candidate": "Write this agent's candidate.py in its isolated research scratch folder.",
    "bbo_preflight_candidate": "Check candidate.py syntax, Optimizer signatures, imports, and obvious duplicate query counting without executing it.",
    "bbo_probe_candidate": "After writing candidate.py, run synthetic ask/tell interface and duplicate-query checks in the offline visible container; no objective or score.",
}
CATALOG_TEXT = (
    "BBO generation-stage tools (available now; YAML may assign any subset to agents):\n"
    + "\n".join(f"- {name}: {description}" for name, description in CATALOG.items())
    + "\nUse the YAML tool lists as the starting point; change them only when useful. "
      "Assign bbo_write_candidate with bbo_preflight_candidate and bbo_probe_candidate to an agent that needs to check its own candidate. "
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
            return "MISSING: your candidate.py is absent; write your own candidate first"
        self._record()
        return path.read_text(encoding="utf-8")[:100_000]


class BBOReadPublicData(_BBOBase):
    name = "bbo_read_public_data"
    description = CATALOG[name]
    inputs = {"item": {"type": "string", "description": "Exactly summary or instance_0 through instance_11"}}

    def forward(self, item: str) -> str:
        if item != "summary" and not re.fullmatch(r"instance_(0|[1-9][0-9]?)", item):
            self._record(False)
            return "DENIED: choose summary or a numbered public visible instance"
        task_root = next((p for p in Path(__file__).resolve().parents
                          if (p / "environment/methods/main/solver.py").is_file()), None)
        if task_root is None:
            self._record(False)
            return "MISSING: public task checkout is unavailable"
        public = task_root / "environment/data/visible.json"
        if public.is_symlink() or not public.is_file():
            self._record(False)
            return "MISSING: official public visible.json is unavailable"
        data = json.loads(public.read_text(encoding="utf-8"))
        instances = data["instances"]
        if item == "summary":
            result = {"task_id": data.get("task_id"), "budget": data.get("budget"),
                      "n_instances": len(instances),
                      "instance_fields": sorted(instances[0]) if instances else []}
        else:
            index = int(item.split("_", 1)[1])
            if index >= len(instances):
                self._record(False)
                return "DENIED: public instance index is out of range"
            result = instances[index]
        answer = json.dumps(result, ensure_ascii=False)
        if len(answer) > 60_000:
            self._record(False)
            return "DENIED: public instance exceeds bounded tool response"
        self._record(True)
        return answer


class BBOReadPriorFeedback(_BBOBase):
    name = "bbo_read_prior_feedback"
    description = CATALOG[name]
    inputs = {"item": {"type": "string", "description": "Exactly summary"}}

    def forward(self, item: str) -> str:
        path = self.root / "tool_workspace" / "prior_feedback.json"
        if item != "summary":
            self._record(False)
            return "DENIED: choose summary"
        if path.is_symlink() or not path.is_file():
            self._record()
            return "NO_PRIOR_VERSION: this MAS starts a fresh research run"
        data = json.loads(path.read_text(encoding="utf-8"))
        self._record()
        return json.dumps({"source_label": data["source_label"],
                           "summary": data["summary"][:11000]}, ensure_ascii=False)


class BBOReadPriorTrace(_BBOBase):
    name = "bbo_read_prior_trace"
    description = CATALOG[name]
    inputs = {"item": {"type": "string", "description": "Exactly checkpoints"}}

    def forward(self, item: str) -> str:
        path = self.root / "tool_workspace" / "prior_feedback.json"
        if item != "checkpoints":
            self._record(False)
            return "DENIED: choose checkpoints"
        if path.is_symlink() or not path.is_file():
            self._record()
            return "NO_PRIOR_VERSION: this MAS starts a fresh research run"
        data = json.loads(path.read_text(encoding="utf-8"))
        self._record()
        return json.dumps(data.get("public_trace") or {
            "status": "no public trajectory available for the previous version"})


class BBOReadHistory(_BBOBase):
    name = "bbo_read_history"
    description = CATALOG[name]
    inputs = {"item": {"type": "string", "description": "Exactly index or a candidate label from the index"}}

    def forward(self, item: str) -> str:
        path = self.root / "tool_workspace" / "history.json"
        if path.is_symlink() or not path.is_file():
            self._record()
            return "NO_PRIOR_VERSION: no candidate from this run has completed"
        history = json.loads(path.read_text(encoding="utf-8"))
        if item == "index":
            self._record()
            return json.dumps([{k: v for k, v in entry.items() if k != "solver_source"}
                               for entry in history], ensure_ascii=False)
        entry = next((e for e in history if e["label"] == item), None)
        if entry is None:
            self._record(False)
            return "DENIED: item must be index or an exact label from this run"
        self._record()
        answer = json.dumps(entry, ensure_ascii=False)
        return answer if len(answer) <= 250_000 else "DENIED: candidate history entry is too long"


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


class BBOProbeCandidate(_BBOBase):
    name = "bbo_probe_candidate"
    description = CATALOG[name]
    inputs = {"item": {"type": "string", "description": "Exactly candidate.py, after this agent writes it"}}

    def __init__(self, root: Path, agent_id: str, container: str | None = None):
        self.container = container
        super().__init__(root, agent_id)

    def forward(self, item: str) -> str:
        import subprocess
        import uuid
        from visible_selfcheck import CONTRACT_PROBE

        path = self.scratch / "candidate.py"
        if item != "candidate.py" or path.is_symlink() or not path.is_file():
            self._record(False)
            return "DENIED: probe requires your own candidate.py"
        audit = self.root / "tool_audit.jsonl"
        written = audit.is_file() and any(
            event.get("agent") == self.agent_id
            and event.get("tool") == "bbo_write_candidate"
            and event.get("event") == "called" and event.get("success") is True
            for line in audit.read_text(encoding="utf-8").splitlines()
            if line.strip() for event in [json.loads(line)]
        )
        if not written or not self.container:
            self._record(False)
            return "DENIED: write your candidate first and use the configured offline visible container"
        static = inspect_candidate(path.read_text(encoding="utf-8"))
        if not static.startswith("PASS"):
            self._record(False)
            return static
        remote = "/tmp/bbo_agent_probe_" + uuid.uuid4().hex + ".py"
        try:
            copied = subprocess.run(
                ["docker", "cp", "-a", str(path), f"{self.container}:{remote}"],
                capture_output=True, text=True, timeout=15, check=False)
            if copied.returncode:
                self._record(False)
                return "FAIL: offline probe could not stage candidate: " + copied.stderr[-350:]
            result = subprocess.run(
                ["docker", "exec", self.container, "python", "-c", CONTRACT_PROBE,
                 remote, "--quick"],
                capture_output=True, text=True, timeout=20, check=False)
            payload = next((json.loads(line.removeprefix("PROBE_JSON:"))
                            for line in result.stdout.splitlines()
                            if line.startswith("PROBE_JSON:")), None)
            if not isinstance(payload, dict) or result.returncode or not payload.get("valid"):
                self._record(False)
                return "FAIL: synthetic ask/tell probe: " + (
                    str(payload.get("error", "")) if isinstance(payload, dict)
                    else (result.stderr or result.stdout)[-500:])
            counts = [case.get("unique") for case in payload.get("cases", [])]
            self._record(True)
            return ("PASS: quick synthetic ask/tell only; unique points per "
                    "case=" + str(counts) + "; warnings="
                    + str(payload.get("warnings", []))
                    + ". Repeats may denoise later choices; compare their benefit with lost exploration.")
        except subprocess.TimeoutExpired:
            self._record(False)
            return "FAIL: quick synthetic probe exceeded 20 seconds; candidate may be too slow for the official aggregate runtime"
        except OSError as exc:
            self._record(False)
            return "FAIL: offline probe unavailable: " + str(exc)[-350:]
        finally:
            try:
                subprocess.run(["docker", "exec", self.container, "rm", "-f", remote],
                               capture_output=True, timeout=5, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass


_FACTORIES = {
    "bbo_read_context": BBOReadContext,
    "bbo_read_public_data": BBOReadPublicData,
    "bbo_read_prior_feedback": BBOReadPriorFeedback,
    "bbo_read_prior_trace": BBOReadPriorTrace,
    "bbo_read_history": BBOReadHistory,
    "bbo_write_candidate": BBOWriteCandidate,
    "bbo_preflight_candidate": BBOPreflightCandidate,
    "bbo_probe_candidate": BBOProbeCandidate,
}


def make_tool(name: str, root: Path, agent_id: str, container: str | None = None) -> Tool:
    if name not in _FACTORIES:
        raise ValueError(f"unknown BBO generation tool: {name}")
    if name == "bbo_probe_candidate":
        return BBOProbeCandidate(root, agent_id, container)
    return _FACTORIES[name](root, agent_id)
