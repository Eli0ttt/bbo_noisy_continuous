"""Harbor agent that stages one selected solver for the separate official verifier."""
from __future__ import annotations
import hashlib
import shlex
from typing import Any
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class EvoSubmissionAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "evomas-bbo-submission"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        check = await environment.exec(
            "test -f /opt/evomas-selected/solver.py && test -d /app/methods/main",
            cwd="/app", timeout_sec=120,
        )
        if check.return_code != 0:
            raise RuntimeError("Selected solver mount or submission directory is missing: "
                               + check.stderr[-1200:])

    def __init__(self, *args: Any, solver_sha256: str, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if len(solver_sha256) != 64 or any(c not in '0123456789abcdef' for c in solver_sha256):
            raise ValueError('Invalid solver SHA-256')
        self.solver_sha256 = solver_sha256

    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None:
        code = (
            "import hashlib,pathlib; "
            "src=pathlib.Path('/opt/evomas-selected/solver.py'); "
            "dst=pathlib.Path('/app/methods/main/solver.py'); "
            "assert src.is_file() and not src.is_symlink(); "
            "data=src.read_bytes(); "
            "assert hashlib.sha256(data).hexdigest()==" + repr(self.solver_sha256) + "; "
            "dst.write_bytes(data); "
            "assert hashlib.sha256(dst.read_bytes()).hexdigest()==" + repr(self.solver_sha256)
        )
        result = await environment.exec("python -c " + shlex.quote(code),
                                        cwd="/app", timeout_sec=120)
        if result.return_code != 0:
            raise RuntimeError("Could not stage selected solver: " + result.stderr[-1200:])
