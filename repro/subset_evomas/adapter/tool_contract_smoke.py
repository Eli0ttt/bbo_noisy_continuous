#!/usr/bin/env python3
"""No-API check: actual smolagents Tool objects through EvoMAS's runner."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2] / "EvoMAS-openai"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.agents.spec import AgentSpec
from src.tools.bbo_generation import CATALOG, inspect_candidate
import src.agents.runners.smolagents as runner_module
import src.models


class FakeAgent:
    def __init__(self, tools, model, additional_authorized_imports):
        self.tools = {tool.name: tool for tool in tools}
        assert set(self.tools) == set(CATALOG)

    def run(self, task):
        assert self.tools["bbo_read_context"].forward("instruction.md") == "public task"
        assert self.tools["bbo_read_context"].forward("../secret").startswith("DENIED")
        assert json.loads(self.tools["bbo_read_public_data"].forward("summary"))["n_instances"] == 12
        assert self.tools["bbo_read_prior_feedback"].forward("summary").startswith("NO_PRIOR_VERSION")
        assert self.tools["bbo_read_prior_trace"].forward("checkpoints").startswith("NO_PRIOR_VERSION")
        assert self.tools["bbo_read_history"].forward("index").startswith("NO_PRIOR_VERSION")
        source = (
            "import numpy as np\nclass Optimizer:\n"
            " def __init__(self, dim, lower, upper, budget, seed, rng): pass\n"
            " def ask(self, n): pass\n"
            " def tell(self, X, y, metadata=None): pass\n"
        )
        assert self.tools["bbo_write_candidate"].forward(source).startswith("WROTE")
        assert self.tools["bbo_preflight_candidate"].forward("candidate.py").startswith("PASS")
        def fake_docker(command, **kwargs):
            if command[1] == "cp":
                return subprocess.CompletedProcess(command, 0, "", "")
            if "python" in command:
                return subprocess.CompletedProcess(command, 0,
                    'CONTRACT_PASS\nPROBE_JSON:{"valid":true,"cases":[{"unique":120,"observed":120}],"warnings":[]}\n', "")
            return subprocess.CompletedProcess(command, 0, "", "")
        with patch("subprocess.run", side_effect=fake_docker):
            assert self.tools["bbo_probe_candidate"].forward("candidate.py").startswith("PASS")
        return "mock agent report"


def main():
    old_model = src.models.get_model
    old_agent = runner_module.CodeAgent
    try:
        src.models.get_model = lambda *args, **kwargs: SimpleNamespace(smolagents_model=object())
        runner_module.CodeAgent = FakeAgent
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "tool_workspace").mkdir()
            (root / "tool_workspace" / "instruction.md").write_text("public task")
            runner = runner_module.SmolagentsRunner()
            runner.bbo_tool_root = root
            runner.bbo_eval_container = "offline-visible"
            spec = AgentSpec(id="worker1", role="worker", model_id="openai:deepseek-v4-flash-s2",
                             tools=list(CATALOG), prompt="worker")
            result = runner.run(spec, "offline test")
            assert result.success, result.error
            events = [json.loads(line) for line in (root / "tool_audit.jsonl").read_text().splitlines()]
            assert len([e for e in events if e["event"] == "loaded"]) == len(CATALOG)
            assert len([e for e in events if e["event"] == "called"]) == 9
            assert inspect_candidate("class Optimizer: pass").startswith("FAIL")
    finally:
        src.models.get_model = old_model
        runner_module.CodeAgent = old_agent
    print("PASS: YAML tool names instantiate, restricted calls work, and audit records usage")


if __name__ == "__main__":
    main()
