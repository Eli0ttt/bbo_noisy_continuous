from __future__ import annotations

from pathlib import Path
from typing import Any


TOOL_SPEC: dict[str, dict[str, Any]] = {
    "python_interpreter": {
        "status": "implemented",
        "scope": "offline numerical reasoning",
        "allowed_imports": [
            "numpy", "numpy.random", "numpy.linalg",
            "math", "statistics"
        ],
        "network": False,
        "filesystem": False,
        "subprocess": False,
    },
    "python_code_interpreter": {
        "status": "implemented",
        "scope": "offline numerical reasoning",
        "allowed_imports": ["numpy", "math", "statistics"],
        "network": False,
        "filesystem": False,
        "subprocess": False,
    },
    "bbo_read_scratch": {
        "status": "pending_executor",
        "scope": "read-only files in agent scratch directory",
        "network": False,
        "filesystem": "scratch-read-only",
        "subprocess": False,
    },
    "bbo_candidate_preflight": {
        "status": "pending_executor",
        "scope": "compile and inspect candidate solver in scratch",
        "network": False,
        "filesystem": "scratch-read-only",
        "subprocess": False,
        "official_selfcheck": False,
    },
}

# 当前还没有真正接入执行器。
EXECUTABLE_TOOLS: set[str] = {
    "python_code_interpreter",
    "python_interpreter",
}


def _agent_entries(config: dict[str, Any]):
    agents = config.get("agents", [])

    if isinstance(agents, dict):
        return [
            (str(name), agent)
            for name, agent in agents.items()
            if isinstance(agent, dict)
        ]

    if isinstance(agents, list):
        result = []
        for index, agent in enumerate(agents):
            if isinstance(agent, dict):
                name = agent.get("name") or agent.get("id") or f"agent_{index}"
                result.append((str(name), agent))
        return result

    return []


def validate_tools(config: dict[str, Any]) -> list[dict[str, Any]]:
    violations = []

    for agent_name, agent in _agent_entries(config):
        tools = agent.get("tools", []) or []

        if not isinstance(tools, list):
            violations.append({
                "agent": agent_name,
                "tool": None,
                "reason": "tools must be a list",
            })
            continue

        for tool in tools:
            tool_name = str(tool)

            if tool_name not in TOOL_SPEC:
                violations.append({
                    "agent": agent_name,
                    "tool": tool_name,
                    "reason": "unknown_tool",
                })
            elif tool_name not in EXECUTABLE_TOOLS:
                violations.append({
                    "agent": agent_name,
                    "tool": tool_name,
                    "reason": "executor_not_wired",
                })

    return violations


def audit_config(path: str | Path) -> dict[str, Any]:
    import yaml

    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text()) or {}
    violations = validate_tools(config)

    return {
        "config": str(config_path),
        "valid_for_execution": not violations,
        "declared_tools": {
            name: spec["status"]
            for name, spec in TOOL_SPEC.items()
        },
        "violations": violations,
    }
